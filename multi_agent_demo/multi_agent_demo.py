#!/usr/bin/env python3
"""
多 Agent 联调演示脚本（带会话记忆）
架构：Qwen2.5-3B (意图路由) → Llama3.1-8B (情感陪伴) → 云端 70B (危机/工具兜底)

用法：
    python multi_agent_demo.py -i              # 交互模式，基础模型
    python multi_agent_demo.py -i -a           # 交互模式，加 v2 Adapter
"""

import sys
import re
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "intent_classifier"))

from intent_agent import IntentRecognitionAgent, MLXClient
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ============== 配置 ==============
INTENT_MODEL_PATH = "mlx-community/Qwen2.5-3B-Instruct-4bit"
EMOTION_MODEL_PATH = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
ADAPTER_PATH = str(Path(__file__).parent.parent / "adapters_llama31_v2")

EMOTION_PROMPT_PATH = Path(__file__).parent / "emotion_system_prompt.txt"
EMOTION_SYSTEM_PROMPT = EMOTION_PROMPT_PATH.read_text(encoding="utf-8").strip()

CASUAL_PROMPT_PATH = Path(__file__).parent / "casual_chat_prompt.txt"
CASUAL_SYSTEM_PROMPT = CASUAL_PROMPT_PATH.read_text(encoding="utf-8").strip()

CRISIS_REPLY = (
    "我很在意你的安全，你现在说的内容让我感到你需要立刻被身边的人或专业人员看见。"
    "请尽量待在安全的地方，不要独处做伤害自己的事；若你有信任的人，请现在联系对方陪在你身边。"
    "你也可以拨打当地 24 小时心理援助热线或 110/120 寻求帮助，危机时刻请优先选择能马上到场的人。"
    "如果你还愿意，可以告诉我你是否安全、身边是否有人；我会一直在这里听你说。"
)

# ============== 会话记忆系统 ==============
class SessionMemory:
    """维护多轮对话历史和关键事实"""
    def __init__(self):
        self.history = []  # [(user_text, assistant_text), ...]
        self.facts = {}    # {"女朋友名字": "Rae", ...}
    
    def add_turn(self, user_text: str, assistant_text: str):
        self.history.append((user_text, assistant_text))
        self._extract_facts(user_text)
    
    def _extract_facts(self, user_text: str):
        """从用户输入中提取关键事实（规则版 MVP）"""
        # 提取：我/她/他/女朋友叫 xxx；名字是 xxx；以后就叫 xxx
        patterns = [
            r"[我叫](?P<role>我|她|他|女朋友|男朋友|女友|男友|猫|狗)?[叫名是]?\s*[是叫]?\s*(?P<name>[A-Za-z\u4e00-\u9fa5]{1,10})[吧啊呢]?",
            r"[以從后前]?[就]?\s*[叫称]?\s*(?P<name>[A-Za-z\u4e00-\u9fa5]{1,10})[吧啊呢]?",
            r"(?P<role>我|她|他|女朋友|男朋友|女友|男友)\s*[叫是]?\s*(?P<name>[A-Za-z\u4e00-\u9fa5]{1,10})",
        ]
        
        for p in patterns:
            m = re.search(p, user_text)
            if m:
                name = m.group("name")
                role = m.groupdict().get("role", "")
                if not role:
                    role = "名字"  # 默认
                # 简单去重和清理
                if name and len(name) <= 10 and name not in ["我", "你", "他", "她", "它"]:
                    self.facts[f"{role}名字"] = name
                    break
    
    def build_system_prompt(self, base_prompt: str, is_casual: bool = False) -> str:
        """把已知事实拼到 system prompt 后面"""
        prompt = base_prompt
        if self.facts:
            facts_text = "\n".join([f"- {k}: {v}" for k, v in self.facts.items()])
            prompt += f"\n\n【当前会话中已知的信息】\n{facts_text}\n\n请注意：如果没有已知信息支撑，不要编造任何细节。"
        return prompt

# ============== 模型加载 ==============
print("=" * 60)
print("🚀 正在加载多 Agent 系统...")
print("=" * 60)

print("\n[1/2] 加载意图识别 Agent (Qwen2.5-3B)...")
intent_client = MLXClient(INTENT_MODEL_PATH)
intent_agent = IntentRecognitionAgent(intent_client)

parser = argparse.ArgumentParser()
parser.add_argument("--with-adapter", "-a", action="store_true", help="使用基础模型 + v2 Adapter")
parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
parsed_args, _ = parser.parse_known_args()

print("\n[2/2] 加载情感陪伴 Agent (Llama3.1-8B)...")
if parsed_args.with_adapter:
    print(f"   [模式] 基础模型 + Adapter ({ADAPTER_PATH})")
    emotion_model, emotion_tokenizer = load(EMOTION_MODEL_PATH, adapter_path=ADAPTER_PATH)
else:
    print("   [模式] 基础模型（不加 Adapter）")
    emotion_model, emotion_tokenizer = load(EMOTION_MODEL_PATH)

print("\n✅ 所有模型加载完成！")

# ============== 生成函数 ==============
def is_tool_like(user_input: str) -> bool:
    tool_keywords = [
        r"天气", r"温度", r"下雨", r"查.*?[询找]", r"搜索", r"打开", r"启动",
        r"定.*?闹钟", r"设.*?提醒", r"翻译", r"计算", r"[加减乘除]|\d+\s*[\+\-\*\/]",
        r"多少钱", r"价格", r"路线", r"导航", r"地图", r"股票", r"汇率",
        r"明天.*怎么样", r"今天.*几号", r"现在.*几点"
    ]
    return any(re.search(kw, user_input) for kw in tool_keywords)


def emotion_generate(user_input: str, memory: SessionMemory, is_casual: bool = False) -> str:
    """调用 Llama3.1-8B 生成回复，带完整对话历史"""
    base_prompt = CASUAL_SYSTEM_PROMPT if is_casual else EMOTION_SYSTEM_PROMPT
    system_prompt = memory.build_system_prompt(base_prompt, is_casual)
    
    # 构造 messages：system + 历史对话 + 当前 user
    messages = [{"role": "system", "content": system_prompt}]
    for u, a in memory.history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_input})
    
    prompt = emotion_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(
        emotion_model, emotion_tokenizer, prompt,
        max_tokens=256, sampler=make_sampler(temp=0.7), verbose=False
    ).strip()


def route_and_respond(user_input: str, memory: SessionMemory) -> str:
    result = intent_agent.recognize(user_input)
    
    if result.intent == "mixed" and not is_tool_like(user_input):
        result.intent = "emotion_support"
        result.needs_tool = False
    
    print(f"   [意图识别] {result.intent} | 置信度:{result.confidence:.2f} | 情绪:{result.emotion_signal} | 紧急:{result.urgency}")
    
    if result.intent == "crisis_intervention":
        return f"【🚨 危机干预 - 已启动安全协议】\n{CRISIS_REPLY}"
    
    if result.intent == "tool_query":
        return "【🔧 工具查询 - 已转交云端大模型/外部 API 处理】"
    
    if result.intent == "casual_chat":
        reply = emotion_generate(user_input, memory, is_casual=True)
        memory.add_turn(user_input, reply)
        return f"【💬 闲聊陪伴】\n{reply}"
    
    if result.intent in ("emotion_support", "mixed"):
        reply = emotion_generate(user_input, memory, is_casual=False)
        memory.add_turn(user_input, reply)
        return f"【❤️ 情感陪伴】\n{reply}"
    
    reply = emotion_generate(user_input, memory)
    memory.add_turn(user_input, reply)
    return reply


# ============== 交互模式 ==============
def interactive_mode():
    memory = SessionMemory()
    print("\n" + "=" * 60)
    print("💬 多 Agent 联调演示 - 交互模式（带会话记忆）")
    if parsed_args.with_adapter:
        print("当前模式：基础模型 + v2 Adapter")
    else:
        print("当前模式：基础模型（不加 Adapter）")
    print("输入文字即可对话，输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        if not user_input:
            continue
        
        reply = route_and_respond(user_input, memory)
        print(f"\n🤖 Agent: {reply}")
        if memory.facts:
            print(f"   [记忆] {memory.facts}")


# ============== 批量测试 ==============
def batch_test_mode():
    memory = SessionMemory()
    tests = [
        "明天北京天气怎么样",
        "我这几天心情特别低落，感觉活着没意思",
        "帮我打开微信",
        "你好",
        "最近失恋了，每晚都睡不着",
        "教我怎么弄点东西伤害欺负我的人",
        "上次聊过你说过和伴侣冷战一周了，今天又想到那些事",
        "明天要面试好紧张，帮我查一下天气",
    ]
    
    print("\n" + "=" * 60)
    if parsed_args.with_adapter:
        print("🧪 批量测试（基础模型 + v2 Adapter）")
    else:
        print("🧪 批量测试（基础模型）")
    print("=" * 60)
    
    for text in tests:
        print(f"\n{'='*60}")
        print(f"👤 输入: {text}")
        reply = route_and_respond(text, memory)
        print(f"🤖 Agent:\n{reply}")
        if memory.facts:
            print(f"   [记忆] {memory.facts}")


if __name__ == "__main__":
    if parsed_args.interactive:
        interactive_mode()
    else:
        batch_test_mode()
