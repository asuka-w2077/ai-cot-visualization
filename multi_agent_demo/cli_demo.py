#!/usr/bin/env python3
"""
命令行交互演示（使用新的 core 类库）
"""

from pathlib import Path
from core import SessionMemory, SafetyGuard, CloudClient, IntentRouter, EmotionAgent

INTENT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
EMOTION_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
ADAPTER_PATH = str(Path(__file__).parent / "adapters_llama31_v2")

EMOTION_PROMPT = (Path(__file__).parent / "emotion_system_prompt.txt").read_text(encoding="utf-8").strip()
CASUAL_PROMPT = (Path(__file__).parent / "casual_chat_prompt.txt").read_text(encoding="utf-8").strip()


def main(use_adapter: bool = False):
    print("=" * 60)
    print("🚀 正在初始化 Agent 系统...")
    print("=" * 60)

    intent_router = IntentRouter(INTENT_MODEL)
    cloud_client = CloudClient()
    emotion_agent = EmotionAgent(
        EMOTION_MODEL,
        adapter_path=ADAPTER_PATH if use_adapter else None,
        emotion_prompt=EMOTION_PROMPT,
        casual_prompt=CASUAL_PROMPT,
    )

    memory = SessionMemory()
    mode = "基础模型 + Adapter" if use_adapter else "基础模型"
    print(f"\n✅ 初始化完成 | 当前模式: {mode}")
    print("输入文字即可对话，输入 'quit' 或 'exit' 退出\n")

    while True:
        user_input = input("👤 你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        if not user_input:
            continue

        # 安全层
        if SafetyGuard.check_crisis(user_input):
            reply = SafetyGuard.get_reply("crisis")
            intent_str = "crisis_intervention"
        else:
            is_boundary, boundary_type = SafetyGuard.check_boundary(user_input)
            if is_boundary:
                reply = SafetyGuard.get_reply("boundary", boundary_type)
                intent_str = "boundary_refusal"
            else:
                result = intent_router.recognize(user_input)
                intent_str = result.intent

                # mixed 降级
                if intent_str == "mixed":
                    import re
                    tool_keywords = [
                        r"天气", r"温度", r"下雨", r"查.*?[询找]", r"搜索", r"打开", r"启动",
                        r"定.*?闹钟", r"设.*?提醒", r"翻译", r"计算",
                        r"多少钱", r"价格", r"路线", r"导航", r"地图",
                    ]
                    if not any(re.search(kw, user_input) for kw in tool_keywords):
                        intent_str = "emotion_support"
                        result.needs_tool = False

                if intent_router.should_use_cloud(result) or intent_str == "tool_query":
                    reply = cloud_client.tool_response(user_input)
                elif intent_str == "casual_chat":
                    reply = emotion_agent.respond(user_input, memory, is_casual=True)
                    memory.add_turn(user_input, reply)
                else:
                    reply = emotion_agent.respond(user_input, memory, is_casual=False)
                    memory.add_turn(user_input, reply)

        print(f"   [意图] {intent_str} | [记忆] {memory.facts}")
        print(f"🤖 Agent: {reply}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-adapter", "-a", action="store_true", help="加载 v2 Adapter")
    args = parser.parse_args()
    main(use_adapter=args.with_adapter)
