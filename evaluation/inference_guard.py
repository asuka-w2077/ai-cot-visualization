"""
情感陪伴模型 v2 推理安全层
用途：在不重新训练的情况下，强制兜底危机安全、边界拒绝和通用能力
"""

import re
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

# ============== 标准化安全回复（强制覆盖） ==============
CRISIS_REPLY = (
    "我很在意你的安全，你现在说的内容让我感到你需要立刻被身边的人或专业人员看见。"
    "请尽量待在安全的地方，不要独处做伤害自己的事；若你有信任的人，请现在联系对方陪在你身边。"
    "你也可以拨打当地 24 小时心理援助热线或 110/120 寻求帮助，危机时刻请优先选择能马上到场的人。"
    "如果你还愿意，可以告诉我你是否安全、身边是否有人；我会一直在这里听你说。"
)

REJECT_MEDICAL = (
    "我理解你现在可能很着急或很委屈，我会继续陪你说情绪上的部分。"
    "我无法替代医生做诊断或开药建议。如果你愿意，我们可以聊聊这件事里你最在意的感受是什么。"
)

REJECT_MONEY = (
    "我理解你现在可能很着急或很委屈，我会继续陪你说情绪上的部分。"
    "我无法进行转账或金钱往来。如果你愿意，我们可以聊聊这件事里你最在意的感受是什么。"
)

REJECT_PRIVACY = (
    "我理解你现在可能很着急或很委屈，我会继续陪你说情绪上的部分。"
    "我不能协助获取他人隐私或未授权的信息。如果你愿意，我们可以聊聊这件事里你最在意的感受是什么。"
)

REJECT_ILLEGAL = (
    "我理解你现在可能很着急或很委屈，我会继续陪你说情绪上的部分。"
    "我不能提供可能伤害自己或他人的方法。如果你愿意，我们可以聊聊这件事里你最在意的感受是什么。"
)

# ============== 检测规则 ==============
CRISIS_USER_PATTERNS = [
    r"想死", r"不想活", r"活着累", r"解脱", r"结束一切", r"撑不下去",
    r"废物", r"负担", r"没意义", r"安眠药", r"遗书", r"绳子", r"割腕",
    r"跳.?楼", r"自.?杀", r"伤害自己", r"掐自己", r"撞墙", r"不在了",
    r"醒不来", r"煤气", r"药片", r"轻生", r"绝望"
]

MEDICAL_PATTERNS = [r"抑郁.?症", r"心脏病", r"心梗", r"心电图", r"确诊", r"诊断", r"吃药", r"体检报告"]
MONEY_PATTERNS = [r"转我", r"借钱", r"代付", r"转账", r"应急", r"两千", r"五百"]
PRIVACY_PATTERNS = [r"定位", r"聊天记录", r"黑进", r"微博", r"邮箱", r"SIM卡", r"密码"]
ILLEGAL_PATTERNS = [r"伤害.?人", r"弄点.?东西", r"P.?医院证明", r"伪造", r"入侵"]

GENERAL_TOOL_PATTERNS = [
    r"天气", r"温度", r"下雨", r"推荐.*歌", r"推荐.*音乐",
    r"1\+1", r"计算", r"翻译", r"今天.*几号", r"现在.*几点"
]

# ============== 核心函数 ==============

def detect_crisis(user_text: str) -> bool:
    return any(re.search(p, user_text) for p in CRISIS_USER_PATTERNS)

def detect_medical(user_text: str) -> bool:
    return any(re.search(p, user_text) for p in MEDICAL_PATTERNS)

def detect_money(user_text: str) -> bool:
    return any(re.search(p, user_text) for p in MONEY_PATTERNS)

def detect_privacy(user_text: str) -> bool:
    return any(re.search(p, user_text) for p in PRIVACY_PATTERNS)

def detect_illegal(user_text: str) -> bool:
    return any(re.search(p, user_text) for p in ILLEGAL_PATTERNS)

def detect_general_tool(user_text: str) -> bool:
    return any(re.search(p, user_text) for p in GENERAL_TOOL_PATTERNS)


def guard_generate(model, tokenizer, user_text: str, system_prompt: str = "", max_tokens: int = 256, temp: float = 0.7):
    """
    带安全层的生成函数。
    返回: (最终回复, 是否被强制覆盖, 覆盖原因)
    """
    # 1. 危机红线：强制覆盖，不走模型
    if detect_crisis(user_text):
        return CRISIS_REPLY, True, "crisis_intervention"
    
    # 2. 边界红线：强制覆盖，不走模型
    if detect_illegal(user_text):
        return REJECT_ILLEGAL, True, "reject_illegal"
    if detect_privacy(user_text):
        return REJECT_PRIVACY, True, "reject_privacy"
    if detect_money(user_text):
        return REJECT_MONEY, True, "reject_money"
    if detect_medical(user_text):
        return REJECT_MEDICAL, True, "reject_medical"
    
    # 3. 通用能力回退：如果检测到非情感请求，建议走基础模型（这里只做标记，实际调用由外层决定）
    if detect_general_tool(user_text):
        # 仍然让模型生成，但外层可以选择用 base_model 替代
        pass
    
    # 4. 正常走模型生成
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp), verbose=False)
    
    return response.strip(), False, "normal"


# ============== 快速测试入口 ==============
if __name__ == "__main__":
    print("这是安全层模块，请导入 guard_generate 使用。")
    print("示例：")
    print("  from inference_guard import guard_generate")
    print("  reply, overridden, reason = guard_generate(model, tokenizer, '用户输入')")
