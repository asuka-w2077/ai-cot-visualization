import re
from typing import Tuple, Optional


class SafetyGuard:
    """危机安全与边界拒绝的兜底层"""

    CRISIS_REPLY = (
        "我很在意你的安全，你现在说的内容让我感到你需要立刻被身边的人或专业人员看见。"
        "请尽量待在安全的地方，不要独处做伤害自己的事；若你有信任的人，请现在联系对方陪在你身边。"
        "你也可以拨打当地 24 小时心理援助热线或 110/120 寻求帮助，危机时刻请优先选择能马上到场的人。"
        "如果你还愿意，可以告诉我你是否安全、身边是否有人；我会一直在这里听你说。"
    )

    BOUNDARY_REPLIES = {
        "medical": (
            "我理解你现在可能很着急或很委屈，我会继续陪你说情绪上的部分。"
            "我无法替代医生做诊断或开药建议。如果你愿意，我们可以聊聊这件事里你最在意的感受是什么。"
        ),
        "money": (
            "我理解你现在可能很着急或很委屈，我会继续陪你说情绪上的部分。"
            "我无法进行转账或金钱往来。如果你愿意，我们可以聊聊这件事里你最在意的感受是什么。"
        ),
        "privacy": (
            "我理解你现在可能很着急或很委屈，我会继续陪你说情绪上的部分。"
            "我不能协助获取他人隐私或未授权的信息。如果你愿意，我们可以聊聊这件事里你最在意的感受是什么。"
        ),
        "illegal": (
            "我理解你现在可能很着急或很委屈，我会继续陪你说情绪上的部分。"
            "我不能提供可能伤害自己或他人的方法。如果你愿意，我们可以聊聊这件事里你最在意的感受是什么。"
        ),
    }

    CRISIS_PATTERNS = [
        r"想死", r"不想活", r"活着累", r"解脱", r"结束一切", r"撑不下去",
        r"废物", r"负担", r"没意义", r"安眠药", r"遗书", r"绳子", r"割腕",
        r"跳.?楼", r"自.?杀", r"伤害自己", r"掐自己", r"撞墙", r"不在了",
        r"醒不来", r"煤气", r"药片", r"轻生", r"绝望"
    ]

    BOUNDARY_PATTERNS = {
        "medical": [r"抑郁.?症", r"心脏病", r"心梗", r"心电图", r"确诊", r"诊断", r"吃药", r"体检报告"],
        "money": [r"转我", r"借钱", r"代付", r"转账", r"金钱"],
        "privacy": [r"定位", r"聊天记录", r"黑进", r"微博", r"邮箱", r"SIM卡", r"密码"],
        "illegal": [r"伤害.?人", r"弄点.?东西", r"P.?医院证明", r"伪造", r"入侵"],
    }

    @classmethod
    def check_crisis(cls, user_input: str) -> bool:
        return any(re.search(p, user_input) for p in cls.CRISIS_PATTERNS)

    @classmethod
    def check_boundary(cls, user_input: str) -> Tuple[bool, Optional[str]]:
        for boundary_type, patterns in cls.BOUNDARY_PATTERNS.items():
            if any(re.search(p, user_input) for p in patterns):
                return True, boundary_type
        return False, None

    @classmethod
    def get_reply(cls, guard_type: str, boundary_subtype: Optional[str] = None) -> str:
        if guard_type == "crisis":
            return cls.CRISIS_REPLY
        if guard_type == "boundary" and boundary_subtype:
            return cls.BOUNDARY_REPLIES.get(boundary_subtype, cls.BOUNDARY_REPLIES["illegal"])
        return ""
