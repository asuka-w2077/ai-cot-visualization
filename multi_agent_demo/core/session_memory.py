import re
from typing import List, Tuple, Dict


class SessionMemory:
    """维护多轮对话历史和关键事实提取"""

    def __init__(self):
        self.history: List[Tuple[str, str]] = []  # [(user, assistant), ...]
        self.facts: Dict[str, str] = {}           # {"女朋友名字": "Rae", ...}

    def add_turn(self, user_text: str, assistant_text: str):
        self.history.append((user_text, assistant_text))
        self._extract_facts(user_text)

    def _extract_facts(self, user_text: str):
        """规则式提取关键实体（更严格的版本）"""
        # 只处理明确介绍名字/称呼的句式，避免把质问、反问当名字提取

        # 模式1: (我女朋友/我男友/她/他) + (叫/名字是) + 名字
        # 要求"叫"或"名字"出现在名字之前，且整句是陈述语气
        pattern_role_name = re.compile(
            r"(?P<role>我女朋友|我男友|我女友|我男友|女朋友|男友|女友|男朋友|她|他|它|我的猫|我的狗)\s*(?:的名字|名字)?\s*(?:叫|是)\s*(?P<name>[A-Za-z\u4e00-\u9fa5]{1,10})[吧啊呢\s]*(?:$|[，。！？])"
        )
        m = pattern_role_name.search(user_text)
        if m:
            role_raw = m.group("role")
            name = m.group("name")
            role_map = {
                "我女朋友": "女朋友名字", "女朋友": "女朋友名字",
                "我女友": "女朋友名字", "女友": "女朋友名字",
                "我男友": "男朋友名字", "男朋友": "男朋友名字",
                "我男友": "男朋友名字", "男友": "男朋友名字",
                "她": "她名字", "他": "他名字", "它": "它名字",
                "我的猫": "猫名字", "我的狗": "狗名字",
            }
            role_label = role_map.get(role_raw, f"{role_raw}名字")
            self.facts[role_label] = name
            return

        # 模式2: 以后/就 + 叫 + 名字（要求前面有明确的被称呼对象）
        pattern_nickname = re.compile(
            r"(?:以后|就|从前|从前)\s*(?:叫|称|喊)\s*(?P<name>[A-Za-z\u4e00-\u9fa5]{1,10})[吧啊呢\s]*(?:$|[，。！？])"
        )
        m = pattern_nickname.search(user_text)
        if m:
            name = m.group("name")
            # 只有当前面句子明确提到了某人时，才把这个名字关联上去
            # 简单推断：如果之前已经提取过某个角色名字，这里可能是别名；否则不提取
            if self.facts:
                # 不覆盖已有信息，避免误提取
                pass
            return

        # 模式3: 我叫 xxx（用户自己名字）
        pattern_self = re.compile(
            r"(?:我|本人)\s*(?:叫|名字是|是)\s*(?P<name>[A-Za-z\u4e00-\u9fa5]{1,10})[吧啊呢\s]*(?:$|[，。！？])"
        )
        m = pattern_self.search(user_text)
        if m:
            name = m.group("name")
            self.facts["用户名字"] = name
            return

    def build_system_prompt(self, base_prompt: str) -> str:
        prompt = base_prompt
        if self.facts:
            facts_text = "\n".join([f"- {k}: {v}" for k, v in self.facts.items()])
            prompt += f"\n\n【当前会话中已知的信息】\n{facts_text}\n\n请注意：如果没有已知信息支撑，不要编造任何细节。"
        return prompt
