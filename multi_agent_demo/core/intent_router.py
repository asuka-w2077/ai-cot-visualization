import sys
from pathlib import Path

# 将仓库根目录下的 intent_classifier 加入路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "intent_classifier"))

from intent_agent import IntentRecognitionAgent, MLXClient, IntentResult


class IntentRouter:
    """封装意图识别逻辑，对外提供简洁接口"""

    def __init__(self, model_path: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"):
        self.client = MLXClient(model_path)
        self.agent = IntentRecognitionAgent(self.client)

    def recognize(self, user_input: str) -> IntentResult:
        return self.agent.recognize(user_input)

    def should_use_cloud(self, result: IntentResult) -> bool:
        """判断是否应转交云端处理"""
        if result.intent == "crisis_intervention":
            return True
        if result.needs_tool and result.intent == "tool_query":
            return True
        if result.confidence < 0.6:
            return True
        return False
