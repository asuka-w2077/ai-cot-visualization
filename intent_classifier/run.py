#!/usr/bin/env python3
# run.py
from intent_agent import IntentRecognitionAgent, MLXClient

# 配置：你的模型路径
MODEL_PATH = "mlx-community/Qwen2.5-3B-Instruct-4bit"  # 或你的本地路径

def main():
    # 初始化（会加载模型，第一次较慢）
    client = MLXClient(MODEL_PATH)
    agent = IntentRecognitionAgent(client)
    
    # 测试用例
    tests = [
        "明天北京天气怎么样",
        "我这几天心情特别低落，感觉活着没意思",
        "帮我打开微信",
        "你好"
    ]
    
    print("=" * 50)
    for text in tests:
        print(f"\n👤 输入: {text}")
        result = agent.recognize(text)
        print(f"🤖 意图: {result.intent} (置信度: {result.confidence:.2f})")
        print(f"   情绪: {result.emotion_signal} | 紧急: {result.urgency}")
        print(f"   🚦 云端接管: {'是' if agent.should_handoff_to_cloud(result) else '否'}")

if __name__ == "__main__":
    main()