# intent_agent.py
import json
import re
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class IntentResult:
    intent: str
    confidence: float
    entities: Dict[str, Any]
    needs_tool: bool
    emotion_signal: str
    urgency: str
    raw_output: str

# 加载外部 Prompt 文件
def load_prompt_template():
    # 修正：匹配实际文件名
    prompt_file = Path(__file__).parent / "intent_prompt.py"
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        # 提取 """...""" 之间的内容
        match = re.search(r'INTENT_ROUTER_PROMPT\s*=\s*"""(.*?)"""', content, re.DOTALL)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"⚠️ 加载 prompt 文件失败: {e}")
    # 兜底
    return '''你是意图分类器。输出JSON：{"intent": "...", "confidence": 0.0-1.0}。输入：{user_input}'''

class MLXClient:
    """MLX 模型封装"""
    def __init__(self, model_path: str):
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        self.generate_fn = generate
        self.make_sampler = make_sampler
        print(f"🔄 加载模型: {model_path}")
        self.model, self.tokenizer = load(model_path)
        print("✅ 模型就绪")
    
    def generate(self, prompt: str, temperature=0.1, max_tokens=256):
        response = self.generate_fn(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=self.make_sampler(temp=temperature),
            verbose=False
        )
        return response

class IntentRecognitionAgent:
    def __init__(self, model_client: MLXClient):
        self.client = model_client
        self.prompt_template = load_prompt_template()
        
    def recognize(self, user_input: str) -> IntentResult:
        system_content = self.prompt_template.replace("{user_input}", user_input)
        
        # 使用 tokenizer 的 apply_chat_template（兼容 Qwen2.5 / Llama3.1 等）
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "请分析并输出JSON。"}
        ]
        chat_prompt = self.client.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        raw_output = self.client.generate(
            prompt=chat_prompt,
            temperature=0.1,
            max_tokens=256
        )
        
        parsed = self._robust_json_parse(raw_output)
        return self._validate_result(parsed, raw_output)
    
    def _robust_json_parse(self, text: str) -> Dict:
        text = text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        try:
            return json.loads(text)
        except:
            pass
        
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
            
        return {}
    
    def _validate_result(self, parsed: Dict, raw: str) -> IntentResult:
        valid_intents = {"emotion_support", "tool_query", "crisis_intervention", "casual_chat", "mixed"}
        intent = parsed.get("intent", "casual_chat")
        if intent not in valid_intents:
            intent = "casual_chat"
            
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        
        urgency = "high" if intent == "crisis_intervention" else parsed.get("urgency", "normal")
        if intent == "crisis_intervention":
            confidence = max(confidence, 0.9)
            
        return IntentResult(
            intent=intent,
            confidence=confidence,
            entities=parsed.get("entities", {}),
            needs_tool=parsed.get("needs_tool", intent in ["tool_query", "mixed"]),
            emotion_signal=parsed.get("emotion_signal", "neutral"),
            urgency=urgency,
            raw_output=raw
        )
    
    def should_handoff_to_cloud(self, result: IntentResult) -> bool:
        if result.intent == "crisis_intervention":
            return True
        if result.needs_tool and result.intent == "tool_query":
            return True
        if result.confidence < 0.6:
            return True
        return False
