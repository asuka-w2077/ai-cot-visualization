from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from .session_memory import SessionMemory


class EmotionAgent:
    """情感陪伴 Agent：加载 Llama3.1-8B，支持会话记忆和多轮对话"""

    def __init__(
        self,
        model_path: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        adapter_path: str = None,
        emotion_prompt: str = "",
        casual_prompt: str = "",
    ):
        print(f"🔄 加载情感陪伴模型: {model_path}")
        if adapter_path:
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
            print(f"   已加载 Adapter: {adapter_path}")
        else:
            self.model, self.tokenizer = load(model_path)
            print("   使用基础模型（无 Adapter）")
        self.emotion_prompt = emotion_prompt
        self.casual_prompt = casual_prompt

    def respond(self, user_input: str, memory: SessionMemory, is_casual: bool = False) -> str:
        system_prompt = memory.build_system_prompt(
            self.casual_prompt if is_casual else self.emotion_prompt
        )

        messages = [{"role": "system", "content": system_prompt}]
        for u, a in memory.history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_input})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return generate(
            self.model, self.tokenizer, prompt,
            max_tokens=256, sampler=make_sampler(temp=0.7), verbose=False
        ).strip()
