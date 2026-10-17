#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
依赖安装（建议在虚拟环境中）：
  pip install -U "mlx-lm>=0.20.0" "huggingface_hub>=0.23.0"

说明：
- MLX 目前通常通过 HuggingFace 上的 MLX 量化权重仓库加载 4bit 版本（mlx-community）。
- 使用 MLX 默认设备映射，适配 Mac M1 Max。
"""

import re
import sys

from mlx_lm import load, generate


MODEL_ID_4BIT = "mlx-community/Qwen2.5-3B-Instruct-4bit"


PROMPT = "将以下工作记录整理为周报格式：本周做了登录优化和会员文档"


SYSTEM_INSTRUCTION = (
    "你是一个严谨的周报生成助手。"
    "你必须严格只输出三个板块，且顺序固定、标题完全一致："
    "【本周进展】、【遇到的问题】、【下周计划】。"
    "每个板块都必须有内容，使用条目化（例如 1. 2. 3. 或 -）。"
    "禁止输出任何其他标题、前后缀说明、客套话或多余文本。"
)


def build_chat_prompt(user_text: str) -> str:
    # 兼容 Llama 系列常见的 chat 模板（简化版）。
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_INSTRUCTION}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_text}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def is_strict_weekly_format(text: str) -> bool:
    text = text.strip()
    pattern = (
        r"^\s*【本周进展】[\s\S]+"
        r"【遇到的问题】[\s\S]+"
        r"【下周计划】[\s\S]+$"
    )
    if not re.match(pattern, text):
        return False
    # 不允许出现第四个同类标题
    extra = re.findall(r"【[^】]+】", text)
    return extra == ["【本周进展】", "【遇到的问题】", "【下周计划】"]


# 替换整个 main 函数为简化版
def main() -> int:
    print("🚀 加载 Qwen 2.5 3B...")
    model, tokenizer = load(MODEL_ID_4BIT)
    print("✅ 加载成功\n")
    
    prompt = build_chat_prompt("本周做了登录优化和会员文档")
    
    print("📝 生成周报...")
    print("-" * 50)
    
    # 只生成一次，不修复
    out = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=200,  # 减少长度避免冗余
    )
    
    # 清理特殊 tokens
    out = out.split("<|")[0].strip()
    
    print(out)
    print("-" * 50)
    
    # 评估
    if "【本周进展】" in out and "【遇到的问题】" in out and "【下周计划】" in out:
        print("✅ 格式检查通过")
    else:
        print("⚠️ 格式不完整")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())