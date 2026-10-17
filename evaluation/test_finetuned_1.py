from mlx_lm import load, generate
import re

# 基础模型路径
BASE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
ADAPTER_PATH = "adapters"

SYSTEM_RULE = """你是一个周报生成助手，必须严格按以下格式输出，不要解释：
【本周进展】
1. [事项]：[成果]，耗时[X天]
【遇到的问题】
[问题或无]
【下周计划】
[计划]
禁止输出任何中括号占位符（如[事项]、[成果]、[问题或无]、[计划]），必须替换为具体内容。
不得编造输入中未出现的事项；事项名称必须来自用户输入或是对输入事项的等价改写。
只输出一次完整结果，输出结束后立即停止，不要重复任何段落。"""


def build_prompt(user_text: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_RULE}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def postprocess_response(text: str) -> str:
    # 若模型继续生成到特殊结束符之后，先截断
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    text = text.strip()

    # 若出现重复生成（第二个【本周进展】），截掉重复部分
    first = text.find("【本周进展】")
    if first != -1:
        second = text.find("【本周进展】", first + 1)
        if second != -1:
            text = text[:second].rstrip()

    return text


def run_once(model, tokenizer, user_text: str, max_tokens: int = 140) -> str:
    prompt = build_prompt(user_text)
    raw = generate(model, tokenizer, prompt, max_tokens=max_tokens)
    return postprocess_response(raw)


def has_all_sections(text: str) -> bool:
    return all(k in text for k in ["【本周进展】", "【遇到的问题】", "【下周计划】"])


def has_duration(text: str) -> bool:
    # 匹配“耗时3天 / 耗时1.5天 / 耗时[3天]”等常见写法
    return re.search(r"耗时\s*\[?\d+(\.\d+)?\s*天\]?", text) is not None


def test_model(model, tokenizer, label):
    print(f"\n{'='*20} {label} {'='*20}")
    user_text = "将以下记录整理为周报：本周做了登录优化和会员文档"
    response = run_once(model, tokenizer, user_text, max_tokens=140)
    print(response)
    print("="*50)

# 测试1：加载基础模型（微调前）
print("🔄 加载基础模型（微调前）...")
model_base, tokenizer = load(BASE_MODEL)
test_model(model_base, tokenizer, "微调前（Base）")

# 测试2：加载微调后的模型（加载适配器）
print("\n🔄 加载微调后模型（加载 LoRA 适配器）...")
model_finetuned, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)
test_model(model_finetuned, tokenizer, "微调后（LoRA）")

print("\n🧪 过拟合测试：用训练集外的输入测试")

test_prompts = [
    "周报：这周啥正经事没干，就天天开会",  # 负面/极端口语
    "总结一下：完成了AI助手的设计，还搞定了数据迁移",  # 关键词不同（总结vs周报）
    "本周工作：1.优化首页 2.修bug 3.写文档",  # 已经是结构化输入
]

total = len(test_prompts)
ok_sections = 0
ok_duration = 0

for p in test_prompts:
    print(f"\n输入：{p}")
    r = run_once(model_finetuned, tokenizer, p, max_tokens=140)
    print(f"输出：{r[:100]}...")
    
    # 检查格式
    if has_all_sections(r):
        ok_sections += 1
        print("✅ 格式遵循")
    else:
        print("❌ 格式丢失（可能过拟合）")

    if has_duration(r):
        ok_duration += 1

print("\n📊 字段覆盖率统计（LoRA）")
print(f"- 三段齐全率: {ok_sections}/{total} = {ok_sections / total:.0%}")
print(f"- 含耗时率: {ok_duration}/{total} = {ok_duration / total:.0%}")