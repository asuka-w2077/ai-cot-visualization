from mlx_lm import load, generate

# 基础模型路径
BASE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
ADAPTER_PATH = "adapters"

def test_model(model, tokenizer, label):
    print(f"\n{'='*20} {label} {'='*20}")
    prompt = """将以下记录整理为周报：本周做了登录优化和会员文档,大部分时间在摸鱼（这个不需要被总结），另外我还在研究大模型微调
格式要求：【本周进展】+【遇到的问题】+【下周计划】"""

    response = generate(model, tokenizer, prompt, max_tokens=200)
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
# 在 test_finetuned.py 最后增加：

print("\n🧪 过拟合测试：用训练集外的输入测试")

test_prompts = [
    "周报：这周啥正经事没干，就天天开会",  # 负面/极端口语
    "总结一下：完成了AI助手的设计，还搞定了数据迁移",  # 关键词不同（总结vs周报）
    "本周工作：1.优化首页 2.修bug 3.写文档",  # 已经是结构化输入
]

for p in test_prompts:
    print(f"\n输入：{p}")
    r = generate(model_finetuned, tokenizer, p, max_tokens=200)
    print(f"输出：{r[:100]}...")
    
    # 检查格式
    if "【本周进展】" in r and "【遇到的问题】" in r:
        print("✅ 格式遵循")
    else:
        print("❌ 格式丢失（可能过拟合）")