import subprocess
import yaml
from pathlib import Path


def main():
    # 指向新版 YAML 配置文件
    config_path = Path("lora_configs/lora_config_llama31_v2.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = config.get("model", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    data = config.get("data", "training_data/emotion_data_v2")
    adapter_path = config.get("adapter_path", "adapters_llama31_v2")
    iters = config.get("iters", 900)
    batch_size = config.get("batch_size", 2)
    grad_accum = config.get("grad_accumulation_steps", 4)
    effective_batch = batch_size * grad_accum
    epoch_steps = 1800 // effective_batch
    total_epochs = iters // grad_accum // epoch_steps

    print("🎭 开始训练情感陪伴模型 v2（Llama 3.1 8B）...")
    print(f"模型：{model}")
    print(f"数据：{data}")
    print(f"配置：{config_path}")
    print(f"适配器输出：{adapter_path}")
    print(f"训练迭代：{iters} | effective batch：{effective_batch} | 约 {total_epochs:.1f} epoch")
    print("预计时间：1-2 小时（M1 Max 64GB）\n")

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--config", str(config_path)
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败，退出码：{e.returncode}")
        raise

    print(f"\n✅ 训练完成！适配器保存在：{adapter_path}")


if __name__ == "__main__":
    main()
