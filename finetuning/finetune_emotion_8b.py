import subprocess
import yaml
from pathlib import Path


def main():
    # 指向 YAML 配置文件，使三个文件真正产生交互
    config_path = Path("lora_configs/lora_config_llama31.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = config.get("model", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    data = config.get("data", "training_data/emotion_300_train.jsonl")
    adapter_path = config.get("adapter_path", "adapters_llama31")
    iters = config.get("iters", 5150)

    print("🎭 开始训练情感陪伴模型（Llama 3.1 8B）...")
    print(f"模型：{model}")
    print(f"数据：{data}")
    print(f"配置：{config_path}")
    print(f"适配器输出：{adapter_path}")
    print(f"训练迭代：{iters}")
    print("预计时间：2-4 小时（视硬件而定）\n")

    # 通过命令行调用 mlx_lm.lora，并传入 YAML 配置
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
