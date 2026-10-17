import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"
DATA_DIR = Path.home() / "ai-cot-visualization"  # /Users/niaofeiliao/ai-cot-visualization
TRAIN_FILE = DATA_DIR / "training_data" / "500_train.jsonl"

# Training hyperparams (as requested)
EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

# LoRA hyperparams (as requested)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

ADAPTER_DIR = Path("adapters")


def count_jsonl_examples(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # basic validation (optional but helps catch broken lines early)
            json.loads(line)
            n += 1
    return n


def build_yaml_config(iters: int) -> str:
    # NOTE: mlx-lm uses `lora_parameters.rank/scale/dropout` (no CLI flags).
    # We map: r -> rank, alpha -> scale, dropout -> dropout.
    return f"""\
model: "{MODEL_NAME}"
train: true
fine_tune_type: lora

data: "{DATA_DIR.as_posix()}"

batch_size: {BATCH_SIZE}
iters: {iters}
learning_rate: {LEARNING_RATE}

adapter_path: "{ADAPTER_DIR.as_posix()}"

# Reasonable defaults; adjust if you hit memory limits
num_layers: 16
max_seq_length: 2048
grad_checkpoint: false
grad_accumulation_steps: 1

lora_parameters:
  keys: ["self_attn.q_proj", "self_attn.v_proj"]
  rank: {LORA_R}
  scale: {LORA_ALPHA}
  dropout: {LORA_DROPOUT}
"""


def build_yaml_config_with_data_dir(iters: int, data_dir: Path) -> str:
    return build_yaml_config(iters).replace(
        f'data: "{DATA_DIR.as_posix()}"',
        f'data: "{data_dir.as_posix()}"',
    )


def main() -> None:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"未找到训练数据文件：{TRAIN_FILE.resolve()}（默认从当前目录读取 ./train.jsonl）"
        )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    n_examples = count_jsonl_examples(TRAIN_FILE)
    if n_examples <= 0:
        raise ValueError(f"{TRAIN_FILE} 为空或没有有效 JSONL 行。")

    steps_per_epoch = math.ceil(n_examples / BATCH_SIZE)
    iters = EPOCHS * steps_per_epoch

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # mlx_lm 本地训练默认读取数据目录下的 train.jsonl。
    # 你的源文件名是 500_train.jsonl，这里自动复制为临时 train.jsonl。
    with tempfile.TemporaryDirectory(prefix="mlx_data_") as tmp_dir:
        tmp_data_dir = Path(tmp_dir)
        tmp_train_file = tmp_data_dir / "train.jsonl"
        shutil.copy2(TRAIN_FILE, tmp_train_file)

        config_path = Path("lora_config.generated.yaml")
        config_path.write_text(
            build_yaml_config_with_data_dir(iters, tmp_data_dir), encoding="utf-8"
        )

        # Use the recommended entrypoint: `python -m mlx_lm lora ...`
        cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    print(f"\n训练完成。LoRA 适配器已保存到：{ADAPTER_DIR.resolve()}")


if __name__ == "__main__":
    main()

