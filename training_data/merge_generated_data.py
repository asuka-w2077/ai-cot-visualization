"""
把外部 AI 生成的数据合并到 emotion_data_v2 训练目录中。
用法：
    python training_data/merge_generated_data.py --input generated_data.jsonl

要求：
    - 输入文件必须是 JSONL 格式
    - 每条记录必须是 {"messages": [...]} 格式
    - 会自动按 9:1 拆分为 train/valid，并追加到现有 emotion_data_v2 中
"""

import json
import argparse
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path("training_data/emotion_data_v2")
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="外部生成的 JSONL 文件路径")
    parser.add_argument("--ratio", type=float, default=0.9, help="训练集占比")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    new_records = load_jsonl(args.input)
    random.shuffle(new_records)
    split_idx = int(len(new_records) * args.ratio)
    train_new = new_records[:split_idx]
    valid_new = new_records[split_idx:]

    # 追加写入
    save_jsonl(TRAIN_FILE, train_new)
    save_jsonl(VALID_FILE, valid_new)

    # 统计
    train_all = load_jsonl(TRAIN_FILE)
    valid_all = load_jsonl(VALID_FILE)

    print(f"✅ 已合并 {len(new_records)} 条新数据")
    print(f"   -> 训练集新增：{len(train_new)} 条（总计：{len(train_all)} 条）")
    print(f"   -> 验证集新增：{len(valid_new)} 条（总计：{len(valid_all)} 条）")
    print(f"\n当前目标：总计 2000 条（train 1800 + valid 200）")
    print(f"目前进度：{len(train_all)} train / {len(valid_all)} valid = {len(train_all) + len(valid_all)} 条")
    print(f"还差：{2000 - len(train_all) - len(valid_all)} 条")


if __name__ == "__main__":
    main()
