# ai-cot-visualization

在 Apple Silicon（M 系列芯片）上，用 MLX 框架本地微调 Llama-3.1-8B 情感陪伴模型，并搭建一套**多 Agent 编排**的完整 demo：小模型做意图路由、大模型做情感陪伴、云端 API 做工具与危机兜底。

> 本项目是学习型 demo：从零生成训练数据 → MLX LoRA 微调 → 微调前后对比评估 → 多 Agent 联调 → Web UI 呈现，全链路可在本地跑通。

## 架构

### 推理期：多 Agent 编排

```
用户输入
    │
    ▼
┌──────────────────┐
│  SafetyGuard      │  规则层：危机关键词 / 边界请求，强制兜底，不走 LLM
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  IntentRouter     │  Qwen2.5-3B-4bit（本地 MLX）输出结构化意图 JSON
└──────────────────┘
    │
    ├─ crisis_intervention ──► 🚨 标准化安全回复（或转云端 crisis 接口）
    ├─ boundary ─────────────► 🚧 固定拒绝话术（规则层处理）
    ├─ tool / realtime ──────► ☁️ CloudClient → DeepSeek / Moonshot
    │                           （抓取 wttr.in 天气 / DuckDuckGo 搜索后投喂）
    └─ casual / emotion ─────► ❤️ EmotionAgent → Llama-3.1-8B（本地 MLX）
                                  + SessionMemory 会话记忆注入 system prompt
```

### 训练期：数据 → 微调 → 评估

```
手工种子样本 / 外部 AI 批量生成数据
        │
        ▼
data_generation/  +  training_data/     # 模板组合生成、去重、硬校验、9:1 拆分
        │
        ▼
lora_configs/ + finetuning/             # mlx_lm lora --config xxx.yaml
        │                                  Llama-3.1-8B-4bit，LoRA r=16
        ▼
evaluation/                             # 微调前后 20 用例对比（危机/边界/记忆/混合）
```

## 目录结构

```
├── multi_agent_demo/        # 多 Agent 联调演示：CLI + Web UI + core/ 类库
├── intent_classifier/       # 意图识别 Agent（Qwen2.5-3B，5 类意图 + 置信度）
├── finetuning/              # MLX LoRA 微调脚本（Qwen 实验线 + Llama 主线）
├── lora_configs/            # LoRA 训练配置（v1: 300 条数据 / v2: 1690 条数据）
├── training_data/           # 训练/验证数据（jsonl）+ 合并脚本
├── data_generation/         # 训练数据生成脚本（模板组合 + 硬校验）
├── evaluation/              # 微调前后对比评估脚本与报告
├── web_ui/                  # CoT 可视化演示页（需本地 Ollama 服务）
├── Kimi_Agent_情感陪伴数据生成/   # 外部 AI 生成的分类对话数据（1700 条）
├── Deepseek 情感陪伴数据生成/     # 外部 AI 生成的对话数据（23 个文件）
└── adapters_llama31_v2/     # v2 最终 LoRA 权重（Git LFS，52MB）
```

更详细的模块文档见 [multi_agent_demo/README.md](multi_agent_demo/README.md) 与
[multi_agent_demo/CLOUD_SETUP.md](multi_agent_demo/CLOUD_SETUP.md)。

## 快速开始

### 环境要求

- macOS + Apple Silicon（M1/M2/M3，推荐 32GB 以上统一内存）
- Python 3.11（建议 conda）
- 可选：本地 Ollama（仅 web_ui 演示页需要）

```bash
conda create -n mlx python=3.11 -y && conda activate mlx
pip install -r requirements.txt
```

首次运行时会自动从 HuggingFace 下载 4bit 模型（Qwen2.5-3B 约 2GB，Llama-3.1-8B 约 4.5GB）。

### 命令行演示

```bash
cd multi_agent_demo
python cli_demo.py            # 基础模型
python cli_demo.py --with-adapter   # 加载 v2 LoRA 权重（见「已知问题」）
```

### Web 界面

```bash
cd multi_agent_demo
python web_app.py
# 浏览器打开 http://127.0.0.1:8080
```

### 接入云端 API（可选）

工具查询与低置信度意图默认返回占位提示。配置任一 Key 即可启用云端：

```bash
export DEEPSEEK_API_KEY="sk-..."     # 默认 provider
export MOONSHOT_API_KEY="sk-..."     # 备选 provider
```

## 微调复现

```bash
python finetuning/finetune_emotion_8b_v2.py   # 读 lora_configs/lora_config_llama31_v2.yaml
# 或直接：
python -m mlx_lm lora --config lora_configs/lora_config_llama31_v2.yaml
```

v2 配置：Llama-3.1-8B-4bit，LoRA（q/k/v/o_proj，r=16，α=32），
数据 `training_data/emotion_data_v2/`（1521 训练 + 169 验证，messages 格式），
900 迭代，M1 Max 64GB 约 1–2 小时。

## 设计决策：为什么最终放弃 v2 Adapter

在 `adapters_llama31_v2` 上做完 LoRA 微调（1690 条数据）后，评估发现三个致命缺陷：

1. **编造记忆**：随机生成从未发生过的"往事"，破坏信任；
2. **危机误触**：普通闲聊也触发危机干预模板；
3. **语气污染**：回复过度文艺化，不像自然口语。

最终方案改为**基础 Llama-3.1-8B + 强约束 system prompt + 工程规则兜底**。
v2 权重仍随仓库保留（Git LFS），便于复现对比实验。

## 模型权重说明

| 目录 | 说明 | 是否入库 |
|------|------|---------|
| `adapters_llama31_v2/adapters.safetensors` | v2 最终 LoRA 权重 | ✅ Git LFS |
| `adapters_llama31_v2/0000*.safetensors` | v2 中间 checkpoint | ❌ 体积过大 |
| `adapters_llama31/` | v1 实验线权重（572MB） | ❌ 历史实验线 |
| `adapters/` | Qwen 周报实验线权重（325MB） | ❌ 历史实验线 |

## 已知问题

以下为代码审查中确认、但**尚未修复**的问题（欢迎 PR）：

1. `evaluation/compare_before_after_n1.py` 文件末尾被截断，存在语法错误，无法运行；
2. `data_generation/generate_emotion_seed.py` 引用了未定义的 `IM_START` / `IM_END`，直接运行会在第一条记录处 `NameError`；
3. `multi_agent_demo/cli_demo.py` 与 `web_app.py` 中 adapter 路径多取了一层目录，`--with-adapter` 模式会加载失败；
4. `multi_agent_demo/core/intent_router.py` 通过 `sys.path` 引用仓库根目录的 `intent_classifier/`，**必须以仓库根目录为工作目录运行**；
5. `web_app.py` 的会话记忆保存在前端页面，刷新即丢失；
6. `finetuning/finetune_lora_mlx_500.py` 的 `DATA_DIR` 硬编码为另一台机器的路径；
7. `README` 中提到的"Web UI 云端模型切换"功能未实现（对应代码为预留的死代码）。

## 协议

[MIT](LICENSE) — 仅供学习和个人研究使用。涉及心理健康的内容，请务必在真实产品中配置专业人工审核和危机干预机制。微调权重基于 Meta Llama 3.1，使用需遵守 Meta 的许可协议。
