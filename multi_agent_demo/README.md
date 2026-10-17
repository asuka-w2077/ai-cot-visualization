# 多 Agent 情感陪伴系统

一个基于本地 Apple Silicon（M 系列芯片）运行的轻量级多 Agent 情感陪伴方案。

## 架构设计

```
用户输入
    │
    ▼
┌─────────────────┐
│  Qwen2.5-3B     │  意图识别（本地，低延迟）
│  意图路由 Agent   │
└─────────────────┘
    │
    ├─ crisis_intervention ──► 🚨 标准化安全回复（强制规则，不走 LLM）
    │
    ├─ tool_query ───────────► ☁️ 云端 70B / 外部 API（Kimi/Moonshot）
    │
    ├─ emotion_support ──────► ❤️ Llama3.1-8B 情感陪伴（基础模型 + 硬约束 Prompt）
    │
    ├─ casual_chat ──────────► 💬 Llama3.1-8B 轻松闲聊（独立 Prompt）
    │
    └─ mixed ────────────────► 🔄 无明确工具词时降级为 emotion_support
```

## 核心特性

- **边缘小模型路由**：3B Qwen 负责意图分类，响应快、占用内存小
- **基础模型情感陪伴**：8B Llama 不加微调 Adapter，避免编造记忆和语气污染
- **会话记忆**：自动维护多轮对话历史 + 提取关键事实（如女朋友名字）
- **安全硬兜底**：危机干预和边界拒绝由规则层强制控制，不依赖生成模型
- **云端弹性扩展**：工具查询和复杂危机场景可转交云端大模型

## 目录结构

```
multi_agent_demo/
├── core/                      # 可复用类库
│   ├── session_memory.py      # 会话记忆与事实提取
│   ├── intent_router.py       # 意图识别封装
│   ├── emotion_agent.py       # 情感陪伴模型封装
│   ├── safety_guard.py        # 危机/边界安全兜底
│   └── cloud_client.py        # 云端 API 客户端
├── web_app.py                 # Web UI（内置 http.server，零依赖）
├── cli_demo.py                # 命令行交互演示
├── multi_agent_demo.py        # 旧版单体 CLI 演示（已被 cli_demo.py 取代）
├── emotion_system_prompt.txt  # 情感陪伴 system prompt
├── casual_chat_prompt.txt     # 闲聊 system prompt
├── CLOUD_SETUP.md             # 云端 API 接入指南
└── README.md                  # 本文档
```

## 环境要求

- macOS + Apple Silicon（M1/M2/M3 Max 等，推荐 32GB 以上统一内存）
- Python 3.11
- `mlx` conda 环境，已安装 `mlx-lm`

## 快速开始

### 1. 命令行交互

```bash
cd multi_agent_demo
python cli_demo.py
```

### 2. Web 界面

```bash
cd multi_agent_demo
python web_app.py
```

浏览器打开 `http://127.0.0.1:8080` 即可使用。

### 3. 接入云端 API（可选）

默认情况下，云端 Agent 会返回提示文本。如需启用真实的 Kimi / Moonshot 云端调用：

```bash
export MOONSHOT_API_KEY="你的 API Key"
python web_app.py
```

## 内存占用参考

| 模型 | 显存/内存占用 |
|------|--------------|
| Qwen2.5-3B-Instruct-4bit | ~2 GB |
| Meta-Llama-3.1-8B-Instruct-4bit | ~4–5 GB |
| 总计（同时加载） | ~6–7 GB |

M1 Max 64GB 可轻松同时运行两个模型并有余量。

## 设计决策记录

### 为什么放弃 v2 Adapter？

我们在 `adapters_llama31_v2` 上进行了 LoRA 微调（1690 条数据），但测试发现三个致命缺陷：

1. **编造记忆**：模型会随机生成从未发生过的"往事"（如"上次你说心脏疼"），严重破坏信任。
2. **危机误触**：普通闲聊也会被触发危机干预模板，用户体验极差。
3. **语气污染**：回复过度文艺化（"行走在玻璃路上""和谐的坟堆"），不像自然口语。

因此最终方案改为：**基础 Llama3.1-8B + 强约束 system prompt + 工程规则兜底**，既保留了模型的通用诚实性，又通过 prompt 和后处理控制了陪伴风格。

## 已知限制

- **记忆仅限单会话**：退出程序后历史清空。跨会话长期记忆需要外接数据库。
- **事实提取是规则式**：只能提取简单的"名字"类事实，复杂关系/事件暂不支持。
- **云端依赖 API Key**：没有配置 `MOONSHOT_API_KEY` 时，工具查询只会返回占位提示。
- **多轮过长会截断**：当前 `max_tokens=256`，如果对话历史超过模型上下文长度（1024–2048 token），早期轮次会被截断。

## 后续可扩展方向

- 接入本地向量数据库（如 Chroma），实现真正的长期记忆检索
- 加入 TTS / STT，支持语音对话
- 为 crisis_intervention 接入真实的人工值班或自动报警系统
- 云端 API 支持多厂商 failover（Kimi / OpenAI / Claude）

## 协议

仅供学习和个人研究使用。涉及心理健康的内容，请务必在真实产品中配置专业人工审核和危机干预机制。
