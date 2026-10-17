# DeepSeek 云端 API 接入指南

本文档说明如何将本地多 Agent 情感陪伴系统接入 **DeepSeek** 云端大模型，实现「本地 3B+8B 兜底 + 云端 DeepSeek-V3 增强」的双轨运行。

---

## 1. 获取 API Key

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册账号
3. 进入「API Keys」→「创建 API Key」
4. 复制 Key（格式为 `sk-xxxxxxxxxxxxxxxx`）

> ⚠️ **安全提醒**：API Key 仅在创建时可见，请妥善保存。不要提交到 Git，不要硬编码在代码中。

---

## 2. 配置 API Key

### 方式一：环境变量（推荐）

```bash
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"
python web_app.py
```

Mac/Linux 可写入 `~/.zshrc` 或 `~/.bashrc` 永久生效：

```bash
echo 'export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"' >> ~/.zshrc
source ~/.zshrc
```

### 方式二：启动时临时传入

```bash
DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx" python web_app.py
```

---

## 3. 验证是否接入成功

启动服务后，在 Web UI 左下角切换到 **「云端 DeepSeek」**，发送任意消息：

- ✅ **成功**：回复开头没有 `【云端大模型未启用】` 字样，且响应正常
- ❌ **失败**：显示认证失败或网络错误，请检查 Key 是否正确配置

也可在命令行快速验证：

```bash
cd multi_agent_demo
python -c "
from core import CloudClient
c = CloudClient(provider='deepseek')
print('云端已启用:', c.enabled)
if c.enabled:
    r = c.chat([{'role': 'user', 'content': '你好'}])
    print('回复:', r)
"
```

---

## 4. 模型选择策略（Web UI 中的两个选项）

| 选项 | 路由逻辑 | 适用场景 |
|---|---|---|
| **本地 Agent** | Qwen 3B 意图识别 → Llama 8B 情感回复 / 云端工具查询 | 日常情感陪伴，零 API 费用，响应快 |
| **云端 DeepSeek** | 跳过本地意图识别，直接把完整上下文给 DeepSeek-V3 | 复杂问题、长上下文、需要强理解力 |

### 什么时候选「云端 DeepSeek」？

- 用户在倾诉很长的一段心事，需要深度共情
- 用户问天气/翻译/计算等工具问题（本地也会自动走云端，但云端模式上下文更完整）
- 本地模型回复质量不够满意，想对比云端的效果

### 什么时候选「本地 Agent」？

- 日常闲聊、简单情绪倾诉
- 不想产生 API 费用（DeepSeek API 价格已很低，但本地完全免费）
- 网络不稳定或担心隐私（纯本地运行）

---

## 5. 路由策略详解

无论选哪个模型，**安全层（SafetyGuard）始终优先执行**：

```
用户输入
    │
    ▼
┌─────────────────┐
│  SafetyGuard    │  ← 危机关键词 / 边界请求
│  安全检查       │     （自杀/自残/医疗/转账/隐私/违法）
└─────────────────┘
    │
    ├─ 危机 ──→ 本地兜底标准回复 或 云端 crisis_response
    │
    ├─ 边界 ──→ 本地拒绝回复
    │
    └─ 正常 ──→ 继续
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
   「本地 Agent」        「云端 DeepSeek」
        │                    │
   Qwen 3B 意图识别     跳过意图识别
        │                    │
   Llama 8B 回复        DeepSeek-V3 直接回复
   或云端工具查询        （带完整上下文+记忆）
```

---

## 6. 费用说明

DeepSeek API 价格非常便宜（以官网最新为准）：

| 模型 | 输入费用 | 输出费用 | 特点 |
|---|---|---|---|
| `deepseek-chat` (V3) | 极低 | 极低 | 默认模型，通用对话，性价比极高 |
| `deepseek-reasoner` (R1) | 低 | 低 | 推理模型，带思维链，适合复杂问题 |

### 省钱技巧

1. **日常用本地**：情感陪伴场景下，本地 Llama 8B 质量已足够好
2. **工具查询走云端**：本地模式会自动识别天气/搜索等意图并调用云端
3. **切换模型**：如需更强的推理能力，可改为 `deepseek-reasoner`

如需切换模型，修改 `web_app.py` 中初始化 `CloudClient` 时的参数：

```python
cloud_client = CloudClient(
    provider="deepseek",
    model="deepseek-reasoner",  # 改为 R1 推理模型
)
```

---

## 7. 故障排查

| 现象 | 原因 | 解决方案 |
|---|---|---|
| `【云端大模型未启用】` | 未配置 `DEEPSEEK_API_KEY` | 按第 2 节配置环境变量 |
| `【认证失败】API Key 无效` | Key 错误或已过期 | 去平台重新创建 Key |
| `【请求过于频繁】429` | 触发限流 | 等几秒再试 |
| `【网络错误】` | 无法访问 api.deepseek.com | 检查网络/代理 |
| `【云端服务异常】500` | DeepSeek 服务器问题 | 稍后再试，或切本地模式 |

---

## 8. 隐私与合规

- **本地模式**：所有数据仅在本地 M1 Max 上处理，**零外泄**
- **云端模式**：用户输入和对话历史会发送到 DeepSeek 服务器
- **危机场景**：即使选本地模式，危机关键词检测也会触发

---

## 9. 技术细节（开发者参考）

### 多 Provider 支持

`CloudClient` 已内置支持 Moonshot 和 DeepSeek，可自由切换：

```python
from core import CloudClient

# DeepSeek（当前默认）
c = CloudClient(provider="deepseek", api_key="sk-xxx")

# Moonshot / Kimi
c = CloudClient(provider="moonshot", api_key="sk-xxx")

# 自定义其他 OpenAI-compatible 服务
c = CloudClient(
    provider="custom",
    api_key="sk-xxx",
    base_url="https://api.xxx.com/v1/chat/completions",
)
```

### 纯标准库实现

`cloud_client.py` 仅依赖 Python 标准库（`urllib`、`json`、`os`），无需安装 `requests` 或 `openai` 包。

---

如有问题，请检查：
1. `DEEPSEEK_API_KEY` 是否已正确导出
2. Key 是否在 [DeepSeek 平台](https://platform.deepseek.com/) 处于有效状态
3. 账号是否有足够余额（新账号通常有赠送额度）
