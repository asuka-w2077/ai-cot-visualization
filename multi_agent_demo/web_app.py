#!/usr/bin/env python3
"""
极简 Web UI：基于 Python 内置 http.server
无需安装 Flask/FastAPI，直接运行即可

用法：
    python web_app.py          # 默认端口 8080
    python web_app.py --port 9000
"""

import json
import argparse
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from core import SessionMemory, SafetyGuard, CloudClient, IntentRouter, EmotionAgent

# ============== 配置 ==============
INTENT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
EMOTION_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
ADAPTER_PATH = str(Path(__file__).parent / "adapters_llama31_v2")

EMOTION_PROMPT = (Path(__file__).parent / "emotion_system_prompt.txt").read_text(encoding="utf-8").strip()
CASUAL_PROMPT = (Path(__file__).parent / "casual_chat_prompt.txt").read_text(encoding="utf-8").strip()

# ============== 全局 Agent 实例（启动时加载） ==============
print("🚀 正在初始化 Agent 系统...")
intent_router = IntentRouter(INTENT_MODEL)
cloud_client = CloudClient(provider="deepseek")
# 默认不加 Adapter；如需测试 Adapter，可手动改成 adapter_path=ADAPTER_PATH
emotion_agent = EmotionAgent(
    EMOTION_MODEL,
    adapter_path=None,
    emotion_prompt=EMOTION_PROMPT,
    casual_prompt=CASUAL_PROMPT,
)
print("✅ 初始化完成，Web 服务即将启动\n")


# ============== 请求处理器 ==============
class ChatHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        if self.path == "/":
            self._set_headers("text/html; charset=utf-8")
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(body)

        user_input = data.get("user_input", "").strip()
        history = data.get("history", [])  # [{user, assistant, intent}, ...]

        # 重建 SessionMemory
        memory = SessionMemory()
        for turn in history:
            if "user" in turn and "assistant" in turn:
                memory.add_turn(turn["user"], turn["assistant"])
                # 恢复已知事实（如果前端传了）
                if "facts" in turn:
                    memory.facts.update(turn["facts"])

        # ===== 自动路由：意图识别决定本地/云端 =====

        # 1. 安全层优先检查
        if SafetyGuard.check_crisis(user_input):
            # 危机场景走云端 DeepSeek，回复质量优于本地兜底模板
            reply = cloud_client.crisis_response(user_input, history)
            intent_str = "crisis_intervention"
        else:
            is_boundary, boundary_type = SafetyGuard.check_boundary(user_input)
            if is_boundary:
                reply = SafetyGuard.get_reply("boundary", boundary_type)
                intent_str = "boundary_refusal"
            else:
                # 2. 意图识别（本地 Qwen 3B）
                result = intent_router.recognize(user_input)
                intent_str = result.intent

                # mixed 降级：没有明确工具关键词 → 按情感陪伴处理
                if intent_str == "mixed":
                    tool_keywords = [
                        r"天气", r"温度", r"下雨", r"查.*?[询找]", r"搜索", r"打开", r"启动",
                        r"定.*?闹钟", r"设.*?提醒", r"翻译", r"计算",
                        r"多少钱", r"价格", r"路线", r"导航", r"地图",
                    ]
                    if not any(re.search(kw, user_input) for kw in tool_keywords):
                        intent_str = "emotion_support"
                        result.needs_tool = False

                # 需要实时信息的问题（日期、时间、天气、新闻等），本地模型无法回答，强制走云端
                realtime_keywords = [
                    r"几号", r"日期", r"星期几", r"周几", r"哪天",
                    r"几点", r"现在.*时间", r"当前.*时间",
                    r"天气", r"温度", r"下雨", r"下雪", r"空气质量",
                    r"新闻", r"最近.*事",
                ]
                needs_realtime = any(re.search(kw, user_input) for kw in realtime_keywords)

                # 3. 路由决策
                if needs_realtime or intent_router.should_use_cloud(result) or intent_str == "tool_query":
                    # 需要实时信息 / 危机 / 工具查询 / 低置信度 → 云端 DeepSeek
                    reply = cloud_client.tool_response(user_input, history)
                elif intent_str == "casual_chat":
                    # 闲聊 → 本地 Llama 8B（轻量 prompt）
                    reply = emotion_agent.respond(user_input, memory, is_casual=True)
                    memory.add_turn(user_input, reply)
                else:
                    # 情感支持及其他 → 本地 Llama 8B（完整 prompt）
                    reply = emotion_agent.respond(user_input, memory, is_casual=False)
                    memory.add_turn(user_input, reply)

        response = {
            "reply": reply,
            "intent": intent_str,
            "facts": memory.facts,
            "history_count": len(memory.history),
        }

        self._set_headers()
        self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        # 简化日志输出
        pass


# ============== HTML 页面（完全内联） ==============
HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>多 Agent 情感陪伴系统</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body {
    height: 100%;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans SC", sans-serif;
    background: #f6f6f6;
    color: #1a1a1a;
}
#app {
    max-width: 800px;
    margin: 0 auto;
    height: 100vh;
    display: flex;
    flex-direction: column;
    background: #fff;
}

/* 顶部简洁标题 */
#header {
    padding: 12px 20px;
    text-align: center;
    font-size: 15px;
    font-weight: 600;
    color: #1a1a1a;
    border-bottom: 1px solid #f0f0f0;
    background: #fff;
    flex-shrink: 0;
}
#header span {
    font-size: 12px;
    color: #999;
    font-weight: 400;
    margin-left: 8px;
}

/* 聊天区域 */
#chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 20px 16px 120px;
    display: flex;
    flex-direction: column;
    gap: 24px;
}

/* 消息行 */
.message-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    max-width: 90%;
}
.message-row.user {
    align-self: flex-end;
    flex-direction: row-reverse;
}
.message-row.agent {
    align-self: flex-start;
}

/* 头像 */
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: 600;
    color: #fff;
}
.avatar.user {
    background: #e0e0e0;
    color: #666;
}
.avatar.agent {
    background: #4a90ff;
}

/* 消息气泡 */
.bubble {
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 15px;
    line-height: 1.6;
    word-break: break-word;
    white-space: pre-wrap;
}
.bubble.user {
    background: #f0f0f0;
    color: #1a1a1a;
    border-bottom-right-radius: 4px;
}
.bubble.agent {
    background: #f8f8f8;
    color: #1a1a1a;
    border-bottom-left-radius: 4px;
}

/* 元信息（意图/记忆） */
.meta {
    font-size: 11px;
    color: #bbb;
    margin-top: 4px;
    padding-left: 48px;
}
.message-row.user .meta {
    text-align: right;
    padding-right: 48px;
    padding-left: 0;
}

/* 正在思考中卡片 */
.thinking-card {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #f8f8f8;
    border: 1px solid #eee;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #666;
    cursor: pointer;
    transition: background 0.2s;
}
.thinking-card:hover {
    background: #f0f0f0;
}
.thinking-card .icon {
    font-size: 16px;
}
.thinking-card .arrow {
    margin-left: auto;
    font-size: 12px;
    color: #999;
    transition: transform 0.2s;
}
.thinking-card.expanded .arrow {
    transform: rotate(90deg);
}
.thinking-detail {
    display: none;
    margin-top: 8px;
    padding: 8px 12px;
    background: #fff;
    border-radius: 8px;
    font-size: 12px;
    color: #888;
    line-height: 1.5;
}
.thinking-detail.show {
    display: block;
}

/* 底部输入区 */
#input-area {
    position: fixed;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100%;
    max-width: 800px;
    background: #fff;
    border-top: 1px solid #f0f0f0;
    padding: 12px 16px 24px;
}
#input-card {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    background: #f8f8f8;
    border: 1px solid #e8e8e8;
    border-radius: 24px;
    padding: 8px 12px;
    min-height: 48px;
}
#plus-btn {
    width: 32px;
    height: 32px;
    border: none;
    background: transparent;
    font-size: 22px;
    color: #666;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
#user-input {
    flex: 1;
    border: none;
    background: transparent;
    font-size: 15px;
    line-height: 1.5;
    outline: none;
    resize: none;
    max-height: 120px;
    padding: 6px 4px;
    font-family: inherit;
}
#model-select {
    font-size: 13px;
    color: #666;
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 4px 8px;
    flex-shrink: 0;
}
#send-btn {
    width: 36px;
    height: 36px;
    border: none;
    border-radius: 50%;
    background: #1a1a1a;
    color: #fff;
    font-size: 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background 0.2s;
}
#send-btn:hover {
    background: #333;
}
#send-btn:disabled {
    background: #ccc;
    cursor: not-allowed;
}

/* 底部免责声明 */
#disclaimer {
    text-align: center;
    font-size: 12px;
    color: #bbb;
    margin-top: 8px;
}

/* 欢迎消息 */
.welcome-msg {
    text-align: center;
    color: #999;
    font-size: 14px;
    margin-top: 40px;
}
</style>
</head>
<body>
<div id="app">
    <div id="header">
        多 Agent 情感陪伴系统<span>本地 3B+8B · 云端 DeepSeek 兜底</span>
    </div>
    <div id="chat-container">
        <div class="welcome-msg">你好，我是你的情感陪伴助手。有什么想聊的都可以跟我说。</div>
    </div>
    <div id="input-area">
        <div id="input-card">
            <button id="plus-btn">+</button>
            <textarea id="user-input" rows="1" placeholder="输入消息..."></textarea>
            <button id="send-btn">➤</button>
        </div>
        <div id="disclaimer">内容由 AI 生成，请仔细甄别</div>
    </div>
</div>

<script>
const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

let history = [];

// 自动调整 textarea 高度
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

function appendMessage(role, text, meta='') {
    const row = document.createElement('div');
    row.className = 'message-row ' + role;

    const avatar = document.createElement('div');
    avatar.className = 'avatar ' + role;
    avatar.textContent = role === 'user' ? '我' : 'AI';

    const content = document.createElement('div');

    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + role;
    bubble.textContent = text;
    content.appendChild(bubble);

    if (meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'meta';
        metaDiv.textContent = meta;
        content.appendChild(metaDiv);
    }

    row.appendChild(avatar);
    row.appendChild(content);
    chatContainer.appendChild(row);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showThinking() {
    const card = document.createElement('div');
    card.className = 'thinking-card';
    card.id = 'thinking-card';
    card.innerHTML = '<span class="icon">💡</span><span>正在思考中</span><span class="arrow">▶</span>';

    const detail = document.createElement('div');
    detail.className = 'thinking-detail';
    detail.id = 'thinking-detail';
    detail.textContent = '意图识别中...';

    card.addEventListener('click', () => {
        card.classList.toggle('expanded');
        detail.classList.toggle('show');
    });

    const wrapper = document.createElement('div');
    wrapper.style.alignSelf = 'flex-start';
    wrapper.style.maxWidth = '90%';
    wrapper.appendChild(card);
    wrapper.appendChild(detail);
    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return wrapper;
}

function updateThinking(wrapper, detailText) {
    const detail = wrapper.querySelector('.thinking-detail');
    if (detail) {
        detail.textContent = detailText;
    }
}

function removeThinking(wrapper) {
    wrapper.remove();
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;
    userInput.value = '';
    userInput.style.height = 'auto';

    // 隐藏欢迎语
    const welcome = chatContainer.querySelector('.welcome-msg');
    if (welcome) welcome.remove();

    appendMessage('user', text);
    sendBtn.disabled = true;

    const thinkingWrapper = showThinking();

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_input: text, history: history })
        });
        const data = await res.json();

        removeThinking(thinkingWrapper);

        const meta = `意图: ${data.intent}${data.facts && Object.keys(data.facts).length ? ' · 记忆: ' + JSON.stringify(data.facts) : ''}`;
        appendMessage('agent', data.reply, meta);

        history.push({ user: text, assistant: data.reply, intent: data.intent, facts: data.facts });
    } catch (e) {
        removeThinking(thinkingWrapper);
        appendMessage('agent', '请求失败: ' + e.message);
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
</script>
</body>
</html>
"""

# ============== 启动入口 ==============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="服务端口，默认 8080")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址，默认 127.0.0.1")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), ChatHandler)
    print(f"🌐 Web 服务已启动: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
