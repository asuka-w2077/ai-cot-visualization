import json
import os
import urllib.request
from typing import List, Dict, Optional


class CloudClient:
    """
    云端大模型客户端（支持 Moonshot / DeepSeek 等 OpenAI-compatible API）
    纯标准库实现，零外部依赖。

    使用方式：
        # DeepSeek（默认）
        client = CloudClient(provider="deepseek", api_key="sk-xxx")

        # Moonshot / Kimi
        client = CloudClient(provider="moonshot", api_key="sk-xxx")
    """

    PROVIDERS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1/chat/completions",
            "default_model": "deepseek-chat",  # DeepSeek-V3
            "models": {
                "deepseek-chat": "DeepSeek-V3，通用对话，性价比高",
                "deepseek-reasoner": "DeepSeek-R1，推理模型，思维链展示",
            },
            "env_key": "DEEPSEEK_API_KEY",
        },
        "moonshot": {
            "base_url": "https://api.moonshot.cn/v1/chat/completions",
            "default_model": "moonshot-v1-8k",
            "models": {
                "moonshot-v1-8k": "轻量上下文（8k），适合日常对话",
                "moonshot-v1-32k": "中长上下文（32k），适合多轮长对话",
                "moonshot-v1-128k": "超长上下文（128k），适合全文分析",
            },
            "env_key": "MOONSHOT_API_KEY",
        },
    }

    def __init__(
        self,
        provider: str = "deepseek",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        cfg = self.PROVIDERS.get(provider, self.PROVIDERS["deepseek"])
        self.provider = provider
        self.base_url = base_url or cfg["base_url"]

        # 优先级：传参 > 环境变量 > 空字符串
        env_key_name = cfg["env_key"]
        self.api_key = api_key or os.environ.get(env_key_name, "")
        self.enabled = bool(self.api_key)

        # 模型：传参 > 默认
        self.model = model or cfg["default_model"]

    # ------------------------------------------------------------------
    # 底层通用调用
    # ------------------------------------------------------------------
    def _request(self, payload: dict) -> dict:
        """发送 HTTP 请求并解析 JSON，失败时抛异常"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        通用对话接口。未配置 API Key 时返回友好的 fallback 文本。

        Args:
            messages: 标准 OpenAI 格式消息列表 [{"role": "user", "content": "..."}, ...]
            temperature: 采样温度，建议情感对话 0.6~0.8，工具查询 0.3~0.5
            max_tokens: 最大生成 token 数
        """
        if not self.enabled:
            env_name = self.PROVIDERS[self.provider]["env_key"]
            return (
                f"【云端大模型未启用】\n"
                f"当前使用本地模型兜底。如需接入云端，请配置环境变量：\n"
                f"  export {env_name}=sk-xxx\n"
                f"或在启动 web_app.py 时传入：\n"
                f"  {env_name}=sk-xxx python web_app.py"
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            data = self._request(payload)
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            try:
                err = json.loads(body)
                msg = err.get("error", {}).get("message", body)
            except json.JSONDecodeError:
                msg = body or e.reason
            if e.code == 401:
                return f"【认证失败】API Key 无效或已过期。请检查环境变量。详情：{msg}"
            if e.code == 429:
                return f"【请求过于频繁】已被限流，请稍后再试。详情：{msg}"
            if e.code == 500:
                return f"【云端服务异常】服务器内部错误，请稍后再试。详情：{msg}"
            return f"【云端调用失败】HTTP {e.code}: {msg}"
        except urllib.error.URLError as e:
            return f"【网络错误】无法连接到服务器，请检查网络。详情：{e.reason}"
        except Exception as e:
            return f"【云端调用异常】{type(e).__name__}: {str(e)}"

    # ------------------------------------------------------------------
    # 实时天气（wttr.in）
    # ------------------------------------------------------------------
    def _extract_city(self, text: str) -> Optional[str]:
        """从用户输入中提取城市名"""
        import re
        # 先去掉时间词，避免干扰城市名提取
        cleaned = re.sub(r"今天|明天|后天|未来几天|未来一周|这周末", "", text)
        # 匹配 "XX天气"、"XX气温"、"查一下XX的天气" 等模式
        patterns = [
            r"(.+?)(?:的?天气|气温|温度|下不下雨)",
            r"(.+?).*?(?:天气|气温)",
            r"(?:查|告诉|查询|查一下|搜一下).*?(.+?)(?:的|怎么样|如何|$)",
        ]
        for p in patterns:
            m = re.search(p, cleaned)
            if m:
                city = m.group(1).strip()
                # 去掉常见前缀词和语气词
                city = re.sub(r"^(请问|告诉我|查询|查一下|搜索|搜一下|我想知道|问一下)", "", city).strip()
                city = re.sub(r"[啊呢吗吧]$", "", city).strip()
                if city and 2 <= len(city) <= 10:  # 城市名通常2-10个字
                    return city
        return None

    def _get_weather(self, city: str) -> str:
        """
        用 wttr.in 获取城市实时天气。
        失败时返回空字符串。
        """
        try:
            import urllib.request
            import urllib.parse
            encoded = urllib.parse.quote(city)
            # format: 天气状况 + 温度 + 风速 + 湿度 + 能见度
            url = f"https://wttr.in/{encoded}?format=%l:+%C+%t+%w+%h+%p"
            req = urllib.request.Request(url, headers={"User-Agent": "curl"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # 联网搜索（DuckDuckGo）
    # ------------------------------------------------------------------
    def _search_web(self, query: str, max_results: int = 3) -> str:
        """
        用 DuckDuckGo 搜索网页信息，返回拼接后的摘要文本。
        搜索失败时返回空字符串。
        """
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region="zh-cn", max_results=max_results))
                if not results:
                    return ""
                snippets = []
                for i, r in enumerate(results, 1):
                    title = r.get("title", "")
                    body = r.get("body", "")
                    if body:
                        snippets.append(f"[{i}] {title}: {body[:150]}")
                return "\n".join(snippets)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # 业务封装（带 system prompt + 历史上下文）
    # ------------------------------------------------------------------
    def tool_response(self, user_input: str, history: Optional[List[Dict]] = None) -> str:
        """处理工具类查询（天气、搜索、翻译等），带历史上下文"""
        from datetime import datetime
        now = datetime.now().strftime("%Y年%m月%d日 %H:%M")

        # 判断是否是天气查询
        import re
        is_weather = bool(re.search(r"天气|气温|温度|下不下雨|rain|weather", user_input))
        weather_info = ""
        search_result = ""

        if is_weather:
            # 1. 提取城市名
            city = self._extract_city(user_input)
            if city:
                # 2. 用 wttr.in 获取实时天气
                weather_info = self._get_weather(city)
            else:
                # 城市名不明确，尝试用搜索确认
                search_result = self._search_web(user_input)
        else:
            # 非天气查询：用 DuckDuckGo 搜索
            search_result = self._search_web(user_input)

        # 构建 system prompt
        parts = ["你是一个 helpful 的通用助手。请直接回答用户的问题，不要反问。保持简洁。"]
        parts.append(f"当前时间：{now}")

        if weather_info:
            parts.append(f"\n【实时天气数据】\n{weather_info}\n请基于以上天气数据回答。")
        if search_result:
            parts.append(f"\n【参考信息】\n{search_result}")

        system = "\n".join(parts)

        messages = [{"role": "system", "content": system}]
        if history:
            for turn in history:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": user_input})
        return self.chat(messages, temperature=0.5, max_tokens=1024)

    def crisis_response(self, user_input: str, history: Optional[List[Dict]] = None) -> str:
        """云端二次确认危机干预，语气更温暖紧迫"""
        system = (
            "你是一位专业的心理危机干预助手。用户可能处于危险状态。"
            "请用温暖但紧迫的语气回应，必须建议用户联系身边可信的人或拨打 24 小时心理援助热线 / 110 / 120。"
            "不要讲大道理，优先确认用户当前是否安全。回复控制在 3 句话以内。"
        )
        messages = [{"role": "system", "content": system}]
        if history:
            for turn in history:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": user_input})
        return self.chat(messages, temperature=0.3, max_tokens=512)

    def emotion_response(
        self,
        user_input: str,
        history: Optional[List[Dict]] = None,
        memory_text: str = "",
    ) -> str:
        """
        情感陪伴模式：将本地记忆和历史注入 system prompt，走云端大模型。
        适合需要更强理解力或长上下文的场景。
        """
        system = (
            "你是一位温柔、耐心的情感陪伴助手。请用口语化中文回复，像朋友在微信上聊天。\n"
            "要求：\n"
            "1. 先命名用户的情绪，再复述感受，然后提 1 个温和的澄清问题\n"
            "2. 不要给建议，不要说'应该'\n"
            "3. 不要编造用户没提过的细节\n"
            "4. 每段回复 2-3 句话\n"
            "5. 如果涉及危机（自杀、自残），必须表达关心并建议联系身边人/热线/110/120\n"
        )
        if memory_text:
            system += f"\n【你已记住的信息】\n{memory_text}\n"

        messages = [{"role": "system", "content": system}]
        if history:
            for turn in history:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": user_input})
        return self.chat(messages, temperature=0.7, max_tokens=1024)

    def full_conversation(
        self,
        user_input: str,
        history: Optional[List[Dict]] = None,
        memory_text: str = "",
    ) -> str:
        """
        前端选择"云端"模式时的统一入口。
        绕过本地意图识别，直接把完整上下文丢给云端大模型，
        让云端模型自己做意图判断和安全兜底。
        """
        system = (
            "你是用户的情感陪伴助手，同时也是一个通用助手。\n"
            "规则：\n"
            "1. 如果用户在闲聊或倾诉情绪：先命名情绪、复述感受、问 1 个温和问题。不给建议。\n"
            "2. 如果用户问天气/搜索/翻译/计算等工具问题：直接回答。\n"
            "3. 如果用户表达自杀/自残/绝望：必须表达关心，并建议联系身边人/心理热线/110/120。\n"
            "4. 如果用户要求医疗诊断/转账/隐私信息/违法行为：明确拒绝。\n"
            "5. 口语化中文，每段 2-3 句话，像微信聊天。\n"
        )
        if memory_text:
            system += f"\n【已记住的用户信息】\n{memory_text}\n"

        messages = [{"role": "system", "content": system}]
        if history:
            for turn in history:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": user_input})
        return self.chat(messages, temperature=0.7, max_tokens=1024)
