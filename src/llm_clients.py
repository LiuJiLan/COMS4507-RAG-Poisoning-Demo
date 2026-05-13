"""
LLM clients, all routed through OpenRouter (Claude / GPT / Gemini / Llama / ...).
LLM 客户端,统一走 OpenRouter 路由各家模型。

Why a single router:
- 4 vendor SDKs have wildly different shapes; one HTTP client cuts work to ~25%.
- One API key, one bill — easy to wipe after the demo.
- OpenRouter is just routing; the underlying model is the real vendor model,
  so research validity is unaffected.
- Adding a new reranker candidate is a config edit, not new code.

为什么统一走 OpenRouter:4 家 SDK 形态各异,统一一个 HTTP client 工程量降到 1/4;
1 个 key 1 笔账单,demo 后清零方便;OpenRouter 只是路由层,底层是真实模型,研究
validity 不受影响;扩 reranker 候选只需要往 config.AVAILABLE_LLMS 里加条目,不写新代码。

To swap to a different provider later (a local LLM, a vendor SDK directly), implement
a new LLMClient subclass; reranker / generator do not need to change.
未来要换 provider(本地 LLM / 商业直接 API),实现一个新的 LLMClient 子类即可,
上层 reranker / generator 不需要改。
"""
from abc import ABC, abstractmethod
import logging
import re
import time

import requests

logger = logging.getLogger(__name__)

# Retry policy for OpenRouter transient errors (429 / 5xx).
# OpenRouter 临时错误(429 / 5xx)的退避重试策略。
# Main experiment hits ~1200 requests, so 429 rate-limit bumps are likely.
# 主实验 600 row × 2 call = 1200 request,撞 RPM cap 概率高,加指数退避兜底。
RETRY_HTTP_STATUSES = {429, 500, 502, 503, 504}
RETRY_MAX_ATTEMPTS = 3       # 3 extra retries on top of the first try (4 total)
RETRY_BACKOFF_BASE = 4.0     # 4s → 8s → 16s


class LLMClient(ABC):
    """
    Common interface for every LLM client.
    所有 LLM client 的统一接口。
    """

    def __init__(self, model_name: str, api_key: str = "", provider_options: dict | None = None):
        self.model_name = model_name
        self.api_key = api_key
        # OpenRouter `provider` sub-routing hint (only OpenRouterClient uses it).
        # OpenRouter 的 provider 子路由 hint(只 OpenRouterClient 用)。
        # Example: {"order": ["google-ai-studio"], "allow_fallbacks": False}
        # forces OpenRouter onto a specific upstream endpoint, bypassing flaky ones.
        # 例 {"order": ["google-ai-studio"], "allow_fallbacks": False}
        # 强制 OpenRouter 走某个上游 endpoint,绕开不稳定的后端。
        self.provider_options = provider_options

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        """
        Generate text. Core method; every subclass must implement it.
        生成文本(核心方法,所有子类必须实现)。
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_name})"


class StubLLMClient(LLMClient):
    """
    Stub client — no real API call, no quota burn.
    占位 client,不调真实 API,不消耗 quota。

    Used for:
    - End-to-end pipeline runs in dev without an API key.
    - Skipping the reranker in experiments to measure pure dense retriever attack effects.

    用途:dev 时无 key 也能跑通 pipeline;实验中跳过 reranker 看纯 dense retriever 的攻击效果。

    Behaviour:
    - Rerank-style prompt (contains "rank" + "document") → returns "1,2,3,...,N",
      i.e. preserves the dense retriever order.
    - Generate-style prompt → returns a fixed placeholder string.

    行为:rerank prompt → 返回 "1,2,3,...,N",即"保留 dense 原序";
    generate prompt → 返回固定占位文本。
    """

    def __init__(self, model_name: str = "stub"):
        super().__init__(model_name=model_name)

    def complete(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        prompt_lower = prompt.lower()
        if "rank" in prompt_lower and "document" in prompt_lower:
            # Count "[N]" markers in the prompt to estimate doc count.
            # 从 prompt 里数 "[N]" 标记估文档数。
            n = len(re.findall(r"\[\d+\]", prompt))
            if n == 0:
                n = 5
            return ",".join(str(i) for i in range(1, n + 1))
        return ("[STUB GENERATOR] Placeholder answer. "
                "Configure OPENROUTER_API_KEY in .env to use real LLM.")


class OpenRouterClient(LLMClient):
    """
    Call any model via OpenRouter's OpenAI-compatible REST API.
    通过 OpenRouter 调任意模型(OpenAI-compatible REST API)。

    model_name follows OpenRouter's "<provider>/<model>" convention, e.g.:
        anthropic/claude-3.5-sonnet
        openai/gpt-4o-mini
        google/gemini-2.0-flash-001
        meta-llama/llama-3.3-70b-instruct

    model_name 用 OpenRouter 的 "<provider>/<model>" 命名规范。
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    REQUEST_TIMEOUT = 60  # seconds — reranker temp=0 should be <10s, leave headroom for slow models

    # Optional metadata shown in the OpenRouter dashboard.
    # OpenRouter 后台显示用,可选。
    HTTP_REFERER = "https://github.com/local/coms4507-rag-poisoning-demo"
    APP_TITLE = "RAG Poisoning Demo (COMS4507)"

    def complete(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Add it to .env in project root."
            )

        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.provider_options:
            body["provider"] = self.provider_options

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.HTTP_REFERER,
            "X-Title": self.APP_TITLE,
        }

        for attempt in range(RETRY_MAX_ATTEMPTS + 1):
            try:
                response = requests.post(
                    self.API_URL, headers=headers, json=body, timeout=self.REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                break  # success
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status in RETRY_HTTP_STATUSES and attempt < RETRY_MAX_ATTEMPTS:
                    wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        f"OpenRouter {self.model_name}: HTTP {status},"
                        f" retry {attempt + 1}/{RETRY_MAX_ATTEMPTS} in {wait:.0f}s"
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"OpenRouter request failed for {self.model_name}: {e}")
                raise
            except requests.RequestException as e:
                # Connection / timeout errors — do not retry, surface to caller.
                # 网络层错误(connection / timeout),不 retry,抛给上层 fallback。
                logger.error(f"OpenRouter request failed for {self.model_name}: {e}")
                raise

        data = response.json()
        # OpenAI-compatible response shape: choices[0].message.content.
        # OpenAI-compatible 响应结构:choices[0].message.content。
        return data["choices"][0]["message"]["content"]


def make_client(llm_config: dict, use_stub: bool = False) -> LLMClient:
    """
    Build an LLMClient from a config.AVAILABLE_LLMS entry.
    根据 config.AVAILABLE_LLMS 配置构造 LLMClient。

    Args:
        llm_config: dict like {"provider": "openrouter", "model": "...", "enabled": True}
        use_stub: force a StubLLMClient (ignores `enabled`) — for local tests without burning quota.

    Returns:
        LLMClient instance (StubLLMClient or OpenRouterClient).
    """
    model = llm_config["model"]

    if use_stub or not llm_config.get("enabled", False):
        return StubLLMClient(model_name=model)

    provider = llm_config["provider"]
    if provider == "openrouter":
        from config import OPENROUTER_API_KEY
        return OpenRouterClient(
            model_name=model,
            api_key=OPENROUTER_API_KEY,
            provider_options=llm_config.get("openrouter_provider"),
        )

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Current design only supports "
        f"'openrouter'. Set provider='openrouter' in config.AVAILABLE_LLMS "
        f"and use OpenRouter's namespace format (e.g., 'anthropic/claude-3.5-sonnet')."
    )
