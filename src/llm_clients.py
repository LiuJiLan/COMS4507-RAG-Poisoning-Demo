"""
LLM 客户端。统一通过 OpenRouter 路由所有模型（Claude / GPT / Gemini / Llama / ...）。

设计理由（详见 notes/session 日志）:
- 4 家 SDK 形态各异,统一走 OpenRouter 工程量降到 1/4
- 一个 key 一笔账单,demo 后清零方便
- OpenRouter 只是路由层,底层还是真实模型,研究 validity 不受影响
- 扩 reranker 候选只需要往 config.AVAILABLE_LLMS 里加条目,不写新代码

如果将来要换 provider(本地 LLM / 商业直接 API), 实现一个新的 LLMClient
子类即可, 上层 reranker / generator 不需要改。
"""
from abc import ABC, abstractmethod
import logging
import re

import requests

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """所有 LLM client 的统一接口"""

    def __init__(self, model_name: str, api_key: str = ""):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        """生成文本(核心方法,所有子类必须实现)"""
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_name})"


class StubLLMClient(LLMClient):
    """
    占位 client,不调真实 API,不消耗 quota。

    用途:
    - dev 时跑通 pipeline 端到端(无 API key 也能跑)
    - 实验时跳过 reranker,看纯 dense retriever 的攻击效果

    行为:
    - rerank prompt(含 "rank" 和 "document") → 返回 "1,2,3,...,N",
      即"不改变 dense retriever 给的顺序"
    - generate prompt → 返回固定占位文本
    """

    def __init__(self, model_name: str = "stub"):
        super().__init__(model_name=model_name)

    def complete(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        prompt_lower = prompt.lower()
        if "rank" in prompt_lower and "document" in prompt_lower:
            # 从 prompt 里找 "[N]" 标记估文档数
            n = len(re.findall(r"\[\d+\]", prompt))
            if n == 0:
                n = 5
            return ",".join(str(i) for i in range(1, n + 1))
        return ("[STUB GENERATOR] Placeholder answer. "
                "Configure OPENROUTER_API_KEY in .env to use real LLM.")


class OpenRouterClient(LLMClient):
    """
    通过 OpenRouter 调用任意模型。OpenAI-compatible REST API。

    model_name 格式: "<provider>/<model>" (OpenRouter 命名规范)。例如:
        anthropic/claude-3.5-sonnet
        openai/gpt-4o-mini
        google/gemini-2.0-flash-001
        meta-llama/llama-3.3-70b-instruct
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    REQUEST_TIMEOUT = 60  # 秒。reranker 用低 temp,响应应该 <10s,留余量给慢模型。

    # 用于 OpenRouter 后台显示项目归属。可选,但加上比较干净。
    HTTP_REFERER = "https://github.com/local/coms4507-rag-poisoning-demo"
    APP_TITLE = "RAG Poisoning Demo (COMS4507)"

    def complete(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Add it to .env in project root."
            )

        try:
            response = requests.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": self.HTTP_REFERER,
                    "X-Title": self.APP_TITLE,
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=self.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"OpenRouter request failed for {self.model_name}: {e}")
            raise

        data = response.json()
        # OpenAI-compatible response 结构: choices[0].message.content
        return data["choices"][0]["message"]["content"]


def make_client(llm_config: dict, use_stub: bool = False) -> LLMClient:
    """
    根据 config.py 里的 AVAILABLE_LLMS 配置创建 client。

    Args:
        llm_config: dict like {"provider": "openrouter", "model": "...", "enabled": True}
        use_stub: 强制返回 stub(忽略 enabled),用于不消耗 quota 的本地测试

    Returns:
        LLMClient 实例(StubLLMClient 或 OpenRouterClient)
    """
    model = llm_config["model"]

    if use_stub or not llm_config.get("enabled", False):
        return StubLLMClient(model_name=model)

    provider = llm_config["provider"]
    if provider == "openrouter":
        from config import OPENROUTER_API_KEY
        return OpenRouterClient(model_name=model, api_key=OPENROUTER_API_KEY)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Current design only supports "
        f"'openrouter'. Set provider='openrouter' in config.AVAILABLE_LLMS "
        f"and use OpenRouter's namespace format (e.g., 'anthropic/claude-3.5-sonnet')."
    )
