"""
LLM 客户端。今天先用 stub，明天接真实 API。

设计：所有 client 实现统一的 `LLMClient` 接口，
这样上层的 reranker 和 generator 不需要关心是哪家的 API。

明天接真实 API 时，只需要在每个 subclass 的 _call_api 里
替换 stub 为真实调用。
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import re

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """所有 LLM client 的统一接口"""

    def __init__(self, model_name: str, api_key: str = ""):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        """生成文本（核心方法，所有子类必须实现）"""
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_name})"


# ============================================================
# Stub Client（今天的占位实现）
# ============================================================
class StubLLMClient(LLMClient):
    """
    一个不调用真实 API 的占位 client。
    用于今天搭骨架时让 pipeline 端到端跑通。
    
    它的行为：
    - 对 rerank prompt（包含 "rank these documents"）→ 返回 "1,2,3,4,5"
    - 对 generate prompt → 返回一段固定的占位文本
    """

    def __init__(self, model_name: str = "stub"):
        super().__init__(model_name=model_name)

    def complete(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        prompt_lower = prompt.lower()
        if "rank" in prompt_lower and "document" in prompt_lower:
            # 尝试从 prompt 里提取文档数量
            # 简单 heuristic：找 "[N]" 这样的标记
            n = len(re.findall(r"\[\d+\]", prompt))
            if n == 0:
                n = 5
            # 返回 1..N 作为默认顺序（即"不改变排名"）
            return ",".join(str(i) for i in range(1, n + 1))
        # generator stub
        return ("[STUB GENERATOR] This is a placeholder answer. "
                "Real LLM integration coming tomorrow.")


# ============================================================
# 真实 Client 的骨架（明天填充）
# ============================================================
class ClaudeClient(LLMClient):
    """Anthropic Claude. 明天填充实际的 SDK 调用。"""

    def complete(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        # TODO: 明天接 anthropic SDK
        # from anthropic import Anthropic
        # client = Anthropic(api_key=self.api_key)
        # response = client.messages.create(
        #     model=self.model_name,
        #     max_tokens=max_tokens,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.content[0].text
        raise NotImplementedError("Claude client not yet implemented")


class GPTClient(LLMClient):
    """OpenAI GPT. 明天填充。"""

    def complete(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        # TODO: 明天接 openai SDK
        raise NotImplementedError("GPT client not yet implemented")


class GeminiClient(LLMClient):
    """Google Gemini. 明天填充。"""

    def complete(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        # TODO: 明天接 google-generativeai SDK
        raise NotImplementedError("Gemini client not yet implemented")


class OpenRouterClient(LLMClient):
    """通用 OpenRouter client，可以跑 Llama / Qwen / DeepSeek 等开源模型。"""

    def complete(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        # TODO: 明天接 requests HTTP 调用
        # POST https://openrouter.ai/api/v1/chat/completions
        raise NotImplementedError("OpenRouter client not yet implemented")


# ============================================================
# 工厂函数
# ============================================================
def make_client(llm_config: dict, use_stub: bool = True) -> LLMClient:
    """
    根据 config.py 里的 AVAILABLE_LLMS 配置创建 client。
    
    Args:
        llm_config: dict like {"provider": "anthropic", "model": "...", "enabled": True}
        use_stub: 如果为 True，无视 provider，统一返回 StubLLMClient
    
    Returns:
        LLMClient 实例
    """
    if use_stub or not llm_config.get("enabled", False):
        return StubLLMClient(model_name=llm_config.get("model", "stub"))

    provider = llm_config["provider"]
    model = llm_config["model"]
    if provider == "anthropic":
        from config import ANTHROPIC_API_KEY
        return ClaudeClient(model_name=model, api_key=ANTHROPIC_API_KEY)
    elif provider == "openai":
        from config import OPENAI_API_KEY
        return GPTClient(model_name=model, api_key=OPENAI_API_KEY)
    elif provider == "google":
        from config import GOOGLE_API_KEY
        return GeminiClient(model_name=model, api_key=GOOGLE_API_KEY)
    elif provider == "openrouter":
        from config import OPENROUTER_API_KEY
        return OpenRouterClient(model_name=model, api_key=OPENROUTER_API_KEY)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
