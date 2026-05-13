"""
Embedding model wrapper around sentence-transformers.
sentence-transformers 嵌入模型的薄包装。

Adds three things on top of the upstream class:
- Singleton — avoid reloading the model on every instantiation
- In-memory cache — same text is not re-embedded
- Uniform numpy float32 output (required by FAISS)

封装三件事:单例、文本级缓存、统一 float32 输出(FAISS 要求)。
"""
from typing import List, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """
    Thin wrapper over SentenceTransformer.
    SentenceTransformer 的薄包装。

    Usage:
        embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectors = embedder.encode(["text 1", "text 2", ...])
        # vectors.shape == (2, 384), dtype float32
    """

    _instance = None  # singleton cache / 单例缓存

    def __new__(cls, model_name: str = None, device: str = None):
        # Simple singleton: the first instantiation pins model + device;
        # later calls return the same instance regardless of args.
        # 单例:第一次实例化决定 model + device,后续调用忽略参数返回同一实例。
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        if self._initialized:
            return
        self.model_name = model_name
        self.device = device
        self._model = None
        self._initialized = True
        self._embed_cache: dict = {}  # text -> np.ndarray

    def _lazy_load(self):
        """
        Load the model on first use to keep import-time cheap.
        延迟加载,避免 import 时阻塞。
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                if "cuda" in self.device.lower():
                    logger.warning(f"GPU load failed ({e}), falling back to CPU")
                    self.device = "cpu"
                    self._model = SentenceTransformer(self.model_name, device="cpu")
                else:
                    raise

    def encode(self, texts: Union[str, List[str]],
               use_cache: bool = True,
               show_progress: bool = False) -> np.ndarray:
        """
        Encode text(s) into embedding vectors.
        把文本编码成向量。

        Args:
            texts: a single string or a list of strings.
            use_cache: reuse cached embeddings for previously-seen texts.
            show_progress: show the sentence-transformers progress bar (helpful for large batches).

        Returns:
            np.ndarray of shape (N, dim), dtype float32.
        """
        self._lazy_load()

        # Normalize input to list.
        # 输入统一成 list。
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        # Cache lookup + encode only the misses.
        # 查缓存,只 encode 未命中。
        if use_cache:
            uncached_idx = []
            uncached_texts = []
            results = [None] * len(texts)
            for i, t in enumerate(texts):
                if t in self._embed_cache:
                    results[i] = self._embed_cache[t]
                else:
                    uncached_idx.append(i)
                    uncached_texts.append(t)
            if uncached_texts:
                new_vecs = self._model.encode(
                    uncached_texts,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                ).astype(np.float32)
                for idx, vec in zip(uncached_idx, new_vecs):
                    self._embed_cache[texts[idx]] = vec
                    results[idx] = vec
            out = np.stack(results)
        else:
            out = self._model.encode(
                texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            ).astype(np.float32)

        if single:
            return out[0]
        return out

    @property
    def dim(self) -> int:
        """
        Embedding vector dimension.
        向量维度。
        """
        self._lazy_load()
        return self._model.get_embedding_dimension()
