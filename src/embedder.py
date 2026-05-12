"""
Embedding 模型包装。底层是 sentence-transformers，做了如下封装：
- 单例模式（避免多次加载模型）
- 缓存（embed 过的文本不再重复 embed）
- 统一的 numpy float32 输出（FAISS 要求）
"""
from typing import List, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """
    Embedder 是个对 SentenceTransformer 的薄包装。
    
    用法：
        embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectors = embedder.encode(["text 1", "text 2", ...])
        # vectors.shape == (2, 384), dtype float32
    """

    _instance = None  # 单例缓存

    def __new__(cls, model_name: str = None, device: str = None):
        # 简单的单例：第一次实例化决定 model 和 device
        # 后续调用直接返回同一个 instance
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
        """第一次用到才真正加载模型，避免启动慢"""
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
        把文本编码成向量。
        
        Args:
            texts: 单个字符串或字符串列表
            use_cache: 是否使用内存缓存（同样的文本不会重复 embed）
            show_progress: 是否显示进度条（大批量时有用）
        
        Returns:
            np.ndarray shape (N, dim), dtype float32
        """
        self._lazy_load()

        # 统一成 list
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        # 查 cache
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
        """向量维度"""
        self._lazy_load()
        return self._model.get_embedding_dimension()
