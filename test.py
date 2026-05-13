"""
环境健康检查脚本。

逐项检查 Python / NumPy / PyTorch+CUDA / FAISS / Streamlit / PyYAML / python-dotenv,
最后做一次 sentence-transformers + FAISS 的 end-to-end smoke test。

每一项独立 try/except,前面失败不影响后面继续。最后给汇总和建议的 pip install 命令。

用法:
    python test.py
"""
from __future__ import annotations

import sys
import time
import platform
import traceback
from typing import Callable

# ----------------------------------------------------------------------
# 框架
# ----------------------------------------------------------------------
OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results: list[tuple[str, str, str]] = []  # (status, name, detail)


def check(name: str, fn: Callable[[], str]) -> None:
    """跑一个检查,捕获异常,记录结果。fn 返回一行/多行 info string。"""
    print(f"\n--- {name} ---")
    t0 = time.time()
    try:
        info = fn() or ""
        dt = time.time() - t0
        print(f"{OK}  {name}  ({dt:.2f}s)")
        if info:
            for line in info.splitlines():
                print(f"       {line}")
        results.append((OK, name, info))
    except Exception as e:
        dt = time.time() - t0
        print(f"{FAIL}  {name}  ({dt:.2f}s)")
        traceback.print_exc()
        results.append((FAIL, name, f"{type(e).__name__}: {e}"))


# ----------------------------------------------------------------------
# 各项检查
# ----------------------------------------------------------------------
def check_python() -> str:
    v = sys.version_info
    info = (f"Python {v.major}.{v.minor}.{v.micro}\n"
            f"platform: {platform.platform()}\n"
            f"executable: {sys.executable}")
    if v < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} < 3.10 expected")
    return info


def check_numpy() -> str:
    import numpy as np
    a = np.random.randn(3, 4).astype(np.float32)
    return f"numpy {np.__version__}, sample {a.shape} {a.dtype}"


def check_torch_cuda() -> str:
    import torch
    lines = [f"torch {torch.__version__}"]
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        lines.append(f"CUDA available: torch.version.cuda={torch.version.cuda}")
        lines.append(f"device 0: {name} (total {n} GPU)")
        # GPU 实际操作 sanity:1k x 1k matmul
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        lines.append(f"GPU matmul OK, peak alloc ~{peak_mb:.1f} MB")
    else:
        lines.append("CUDA NOT available -- will fall back to CPU")
        lines.append("(this is OK for demo but slower; check torch install if RTX 4070 expected)")
    return "\n".join(lines)


def check_faiss() -> str:
    import faiss
    import numpy as np
    d = 16
    xb = np.random.randn(100, d).astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    xq = np.random.randn(3, d).astype(np.float32)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, 5)
    return (f"faiss {faiss.__version__}\n"
            f"IndexFlatIP smoke: D={D.shape}, I={I.shape}, top-1 scores={D[:,0].round(3).tolist()}")


def check_streamlit() -> str:
    # 只 import,不启动服务
    import streamlit
    return f"streamlit {streamlit.__version__} (import-only check; UI not started)"


def check_pyyaml() -> str:
    import yaml
    data = yaml.safe_load("- a\n- b\n- c\n")
    return f"pyyaml {yaml.__version__}, parsed {data!r}"


def check_dotenv() -> str:
    import dotenv
    ver = getattr(dotenv, "__version__", "unknown")
    return f"python-dotenv {ver}"


def check_sentence_transformers() -> str:
    """加载 all-MiniLM-L6-v2 模型并 embed 几句话。首次会从 HuggingFace 下载 ~80 MB。"""
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"       loading all-MiniLM-L6-v2 on {device}...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    sample = ["Brisbane is a city in Queensland.",
              "FAISS is a library for vector search."]
    vec = model.encode(sample)
    return (f"sentence-transformers OK on {device}\n"
            f"embedding shape: {vec.shape}, dim: {vec.shape[1]}")


def check_datasets() -> str:
    """ADJ-001:HuggingFace `datasets` 库,用于拉 MS MARCO 背景集合。仅 import 检查,不下载数据。"""
    import datasets
    from datasets import load_dataset  # noqa: F401  仅做 import 验证
    return f"datasets {datasets.__version__}, load_dataset import OK"


def check_end_to_end() -> str:
    """组合 sentence-transformers + FAISS 模拟一次真实 retrieval,看 top-1 是否合理。"""
    import torch
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    docs = [
        "South Bank Parklands is a popular Brisbane riverside attraction.",
        "Brisbane is the capital of Queensland, Australia.",
        "Sydney Opera House is in Sydney, not Brisbane.",
        "The University of Queensland is located in Brisbane.",
        "FAISS is a library for efficient similarity search.",
    ]
    query = "Where is the University of Queensland?"

    doc_vecs = model.encode(docs).astype(np.float32)
    q_vec = model.encode([query]).astype(np.float32)
    faiss.normalize_L2(doc_vecs)
    faiss.normalize_L2(q_vec)

    index = faiss.IndexFlatIP(doc_vecs.shape[1])
    index.add(doc_vecs)
    D, I = index.search(q_vec, 3)

    lines = [f"query: {query}"]
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        lines.append(f"  rank {rank}: ({score:.4f}) {docs[idx]}")

    top1 = docs[int(I[0][0])]
    if "University" in top1 and "Queensland" in top1:
        lines.append("top-1 hits expected doc (UQ)")
    else:
        lines.append("WARN: top-1 is not the UQ doc, pipeline runs but embedding quality may be poor")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> int:
    print("=" * 64)
    print("RAG Poisoning Demo -- environment check")
    print("=" * 64)

    # 顺序:从轻到重。任何一项失败都不阻塞后续,但 sentence-transformers
    # 失败时 end-to-end 一定也会失败,这是预期。
    check("Python version (>=3.10)", check_python)
    check("NumPy", check_numpy)
    check("PyTorch + CUDA", check_torch_cuda)
    check("FAISS (faiss-cpu)", check_faiss)
    check("Streamlit", check_streamlit)
    check("PyYAML", check_pyyaml)
    check("python-dotenv", check_dotenv)
    check("sentence-transformers (downloads ~80 MB on first run)", check_sentence_transformers)
    check("HuggingFace datasets (ADJ-001, import only)", check_datasets)
    check("End-to-end smoke test (embedder + FAISS)", check_end_to_end)

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    for status, name, _ in results:
        print(f"  {status}  {name}")

    n_pass = sum(1 for r in results if r[0] == OK)
    n_fail = sum(1 for r in results if r[0] == FAIL)
    print(f"\n{n_pass} passed, {n_fail} failed")

    if n_fail == 0:
        print("\nAll checks passed. Environment is ready.")
        return 0

    # 失败的话给点提示
    print("\nFailed items above. Hints:")
    failed_names = {name for status, name, _ in results if status == FAIL}
    if any("PyTorch" in n for n in failed_names):
        print("  - torch + CUDA install: visit https://pytorch.org/get-started/locally/")
        print("    For RTX 4070 + Win + Py 3.10, typically:")
        print("      pip install torch --index-url https://download.pytorch.org/whl/cu121")
    if any("FAISS" in n for n in failed_names):
        print("  - faiss-cpu install:  pip install faiss-cpu")
        print("    If wheel fails, try conda: conda install -c conda-forge faiss-cpu")
    if any("sentence-transformers" in n for n in failed_names):
        print("  - sentence-transformers:  pip install sentence-transformers")
        print("    If model download is slow, try HF mirror:")
        print("      set HF_ENDPOINT=https://hf-mirror.com")
    if any("datasets" in n for n in failed_names):
        print("  - HuggingFace datasets:  pip install datasets")
    if any(p in failed_names for p in ["Streamlit", "NumPy", "PyYAML", "python-dotenv"]):
        print("  - misc:  pip install streamlit numpy pyyaml python-dotenv pandas")
    return 1


if __name__ == "__main__":
    sys.exit(main())
