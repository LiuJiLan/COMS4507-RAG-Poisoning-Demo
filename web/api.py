"""
FastAPI app — backend for the RAG Poisoning Visualizer.
FastAPI 后端 — RAG Poisoning Visualizer。

Replaces the Streamlit Dashboard as of v2_008 (route pivot:
Streamlit's nested DOM made mockup alignment too costly).
The pipeline itself in src/ is unchanged; this file is purely the
HTTP/SSE transport layer plus a startup lifespan hook that loads
the RAGPipeline once.

v2_008 路线切换:从 Streamlit 改用 FastAPI。src/ 中的 pipeline 完全
不动,本文件只负责 HTTP/SSE 传输 + startup 时 load 一次 pipeline。

Design contract (frontend ↔ backend):
  - Backend is STATELESS about Runs. No server-side run cache.
    Frontend owns the cache (top_k1, top_k2, generator output).
  - Every endpoint takes a `ts` (frontend timestamp) and every SSE
    event / response echoes it. Frontend drops events whose ts
    doesn't match its current ts → race-safe even with rapid clicks.
  - Top-k payloads include doc.content (~few KB extra per request,
    negligible). Frontend stashes full docs so /api/rerank/stream
    and /api/generate can be invoked without server-side lookup.
  - retriever.{inject,remove}_poison still lives in src/; here we
    bracket each /api/run/stream Run with `reset_all_poison()` on
    entry (defence) + `asyncio.shield(remove_poison)` in finally
    (so client-abort GeneratorExit doesn't cancel the cleanup).
"""
import os
import sys
import time
import json
import asyncio
import warnings
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

# === Noise suppression — must precede HF / transformers imports ===
# === 噪音抑制(必须在 HF / transformers 加载前) ===
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

# Make project root importable so `import config` / `from src... import ...`
# resolves regardless of uvicorn's launch cwd.
# 把项目根加入 sys.path,确保 uvicorn 在任意 cwd 启动时都能 import config / src。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from huggingface_hub.utils import logging as hub_logging
hub_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config
from src.corpus import Document, load_corpus, load_poison_set
from src.pipeline import RAGPipeline
from src.retriever import RetrievalResult
from src.evaluator import AttackMetrics, compare_rankings
from src.llm_clients import make_client, StubLLMClient


# Module-level pipeline singleton, populated LAZILY by /api/init/stream
# on first client request. Startup lifespan no longer builds it — that
# way the user can watch the build animation in the Database box and
# the Run button stays locked until init completes.
# Pipeline 单例改为 lazy 触发(由 /api/init/stream 第一次客户端请求时
# 构造);startup lifespan 不再 build,让用户能看到 init 动画 + Run
# 按钮在 init 完成前锁定。
_pipeline: Optional[RAGPipeline] = None
_init_elapsed: Optional[float] = None
# Async lock so concurrent /api/init/stream calls don't double-build.
# 异步锁,防止并发 /api/init/stream 多次构造。
_init_lock: asyncio.Lock = asyncio.Lock()


# ============================================================
# Pipeline build / lifespan
# ============================================================
def _build_pipeline() -> RAGPipeline:
    """
    Construct (or load cached) RAGPipeline. Mirrors app.py's get_pipeline().
    Both the cache-hit and from-scratch paths converge before the embedder
    warm-up at the bottom — so the SentenceTransformer model load +
    CUDA-JIT cost (~10s on RTX 4070) is always paid here inside
    /api/init/stream's window (driving the Database "first build" timer)
    rather than leaking into the first user Run's stage_embed.
    构造或从 cache 加载 RAGPipeline;两种 path(命中/从零)汇合后统一
    做一次 warm-up,把 ~10s 的 model load + CUDA JIT 算进 init 窗口。
    """
    pipeline = RAGPipeline(
        embedding_model=config.EMBEDDING_MODEL,
        embedding_device=config.EMBEDDING_DEVICE,
        top_k_1=config.TOP_K_1,
        top_k_2=config.TOP_K_2,
    )

    loaded_from_cache = False
    if config.FAISS_CACHE.exists() and config.DOCS_CACHE.exists():
        try:
            pipeline.load_cached_index(config.FAISS_CACHE, config.DOCS_CACHE)
            log.info("Pipeline loaded from cache.")
            loaded_from_cache = True
        except Exception as e:
            log.warning(f"Cache load failed ({e}), rebuilding...")

    if not loaded_from_cache:
        for f in (config.BASE_CORPUS_FILE, config.BACKGROUND_CORPUS_FILE):
            if not f.exists():
                raise RuntimeError(f"Corpus file not found: {f}")

        base_docs = load_corpus(config.BASE_CORPUS_FILE)
        background_docs = load_corpus(config.BACKGROUND_CORPUS_FILE)
        pipeline.initialize(base_docs + background_docs)

        try:
            pipeline.retriever.save(config.FAISS_CACHE, config.DOCS_CACHE)
            log.info("Pipeline cache saved.")
        except Exception as e:
            log.warning(f"Failed to save cache: {e}")

    # Warm-up: pay model-load + CUDA-JIT cost now (~10s first time)
    # regardless of which path got us here.
    # 预热:无论走哪条 path 都付一次 model load + CUDA JIT 成本。
    log.info("Warming up embedder model (first GPU forward)...")
    pipeline.embedder.encode("warmup")
    log.info("Embedder warmed up.")

    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pipeline init is deferred to the first /api/init/stream request so
    the user gets a visible build-progress animation in the Database
    box. Lifespan only does cheap setup (loggers etc.).
    Pipeline init 延迟到第一次 /api/init/stream 触发,这样用户能在
    Database 框看到 build 动画。Lifespan 只做轻量 setup。
    """
    log.info("Lifespan startup — pipeline init deferred until first "
             "/api/init/stream request.")
    yield


app = FastAPI(title=config.APP_TITLE, lifespan=lifespan)

_WEB_DIR = Path(__file__).resolve().parent
app.mount(
    "/static",
    StaticFiles(directory=_WEB_DIR / "static"),
    name="static",
)
templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))


# ============================================================
# Pydantic request models
# ============================================================
class WireDoc(BaseModel):
    """
    On-the-wire form of a RetrievalResult — frontend's cache uses
    exactly this shape, sent back on /api/rerank/stream and
    /api/generate so the server doesn't need its own Run cache.
    RetrievalResult 的 wire 形式 — 前端 cache 用同一 shape,partial
    rerank / generate 时送回 server,免去 server-side run cache。
    """
    rank: int
    doc_id: str
    title: str
    content: str
    source: str
    is_poison: bool = False
    score: float


class RunRequest(BaseModel):
    query: str
    poison: str       # poison-set key, e.g. "P_keyword_stuffing"
    reranker: str     # reranker key, or "stub"
    ts: str           # frontend timestamp — echoed on every SSE event


class RerankRequest(BaseModel):
    query: str
    top_k1_clean: List[WireDoc]
    top_k1_poisoned: List[WireDoc]
    reranker: str
    ts: str


class GenerateRequest(BaseModel):
    query: str
    top_k2_clean: List[WireDoc]
    top_k2_poisoned: List[WireDoc]
    ts: str


# ============================================================
# Serialization / deserialization helpers
# ============================================================
def _rr_to_json(rr: RetrievalResult) -> dict:
    """
    Serialize one RetrievalResult to the WireDoc shape. doc.content is
    INCLUDED so the frontend can cache full docs and post them back to
    /api/rerank/stream and /api/generate later — server stays stateless
    about Runs.
    序列化 RetrievalResult 为 WireDoc shape;含 doc.content,前端 cache
    后可送回 server,server 不需 Run cache。
    """
    return {
        "rank": rr.rank,
        "doc_id": rr.doc.doc_id,
        "title": rr.doc.title,
        "content": rr.doc.content,
        "source": rr.doc.source,
        "is_poison": bool(rr.doc.is_poison),
        "score": float(rr.score),
    }


def _wire_to_rr(w: WireDoc) -> RetrievalResult:
    """
    Reverse of _rr_to_json — rebuild a RetrievalResult from a frontend-
    supplied WireDoc. Used in /api/rerank/stream and /api/generate to
    avoid any server-side lookup.
    _rr_to_json 的反向 — 从前端传来的 WireDoc 重建 RetrievalResult,
    partial rerank / generate 用,server 端无需查找。
    """
    doc = Document(
        doc_id=w.doc_id,
        title=w.title,
        content=w.content,
        source=w.source,
        topic="",
        url="",
    )
    doc.is_poison = w.is_poison
    return RetrievalResult(doc=doc, score=w.score, rank=w.rank)


def _metrics_to_json(m: AttackMetrics) -> dict:
    return {
        "poison_in_topk": bool(m.poison_in_topk),
        "poison_rank": m.poison_rank,
        "n_poison_in_topk": int(m.n_poison_in_topk),
        "displaced_docs": list(m.displaced_docs),
        "score_gap": m.score_gap,
    }


def _sse(payload: dict) -> str:
    """
    Format one Server-Sent Events frame. The empty line at the end is
    part of the SSE wire format (event terminator).
    格式化一条 SSE 帧;末尾空行是 SSE 协议要求的 event 终止符。
    """
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return f"data: {body}\n\n"


# ============================================================
# Meta builder — shared by GET / (Jinja2 context) and /api/meta
# ============================================================
def _compute_meta() -> dict:
    # Tolerant build: pipeline may not be initialized yet (lazy init).
    # When that's the case we return n_background = n_base = None;
    # the GET / template falls back to "—" placeholders and the
    # frontend fills in real numbers from /api/init/stream's done event.
    # Pipeline 未 ready 时容忍:n_* = None,GET / 显示 "—",init 完成
    # 后前端从 done event 的 meta 字段更新。
    if _pipeline is None:
        n_background = None
        n_base = None
    else:
        docs = _pipeline.retriever.documents
        n_background = sum(1 for d in docs if d.source == "msmarco")
        n_base = sum(
            1 for d in docs if d.source != "msmarco" and not d.is_poison
        )

    poison_sets = []
    if config.POISON_DIR.exists():
        for p in sorted(config.POISON_DIR.glob("*.json")):
            key = p.stem
            label = key.removeprefix("P_").replace("_", " ").title()
            poison_sets.append({"key": key, "label": label})

    rerankers = [
        {"key": k, "label": v["model"]}
        for k, v in config.AVAILABLE_LLMS.items()
        if v.get("enabled")
    ]
    rerankers.append({"key": "stub", "label": "stub (skip API call)"})

    return {
        "n_background": n_background,
        "n_base": n_base,
        "poison_sets": poison_sets,
        "rerankers": rerankers,
        "generator_model": config.AVAILABLE_LLMS[config.GENERATOR_LLM]["model"],
        "top_k_1": config.TOP_K_1,
        "top_k_2": config.TOP_K_2,
        "default_query": "best Chinese restaurant in Brisbane",
    }


# ============================================================
# Endpoints
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Render dashboard.html with server-side initial state inlined.
    When the pipeline is not yet initialized (lazy init mode, before
    the first /api/init/stream), cylinder numbers fall back to "—"
    placeholders; the frontend fills them in from the init done event.
    Pipeline 未 init 时,cylinder 数字 fallback "—",前端从 init done
    event 的 meta 拿到实数填回。
    """
    meta = _compute_meta()
    default_poison_short = "—"
    if meta["poison_sets"]:
        first = meta["poison_sets"][0]["key"]
        first = first[2:] if first.startswith("P_") else first
        default_poison_short = first[:6] + "…"
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "app_title": config.APP_TITLE,
            "n_background": meta["n_background"] if meta["n_background"] is not None else "—",
            "n_base": meta["n_base"] if meta["n_base"] is not None else "—",
            "poison_sets": meta["poison_sets"],
            "rerankers": meta["rerankers"],
            "generator_model": meta["generator_model"],
            "top_k_1": meta["top_k_1"],
            "top_k_2": meta["top_k_2"],
            "default_query": meta["default_query"],
            "default_poison_short": default_poison_short,
            "meta_json": json.dumps(meta, ensure_ascii=False),
        },
    )


@app.post("/api/init/stream")
async def api_init_stream() -> StreamingResponse:
    """
    Lazy-trigger the RAGPipeline build on the first client request.
    Emits SSE events the frontend uses to drive the Database "first
    build" animation + Run-button lock. If a previous client already
    triggered the build (or this is a page reload after build done),
    we report `elapsed: null` so the frontend's `first build database`
    row stays hidden (matches the user's "undefined → don't show"
    contract — that row is meant to record what THIS client observed).

    第一次 client 请求时 lazy 触发 pipeline 构建。SSE event 驱动前端
    Database 的 first build 动画 + Run 按钮锁。如果其他客户端已触发过
    (或这是 build 完成后的 reload),返回 elapsed=null,前端 first
    build 行保持隐藏 — 该行记录的是"本客户端观测到的"那次 build。
    """
    async def gen():
        global _pipeline, _init_elapsed
        try:
            async with _init_lock:
                if _pipeline is not None:
                    # Already built by a previous client / earlier
                    # request. Hand back fresh meta + elapsed=null so
                    # the UI hides the "first build" row.
                    yield _sse({
                        "type": "done",
                        "elapsed": None,
                        "meta": _compute_meta(),
                    })
                    return

                yield _sse({"type": "stage", "state": "running"})
                t = time.perf_counter()
                _pipeline = await asyncio.to_thread(_build_pipeline)
                _init_elapsed = time.perf_counter() - t
                log.info(f"Pipeline initialized in {_init_elapsed:.2f}s.")
                yield _sse({
                    "type": "stage", "state": "complete",
                    "elapsed": _init_elapsed,
                })
                yield _sse({
                    "type": "done",
                    "elapsed": _init_elapsed,
                    "meta": _compute_meta(),
                })
        except Exception as e:
            log.exception("Init failed")
            yield _sse({
                "type": "error",
                "message": f"{type(e).__name__}: {e}",
            })

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/meta")
async def api_meta() -> dict:
    return _compute_meta()


def _swap_reranker(reranker_key: str) -> None:
    """Mutate the singleton pipeline's reranker LLM client."""
    if reranker_key == "stub":
        _pipeline.reranker.llm = StubLLMClient()
    elif reranker_key in config.AVAILABLE_LLMS:
        _pipeline.reranker.llm = make_client(
            config.AVAILABLE_LLMS[reranker_key]
        )
    else:
        raise HTTPException(400, f"Unknown reranker: {reranker_key}")


@app.post("/api/run/stream")
async def api_run_stream(req: RunRequest) -> StreamingResponse:
    """
    Run one clean-vs-poisoned comparison; stream stage events as SSE.

    Event order (each event payload includes `ts` for race-safe
    frontend matching):
      stage_embed   running → complete
      stage_search  running          (after search-clean is computed,
                                     before inject)
      stage_build   running → complete  (around inject_poison —
                                     drives the Database box blue
                                     shimmer in the UI)
      stage_search  complete  (with clean + poisoned top-k₁ data)
      stage_rerank  running → complete (with clean + poisoned top-k₂)
      done           (with metrics_k1 + metrics_k2)

    Defence: reset_all_poison() on entry purges any leftover poison
    from a previously aborted Run (without this, the next Run's
    clean-side search could pick up old poison docs because the
    GeneratorExit on client-abort can cancel finally cleanup).
    The finally itself uses asyncio.shield to make remove_poison
    immune to GeneratorExit cancellation.

    防御:入口先 reset_all_poison(清掉上轮 abort 漏的 poison);finally
    用 asyncio.shield 让 remove_poison 不被 GeneratorExit 取消。
    """
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")

    poison_path = config.POISON_DIR / f"{req.poison}.json"
    if not poison_path.exists():
        raise HTTPException(400, f"Unknown poison set: {req.poison}")
    poison_docs = load_poison_set(poison_path)

    # Defence (see docstring): purge any leftover poison BEFORE we start
    # streaming so a previous aborted Run can't contaminate this one.
    # 防御:在开始 streaming 前清掉残留 poison。
    await asyncio.to_thread(_pipeline.retriever.reset_all_poison)

    _swap_reranker(req.reranker)

    async def gen():
        # Two precise stage_build windows — Database shimmers ONLY when
        # the index is actually changing (inject + rebuild), and stays
        # quiet during rerank (which only reads the index).
        # 两段精确 shimmer:Database 只在索引真改变时蠕动(inject 和 rebuild),
        # rerank 期间安静(rerank 只读索引)。
        #
        # `token` lives outside try so the finally safety-net can see it.
        token = None
        try:
            # ---- stage_embed ----
            yield _sse({
                "type": "stage", "stage_id": "stage_embed",
                "state": "running", "ts": req.ts,
            })
            t = time.perf_counter()
            await asyncio.to_thread(_pipeline.embedder.encode, req.query)
            yield _sse({
                "type": "stage", "stage_id": "stage_embed",
                "state": "complete", "ts": req.ts,
                "elapsed": time.perf_counter() - t,
            })

            # ---- stage_search (clean side first, no poison in index yet) ----
            yield _sse({
                "type": "stage", "stage_id": "stage_search",
                "state": "running", "ts": req.ts,
            })
            t_search = time.perf_counter()
            top_k1_clean = await asyncio.to_thread(
                _pipeline.retriever.search, req.query, k=config.TOP_K_1,
            )

            # ---- stage_build window #1: inject poison ----
            yield _sse({
                "type": "stage", "stage_id": "stage_build",
                "state": "running", "phase": "inject", "ts": req.ts,
            })
            t_inject = time.perf_counter()
            token = await asyncio.to_thread(
                _pipeline.retriever.inject_poison, poison_docs,
            )
            inject_elapsed = time.perf_counter() - t_inject
            yield _sse({
                "type": "stage", "stage_id": "stage_build",
                "state": "complete", "phase": "inject", "ts": req.ts,
                "elapsed": inject_elapsed,
            })

            # ---- stage_search (poisoned side, poison now indexed) ----
            top_k1_poisoned = await asyncio.to_thread(
                _pipeline.retriever.search, req.query, k=config.TOP_K_1,
            )
            yield _sse({
                "type": "stage", "stage_id": "stage_search",
                "state": "complete", "ts": req.ts,
                "elapsed": time.perf_counter() - t_search,
                "data": {
                    "clean": [_rr_to_json(r) for r in top_k1_clean],
                    "poisoned": [_rr_to_json(r) for r in top_k1_poisoned],
                },
            })

            # ---- stage_rerank (clean + poisoned) — Database is quiet,
            #      reranker only READS the index. ----
            yield _sse({
                "type": "stage", "stage_id": "stage_rerank",
                "state": "running", "ts": req.ts,
            })
            t_rerank = time.perf_counter()
            top_k2_clean, padded_clean = await asyncio.to_thread(
                _pipeline.reranker.rerank,
                req.query, top_k1_clean, config.TOP_K_2,
            )
            top_k2_poisoned, padded_poisoned = await asyncio.to_thread(
                _pipeline.reranker.rerank,
                req.query, top_k1_poisoned, config.TOP_K_2,
            )
            yield _sse({
                "type": "stage", "stage_id": "stage_rerank",
                "state": "complete", "ts": req.ts,
                "elapsed": time.perf_counter() - t_rerank,
                "data": {
                    "clean": [_rr_to_json(r) for r in top_k2_clean],
                    "poisoned": [_rr_to_json(r) for r in top_k2_poisoned],
                },
                "padded": {
                    "clean": padded_clean,
                    "poisoned": padded_poisoned,
                },
            })

            # ---- stage_build window #2: rebuild (remove_poison) ----
            # Distinguished from inject via the `phase` field; frontend
            # labels them "injecting poison…" vs "rebuilding index…"
            # so the second shimmer is semantically clear, not noise.
            # 用 phase 字段区分 inject 和 rebuild,前端显示不同 corner 文字。
            yield _sse({
                "type": "stage", "stage_id": "stage_build",
                "state": "running", "phase": "rebuild", "ts": req.ts,
            })
            t_rebuild = time.perf_counter()
            await asyncio.shield(asyncio.to_thread(
                _pipeline.retriever.remove_poison, token,
            ))
            rebuild_elapsed = time.perf_counter() - t_rebuild
            token = None
            yield _sse({
                "type": "stage", "stage_id": "stage_build",
                "state": "complete", "phase": "rebuild", "ts": req.ts,
                "elapsed": rebuild_elapsed,
            })

            # ---- Metrics + done ----
            metrics_k1 = compare_rankings(top_k1_clean, top_k1_poisoned)
            metrics_k2 = compare_rankings(top_k2_clean, top_k2_poisoned)
            yield _sse({
                "type": "done", "ts": req.ts,
                "metrics_k1": _metrics_to_json(metrics_k1),
                "metrics_k2": _metrics_to_json(metrics_k2),
            })
        except Exception as e:
            log.exception("Run failed (ts=%s)", req.ts)
            yield _sse({
                "type": "error", "ts": req.ts,
                "message": f"{type(e).__name__}: {e}",
            })
        finally:
            # Safety net: an exception (or client abort before the
            # explicit remove above) leaves token set; shielded clean
            # up so the index doesn't carry poison into the next Run.
            # 异常或客户端在显式 remove 之前 abort 时兜底清理。
            if token is not None:
                try:
                    await asyncio.shield(asyncio.to_thread(
                        _pipeline.retriever.remove_poison, token,
                    ))
                except Exception:
                    log.exception("remove_poison cleanup failed")

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/rerank/stream")
async def api_rerank_stream(req: RerankRequest) -> StreamingResponse:
    """
    Re-run ONLY the rerank stage with a (possibly) different reranker.
    Frontend supplies the cached top_k1 — embed + search are skipped
    entirely, no retriever state involved. SSE event order:
      stage_rerank running → complete  (with new top-k₂)
      done                             (with metrics_k2)

    仅重跑 rerank;前端传 cached top_k1,server 完全不动 retriever。
    """
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")

    _swap_reranker(req.reranker)

    top_k1_clean = [_wire_to_rr(w) for w in req.top_k1_clean]
    top_k1_poisoned = [_wire_to_rr(w) for w in req.top_k1_poisoned]

    async def gen():
        try:
            yield _sse({
                "type": "stage", "stage_id": "stage_rerank",
                "state": "running", "ts": req.ts,
            })
            t = time.perf_counter()
            top_k2_clean, padded_clean = await asyncio.to_thread(
                _pipeline.reranker.rerank,
                req.query, top_k1_clean, config.TOP_K_2,
            )
            top_k2_poisoned, padded_poisoned = await asyncio.to_thread(
                _pipeline.reranker.rerank,
                req.query, top_k1_poisoned, config.TOP_K_2,
            )
            yield _sse({
                "type": "stage", "stage_id": "stage_rerank",
                "state": "complete", "ts": req.ts,
                "elapsed": time.perf_counter() - t,
                "data": {
                    "clean": [_rr_to_json(r) for r in top_k2_clean],
                    "poisoned": [_rr_to_json(r) for r in top_k2_poisoned],
                },
                "padded": {
                    "clean": padded_clean,
                    "poisoned": padded_poisoned,
                },
            })

            metrics_k2 = compare_rankings(top_k2_clean, top_k2_poisoned)
            yield _sse({
                "type": "done", "ts": req.ts,
                "metrics_k2": _metrics_to_json(metrics_k2),
            })
        except Exception as e:
            log.exception("Partial rerank failed (ts=%s)", req.ts)
            yield _sse({
                "type": "error", "ts": req.ts,
                "message": f"{type(e).__name__}: {e}",
            })

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/generate")
async def api_generate(req: GenerateRequest) -> dict:
    """
    Generate clean + poisoned answers from frontend-supplied top_k2.
    No server-side cache; server is fully stateless about Runs.
    Response echoes `ts` for frontend staleness checking.
    生成 clean + poisoned 答案;前端传 top_k2,server 无 cache。响应
    带 ts,前端校验是否过时。
    """
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")

    top_k2_clean = [_wire_to_rr(w) for w in req.top_k2_clean]
    top_k2_poisoned = [_wire_to_rr(w) for w in req.top_k2_poisoned]

    _pipeline.generator.llm = make_client(
        config.AVAILABLE_LLMS[config.GENERATOR_LLM]
    )

    t = time.perf_counter()
    answer_clean = await asyncio.to_thread(
        _pipeline.generator.generate, req.query, top_k2_clean,
    )
    answer_poisoned = await asyncio.to_thread(
        _pipeline.generator.generate, req.query, top_k2_poisoned,
    )
    return {
        "ts": req.ts,
        "clean": answer_clean,
        "poisoned": answer_poisoned,
        "elapsed": time.perf_counter() - t,
    }
