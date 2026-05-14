"""
Streamlit UI entry point.
Streamlit UI 入口。

Launch:
    streamlit run app.py

Layout:
- Left:  sidebar navigation + knowledge-base info
- Right: query input + side-by-side comparison (clean vs poisoned) + metrics

UI 结构:左 sidebar 导航 + 知识库信息;右 query 输入 + clean vs poisoned 双栏对比 + 指标。
"""
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# === Noise suppression — must precede HF / transformers imports ===
# === 噪音抑制(必须在 HF / transformers 加载前) ===
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Silence transformers v5 lazy-load warnings (image_processing_yolos / zoedepth etc.).
# 抑制 transformers v5 lazy-load 的噪音 warning(image_processing_yolos / zoedepth 等)。
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# Silence huggingface_hub's HF_TOKEN unauthenticated warning (its own logger, not Python warnings).
# 抑制 huggingface_hub 的 HF_TOKEN unauthenticated 警告(走自己的 logger,不是 Python warnings)。
from huggingface_hub.utils import logging as hub_logging
hub_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import streamlit as st
import json

import config
from src.corpus import load_corpus, load_poison_set
from src.pipeline import RAGPipeline
from src.llm_clients import make_client, StubLLMClient


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# Resource caching: pipeline is loaded once and reused.
# 资源缓存:pipeline 只加载一次,session 内复用。
# ============================================================
@st.cache_resource(show_spinner="Loading RAG pipeline...")
def get_pipeline():
    """
    Build (or load cached) RAG pipeline. Streamlit caches this across reruns,
    so it only runs once per session.
    构造或从 cache 加载 RAG pipeline。Streamlit 跨 rerun 缓存,session 内只跑一次。
    """
    pipeline = RAGPipeline(
        embedding_model=config.EMBEDDING_MODEL,
        embedding_device=config.EMBEDDING_DEVICE,
        top_k_1=config.TOP_K_1,
        top_k_2=config.TOP_K_2,
    )

    # Try cached index first.
    # 先尝试加载缓存索引。
    if config.FAISS_CACHE.exists() and config.DOCS_CACHE.exists():
        try:
            pipeline.load_cached_index(config.FAISS_CACHE, config.DOCS_CACHE)
            return pipeline
        except Exception as e:
            st.warning(f"Cache load failed ({e}), rebuilding...")

    # Build from scratch (BASE + BACKGROUND combined; ADJ-001).
    # 从零构建(BASE + BACKGROUND 合并,ADJ-001)。
    for f in (config.BASE_CORPUS_FILE, config.BACKGROUND_CORPUS_FILE):
        if not f.exists():
            st.error(f"Corpus file not found: {f}")
            st.stop()

    base_docs = load_corpus(config.BASE_CORPUS_FILE)
    background_docs = load_corpus(config.BACKGROUND_CORPUS_FILE)
    pipeline.initialize(base_docs + background_docs)

    # Save cache for next time.
    # 写一份 cache 供下次启动用。
    try:
        pipeline.retriever.save(config.FAISS_CACHE, config.DOCS_CACHE)
    except Exception as e:
        st.warning(f"Failed to save cache: {e}")

    return pipeline


def _row_decoration(doc) -> tuple[str, str]:
    """
    Return (row_background_css, source_badge_html) for a top-k row, color-coded
    by document origin (ADJ-001). RGBA backgrounds so both light/dark themes work.
    返回 top-k 一行的 (背景 CSS, 来源 badge HTML),按来源着色(ADJ-001);
    用 RGBA 半透明背景,亮/暗 Streamlit 主题都能显示。
        - BG (MS MARCO 背景):浅绿
        - BASE (Brisbane 基准):稍深绿
        - POISON (注入文档):红底 + ☣ 标记 + 左侧红条
    """
    if doc.is_poison:
        # Hazard palette tuned for Streamlit themes: amber/orange base instead
        # of pure yellow, softer translucent red left border.
        # Streamlit 主题专调的警示色:琥珀/橘代替纯黄,左条用更柔和的半透明红。
        bg = "background: rgba(255,152,0,0.20); border-left: 3px solid rgba(198,40,40,0.55);"
        badge = (
            "<span style='background:#FF9800;color:#000;padding:1px 7px;"
            "border-radius:3px;font-size:0.78em;font-weight:bold;"
            "border:1px solid #000;'>&#9763; POISON</span>"
        )
    elif doc.source == "msmarco":
        bg = "background: rgba(76,175,80,0.12);"
        badge = (
            "<span style='background:rgba(76,175,80,0.35);color:#1B5E20;padding:1px 6px;"
            "border-radius:3px;font-size:0.75em;font-weight:600;'>BG</span>"
        )
    else:
        bg = "background: rgba(46,125,50,0.28);"
        badge = (
            "<span style='background:#2E7D32;color:white;padding:1px 6px;"
            "border-radius:3px;font-size:0.75em;font-weight:bold;'>BASE</span>"
        )
    return bg, badge


def _topk_row_html(r, score_label: str = "score") -> str:
    """
    Return one top-k row as a self-contained HTML string. The card is built
    with height:100% so that, when placed inside a CSS-grid cell with
    align-items:stretch, it expands to the row's tallest cell. r=None
    renders an empty placeholder that still participates in grid sizing.
    返回单行 top-k 的自包含 HTML(不调 st.markdown)。卡片用 height:100%,
    放在 align-items:stretch 的 CSS grid cell 里会自动撑满该行最高 cell。
    r=None 渲染一个空占位 cell,仍参与 grid 布局。
    """
    if r is None:
        return "<div></div>"
    _bg, _badge = _row_decoration(r.doc)
    return (
        f"<div style='{_bg} padding:8px 12px; border-radius:4px; "
        f"min-height:64px; height:100%; box-sizing:border-box; "
        f"display:flex; flex-direction:column; justify-content:flex-start;'>"
        f"<div><b>{r.rank}.</b> {r.doc.title}</div>"
        f"<div style='margin-top:auto; padding-top:4px;'>{_badge}"
        f"<small style='opacity:0.7; margin-left:8px;'>"
        f"{score_label}: {r.score:.4f}</small></div>"
        f"</div>"
    )


def _render_topk_pair(clean_list, poisoned_list,
                      score_label: str = "score") -> None:
    """
    Render clean vs poisoned top-k as ONE CSS grid block so the two cells
    in the same grid row stretch to equal height. This avoids the bottom-
    padding mismatch you'd otherwise see when one side's title wraps to a
    new line and the other side doesn't.
    把 clean / poisoned 双栏整张表用单个 CSS grid 渲染:同一 grid row 的
    两 cell 自动等高,避免一侧 title wrap / 另一侧不 wrap 时底部 padding 不齐。
    """
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Without poison")
    with col_r:
        st.markdown("#### With poison")

    k = max(len(clean_list), len(poisoned_list))
    parts = [
        "<div style='display:grid; grid-template-columns:1fr 1fr; "
        "gap:6px 16px; align-items:stretch;'>"
    ]
    for i in range(k):
        parts.append(_topk_row_html(
            clean_list[i] if i < len(clean_list) else None,
            score_label,
        ))
        parts.append(_topk_row_html(
            poisoned_list[i] if i < len(poisoned_list) else None,
            score_label,
        ))
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


@st.cache_data
def load_poison_options() -> dict:
    """
    Load all available poison sets. Returns {filename_stem: [Document, ...]}.
    加载所有可用 poison 集合,返回 {文件名: [Document, ...]}。
    """
    options = {}
    if config.POISON_DIR.exists():
        for p in sorted(config.POISON_DIR.glob("*.json")):
            try:
                options[p.stem] = load_poison_set(p)
            except Exception as e:
                logging.warning(f"Failed to load {p}: {e}")
    return options


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Page",
        config.SIDEBAR_PAGES,
        label_visibility="collapsed",
    )

# Pipeline must be initialized at page level — used by Dashboard for retriever
# stats, reranker / generator client swapping, and run_experiment dispatch.
# pipeline 必须在 page-level 初始化:Dashboard 多处使用(retriever 统计、reranker
# / generator client 替换、run_experiment 调度)。
pipeline = get_pipeline()


# ============================================================
# Main: Dashboard page
# ============================================================
st.title(config.APP_TITLE)

if page == "Dashboard":
    st.markdown(
        "Demonstrates how poison documents can hijack the retrieval ranking "
        "of a RAG system. Compare *clean* vs *poisoned* knowledge bases."
    )

    # ----- Knowledge base (built into the FAISS index at startup) -----
    # ----- 知识库(启动时构建进 FAISS 索引)-----
    st.markdown("## Knowledge base")
    _docs = pipeline.retriever.documents
    _n_background = sum(1 for d in _docs if d.source == "msmarco")
    _n_base = sum(1 for d in _docs
                  if d.source != "msmarco" and not d.is_poison)
    _n_poison_sets = len(load_poison_options())

    kb_c1, kb_c2, kb_c3 = st.columns(3)
    kb_c1.metric("Background docs (BG)", _n_background)
    kb_c2.metric("Base docs (BASE)", _n_base)
    kb_c3.metric(
        "Poison sets available", _n_poison_sets,
        help="How many poison sets you can pick below. Poison documents "
             "are injected into the FAISS index ONLY during a Run experiment "
             "and removed afterwards — 'active poison' in the index is "
             "always 0 outside a run.",
    )
    st.caption(
        f"Embedding: `{config.EMBEDDING_MODEL.split('/')[-1]}` · "
        f"Top-K1: {config.TOP_K_1} → Top-K2: {config.TOP_K_2}"
    )

    st.markdown("---")

    # ----- Inputs -----
    col_input, col_attack, col_reranker = st.columns([3, 2, 2])

    with col_input:
        query = st.text_input(
            "Query",
            value="best Chinese restaurant in Brisbane",
            help="The user's question.",
        )

    poison_name: str | None = None
    with col_attack:
        poison_options = load_poison_options()
        if not poison_options:
            st.warning("No poison sets found in data/poison_sets/")
            selected_poison = None
        else:
            poison_name = st.selectbox(
                "Poison set",
                list(poison_options.keys()),
                format_func=lambda x: x.removeprefix("P_").replace("_", " ").title(),
                help="Which poison documents to inject.",
            )
            selected_poison = poison_options[poison_name]

    with col_reranker:
        enabled_llms = [k for k, v in config.AVAILABLE_LLMS.items() if v.get("enabled")]
        rerank_options = enabled_llms + ["stub"]
        selected_reranker = st.selectbox(
            "Reranker model",
            rerank_options,
            index=0,
            help="Which LLM does the listwise reranking. 'stub' skips the API "
                 "call and preserves dense retriever order (useful for debugging "
                 "without spending quota).",
        )

    # ----- Form-drift detection: clear stale results when any input changes -----
    # ----- 表单漂移检测:任一 input 变化时清掉上一次结果(避免显示误导)-----
    current_key = (query, poison_name, selected_reranker)
    if (
        "last_run_key" in st.session_state
        and st.session_state.last_run_key != current_key
    ):
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_run_key", None)
        st.session_state.pop("generator_cache", None)

    run_button = st.button("Run experiment", type="primary",
                           use_container_width=False)

    # ----- Run pipeline (only writes session_state; rendering is below) -----
    # ----- 运行 pipeline:只写 session_state,渲染统一在下方读 -----
    if run_button and selected_poison:
        # Swap in the chosen reranker client (cached pipeline keeps embedder /
        # retriever / FAISS index; only the LLM client is replaced per run).
        # 替换选定的 reranker client(cache 的 pipeline 保留 embedder / retriever
        # / FAISS,每次只换 LLM client)。
        if selected_reranker == "stub":
            pipeline.reranker.llm = StubLLMClient()
        else:
            pipeline.reranker.llm = make_client(config.AVAILABLE_LLMS[selected_reranker])

        spin_msg = (
            f"Running retrieval comparison "
            f"(reranker: {selected_reranker})..."
        )

        # st.status: spinner while running; auto-marks ✗ if the block raises.
        # Stage 3 is now its own toggle below — never run as part of this call.
        # st.status:运行时转圈;块内抛异常自动变 ✗。
        # Stage 3 不再在这里跑,由下方独立 toggle 触发。
        _t_start = time.perf_counter()
        with st.status(spin_msg, expanded=False) as _status:
            result = pipeline.run_experiment(
                query=query,
                poison_docs=selected_poison,
                include_generator=False,
            )
            _elapsed = time.perf_counter() - _t_start
            _status.update(
                label=f"✓ Pipeline complete in {_elapsed:.1f}s",
                state="complete",
            )

        st.session_state.last_result = result
        st.session_state.last_run_key = current_key
        # Reset generator cache on every new run (run_key just changed anyway).
        # 每次新 run 重置 generator cache(run_key 已变)。
        st.session_state.generator_cache = {}

    elif run_button:
        st.error("No poison set selected.")

    # ----- Render: stage 1 + 2 (+ optional stage 3) from session_state -----
    # ----- 渲染区:从 session_state 读 stage 1 + 2(可选 stage 3)-----
    if "last_result" in st.session_state:
        result = st.session_state.last_result

        st.markdown(f"## Stage 1: Dense retrieval (top-{config.TOP_K_1})")
        _render_topk_pair(
            result.top_k1_clean, result.top_k1_poisoned,
            score_label="score",
        )

        st.markdown("**Stage 1 metrics**")
        m1 = result.metrics_k1
        c1, c2, c3 = st.columns(3)
        c1.metric("Attack success", "Yes" if m1.poison_in_topk else "No")
        c2.metric("Best poison rank",
                  m1.poison_rank if m1.poison_rank is not None else "—")
        c3.metric("Docs displaced", len(m1.displaced_docs))

        st.markdown("---")

        st.markdown(f"## Stage 2: LLM reranker (top-{config.TOP_K_2})")
        if selected_reranker == "stub":
            st.caption("Reranker: **STUB** — no API call, order = dense retriever output")
        else:
            _model_name = config.AVAILABLE_LLMS[selected_reranker]["model"]
            st.caption(f"Reranker: `{_model_name}` via OpenRouter")

        _render_topk_pair(
            result.top_k2_clean, result.top_k2_poisoned,
            score_label="orig score",
        )

        st.markdown("**Stage 2 metrics**")
        m2 = result.metrics_k2
        c1, c2, c3 = st.columns(3)
        c1.metric("Attack success", "Yes" if m2.poison_in_topk else "No")
        c2.metric("Best poison rank",
                  m2.poison_rank if m2.poison_rank is not None else "—")
        c3.metric("Docs displaced", len(m2.displaced_docs))

        # ----- Stage 3 toggle (decoupled from stage 1+2 to avoid re-running them) -----
        # Toggling does NOT re-run stage 1+2. First time ON for this run_key:
        # call generator and cache the answers. Subsequent toggles read cache.
        # Stage 3 toggle 与主 run 解耦:切换它不会重跑 stage 1+2。
        # 当前 run_key 首次 ON 时调 generator 并缓存;后续切换直接读 cache。
        st.markdown("---")
        show_stage3 = st.toggle(
            "Show generated answer (Stage 3)",
            value=False,
            key="show_stage3",
            help=(
                f"Run the LLM generator on top-k₂ docs to produce a natural-"
                f"language answer (clean vs poisoned). **Generator model is "
                f"fixed** to `{config.AVAILABLE_LLMS[config.GENERATOR_LLM]['model']}` "
                f"— it does NOT change with the reranker selection above "
                f"(avoids reranker × generator cartesian product; see README). "
                f"Adds 2 LLM calls (~$0.02). Cached per (query, poison set, "
                f"reranker) — flipping OFF then back ON is free."
            ),
        )

        if show_stage3:
            generator_cache = st.session_state.setdefault("generator_cache", {})
            run_key = st.session_state.last_run_key

            if run_key not in generator_cache:
                pipeline.generator.llm = make_client(
                    config.AVAILABLE_LLMS[config.GENERATOR_LLM]
                )
                _gt = time.perf_counter()
                with st.status(
                    f"Generating answers via {config.GENERATOR_LLM}...",
                    expanded=False,
                ) as _gs:
                    ans_clean = pipeline.generator.generate(
                        query, result.top_k2_clean,
                    )
                    ans_poisoned = pipeline.generator.generate(
                        query, result.top_k2_poisoned,
                    )
                    _ge = time.perf_counter() - _gt
                    _gs.update(
                        label=f"✓ Answers generated in {_ge:.1f}s",
                        state="complete",
                    )
                generator_cache[run_key] = (ans_clean, ans_poisoned)

            ans_clean, ans_poisoned = generator_cache[run_key]

            st.markdown("## Stage 3: Generated answer")
            _gen_model = config.AVAILABLE_LLMS[config.GENERATOR_LLM]["model"]
            st.caption(
                f"Generator: `{_gen_model}` via OpenRouter · "
                f"**Fixed across all runs** — independent of the reranker "
                f"selection (avoids reranker × generator cartesian product)."
            )
            if selected_reranker == config.GENERATOR_LLM:
                # Reranker and generator share the same model in this run by
                # coincidence — surface it so the audience doesn't conclude
                # the generator follows the reranker selection.
                # 本次 reranker 与 generator 恰好同模型 — 主动提示,避免观众误
                # 以为 generator 跟随 reranker 选择。
                st.info(
                    f"ℹ️ Note: reranker and generator are the **same model** "
                    f"this run ({_gen_model}). This is a coincidence — switch "
                    f"the reranker to a different LLM above to see the two "
                    f"roles use distinct models."
                )
            col_clean3, col_poison3 = st.columns(2)
            with col_clean3:
                st.markdown("#### Without poison")
                st.markdown(ans_clean or "_(empty)_")
            with col_poison3:
                st.markdown("#### With poison")
                st.markdown(ans_poisoned or "_(empty)_")

elif page == "Attack Module":
    st.info("TODO: Attack design / custom poison upload. Coming soon.")

elif page == "Experiment":
    st.info("TODO: Batch experiment mode (all queries × all poison types). Coming soon.")

elif page == "History":
    st.info("TODO: Past experiment results. Coming soon.")
