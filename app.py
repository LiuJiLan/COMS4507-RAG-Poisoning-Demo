"""
Streamlit UI 入口。

启动：
    streamlit run app.py

UI 结构：
- 左侧：sidebar 导航 + 知识库信息
- 右侧：query 输入 + 双栏对比（clean vs poisoned）+ 指标
"""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# === 噪音抑制（必须在 HF / transformers 加载前）===
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# 抑制 transformers v5 lazy-load 的噪音 warning（image_processing_yolos / zoedepth 等）
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# 抑制 huggingface_hub 的 HF_TOKEN unauthenticated 警告（走自己的 logger,不是 Python warnings）
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
# Resource caching: pipeline is loaded once and reused
# ============================================================
@st.cache_resource(show_spinner="Loading RAG pipeline...")
def get_pipeline():
    """
    Build (or load cached) RAG pipeline.
    Streamlit caches this across reruns, so it only runs once per session.
    """
    pipeline = RAGPipeline(
        embedding_model=config.EMBEDDING_MODEL,
        embedding_device=config.EMBEDDING_DEVICE,
        top_k_1=config.TOP_K_1,
        top_k_2=config.TOP_K_2,
    )

    # Try to load cached index first
    if config.FAISS_CACHE.exists() and config.DOCS_CACHE.exists():
        try:
            pipeline.load_cached_index(config.FAISS_CACHE, config.DOCS_CACHE)
            return pipeline
        except Exception as e:
            st.warning(f"Cache load failed ({e}), rebuilding...")

    # Build from scratch (combine BASE + BACKGROUND, ADJ-001)
    for f in (config.BASE_CORPUS_FILE, config.BACKGROUND_CORPUS_FILE):
        if not f.exists():
            st.error(f"Corpus file not found: {f}")
            st.stop()

    base_docs = load_corpus(config.BASE_CORPUS_FILE)
    background_docs = load_corpus(config.BACKGROUND_CORPUS_FILE)
    pipeline.initialize(base_docs + background_docs)

    # Save cache for next time
    try:
        pipeline.retriever.save(config.FAISS_CACHE, config.DOCS_CACHE)
    except Exception as e:
        st.warning(f"Failed to save cache: {e}")

    return pipeline


def _source_tag(doc) -> str:
    """ADJ-001: 给 top-k 列表里的文档贴一个来源 tag。poison 文档用 ☣,这里返回 "" 让 marker 处理。"""
    if doc.is_poison:
        return ""
    return " `BG`" if doc.source == "msmarco" else " `BASE`"


@st.cache_data
def load_poison_options() -> dict:
    """Load all available poison sets, return {filename: [Document, ...]}"""
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

    st.markdown("---")
    st.markdown("### Pipeline status")
    pipeline = get_pipeline()
    _docs = pipeline.retriever.documents
    n_background = sum(1 for d in _docs if d.source == "msmarco")
    n_base = sum(1 for d in _docs if d.source != "msmarco" and not d.is_poison)
    st.metric("Background docs", n_background)
    st.metric("Base docs", n_base)
    st.metric("Poison docs", pipeline.retriever.n_poison)

    st.markdown("---")
    st.caption(f"Embedding: `{config.EMBEDDING_MODEL.split('/')[-1]}`")
    st.caption(f"Top-K1: {config.TOP_K_1} → Top-K2: {config.TOP_K_2}")


# ============================================================
# Main: Dashboard page
# ============================================================
st.title(config.APP_TITLE)

if page == "Dashboard":
    st.markdown(
        "Demonstrates how poison documents can hijack the retrieval ranking "
        "of a RAG system. Compare *clean* vs *poisoned* knowledge bases."
    )

    # ----- Inputs -----
    col_input, col_attack, col_reranker = st.columns([3, 2, 2])

    with col_input:
        query = st.text_input(
            "Query",
            value="best Chinese restaurant in Brisbane",
            help="The user's question.",
        )

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
            help="Which LLM does the listwise reranking. 'stub' skips the API call and preserves dense retriever order (useful for debugging without spending quota).",
        )
        use_generator = st.checkbox(
            "Include generator",
            value=False,
            key="include_generator",
            help=(
                f"Run the LLM generator on top-k₂ docs to produce a natural-language "
                f"answer (clean vs poisoned, shown as Stage 3). Adds 2 LLM calls per "
                f"run (~$0.02 with {config.GENERATOR_LLM})."
            ),
        )
        st.caption("+2 calls, ~$0.02")

    run_button = st.button("Run experiment", type="primary", use_container_width=False)

    # ----- Run pipeline -----
    if run_button and selected_poison:
        # Swap in the chosen reranker client (cached pipeline keeps embedder /
        # retriever / FAISS index, only the LLM client gets replaced per run).
        if selected_reranker == "stub":
            pipeline.reranker.llm = StubLLMClient()
        else:
            pipeline.reranker.llm = make_client(config.AVAILABLE_LLMS[selected_reranker])

        # Swap generator client only when the user opts in. When the toggle is OFF
        # we leave the cached pipeline's generator alone (run_experiment skips it).
        if use_generator:
            pipeline.generator.llm = make_client(config.AVAILABLE_LLMS[config.GENERATOR_LLM])

        spin_msg = f"Running retrieval comparison (reranker: {selected_reranker}"
        if use_generator:
            spin_msg += f", generator: {config.GENERATOR_LLM}"
        spin_msg += ")..."

        with st.spinner(spin_msg):
            result = pipeline.run_experiment(
                query=query,
                poison_docs=selected_poison,
                include_generator=use_generator,
            )

        # ----- Display: k_1 stage (dense retriever) -----
        st.markdown(f"## Stage 1: Dense retrieval (top-{config.TOP_K_1})")
        col_clean, col_poison = st.columns(2)

        with col_clean:
            st.markdown("#### Without poison")
            for r in result.top_k1_clean:
                st.markdown(
                    f"**{r.rank}.**{_source_tag(r.doc)} {r.doc.title}  \n"
                    f"&nbsp;&nbsp;&nbsp;<small>score: {r.score:.4f}</small>",
                    unsafe_allow_html=True,
                )

        with col_poison:
            st.markdown("#### With poison")
            for r in result.top_k1_poisoned:
                marker = " ☣" if r.doc.is_poison else ""
                color = "#A32D2D" if r.doc.is_poison else "inherit"
                st.markdown(
                    f"<span style='color: {color}'>"
                    f"**{r.rank}.**{marker}{_source_tag(r.doc)} {r.doc.title}</span>  \n"
                    f"&nbsp;&nbsp;&nbsp;<small>score: {r.score:.4f}</small>",
                    unsafe_allow_html=True,
                )

        # Metrics row for k_1
        st.markdown("**Stage 1 metrics**")
        m1 = result.metrics_k1
        c1, c2, c3 = st.columns(3)
        c1.metric("Attack success", "Yes" if m1.poison_in_topk else "No")
        c2.metric("Best poison rank",
                  m1.poison_rank if m1.poison_rank is not None else "—")
        c3.metric("Docs displaced", len(m1.displaced_docs))

        st.markdown("---")

        # ----- Display: k_2 stage (LLM reranker) -----
        st.markdown(f"## Stage 2: LLM reranker (top-{config.TOP_K_2})")
        if selected_reranker == "stub":
            st.caption("Reranker: **STUB** — no API call, order = dense retriever output")
        else:
            _model_name = config.AVAILABLE_LLMS[selected_reranker]["model"]
            st.caption(f"Reranker: `{_model_name}` via OpenRouter")

        col_clean2, col_poison2 = st.columns(2)
        with col_clean2:
            st.markdown("#### Without poison")
            for r in result.top_k2_clean:
                st.markdown(
                    f"**{r.rank}.**{_source_tag(r.doc)} {r.doc.title}  \n"
                    f"&nbsp;&nbsp;&nbsp;<small>orig score: {r.score:.4f}</small>",
                    unsafe_allow_html=True,
                )

        with col_poison2:
            st.markdown("#### With poison")
            for r in result.top_k2_poisoned:
                marker = " ☣" if r.doc.is_poison else ""
                color = "#A32D2D" if r.doc.is_poison else "inherit"
                st.markdown(
                    f"<span style='color: {color}'>"
                    f"**{r.rank}.**{marker}{_source_tag(r.doc)} {r.doc.title}</span>  \n"
                    f"&nbsp;&nbsp;&nbsp;<small>orig score: {r.score:.4f}</small>",
                    unsafe_allow_html=True,
                )

        # Metrics row for k_2
        st.markdown("**Stage 2 metrics**")
        m2 = result.metrics_k2
        c1, c2, c3 = st.columns(3)
        c1.metric("Attack success", "Yes" if m2.poison_in_topk else "No")
        c2.metric("Best poison rank",
                  m2.poison_rank if m2.poison_rank is not None else "—")
        c3.metric("Docs displaced", len(m2.displaced_docs))

        # ----- Display: Stage 3 (LLM generator answer) -----
        if use_generator:
            st.markdown("---")
            st.markdown("## Stage 3: Generated answer")
            _gen_model = config.AVAILABLE_LLMS[config.GENERATOR_LLM]["model"]
            st.caption(f"Generator: `{_gen_model}` via OpenRouter")
            col_clean3, col_poison3 = st.columns(2)
            with col_clean3:
                st.markdown("#### Without poison")
                st.markdown(result.answer_clean or "_(empty)_")
            with col_poison3:
                st.markdown("#### With poison")
                st.markdown(result.answer_poisoned or "_(empty)_")

    elif run_button:
        st.error("No poison set selected.")

elif page == "Attack Module":
    st.info("TODO: Attack design / custom poison upload. Coming soon.")

elif page == "Experiment":
    st.info("TODO: Batch experiment mode (all queries × all poison types). Coming soon.")

elif page == "History":
    st.info("TODO: Past experiment results. Coming soon.")
