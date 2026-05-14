# RAG 投毒攻击 Demo

> Also available in: [English](./README.md)

本项目是 UQ COMS4507 课程小组项目,用于评估数据投毒攻击对 RAG 系统在 retrieval 与 rerank 阶段的影响。

当前版本为 **v1.1 final**。主实验已完成:`30 queries × 5 attacks × 4 reranker LLMs = 600 rows`。实验数据与图表已纳入 Git 跟踪,位于 `data/results/expr_v1_final.*`。

攻击者向知识库中注入少量 **poison documents**,试图改变 retrieval 阶段的 top-k 排名。本 demo 提供一个可视化对比界面,用于展示 **干净知识库 S** 与 **被投毒知识库 S + P_x** 在两个检索阶段中的排名差异。

---

## 研究目标

本项目关注的问题是:少量投毒文档能否在 RAG pipeline 的 retrieval 或 rerank 阶段进入 top-k,从而影响后续 LLM 可见的 context。

本实验中的攻击成功定义为:

> **只要排名发生改变,即视为攻击成功。**

因此,本项目的评估重点是 retrieval 层面的 **rank shift**,而不是最终 LLM 生成答案是否被误导。

### 威胁背景

RAG 系统的回答质量高度依赖底层知识库。如果攻击者能够向 corpus 中注入少量精心构造的文档,即使这些文档数量很少,也可能在 retrieval 或 rerank 阶段被推入 top-k,从而污染下游 LLM 看到的 context。

在本实验中,投毒文档只占整个知识库 S 的很小比例(每套 P_x 约 22–30 条,小于 1%),但仍可能显著影响检索排名。本项目系统性比较了 5 种 attack 在 4 个主流 LLM reranker 下的攻击效果与鲁棒性差异。

---

## Pipeline 架构

```mermaid
flowchart TB
    %% ===== 输入层 =====
    Q([User Query]):::input
    subgraph KB[静态知识库 S]
        direction TB
        BG[("背景集合<br/>MS MARCO ~5000<br/><span style='font-size:0.85em'>通用噪声</span>")]:::input
        BASE[("基准集合<br/>Brisbane ~300<br/><span style='font-size:0.85em'>主题对齐</span>")]:::input
    end
    PX[(污染集 P_x)]:::input

    %% ===== RAG Pipeline =====
    subgraph RAG[RAG Pipeline]
        direction TB
        EMB["Embedding<br/><span style='font-size:0.85em'>sentence-transformers</span>"]:::internal
        FAISS["FAISS 向量检索<br/><span style='font-size:0.85em'>top-k₁ = 10</span>"]:::internal
        K1[/"top-k₁ 排名对比<br/>clean vs poisoned"/]:::display
        RER["LLM Reranker<br/><span style='font-size:0.85em'>多模型对比维度</span>"]:::internal
        K2[/"top-k₂ 排名对比<br/>clean vs poisoned"/]:::display
        GEN["LLM Generator<br/><span style='font-size:0.85em'>固定单模型</span>"]:::internal
        ANS[/"自然语言答案<br/>clean vs poisoned"/]:::display

        EMB --> FAISS --> K1 --> RER --> K2 --> GEN --> ANS
    end

    Q --> EMB
    BG --> EMB
    BASE --> EMB
    PX -. 运行时注入 .-> EMB

    classDef input fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a,stroke-width:1.5px
    classDef internal fill:#f3f4f6,stroke:#4b5563,color:#1f2937,stroke-width:1.2px
    classDef display fill:#fde68a,stroke:#b45309,color:#78350f,stroke-width:1.8px
```

### 关键设计

- 每次实验并行运行两条 pipeline:一条使用干净知识库,另一条在运行时注入 poison set。
- 静态知识库 S 在启动时一次性建立索引;污染集 P_x 由 UI 下拉菜单切换,并在运行时注入 FAISS 索引,实验结束后清理。
- k₁ 阶段对应 dense retriever 的输出;k₂ 阶段对应 LLM reranker 的输出。
- UI 同时展示 clean 与 poisoned 两侧结果,便于观察 reranker 介入前后的排名变化。

---

## LLM 在 Pipeline 中的角色

| 角色 | 模型策略 | Temperature | 研究定位 |
|---|---|---:|---|
| **Reranker** | 多模型对比:Claude / GPT-4o-mini / Gemini / Llama | 0.0 | 核心研究维度,用于比较不同 LLM 作为 reranker 时的鲁棒性 |
| **Generator** | 固定单一模型,默认 Claude | 0.3 | 用于 demo 展示,避免 reranker × generator 的组合数量爆炸 |

4 个 reranker 均通过 OpenRouter 调用:

- `anthropic/claude-sonnet-4.5`
- `openai/gpt-4o-mini`
- `google/gemini-2.5-flash-lite`
- `meta-llama/llama-3.3-70b-instruct`

### Gemini 注记

在 v1 主实验进行过程中,`google/gemini-2.0-flash-001` 在 OpenRouter 上接近下架,并触发 RPM hard cap。因此,Gemini 相关实验改用同价位、同 provider 的 `google/gemini-2.5-flash-lite` 重新运行全部 150 rows。`expr_v1_final.csv` 中的 Gemini 数据全部来自 `google/gemini-2.5-flash-lite`。

---

## 评估指标

本项目的主指标全部聚焦 retrieval 层面:

- `poison_in_topk`:poison 是否进入 top-k。
- `poison_rank`:poison 的具体排名,排名越靠前代表攻击越强。
- `displaced_docs`:被挤出 top-k 的原始文档列表。
- `score_gap`:poison 分数与 clean top-1 分数之间的差距。

LLM 生成答案是否被误导不是本项目的评估指标。Generator 阶段仅用于 demo 展示。

---

## 数据来源与复现说明

本项目的主要数据 artifact 均已纳入 Git 跟踪。默认建议直接使用 committed 文件,无需重新生成数据或重新运行实验。

下表列出了各 artifact 的来源、关键参数与复现性。含 LLM 生成的步骤无法做到字节级复现,但应能得到语义等价的结果。

| Artifact | 路径 | 生成脚本 | 关键参数 | 模型版本 | 预估成本 | 严格可复现 |
|---|---|---|---|---|---:|:---:|
| MS MARCO 背景语料<br/>5000 docs | `data/corpus_static/msmarco_background.json` | `scripts/prepare_msmarco.py` | HuggingFace `ms_marco` v2.1 train split;`seed=42`;oversample 5200 后取每行第一个非空 `passage_text`,最终保留 5000 条 | 无,纯数据抽样 | $0 | 是 |
| Brisbane 主题语料<br/>290 docs | `data/corpus_static/brisbane_corpus.json` | `data/corpus_static/_brisbane_source/main.py`<br/>辅助参考,详见同目录 README | Wikipedia 抓取、手工 curated 餐厅、UQ 模板填充;共 5 类 topic | 无 | $0 | 否 |
| 测试 query<br/>30 条 | `data/test_queries.yaml`<br/>`data/query_targets.yaml` | 手工 curated,GPT 辅助修订 | 5 类 category 均衡;`query_targets.yaml` 为每条 query 标注 `poison_target` 与 `target_type`,包括 18 条 fictional_entity、8 条 false_fact、4 条 misleading_recommendation | 无 | $0 | 是 |
| Poison 文档<br/>129 条 | `data/poison_sets/P_keyword_stuffing.json`<br/>`data/poison_sets/P_structured_format.json`<br/>`data/poison_sets/P_semantic_mimicry.json`<br/>`data/poison_sets/P_authority_spoof.json`<br/>`data/poison_sets/P_contradiction.json` | `scripts/generate_poisons.py --all` | 输出 100–200 词,validator 允许 220 词作为 10% buffer;per-attack temperature 为 0.5 或 0.7;5 attacks × 30 queries 覆盖矩阵,其中 `keyword_stuffing` 与 `structured_format` 跳过 `false_fact` target_type,`contradiction` 在 corpus 缺少可矛盾事实的 5 条 query 上 skip | `openai/gpt-4o`,固定 model ID;不使用会自动更新的 `gpt-chat-latest` | ~$0.51 | 否 |
| 主实验结果<br/>600 rows | `data/results/expr_v1_final.csv`<br/>`data/results/expr_v1_final.png`<br/>`data/results/expr_v1_final_by_llm.png` | `scripts/run_experiment.py` | 30 queries × 5 attacks × 4 reranker LLMs = 600 组;`TOP_K_1=10`,`TOP_K_2=5` | Reranker:`anthropic/claude-sonnet-4.5`、`openai/gpt-4o-mini`、`google/gemini-2.5-flash-lite`、`meta-llama/llama-3.3-70b-instruct`<br/>Generator:`anthropic/claude-sonnet-4.5` | ~$1.50 | 否 |

### 复现注记

- **Brisbane 语料不可严格复现**:`_brisbane_source/main.py` 是早期生成脚本的副本。当前 committed 的 `brisbane_corpus.json` 经过外部 post-processing,修复了模板填充时出现的重复句问题。因此,直接重跑 `main.py` 不会得到与 committed 文件严格一致的输出。详见 `data/corpus_static/_brisbane_source/README.md`。
- **Poison 与主实验不可字节级复现**:即使 temperature 较低,LLM 输出也不是完全确定的。重跑后应得到语义等价但字节不同的结果。主实验 ASR 数字预计会在几个百分点范围内波动,但核心 finding 应保持稳定。
- **Gemini 列的额外说明**:`expr_v1_final.csv` 的 Gemini 数据在收集过程中曾从 `google/gemini-2.0-flash-001` 切换为 `google/gemini-2.5-flash-lite`,原因是前者在 OpenRouter 上接近下架,容量收缩导致 RPM 撞顶。当前 final CSV 的 Gemini 数据全部来自后者。

### 使用建议

1. 默认使用 Git 中已提交的 artifact,直接读取 `data/` 下的文件即可复现所有下游分析。
2. 如需重跑不可严格复现的步骤,输出可能语义等价但字节不同;主实验 ASR 数字预计会有几个百分点的波动。
3. 全量重跑成本约为 **$2 USD**,其中 poison 生成约 $0.51,主实验约 $1.50。MS MARCO 抽样与 query 文件不产生 LLM 成本。

---

## 核心发现

以下结果来自 `data/results/expr_v1_final.csv`,共 600 rows,即 30 queries × 5 attacks × 4 reranker LLMs。

![5 种 attack 在 4 个 reranker LLM 上的平均 ASR](data/results/expr_v1_final.png)

*图 1:5 种 attack 的平均 ASR(across 4 个 reranker LLM)。数据来源:`expr_v1_final.csv`。*

### 1. Authority Spoof 对 4 个 LLM 几乎通杀,ASR 达 97%

Authority Spoof 通过伪装权威来源实现攻击,例如虚构 institution、DOI 或 "Cambridge study" 等签名词。

该攻击在 4 个 reranker LLM 上的 ASR 均为 **97%**。这是 5 种 attack 中平均 ASR 最高、模型间差异最小的一类。结果说明,主流 LLM reranker 对"权威信号"的抵抗能力非常有限。

### 2. Contradiction 的模型差异最大,Claude 最稳健

![不同 LLM × attack 的 ASR 细分](data/results/expr_v1_final_by_llm.png)

*图 2:同一 attack 在不同 reranker LLM 下的 ASR 分布。Contradiction 一列的 13pp gap 在图中尤为明显。*

Contradiction 构造与 corpus 中已有事实直接冲突的 poison 文档。该攻击在不同 LLM 之间产生了最大的 ASR 差异。

| Reranker | ASR |
|---|---:|
| `claude-sonnet-4.5` | **77%**,最稳健 |
| `gpt-4o-mini` | 83% |
| `gemini-2.5-flash-lite` | 87% |
| `llama-3.3-70b` | **90%**,最容易被攻击 |

13 个百分点的模型间差距远高于后续讨论中的 API artifact 影响,因此这是一个稳健的主 finding。Claude 在面对与已有 fact 直接冲突的 poison 时,表现出比其他三个 LLM 更强的辨识能力。

### 3. Keyword Stuffing 是唯一被 reranker 明显削弱的攻击

| 阶段 | ASR |
|---|---:|
| dense retriever,即 k₁ 阶段 | 73% |
| LLM reranker,即 k₂ 阶段 | **59%** |

ASR 下降 14 个百分点,是 5 种 attack 中唯一在 reranker 阶段被显著削弱的一类。结果说明,LLM reranker 对明显的堆词式 surface attack 具备一定辨识能力,但对 Authority Spoof、Semantic Mimicry 与 Contradiction 等语义层攻击几乎不能形成有效防御。

### 4. 不同 LLM 的 reranker raw output 完整性存在差异

跑批 telemetry 中的 `reranker_padded_clean` 与 `reranker_padded_poisoned` 列显示,4 个 LLM 在 listwise rerank prompt 上的输出完整性存在明显差异。

| LLM | anomaly rows / 150 | 占比 | 主要类型 |
|---|---:|---:|---|
| `claude-sonnet-4.5` | 1 | 0.7% | 单次 padded 7,极罕见 |
| `gemini-2.5-flash-lite` | 3 | 2.0% | LLM under-output;全为 padded 5;无 API fail |
| `llama-3.3-70b` | 4 | 2.7% | 偶发 padded 1;单次 padded 9 |
| `gpt-4o-mini` | **20** | **13.3%** | 全部为 LLM under-output,padded 5 最常见(8 rows),其次 padded 1(5 rows)|

口径:任一侧(`reranker_padded_clean` 或 `reranker_padded_poisoned`)大于 0 即计为 anomaly row。

GPT-4o-mini 存在稳定的 under-output 现象:在对 10 项进行 listwise rerank 时,它经常只输出前 1–9 个结果,后续项目由 parser 按原顺序补齐。Claude 几乎没有此问题;Gemini 切换到 `2.5-flash-lite` 后,API fail 降为 0。

需要强调的是,我们的最终指标只看 top-5(`TOP_K_2 = 5`)。只有 LLM 输出 **少于 5 个**排名(即 `padded > 5`)时,top-5 才会被 parser 的 dense-fill 污染。在 final 主实验 600 row 中,只有 3 row(0.5%)真正受此影响。详细 sensitivity 分析见 Limitations 节。

### 5. Gemini 并非完全透传 dense retriever 顺序

剔除 Gemini 列中 LLM under-output 的 3 rows 后,Gemini 在 surface attack(即 Keyword Stuffing 与 Structured Format)上的 ASR 略低于跨 4 个 LLM 的平均 ASR,差距为几个百分点。

这说明 Gemini 并不是完全放任 dense retriever 的原始顺序,而是做了一定独立判断。不过,这一差异远小于 Contradiction 中 13 个百分点的主差距。

---

## 运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

本项目需要 Python 3.10。Windows、macOS 与 Linux 均可运行。GPU 为可选项;`build_index.py` 在 CPU 上也只会多花几秒。

### 2. 配置环境变量

复制 `.env.example` 为 `.env`,并填写以下环境变量:

- `OPENROUTER_API_KEY`:必需。所有 LLM 调用均通过 OpenRouter。
- `HF_TOKEN`:可选。仅在重跑 `prepare_msmarco.py` 并下载 HuggingFace dataset 时需要。

### 3. 运行环境健康检查

```bash
python test.py
```

正常情况下应输出 10/10 通过。

### 4. 准备 corpus 与索引

```bash
# 通常无需执行:msmarco_background.json 已提交到仓库
python scripts/prepare_msmarco.py

# 构建 FAISS 索引,GPU 约 3 秒,CPU 也只需几秒
python scripts/build_index.py
```

### 5. 启动 demo UI

```bash
python -m web.main
```

浏览器会自动打开 `http://127.0.0.1:8000`。交互式 dashboard(FastAPI + 纯 vanilla JS,无额外打包步骤)展示:

- 实时 pipeline 时间线:query → embedding → FAISS top-k₁ → LLM reranker → top-k₂ →(可选)LLM generator,每个 stage 通过 Server-Sent Events 同步动画。
- 每个 stage 都以 clean / poisoned 双列并列呈现。
- 下拉框切换 5 套 poison set;另一个下拉框切换 reranker LLM;再次切回之前看过的 reranker 命中前端缓存,免去 API 调用。
- 顶部 Database 框显示三个生命周期 timer(first build / last inject poison / last remove poison),不同阶段以不同颜色 shimmer。
- Stage 3 toggle 打开 generator,显示 clean / poisoned 的自然语言答案(约 `$0.02/run`,按 `(ts, reranker)` 缓存)。

![Dashboard 截图 — 顶部 Query / Database row,下接完整 5 stage pipeline(Embedding → FAISS Vector Retrieval → LLM Reranker → LLM Generator),clean 与 poisoned 两列并列,Stage 3 toggle 已打开](docs/dashboard.png)

### 6. 开发期 smoke test

```bash
python scripts/quickrun.py
```

默认使用 `data/dev_fixtures/P_demo.json`,即单 query × 单 LLM × dummy poison。该流程需要 OpenRouter API key。如果将文件顶部的 `USE_STUB_RERANKER` 改为 `True`,则可在零 API 消耗的情况下纯本地跑通。

### 7. 全量复跑主实验,可选

```bash
python scripts/run_experiment.py
```

该命令会运行 30 queries × 5 attacks × 4 LLMs,共 600 rows。预估成本约为 `$1.50 USD`,耗时约 30–60 分钟,具体取决于 OpenRouter latency。

结果将写入:

- `data/results/expr_<timestamp>.csv`
- `data/results/expr_<timestamp>.meta.json`
- `data/results/expr_<timestamp>.errors.log`
- `data/results/expr_<timestamp>.stdout.log`

输出中包含 git hash 与 config snapshot,便于后续追溯。

---

## Limitations

- **Corpus 规模有限**:基准语料仅包含 290 篇 Brisbane 文档与 5000 篇 MS MARCO 噪声文档,远小于真实 RAG 系统。实验结论是否能外推到 10k+ 规模的语料仍需进一步验证。
- **Generator 固定为单一模型**:为避免 reranker × generator 的组合数量爆炸,generator 固定为 Claude。因此,generator 阶段的攻击表现不属于本研究覆盖范围。
- **4 个 reranker LLM 不是完全等价的 baseline**:`gpt-4o-mini` 在 13.3% 的 rows 上 LLM 输出不完整,后续由 parser 按 dense 原序补齐。但因为我们的最终指标只看 top-5(`TOP_K_2 = 5`),只有 LLM 输出**少于 5 个**排名时(即 `padded > 5`)top-5 才会被污染。在 final 主实验 600 row 中,只有 3 row(0.5%)真正受此影响。Sensitivity 分析:剔除这 3 row 后,各 attack × LLM 组合的 ASR delta 全部 ≤ 0.6 pp,F1(Authority Spoof 通杀)/ F2(Contradiction 13 pp gap)/ F3(Keyword Stuffing 唯一被识破)三条主 finding 与全 600 row 完全一致。CSV 中的 `reranker_padded_*` 列暴露此信号,便于复现该 sensitivity 分析。
- **OpenRouter 容量存在抖动**:第三方 LLM gateway 偶尔会出现 429 或 504。主实验已使用指数退避与 retry-with-backoff 缓解该问题,但极端情况下仍可能 fallback 到 dense retriever order。CSV schema 已暴露相关信号,可在 post-hoc 分析中剔除受影响 rows。
- **项目定位为研究 demo**:本项目不是生产级 RAG 系统,未实现 auth、persistence、chunking、用户上传 corpus 等工程能力。
