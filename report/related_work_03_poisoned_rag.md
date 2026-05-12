# PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models

## Bibliographic info

- **Authors**: Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia
- **Venue**: USENIX Security Symposium 2025 (34th)
- **Year**: 2025 (arXiv preprint February 2024)
- **URL**: https://arxiv.org/abs/2402.07867

------

## Problem

The security of RAG systems has been largely overlooked, with prior work focused almost exclusively on improving retrieval accuracy and generation quality. This paper identifies the knowledge database itself as a new and practically exploitable attack surface: an attacker can inject a small number of carefully crafted malicious texts into the database to induce the LLM to generate attacker-chosen false answers for attacker-chosen target questions. Such attacks can be used to spread misinformation, manipulate commercial recommendations, or propagate false financial and medical information, posing severe threats to RAG deployments in high-stakes domains.

------

## Threat model

- **Attacker's goal**: Cause the RAG system to generate a specific attacker-chosen target answer for a specific attacker-chosen target question — a misinformation injection goal, distinct from the denial-of-service goal of Paper 1

- **Control over the LLM**: None. The attacker cannot access LLM parameters or query it directly

- **Control over the knowledge base**: Cannot read or modify existing content; can only **inject new texts** (e.g., by maliciously editing Wikipedia pages or hosting fake web content)

- Knowledge of the retriever

  : Two settings:

  - **White-box**: Attacker knows the retriever's parameters (realistic when the RAG uses a publicly available retriever, e.g., ChatRTX defaults to the publicly released UAE-Large-V1)
  - **Black-box**: Attacker has no knowledge of and cannot query the retriever — described by the authors as a strong threat model

- **Number of injected documents**: N malicious texts per target question (default N=5), a negligible fraction of a multi-million-text knowledge base

- **Attacker's own LLM**: The attacker may use any LLM of their choosing (e.g., GPT-4) to craft malicious texts, independently of whatever LLM is used inside the target RAG system

------

## Method

PoisonedRAG's core insight is that an effective malicious text must simultaneously satisfy two conditions: a **retrieval condition** (the text must be retrieved into the top-k results) and a **generation condition** (when used as context, the text must induce the LLM to output the target answer). The malicious text P is decomposed into two concatenated sub-texts: `P = S ⊕ I`, where S handles retrieval and I handles generation.

### Key steps

**1. Crafting I (generation sub-text)**

An attacker-controlled LLM (GPT-4 by default) generates a fluent natural-language text I such that, when I alone is used as context, the target RAG's LLM will produce the target answer R for the target question [Question]. The prompt used is:

> "*This is my question: [Question]. This is my answer: [Answer]. Please craft a corpus such that the answer is [Answer] when prompting with the question [Question]. Please limit the corpus to V words.*"

After generation, I is verified: if the LLM does not produce R with I as context, the process is repeated up to L=50 times. In practice, ~2 queries suffice on average. Crucially, I is semantically coherent natural language containing no explicit instructions, so its perplexity is indistinguishable from clean documents — defeating perplexity-based detection.

**2. Crafting S (retrieval sub-text)**

- **Black-box**: Simply set `S = Q` (the target question itself). Since a query is maximally similar to itself in embedding space, this naive strategy already achieves F1-Score > 0.96 for retrieval across all tested datasets and retrievers
- **White-box**: With retriever parameter access, use gradient-based adversarial text methods (HotFlip by default) to optimize S so that `Sim(fQ(Q), fT(S⊕I))` is maximized, achieving F1-Score = 1.0

**3. Injection**

`P = S ⊕ I` is injected into the knowledge base. Because P is semantically close to Q and contains content that guides the LLM toward R, the RAG system retrieves P and generates the target answer accordingly.

------

## Key results

**Experimental scope**:

- Datasets: NQ (2.68M texts), HotpotQA (5.23M texts), MS-MARCO (8.84M texts)
- LLMs: PaLM 2, GPT-3.5-Turbo, GPT-4, LLaMA-2-7B/13B, Vicuna-7B/13B/33B (8 total)
- Retrievers: Contriever, Contriever-ms, ANCE
- Default: N=5 injected texts per target question, k=5 retrieved documents

**Attack success rates — black-box setting (Table 1)**:

- NQ: **97%** (PaLM 2 / GPT-4), 92% (GPT-3.5), 97% (LLaMA-2-7B)
- HotpotQA: **99%** (PaLM 2), 98% (GPT-3.5), 93% (GPT-4)
- MS-MARCO: **91%** (PaLM 2 / GPT-4), 89% (GPT-3.5)

**Comparison with baselines (Table 4, NQ)**:

- Naive attack (query only): ASR **3%**
- Corpus poisoning (Zhong et al.): ASR **1%**
- Prompt injection attack: ASR **62%**
- PoisonedRAG black-box: ASR **97%** — large margin over all baselines

**Computational efficiency**: Under the black-box setting, crafting each malicious text takes under 1 second and requires ~2 LLM queries on average

**Defense evaluations**:

- Perplexity filtering: Fully ineffective (AUC ≈ 0.25); I is GPT-4-generated fluent text, S is a normal question, so malicious texts have normal perplexity
- Query paraphrasing: Minor reduction only (NQ black-box: 97% → 87%)
- Duplicate text filtering: Ineffective; each independently generated I is unique
- Knowledge expansion (k=50): Still achieves 41%–43% ASR on HotpotQA

------

## How this relates to OUR project

PoisonedRAG is the closest prior work to our project: its black-box attack (`P = Q ⊕ I`, attacker adds documents to the knowledge base) operates under exactly the same threat model we assume, and our poison document construction directly adopts the query-prefix + generated-content two-part structure it introduces. The paper's retrieval evaluation metrics (Precision/Recall/F1-Score over the top-k) provide a standardized framework we can use to report rank shift in our own experiments. The key difference is in evaluation focus: PoisonedRAG treats ASR — whether the final answer matches the target — as the sole metric, while our project simultaneously tracks retrieval-stage rank shift and generation-stage output changes across four LLMs (Claude, GPT-4o, Gemini, Llama), aiming to characterize how different models respond to the same poison input rather than optimizing attack success rate against any single target.

------

## Quotable findings

> "We find that the knowledge database in a RAG system introduces a new and practical attack surface. Based on this attack surface, we propose PoisonedRAG, the first knowledge corruption attack to RAG." (Abstract, page 1)

> "PoisonedRAG could achieve a 97% attack success rate when injecting five malicious texts for each target question into a knowledge database with millions of texts." (Abstract, page 1)

> "Our results show these defenses are insufficient to defend against PoisonedRAG, highlighting the need for new defenses." (Abstract, page 1)