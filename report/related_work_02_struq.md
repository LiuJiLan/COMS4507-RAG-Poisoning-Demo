# StruQ: Defending Against Prompt Injection with Structured Queries

## Bibliographic info

- **Authors**: Sizhe Chen, Julien Piet, Chawin Sitawarin, David Wagner
- **Venue**: USENIX Security Symposium 2025 (34th)
- **Year**: 2025
- **URL**: https://arxiv.org/abs/2402.06363

------

## Problem

LLM-integrated applications are vulnerable to prompt injection attacks because they concatenate developer instructions (the prompt) and untrusted user data into a single string before passing it to the LLM — there is no structural separation between control and content. Attackers can embed hidden instructions in the data portion (e.g., a resume, a retrieved document, a web page) that override the application's intended behavior, effectively hijacking the LLM's output. This is ranked the #1 security risk for LLM applications by OWASP, and no prior defense was both fully general-purpose and robust against the strongest known attacks.

------

## Threat model

- **Attacker's goal**: Hijack an LLM-integrated application by injecting instructions into the *data* portion of a query, causing the LLM to follow those instructions instead of the developer's prompt
- **Control over the LLM**: None. The attacker cannot modify the model weights, system prompt, or application logic
- **What the attacker can modify**: Only the data input (e.g., the content of a document, email, or user submission) — not the prompt
- **Knowledge of the system**: The prompt and application formatting are assumed to be known to the attacker; only the ability to change them is denied
- **Black-box vs. white-box**: Ranges from fully black-box (manual attacks, TAP) to white-box (GCG, which requires gradient access to the LLM)
- **Attack is successful if**: The LLM follows the injected instruction rather than the developer's prompt, regardless of whether the original task is also completed
- **Scope**: Programmatic LLM-integrated applications with a structured API; not applicable to open-ended conversational chatbots

------

## Method

StruQ's core insight is the same principle that solved SQL injection and XSS decades ago: strictly separate control from data. Rather than expecting developers to safely concatenate prompts and user data into one string, StruQ changes the LLM's API to accept them as two distinct inputs — a *structured query*. The system has two components:

**1. Secure Front-End**

The front-end encodes the structured query into a special format using reserved tokens that cannot appear in user data: `[MARK]`, `[INST]`, `[INPT]`, `[RESP]`, `[COLN]`. These tokens are initialized with embeddings from semantically related normal tokens (e.g., `[MARK]` is initialized from `"###"`) and then updated during training. Critically, the front-end runs a recursive filter over all user-supplied data to strip out any of these special delimiter strings before they reach the LLM, making Completion attacks — where an attacker spoofs a fake "end of response / new instruction" delimiter — structurally impossible when using the real delimiters.

**2. Structured Instruction Tuning**

A base (non-instruction-tuned) LLM is fine-tuned on a dataset that contains both clean samples and *attacked* samples. For attacked samples, an injected instruction is placed in the data portion of the input, but the desired output is always the response to the legitimate prompt — teaching the LLM to ignore instructions that appear after the separator. The training mix is: 50% clean, 25% attacked with a Naive injection (another instruction appended to data), 25% attacked with a Completion-Other injection (fake delimiters + fake response + new instruction). No manually crafted malicious content is needed; injections are drawn from the same instruction tuning distribution as the original prompts.

### Key steps

1. Developer provides `(prompt, data)` as two separate inputs to the StruQ front-end
2. Front-end recursively filters special delimiter tokens from the data, then encodes the structured query using reserved-token delimiters
3. The structured-instruction-tuned LLM processes the encoded input and generates a response, having been trained to respond only to instructions in the prompt region and ignore any instructions in the data region

------

## Key results

**Models tested**: Llama-7B, Mistral-7B

**Attacks evaluated**: 15+ attack types including Naive, Ignore, Escape (Deletion/Separation), Completion (Real/Close/Other/combined variants), HackAPrompt, TAP (LLM-optimized), GCG (gradient-guided white-box)

**Security (attack success rate, Table 2)**:

- Against all manual and completion-based attacks: StruQ reduces ASR to **0%** on Llama and **≤2%** on Mistral (from baselines of 29%–96%)
- Against TAP (LLM-crafted): reduced from **97% → 9%** on Llama, **100% → 36%** on Mistral
- Against GCG (gradient white-box): reduced from **97% → 58%** on Llama, **99% → 56%** on Mistral — not fully secure, identified as an open problem
- Cross-language injections (Chinese, Spanish): all reduced to **0%**

**Utility (AlpacaEval win rate, Table 3)**:

- Llama: 67.2% → 67.6% (no loss; slight improvement)
- Mistral: 80.0% → 78.7% (borderline significant ~1.3% drop)

**Comparison with BIPIA (closest prior defense, Table 7)**:

- BIPIA reduces utility from 53.9% → 26.0% (large loss); StruQ 67.2% → 67.7% (none)
- BIPIA ASR against unseen attack formats: up to 54%; StruQ: 0%
- GCG achieves 100% ASR against BIPIA; StruQ holds at 58%

**Key ablation finding**: The combination of Naive + Completion-Other training augmentations is essential; either alone leaves residual vulnerability. Special reserved tokens (vs. standard textual delimiters) are critical for achieving 0% ASR on Completion-Real attacks.

------

## How this relates to OUR project

StruQ is a defense paper, not an attack paper, so our project does not adopt its methods directly — but it establishes the baseline for what "defending against instruction injection in RAG" looks like. The Machine Against the RAG paper (Paper 1) explicitly evaluates StruQ as a defense against its blocker documents and finds that StruQ defeats instruction injection-based blockers while actually *increasing* the success rate of BBO-optimized blockers, which is a key finding we can cite to motivate why optimization-based attacks deserve attention even in defended systems. Our project sits on the attack side of this boundary: we do not assume any StruQ-style defense is in place, which corresponds to the "undefended" baseline in StruQ's own evaluation. StruQ's threat model also clarifies the scope of our work — we study the RAG-specific indirect injection scenario (attacker controls documents in the knowledge base), which StruQ frames as one instantiation of the broader prompt injection problem.

------

## Quotable findings

> "Prompt injection attacks are an important threat: they trick the model into deviating from the original application's instructions and instead follow user directives. These attacks rely on the LLM's ability to follow instructions and inability to separate prompts and user data." (Abstract, page 1)

> "StruQ decreases the attack success rate of Tree-of-Attacks with Pruning (TAP) from 97% to 9% and that of Greedy Coordinate Gradient (GCG) from 97% to 58% on Llama." (Abstract, page 1)

> "Vulnerability to prompt injection stems from models' ability to follow instructions and inability to distinguish between instructions and data." (Section 6, page 14)