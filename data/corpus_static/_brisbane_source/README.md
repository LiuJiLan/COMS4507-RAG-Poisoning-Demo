# Brisbane Corpus — Generator Scripts (auxiliary / reference only)

These two scripts were used by Task A teammate to generate
`data/corpus_static/brisbane_corpus.json` (290 documents).

| File | Role |
|---|---|
| `main.py` | Wikipedia + curated restaurants + UQ template-fill pipeline that emits the 290-doc JSON |
| `check_corpus.py` | Schema + diversity validator (200-300 docs / Wikipedia ≥30 unique URLs / no template leaks / etc.) |

## Why they are here

The committed `brisbane_corpus.json` is the **ground-truth deliverable**. These
scripts are kept alongside it so a future reviewer or grader can inspect the
generation methodology.

## Known limitation (not blocking the deliverable)

`main.py:316-349` (`ensure_word_range`) contains a `while content += extra`
padding loop. On short base templates it appends the *same* sentence multiple
times, which originally produced visible duplicate sentences in restaurant
docs. The committed `brisbane_corpus.json` does **not** exhibit this issue —
the corpus was post-processed externally before delivery.

Re-running `python main.py` will therefore regenerate a corpus with the
duplicate-sentence artifact. The committed JSON is canonical; do not
overwrite from a fresh run unless you understand and re-apply the
post-processing step.

## Verifying the committed corpus

```bash
# in the project root
python scripts/check_corpus_diversity.py data/corpus_static/brisbane_corpus.json
# expect exit code 0
```
