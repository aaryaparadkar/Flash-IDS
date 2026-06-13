# LLM vs Word2Vec Semantic Separation — 60-Second Walkthrough

## Files to Open (in order)

| Step | File | What to Show |
|------|------|-------------|
| 1 | `flash_embed.py` lines 17–37, 59–81, 162–214 | The pluggable architecture: EmbeddingProvider base class, Word2VecProvider vs LLMEmbedProvider |
| 2 | `flash_embed.py` lines 131–158 | event_to_text() — the shared semantic text conversion that bridges structured CDM data to LLM-compatible text |
| 3 | `results/benchmark_baseline.json` lines 4–28 | Word2Vec benchmark: P=0.553, R=1.000, F1=0.712 |
| 4 | `results/benchmark_hf.json` lines 4–28 | HF Transformer benchmark: P=0.503, R=1.000, F1=0.669 |
| 5 | `flash_embed.py` lines 393–404 | get_provider() factory — run_experiments.py swaps embeddings with one CLI flag |
| 6 | `flash_llm_filter.py` lines 264–321 | LLM Filter — borderline detection review, 15.5x precision improvement |
| 7 | `results/llm_filter_demo.json` lines 6–19 | Raw vs LLM-filtered: F1 0.071 → 0.320 |
| 8 | `flash_explain.py` lines 29–38 | flash_explain.py signature — supports `--embed-mode baseline` or `hf` |
| 9 | `results/explain_nl.md` lines 10–24 | One sample NL explanation from Mistral |

---

## Script (~55 seconds)

**[Step 1–2 | ~10s — flash_embed.py]**
"Here's the pluggable embedding layer. Every provider — Word2Vec, Random, TokenMean, LLM — implements the same `embed(document)` interface. The key innovation for LLM is `event_to_text()`: it converts structured CDM audit data into natural language so transformers can encode the full semantics of an event, not just averaged word vectors."

**[Step 3–4 | ~10s — benchmark JSONs]**
"The benchmark numbers: Word2Vec gets 0.712 F1, HF Transformer gets 0.669. A 4-point gap. But here's the nuance — the GNN hidden dimension is 32, compressing 384-dim transformer embeddings 12-to-1 in the first layer. The fact that we're within 4 points despite that bottleneck is actually promising."

**[Step 5 | ~5s — get_provider()]**
"The important architectural point: swapping embeddings is a one-line config change. The same experiment runner, same GNN, same evaluation — just `--embed-mode hf` vs `--embed-mode baseline`."

**[Step 6–7 | ~15s — flash_llm_filter.py + results]**
"But the real win isn't at the embedding layer — it's the LLM post-filter. 15.5x precision improvement, from 0.043 to 0.667. Eight routine processes — cron jobs, log rotation, SSH keepalives — correctly identified as benign. Four genuinely suspicious entities confirmed. The LLM brings operational context the statistical model cannot."

**[Step 8–9 | ~15s — flash_explain.py + explain_nl.md]**
"And finally, the explanation layer. Same event text feeds the LLM explainer. Here's a Mistral-generated report — four parts: summary, risk level, explanation of why this is anomalous, and concrete recommended action. The semantic text representation is the single shared foundation that powers all three LLM components."

**Closing (~5s):**
"Bottom line: comparable detection performance, plus zero-shot novel token handling, intelligent false positive suppression, and analyst-readable explanations — all from the same semantic text pipeline. Word2Vec cannot do any of that."
