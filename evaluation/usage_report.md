# Token Usage & Cost Report — Buy or Wait?

All figures below are read directly from the instrumented call log
(`evaluation/usage_log.jsonl`); none are estimated by hand except the per-token
prices used to convert tokens to dollars.

**Provider / model:** OpenAI · `gpt-4o` (the only model used).
**Pricing used for cost:** $2.50 / 1M input tokens, $10.00 / 1M output tokens.

## 1. Summary (final full-dataset run that produced `output.csv`)

The full-dataset run over the 250 evaluation requests makes LLM calls only to read
amounts off the blank-amount images that appear in that set (11 unique images).

| Metric | Value |
|---|---|
| LLM calls | 11 |
| Prompt (input) tokens | 11,700 |
| Completion (output) tokens | 156 |
| Total tokens | 11,856 |
| Estimated cost | **$0.0308** |
| Avg tokens per LLM call | 1,077.8 |
| Avg tokens per request (over all 250) | 47.4 |
| Avg cost per request (over all 250) | $0.000123 |

## 2. Breakdown by purpose

| Purpose | Calls | Prompt tok | Completion tok | Total tok | Cost |
|---|---|---|---|---|---|
| Vision / image-amount resolution | 11 | 11,700 | 156 | 11,856 | $0.0308 |
| Forecast, decision, ranking, explanation | 0 | 0 | 0 | 0 | $0.00 |

Vision is the **only** use of an LLM. The financial-state reconstruction, the 90-day
balance forecast, the affordability decision, the plan ranking (the spec's six
tie-break levels), and the `decision_explanation` text are **fully deterministic** by
design — they make zero model calls.

## 3. Caching

Each image is resolved once and cached by `image_id`
(`evaluation/vision_cache.json`); the per-call usage above is logged only on a real
call. Re-running the pipeline makes **0 new calls** and adds **$0.00** — so total cost
is bounded by the number of unique blank-amount images, not by how many times the
pipeline runs. On a cold cache the `output.csv` run costs the $0.0308 above; on a warm
cache it costs nothing.

(For completeness: across development and sample-set validation, all 16 unique images
in the dataset were resolved at least once — 16 calls, 17,314 tokens, ~$0.045 total —
including the 5 images that belong to the sample requests rather than the 250-row
evaluation set. Those 5 are not part of producing `output.csv`.)

## 4. Design rationale

The LLM is used surgically — only where the input is genuinely unstructured, i.e.
reading a printed monetary figure off a document image, and even there the image is
treated as untrusted (embedded instructions are ignored). Everything downstream —
reconstruction, forecasting, decision, ranking, and explanations — is deterministic.
This keeps the system accurate and reproducible (identical inputs give identical
outputs), auditable, and extremely cheap: about $0.03 for the whole dataset, spent on
~12k tokens total.
