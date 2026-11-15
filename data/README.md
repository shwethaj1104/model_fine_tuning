# Authored pilot dataset

228 fictional English support tickets: **128 training, 34 validation, and 66 test examples**. All original 18 seed examples are preserved. The 210 added examples were individually authored with AI assistance, rather than created by multiplying a template or changing names/numbers. They are **AI-authored synthetic data, not human-handcrafted customer records**. No person has yet reviewed or approved every label. Independent agent review and automated validation help find mistakes, but do not substitute for human review or a real-world benchmark.

Each JSONL record contains `id`, `scenario_family`, `ticket`, and `expected`. Labels follow `docs/TASK_SPEC.md`. The `pilot-` scenario-family namespace identifies this expansion. Each added scenario has a distinct case; related paraphrases introduced later must remain in the same split. The demo tickets are UI examples, outside the training/evaluation corpus.

## Split coverage

| Category | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| authentication | 22 | 6 | 11 |
| billing | 22 | 5 | 11 |
| rate_limit | 21 | 6 | 11 |
| integration | 21 | 6 | 11 |
| bug | 21 | 5 | 11 |
| other | 21 | 6 | 11 |
| **Total** | **128** | **34** | **66** |

Coverage includes production/staging/development/unknown environments, missing information, multiple/conflicting environments, terse/typo-containing input, mixed independent issues, misleading category words, instruction-like text, negative examples, and setup questions with no reported impact. Impact labels paraphrase only stated consequences; `null` is intentional when no consequence is stated. All text is fictional; no real credentials or customer records are included.

## Intended use and limits

This is sufficient to start tokenization, baseline inference, and a small LoRA training smoke/pilot experiment. It is not evidence that the resulting model is deployment-ready. Expand training diversity based on **validation** errors, not by inspecting and tuning against the test labels. Sixty-six test examples (11 per class) provide only a small, authored-domain estimate. Use additional independently sourced or human-authored held-out tickets before strong portfolio performance claims.

The deliberately balanced category distribution does not represent production class frequencies. Environment wording and category wording may still be easier than real customer language. The router's team label is deterministically derived from category, so team accuracy is not an independent reasoning benchmark. Impact summaries require human factual/semantic review; exact text match is unsuitable.

## Validation and review

Run:

```bash
python3 -m evaluation.validate_data
```

The validator checks output schema, unique IDs, normalized exact duplicate tickets, and named scenario-family separation across splits. It does not detect every semantic paraphrase or establish that labels are correct. Before training, manually inspect a balanced sample from each class and all ambiguous cases; record reviewer/date and corrections here. Keep the held-out test set frozen after final review.

Review status: AI-authored draft expanded on 2026-11-15; automated structure checks passed. Independent AI supervision is complete: all 228 records were inspected and identified corrections were rechecked. See `docs/DATA_REVIEW.md` for the review record and limitations. Human review remains pending. No baseline or fine-tuned model scores have been computed from these examples yet.

## Explanatory file headers

The three dataset files have a leading `#` explanation requested for readability. These are commented JSONL files, not strict JSONL. Use `evaluation.validate_data.read_jsonl` to read them; it skips only leading comment headers. Before passing files directly to an external JSONL loader, export just the parsed records as JSONL. Future training code must use the shared reader or that export, rather than loading the commented files directly.

User review update (2026-11-15): the user confirmed the prepared data looks good and marked the data-review step complete. This records acceptance for the pilot, not a claim that every label received independent human annotation. Keep the test set unchanged for final evaluation.
