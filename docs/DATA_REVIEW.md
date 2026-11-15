# Dataset supervision report

Review date: 2026-11-15. Reviewer: an independent AI supervisor agent, separate from the authoring agent. This is AI review, not human annotation approval.

## Verdict

**Ready for baseline inference, tokenizer checks, and a small LoRA pilot.** No unresolved schema or labeling errors were identified in this review. The dataset is not a production benchmark, and this verdict does not establish model quality. Human review remains pending.

The supervisor read all 228 ticket/target records against `TASK_SPEC.md`, requested corrections, and rechecked the corrected records. The original 18 seed examples were retained; 210 fictional examples were added. No model was trained or scored during this data review.

## Final coverage

| Category | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| authentication | 22 | 6 | 11 |
| billing | 22 | 5 | 11 |
| rate_limit | 21 | 6 | 11 |
| integration | 21 | 6 | 11 |
| bug | 21 | 5 | 11 |
| other | 21 | 6 | 11 |
| **Total** | **128** | **34** | **66** |

| Environment | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| production | 27 | 7 | 15 |
| staging | 21 | 5 | 11 |
| development | 21 | 5 | 10 |
| unknown | 59 | 17 | 30 |

Null impact summaries: 51 training, 15 validation, and 23 test. Every record has a distinct named scenario family. Maximum ticket lengths are 166, 136, and 156 characters respectively; these are not token counts.

## Corrections made during supervision

- Replaced a held-out webhook-signing setup example that was too close to the original training webhook-signing case with an idempotency-header setup case.
- Replaced a validation proxy-configuration question that overlapped the training proxy setup family with a client thread-sharing question.
- Changed two authentication examples from permission-management questions to explicit permission-denied failures, so labels match the current authentication contract without widening it.
- Added explicit blocked/paused setup consequences to four training, one validation, and one test integration examples. Previously every integration label had a null impact, creating an avoidable category-to-output shortcut.
- Removed an inferred blocked-query consequence from a mixed-issue summary; it now reports quota exhaustion and overlapping navigation text only.

The review also checked negative or misleading category words, credential/configuration boundaries, contradictory or multiple environments, unknown environment despite urgency, mixed independent issues, and instruction-like text embedded in tickets. Bug examples retain factual non-null defect summaries; null targets were not invented simply to balance labels.

## Checks and evidence

`python3 -m evaluation.validate_data` passes: no malformed records, schema violations, inconsistent category/team labels, duplicate IDs, normalized exact duplicate tickets, or named scenario-family overlap across splits.

`python3 -m evaluation.data_report` confirms the counts above. A supplemental cross-split lexical scan using lowercased alphabetic token sets with common function words removed found only one candidate at token Jaccard similarity ≥0.27: training subscription cancellation versus held-out renewal quotation. These are different intents and were retained. The scan is a heuristic, not proof of semantic independence; the supervisor also read the records for paraphrases and related scenarios.

SHA256 fingerprints of the reviewed files:

```text
train.jsonl       1f4dc48e60a8ffbe29ca699ed0635fdf708a2f12c7a7f7a0117fb3cc30228ac6
validation.jsonl  d97a26dde5dcfb7624383999810fb859df54cf5120a4cbd553686b96513be6ee
test.jsonl        b75695553dbc66fee47477e21ae7c8e13d16c7cefacc9d2b7616c3f4e824082e
```

## Remaining limits and next action

- Examples are AI-authored, mostly short and clearly worded. They do not represent real support traffic, longer threads, logs, multilingual input, or realistic class frequencies. Typo/shorthand coverage is sparse.
- There are only 11 test examples per category. Report uncertainty and avoid strong production accuracy claims. Keep test data frozen after final review and use validation errors for development decisions.
- Environment and category cues may remain easier than real customer reports. Automated duplicate checks and an AI review cannot guarantee absence of semantic leakage.
- Routing team is determined by category; its accuracy is not an independent model capability.
- Impact summaries need human factual and semantic review. Exact-string scoring cannot establish that a generated paraphrase is correct.
- Tokenization, truncation, model compatibility, runtime memory, training duration, and model performance have not been measured.

Next: review a balanced sample with a person, run the original model on validation data, inspect token lengths using its actual tokenizer, and save baseline predictions with exact model/prompt settings before attempting a small training run. No additional user information was required to finish this dataset preparation.

## Readability headers added after review

At the user's request, a leading `#` explanation was added to each dataset file. The JSON records were verified byte-for-byte unchanged after removing that first line; the reviewed fingerprints above refer to the original header-free files. The shared project reader now skips leading headers. External strict JSONL loaders require a header-free export. Current file fingerprints:

```text
train.jsonl  a3542b1078b55a2835738431b45258858ece996736179931c50c3416f9b12165
validation.jsonl  2f20603552c1e8b781cdfac37ff176b6ad054bdd7028c0ad39ce57b982df026b
test.jsonl  96ea456e23d5ece803fe82d44d17cc0ce46ac575ae0993429d1dac900f83ed8c
```
