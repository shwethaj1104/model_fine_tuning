# Project plan

## 1. Data and task foundation — current stage

- [x] Choose Qwen3-0.6B as the first experimental model.
- [x] Define a small extraction/routing schema and labeling rules.
- [x] Create seed data and an offline scorer.
- [ ] Human review of a balanced sample and ambiguous cases; confirm categories match the intended workflow.
- [x] Expand an AI-authored pilot with missing information, mixed issues, and instruction-like content.
- [ ] Review each split for semantic overlap and class balance.

Suggested eventual starting scale: 600–1,000 training examples, 100–150 validation examples, and 200–300 test examples, adjusted to labeling time. Quality and coverage matter more than hitting these counts. The expanded pilot is documented in `data/README.md`; run `python3 -m evaluation.data_report` for current counts. Start with a pilot baseline and short training experiment before deciding whether more examples are needed. Never train on validation/test data.

## 2. Baseline

Implement a model runner using the shared prompt in `training/system_prompt.txt`, the model's chat template, and non-thinking mode. Save raw model text as `prediction` with the example ID. Keep raw output so invalid JSON remains visible. Run experiments on validation data first. Record model revision, dependencies, prompt, generation settings, hardware, latency, and peak memory.

Compare a simple category baseline where useful. Run the original model on the held-out test set for the final comparison, without using those results to tune prompts or training.

## 3. Colab training

Create a reproducible notebook and pin dependencies after a working smoke test. Use the training file only for optimization and validation for checkpoint selection. Start with LoRA (rank 8 or 16), a short sequence length, microbatch 1–2, and one epoch. Use QLoRA if memory requires it. Benchmark a short run before estimating duration. Save checkpoints outside the temporary runtime and test resume. Save the adapter, tokenizer, config, and metadata; preserve the base-model revision.

## 4. Final evaluation

Choose settings using validation, then freeze the experiment. Compare base and adapted model on identical held-out test inputs and generation settings. Report JSON/schema validity, category macro-F1, environment accuracy, team accuracy, and human review of impact summaries. Report latency and memory separately with the measurement conditions. Use bootstrap confidence intervals for final results once the test set is large enough; they are not yet implemented. If fine-tuning fails to improve the task, report that honestly and analyze the errors.

Provisional goals, to review before experiments: schema validity ≥98%, category macro-F1 ≥0.85, environment accuracy ≥0.90, and no invented impact in at least 95% of reviewed outputs. These are targets, not observed results. Choose an inference latency budget after measuring local hardware. The small authored-domain test set cannot support broad production claims, even if these targets are met.

## 5. Application and portfolio

Build React + FastAPI. Add local inference using Qwen plus the adapter, ticket examples, side-by-side base/adapted predictions, and an evaluation dashboard. No database. Keep generated outputs in session state. Validate requests and responses; cap input length and generation length.

Publish code on GitHub and adapter/model card on Hugging Face. Provide local setup instructions and a recorded demo. Decide optional public hosting separately; a model repository alone does not serve predictions. Keep secrets server-side. Publish actual results, training costs, limitations, and representative failures.
