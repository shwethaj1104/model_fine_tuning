# Ticket Triage - model fine-tuning portfolio

Turn a support ticket into structured routing and extraction results using a small language model. Start with `Qwen/Qwen3-0.6B`, compare a prompted baseline with a LoRA adapter, and publish measured results.

## Current state

The initial foundation is ready: task definition, output schema, 228 AI-authored fictional tickets (128 train / 34 validation / 66 test), dataset validation, and an offline evaluator. **No model has been trained or evaluated yet.** The baseline model runner and notebook were run successfully on Colab by the user; predictions and metadata are saved locally. LoRA training code and a Colab training notebook are prepared, with GPU training verification pending. React and FastAPI are not implemented.

## Run the initial checks

Use Python 3.11+ for these standard-library scripts. For future ML dependencies, create a separate Python 3.11 or 3.12 environment; support for newer Python versions depends on the libraries.

```bash
python3 -m evaluation.validate_data
python3 -m evaluation.data_report
python3 -m evaluation.evaluate --help
```

The evaluator consumes predictions from a model run (one JSON object per line):

```json
{"id":"test-001","prediction":{"category":"authentication","environment":"production","customer_reported_impact":"One newly rotated API credential fails while the previous credential still works.","suggested_team":"developer_support"}}
```

```bash
python3 -m evaluation.evaluate --gold data/test.jsonl --predictions artifacts/base-test.jsonl --output artifacts/base-test-metrics.json
```

The prediction file above is an example path; a model runner will create it in a later step. Never substitute gold labels for model predictions when reporting results.

## Project layout

- `docs/`: scope, labeling rules, success criteria, and next steps.
- `data/`: independent train/validation/test files plus demo tickets.
- `evaluation/`: shared output contract, dataset checks, and scoring CLI.
- `training/`: executable Colab baseline notebook and future training specification.
- `backend/`: planned Python/FastAPI service.
- `frontend/`: planned React dashboard.
- `model/`: local downloaded weights/adapters (ignored by Git).
- `artifacts/`: local predictions and results (ignored by Git).

## Execution and cost

Author code/data here. Run fine-tuning on Colab: it downloads Qwen into its own runtime, trains an adapter, then saves that adapter to your Hugging Face account and a backup. Running the finished app locally downloads the base model and adapter onto the user's machine. Publishing model files does not automatically create a hosted inference API.

No database is needed for this version. Store demo examples and evaluation reports in files, and keep session predictions in UI state. No paid service is required by this scaffold. Free training access and public demo hosting are subject to provider availability and eligibility.

Do not commit tokens or downloaded weights. Use `.env.example` as a template; load actual secrets through the environment or Colab secrets. A `.env` file is not automatically loaded by these scripts.

## Next steps

Follow [the project plan](docs/PROJECT_PLAN.md) and [next-step review](docs/NEXT_STEPS.md). Dataset statistics are available with `python3 -m evaluation.data_report`; they do not establish model quality. Review the data and measure token lengths before training.

## Run the baseline next

Follow [the Colab guide](docs/COLAB_GUIDE.md). The upload ZIP is prepared at `artifacts/colab-project.zip`; regenerate it with `python3 scripts/prepare_colab.py` after edits. Open `training/baseline_colab.ipynb` in Colab and run its cells in order. It evaluates validation data only; no training starts yet.

## Fine-tuning after the baseline

The baseline was run on Colab and its results downloaded locally. Follow [the fine-tuning guide](docs/FINETUNE_GUIDE.md) using `training/finetune_colab.ipynb` and the new upload ZIP. Training code passes offline checks but needs its first GPU smoke run; no adapter improvement has been measured yet.

## Final held-out test

The downloaded epoch-2 adapter (checkpoint-32) is frozen for final evaluation. Follow [the final test guide](docs/FINAL_TEST_GUIDE.md) with `training/final_test_colab.ipynb`. Separate code and adapter-restore ZIPs are ready under `artifacts/`. Final CUDA predictions have not yet been run.
