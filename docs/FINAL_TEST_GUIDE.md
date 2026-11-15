# Final held-out test comparison in Colab

The selected adapter is frozen: epoch 2, checkpoint-32 from run `20261115T073319089436Z`. It had the lowest validation loss, 0.25631797313690186. The original model is Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`. No final test predictions have been run locally.

## Your steps

1. Open `training/final_test_colab.ipynb` in Colab and connect a T4 GPU. You can also copy its cells, in order, into the existing connected notebook if you want to retain that runtime.
2. Run Step 2 and upload **`artifacts/colab-final-test-project.zip`**. This is a separate package; your earlier upload ZIP is preserved.
3. Run Step 3 to use the pinned training/evaluation library versions.
4. Run Step 4. If the original temporary training folder exists, it is used. Otherwise upload **`artifacts/final-test-adapter.zip`** when prompted. This restores the selected weights/config/tokenizer from the local backup; no training is repeated.
5. Confirm it reports epoch 2, checkpoint-32, 66 examples, and verified hashes. Then run Step 5 to evaluate original and adapted models.
6. Run Step 6 to download the final results ZIP. Keep both the earlier adapter backup and these new results locally.

No Google Drive permission is needed. All outputs are temporary until downloaded. The runner writes to a new output folder and refuses to overwrite previous evidence.

## What runs

- Verify prompt, test dataset, scoring implementation, model revision, and adapter-file fingerprints against `training/final_test_reference.json`.
- Run original Qwen on all 66 test tickets.
- Run the selected adapter on the same tickets using the same inference code/settings.
- Confirm the original and adapted runs used matching device, dtype, packages, and generation settings.
- Save raw predictions, individual metrics, confusion matrices, a comparison, and a final experiment manifest.
- Export a CSV for reviewing factuality/completeness of both models' impact summaries. Those review fields start blank; automated metrics do not score semantic extraction quality.

The inference model sees system instructions and ticket text only. Gold answers are used afterward by the scorer. No fine-tuning, checkpoint selection, or publication occurs here.

## Rebuilding the upload packages

From the local project root:

```bash
python3 -m scripts.prepare_final_test
```

This verifies the downloaded selected adapter and writes two allowlisted packages. It does not modify the original model, adapter, data, or earlier results. Local source data and model artifacts must still match the frozen reference.

## After testing

Save and inspect the results. Do not tune using test errors and then present a rerun on these same tickets as an independent held-out result. If a new model is developed using those findings, create a new untouched test set for a fresh final claim. Describe this test honestly as 66 AI-authored tickets, not real support traffic. Team accuracy follows category, and impact summaries still require human review.

Verification: local freeze checks, unit tests, ZIP integrity, and notebook syntax checks pass. The actual final CUDA predictions remain to be run in Colab. No test scores or improvements are invented.

## Portfolio visual exports

Step 5b creates `final_test_dashboard.png/.pdf` with model chips, explained metrics, category-level correct counts and outcome bars. It also exports `final_test_confusion_details.png/.pdf` for technical readers. All counts come from saved metrics; the plots do not rerun models or change results. The updated dashboard has also been rendered locally against the downloaded final-test run.
