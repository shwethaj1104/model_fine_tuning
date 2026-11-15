# Fine-tune after recording the baseline

## What is prepared

- `training/finetune_colab.ipynb`: run this notebook in Colab.
- `training/train_lora.py`: trains a LoRA adapter with answer-only labels.
- `training/baseline_reference.json`: original model name, exact revision, prompt/data fingerprints, and measured scores extracted from your downloaded baseline.
- `evaluation/run_model.py --adapter PATH`: uses the same prediction code with a trained adapter loaded.
- `artifacts/colab-project.zip`: regenerated upload, now containing training code too.

Model: `Qwen/Qwen3-0.6B`. Original revision: `c1899de289a04d12100db370d81485cdf75e47ca`. This is the same model evaluated in your first validation run. The original output/metadata files remain in `artifacts/triage-baseline-results/`.

## What you do

1. Keep the original results backup.
2. Open `training/finetune_colab.ipynb` in Colab, using a fresh GPU runtime.
3. Run cells 1–3: check GPU, upload the **new** ZIP, and install training libraries.
4. Run cell 4 and authorize your Google Drive mount. This saves checkpoints outside the temporary runtime.
5. Run cell 5 to inspect labels. Only the correct JSON answer and end-of-turn token should be supervised; instructions and ticket tokens are masked with `-100`.
6. Run cell 6 for two optimizer steps on a small subset. Stop if it fails or reports non-finite loss.
7. Run cell 7 for the full three-epoch pilot. Only training examples update the weights. Validation loss chooses among saved checkpoints.
8. Run cell 8 to predict validation tickets with both the original model and selected adapter in the same current environment.
9. Run cell 9 to compare scores and inspect summaries. Lower validation loss does not guarantee better routing or factuality.
10. Run cell 10 to download the adapter/results backup. Checkpoints remain on Drive for resuming.

If a command fails, stop and share the full output. No actual CUDA training, adapter inference, or checkpoint resume has been tested locally. Passing offline checks is not a training result.

## Initial settings and purpose

| Setting | Initial choice | Purpose |
| --- | --- | --- |
| LoRA rank | 16 | Train small extra matrices while freezing original weights |
| Learning rate | 0.0002 | Control how large the learning updates are |
| Epochs | 3 | Try three passes through training data |
| Microbatch | 1 | Keep GPU memory modest |
| Gradient accumulation | 8 | Combine eight microbatches before an update |
| Maximum length | 512 | Already fits the baseline token checks; training rechecks the exact answer-only construction |
| Checkpoint selection | Lowest validation loss | Select a saved version without looking at test performance |
| Precision | Float32 parameters + FP16 mixed precision | Avoid assuming native BF16 training on T4 |

These are starting settings, not values already chosen through validation experiments. Only the LoRA parameters change. The selected adapter depends on the original weights and exact revision; it is not a complete standalone model. It is exported with its tokenizer and `training_identity.json`.

## If Colab disconnects

Return to a GPU runtime, upload the code and install dependencies if needed, then remount Drive. Find the previous run directory and its saved `training/checkpoint-N` folder. Use the original training arguments and explicit paths:

```python
run('-m', 'training.train_lora',
    '--output-dir', '/content/drive/MyDrive/ticket-triage-runs/YOUR_RUN/training',
    '--epochs', '3',
    '--resume-from-checkpoint', '/content/drive/MyDrive/ticket-triage-runs/YOUR_RUN/training/checkpoint-N')
```

Replace `YOUR_RUN` and `N` with existing folders. Settings/model/data must match saved metadata. Checkpoints include optimizer/trainer state; downloading only the adapter does not preserve full resumable training state. Work since the last saved epoch may need repeating. Do not resume a completed experiment just to overwrite results.

## What we do afterward

Use validation results to decide whether more data or different settings are needed. Keep each run's evidence. Once choices are fixed, compare original and adapted models on the same held-out test set. Do not evaluate test tickets while choosing settings. The current notebook does not publish anything or run test predictions.

References: [Trainer API used](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer), [PEFT](https://huggingface.co/docs/peft/quicktour).
