# Run the original model in Colab

This is the next step after reviewing data. It measures the starting model; it does not fine-tune yet.

## 1. Files are prepared

- `training/baseline_colab.ipynb`: the notebook you open in Colab.
- `evaluation/run_model.py`: Python code the notebook calls to get model predictions.
- `artifacts/colab-project.zip`: prepared upload containing code and data, without secrets or weights.

If you change the code or data, regenerate the upload ZIP from the project terminal:

```bash
python3 scripts/prepare_colab.py
```

## 2. Open the notebook

Open https://colab.research.google.com/ and sign in. Choose **Open notebook → Upload**, then select `training/baseline_colab.ipynb` from this project. Menu labels may vary slightly.

## 3. Connect a GPU

Choose **Runtime → Change runtime type**, select a GPU if available, and connect. Use the free runtime; no paid GPU is required to try this workflow. Availability varies. Run the notebook's first cell to confirm a GPU is present.

## 4. Upload the project files

Run cell 2. When it asks for a file, choose `artifacts/colab-project.zip`. This is separate from uploading the notebook. It gives Colab the dataset and Python scripts.

## 5. Run cells in order

Use the play button beside each code cell:

1. Check GPU.
2. Upload project ZIP.
3. Install Transformers and check dataset files.
4. Download the tokenizer and measure prompt + ticket + answer lengths.
5. Download Qwen onto Colab and try two validation tickets.
6. Run all 34 validation tickets and calculate starting scores.
7. Read the predictions beside the expected answers.
8. Download the results ZIP.

If a cell fails, stop there and share its error text. If Colab explicitly requests a restart after installation, restart and continue from cell 3; the uploaded folder remains on disk within the same runtime. If the entire runtime is replaced, upload again.

The runner refuses to overwrite an existing run folder. For a fresh experiment rerun cell 4 to create a new run ID, then continue. The notebook intentionally does not use test tickets.

## 6. Keep the results

Save `triage-baseline-results.zip` locally in `artifacts/`. Colab's disk is temporary; download before disconnecting. Inside are:

- `predictions.jsonl`: raw model answers, timings, and output-limit flags.
- `metrics.json`: JSON validity, routing F1, environment accuracy, and team accuracy.
- `token_lengths.json`: whether examples fit the proposed length budget.
- `metadata.json`: model commit, settings, hashes, device, and package versions.
- `colab-packages.txt`: installed packages for reproducing the run.

Impact summaries still need human review; wording can differ and remain correct. Team follows category. File validation is not evidence of model accuracy.

## 7. What happens afterward

Inspect these baseline results. Then build the LoRA training notebook, choose settings using validation, and reserve test data for the final original-vs-adapted comparison. No training or publishing occurs in this baseline notebook.

## Verification status

Local offline tests check input isolation, token-budget reporting, header support, and upload packaging. The notebook structure and Python cells are checked locally. Actual model loading, tokenizer behavior, CUDA compatibility, and GPU predictions still require the first Colab run; no model performance or GPU timings are claimed yet.

References: [Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B), [Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation), [Colab FAQ](https://research.google.com/colaboratory/faq.html).
