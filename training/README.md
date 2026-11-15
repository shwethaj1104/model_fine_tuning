# Training — next implementation stage

Training code belongs in this repository, but training executes in a Colab GPU runtime. The baseline notebook (`baseline_colab.ipynb`) and runner (`evaluation/run_model.py`) are prepared; follow `docs/COLAB_GUIDE.md`. The user completed baseline GPU inference on Colab and downloaded the results. The fine-tuning notebook (`finetune_colab.ipynb`) and script (`train_lora.py`) are now implemented; follow `docs/FINETUNE_GUIDE.md`. Their first real GPU training smoke run is still pending.

1. Open the project notebook in Colab and select an available GPU runtime.
2. Clone/upload this project and the reviewed data.
3. Install a verified, pinned stack: PyTorch, Transformers, PEFT, TRL, Accelerate; add bitsandbytes if using QLoRA.
4. Validate the data. Load `Qwen/Qwen3-0.6B` directly from Hugging Face inside Colab. Capture its revision.
5. Run a small inference/training smoke test. Use the Qwen chat template with thinking disabled for this structured task. Use the system prompt in this folder. Ensure only assistant targets contribute to loss, and inspect tokenization/truncation.
6. Train a LoRA adapter with training data, selecting settings/checkpoints using validation data. On GPUs without native BF16, use supported precision rather than assuming BF16 is available.
7. Save resumable checkpoints externally. Save adapter/tokenizer/config, metrics, dependency versions, and seeds.
8. Upload the adapter using a narrowly scoped write token from Colab secrets. Download a backup before the runtime expires.
9. Run final evaluation using the frozen base/adapted configuration on test data.

Do not paste tokens into notebooks or commit outputs containing secrets. No Hugging Face token is normally required to download this public ungated base model. Authentication is needed to upload your adapter. Free GPU access is not guaranteed; checkpoint often.

Dataset files contain leading `#` explanations. Load them with `evaluation.validate_data.read_jsonl` or export its parsed records to strict JSONL before using a third-party dataset loader. Do not pass the commented files directly to a strict JSONL loader.

## Original model record

`baseline_reference.json` records Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca` and the original validation scores, extracted from the user-downloaded run. Initial training settings are experimental; validation has not yet selected any adapter.
