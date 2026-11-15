"""Run original Qwen on saved tickets; use --check-only to inspect token lengths."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import statistics
import time

from evaluation.validate_data import ROOT, read_jsonl


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def messages_for(ticket, prompt):
    # Gold answers are deliberately excluded from inference messages.
    return [{"role": "system", "content": prompt}, {"role": "user", "content": ticket}]


def load_tickets(path):
    rows = read_jsonl(path)
    seen = set()
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("ticket"), str) or not row["ticket"].strip():
            raise ValueError(f"{path}: every row needs a string id and nonempty ticket")
        if row["id"] in seen:
            raise ValueError(f"{path}: duplicate id {row['id']}")
        seen.add(row["id"])
    return rows


def token_lengths(tokenizer, rows, prompt, budget):
    result = []
    for row in rows:
        messages = messages_for(row["ticket"], prompt)
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        full_length = None
        if "expected" in row:
            answer = json.dumps(row["expected"], ensure_ascii=False)
            full_ids = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}], tokenize=True, add_generation_prompt=False, enable_thinking=False)
            full_length = len(full_ids)
        result.append({"id": row["id"], "input_tokens": len(input_ids), "full_training_tokens": full_length})
    full = [r["full_training_tokens"] for r in result if r["full_training_tokens"] is not None]
    return {"budget": budget, "examples": len(rows), "max_input_tokens": max(r["input_tokens"] for r in result), "max_full_training_tokens": max(full) if full else None, "over_budget_ids": [r["id"] for r in result if r["full_training_tokens"] is not None and r["full_training_tokens"] > budget], "rows": result}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/validation.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for this run; existing directories are rejected")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="main", help="Hub branch or commit; resolved commit is saved")
    parser.add_argument("--adapter", type=Path, help="Local adapter folder exported by training.train_lora")
    parser.add_argument("--prompt", type=Path, default=ROOT / "training/system_prompt.txt")
    parser.add_argument("--check-data", type=Path, nargs="*", default=[], help="Additional datasets for tokenizer checks only; not used as inference input")
    parser.add_argument("--sequence-budget", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--check-only", action="store_true", help="Download tokenizer only, not model weights")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.sequence_budget < 1 or args.max_new_tokens < 1:
        parser.error("Token budgets must be positive")
    rows = load_tickets(args.input)
    prompt = args.prompt.read_text().strip()
    if not prompt:
        parser.error("System prompt is empty")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new run name to preserve previous evidence")
    # Heavy imports are deferred so --help and local unit checks work without ML packages.
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

    revision = model_info(args.model, revision=args.revision).sha
    if not revision:
        raise ValueError("Could not resolve a fixed model commit")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    paths = list(dict.fromkeys([args.input, *args.check_data]))
    lengths = {str(p): token_lengths(tokenizer, load_tickets(p), prompt, args.sequence_budget) for p in paths}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "token_lengths.json", lengths)
    packages = {}
    for package in ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    metadata = {"status": "tokenizer_checked", "started_utc": datetime.now(timezone.utc).isoformat(), "model_id": args.model, "requested_revision": args.revision, "resolved_revision": revision, "prompt_sha256": sha256(args.prompt), "dataset_sha256": {str(p): sha256(p) for p in paths}, "python": platform.python_version(), "packages": packages, "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens, "enable_thinking": False, "seed": 42}, "notes": ["Greedy decoding is an experimental baseline choice for this short structured task, not Qwen's suggested sampling recipe.", "No JSON repair or gold answers are used during inference.", "Token checks may inspect expected answers for length; these are never included in inference messages.", "CUDA memory is PyTorch peak allocated/reserved, not total device usage; timings exclude download and tokenization."]}
    write_json(args.output_dir / "metadata.json", metadata)
    for path, report in lengths.items():
        print(f"{path}: longest input={report['max_input_tokens']}, full training={report['max_full_training_tokens']}, over {args.sequence_budget}={len(report['over_budget_ids'])}", flush=True)
    if args.check_only:
        return
    try:
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is available")
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)
        set_seed(42)
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
        if args.adapter:
            identity_path = args.adapter / "training_identity.json"
            identity = json.loads(identity_path.read_text())
            if identity["model_id"] != args.model or identity["resolved_revision"] != revision:
                raise ValueError("Adapter requires a different original model or revision")
            if identity["prompt_sha256"] != sha256(args.prompt):
                raise ValueError("Adapter was trained with a different system prompt")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False).eval()
            metadata.update(adapter_path=str(args.adapter), adapter_identity=identity,
                            adapter_weights_sha256=sha256(args.adapter / "adapter_model.safetensors"))
        context = model.config.max_position_embeddings
        for row in lengths[str(args.input)]["rows"]:
            if row["input_tokens"] + args.max_new_tokens > context:
                raise ValueError(f"{row['id']}: input plus generation exceeds context; no truncation performed")
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens, eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, use_cache=True)
        metadata.update(status="running", device=device, dtype=str(dtype), hardware=torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(), context_tokens=context, generation_config=config.to_dict())
        write_json(args.output_dir / "metadata.json", metadata)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        timings = []
        with (args.output_dir / "predictions.jsonl").open("w") as stream:
            for index, row in enumerate(rows, 1):
                inputs = tokenizer.apply_chat_template(messages_for(row["ticket"], prompt), tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors="pt").to(device)
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(**inputs, generation_config=config)
                if device == "cuda":
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                generated = output[0, inputs["input_ids"].shape[-1]:]
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                count = len(generated)
                prediction = {"id": row["id"], "prediction": raw, "generation_seconds": seconds, "generated_tokens": count, "hit_output_limit": count >= args.max_new_tokens}
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                stream.flush()
                timings.append(seconds)
                print(f"{index}/{len(rows)} {row['id']} ({seconds:.2f}s)", flush=True)
        metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), examples=len(rows), mean_generation_seconds=statistics.mean(timings), median_generation_seconds=statistics.median(timings), cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None, cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None)
    except Exception as exc:
        metadata.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(args.output_dir / "metadata.json", metadata)


if __name__ == "__main__":
    main()
