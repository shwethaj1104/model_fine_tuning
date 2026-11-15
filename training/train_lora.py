"""Train a small Qwen LoRA adapter; never use test data."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path

from evaluation.contract import output_errors
from evaluation.run_model import load_tickets, messages_for, sha256, write_json
from evaluation.validate_data import ROOT, validate


def encode_example(tokenizer, row, prompt, max_length):
    if output_errors(row.get('expected')):
        raise ValueError(f"Invalid expected answer: {row['id']}")
    # Use the EXACT inference prefix, including Qwen's empty thinking block.
    prefix = tokenizer.apply_chat_template(messages_for(row['ticket'], prompt), tokenize=True, add_generation_prompt=True, enable_thinking=False)
    answer = tokenizer.encode(json.dumps(row['expected'], ensure_ascii=False), add_special_tokens=False)
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if eos is None or eos == tokenizer.unk_token_id:
        raise ValueError('Qwen end-of-turn token not available')
    ids = prefix + answer + [eos]
    if len(ids) > max_length:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceed {max_length}; no truncation performed")
    # -100 means do not calculate loss on this position. Learn ONLY the answer
    # and end-of-turn token, never the system instructions or user ticket.
    labels = [-100] * len(prefix) + answer + [eos]
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}


class AnswerCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        import torch
        width = max(len(row['input_ids']) for row in rows)
        batch = {key: [] for key in ('input_ids', 'attention_mask', 'labels')}
        for row in rows:
            padding = width - len(row['input_ids'])
            batch['input_ids'].append(row['input_ids'] + [self.pad_id] * padding)
            batch['attention_mask'].append(row['attention_mask'] + [0] * padding)
            batch['labels'].append(row['labels'] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New run folder, preferably on mounted Google Drive')
    parser.add_argument('--reference', type=Path, default=ROOT/'training/baseline_reference.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--prompt', type=Path, default=ROOT/'training/system_prompt.txt')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--smoke', action='store_true', help='Two optimizer steps on eight training examples, not a final adapter')
    parser.add_argument('--check-only', action='store_true', help='Tokenizer and loss masking inspection only; no weights downloaded')
    parser.add_argument('--resume-from-checkpoint', type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.max_length < 1:
        parser.error('Epochs, learning rate and max length must be positive')
    counts, errors = validate(args.data_dir)
    if errors:
        raise ValueError(errors)
    reference = json.loads(args.reference.read_text())
    if sha256(args.prompt) != reference['prompt_sha256'] or sha256(args.data_dir/'validation.jsonl') != reference['validation_sha256']:
        raise ValueError('Prompt/validation files differ from the original baseline; preserve comparability before training')
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset

    prompt = args.prompt.read_text().strip()
    model_id, revision = reference['model_id'], reference['resolved_revision']
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    rows = {split: load_tickets(args.data_dir/f'{split}.jsonl') for split in ('train','validation')}
    encoded = {split: [encode_example(tokenizer, row, prompt, args.max_length) for row in records] for split, records in rows.items()}
    inspection = {}
    for split, records in encoded.items():
        sample = records[0]
        answer_ids = [token for token, label in zip(sample['input_ids'], sample['labels']) if label != -100]
        inspection[split] = {'examples': len(records), 'max_tokens': max(len(r['input_ids']) for r in records), 'sample_id': rows[split][0]['id'], 'masked_prompt_tokens': sum(x == -100 for x in sample['labels']), 'supervised_answer': tokenizer.decode(answer_ids, skip_special_tokens=False)}
    identity = {'model_id': model_id, 'resolved_revision': revision, 'prompt_sha256': sha256(args.prompt), 'train_sha256': sha256(args.data_dir/'train.jsonl'), 'validation_sha256': sha256(args.data_dir/'validation.jsonl'), 'epochs': args.epochs, 'learning_rate': args.learning_rate, 'max_length': args.max_length, 'smoke': args.smoke, 'seed': 42, 'lora': {'r':16,'lora_alpha':32,'lora_dropout':0.05,'target_modules':['q_proj','k_proj','v_proj','o_proj']}, 'gradient_accumulation_steps':8, 'batch_size':1}
    if args.output_dir.exists():
        if not args.resume_from_checkpoint:
            raise ValueError('Run directory exists; choose a new run name or explicitly resume')
        previous = json.loads((args.output_dir/'training_metadata.json').read_text())
        if previous['identity'] != identity:
            raise ValueError('Cannot resume with changed model, data, or training configuration')
        if args.output_dir.resolve() not in args.resume_from_checkpoint.resolve().parents:
            raise ValueError('Resume checkpoint must belong to this run folder')
    elif args.resume_from_checkpoint:
        raise ValueError('Resume requires the original run folder and metadata')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir/'label_inspection.json', inspection)
    print(json.dumps(inspection, indent=2), flush=True)
    metadata = {'identity':identity, 'status':'labels_checked', 'packages':{p:version(p) for p in ('torch','transformers','peft','accelerate')}, 'selection':'lowest validation answer-token loss among saved epochs; task metrics evaluated afterward', 'test_used':False}
    write_json(args.output_dir/'training_metadata.json', metadata)
    if args.check_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for this training workflow')
    class TicketDataset(Dataset):
        def __init__(self, examples): self.examples = examples
        def __len__(self): return len(self.examples)
        def __getitem__(self, index): return self.examples[index]
    try:
        set_seed(42)
        # Float32 base parameters + CUDA FP16 autocast avoid relying on emulated
        # BF16 support reported by some recent Torch builds on T4 hardware.
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.float32, attn_implementation='eager')
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', bias='none', **identity['lora']))
        model.enable_input_require_grads()
        model.print_trainable_parameters()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        metadata.update(status='training', trainable_parameters=trainable, total_parameters=total, hardware=torch.cuda.get_device_name(0), precision='float32 parameters with fp16 autocast')
        write_json(args.output_dir/'training_metadata.json', metadata)
        options = TrainingArguments(output_dir=str(args.output_dir), num_train_epochs=args.epochs, max_steps=2 if args.smoke else -1, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8, learning_rate=args.learning_rate, warmup_ratio=0.1, lr_scheduler_type='linear', weight_decay=0.01, max_grad_norm=1.0, fp16=True, bf16=False, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False}, eval_strategy='steps' if args.smoke else 'epoch', save_strategy='steps' if args.smoke else 'epoch', eval_steps=2 if args.smoke else None, save_steps=2 if args.smoke else 500, logging_steps=1, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, report_to='none', seed=42, data_seed=42, label_names=['labels'], dataloader_num_workers=0)
        trainer = Trainer(model=model, args=options, train_dataset=TicketDataset(encoded['train'][:8] if args.smoke else encoded['train']), eval_dataset=TicketDataset(encoded['validation'][:8] if args.smoke else encoded['validation']), data_collator=AnswerCollator(tokenizer.pad_token_id), processing_class=tokenizer)
        result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
        if not math.isfinite(result.training_loss):
            raise RuntimeError('Training loss is not finite; do not use this adapter')
        selected_loss = trainer.state.best_metric
        if selected_loss is None or not math.isfinite(selected_loss):
            raise RuntimeError('No finite validation checkpoint was selected')
        adapter = args.output_dir/'adapter'
        model.config.use_cache = True
        model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        # Save an explicit immutable base revision alongside the adapter.
        write_json(adapter/'training_identity.json', identity)
        trainer.save_state()
        write_json(args.output_dir/'training_log.json', trainer.state.log_history)
        metadata.update(status='complete', best_checkpoint=trainer.state.best_model_checkpoint, best_validation_loss=selected_loss, training_metrics=result.metrics, adapter_path=str(adapter))
    except Exception as exc:
        metadata.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(args.output_dir/'training_metadata.json', metadata)


if __name__ == '__main__':
    main()
