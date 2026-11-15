"""Evaluate a frozen original/adapter pair on the held-out test set, without training."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from evaluation.evaluate import score
from evaluation.run_model import ROOT, sha256, write_json, load_tickets
from evaluation.validate_data import read_jsonl


def verify(bundle, reference_path):
    reference = json.loads(reference_path.read_text())
    checks = {'training/system_prompt.txt':reference['prompt_sha256'], 'data/test.jsonl':reference['test_sha256'], **reference['scoring_files_sha256']}
    for name, expected in checks.items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f'Frozen file changed: {name}; do not silently change the final experiment')
    adapter = bundle/'training/adapter'
    for name, expected in reference['adapter_files_sha256'].items():
        if sha256(adapter/name) != expected:
            raise ValueError(f'Frozen adapter file changed: {name}')
    identity = json.loads((adapter/'training_identity.json').read_text())
    for key in ('model_id','resolved_revision','prompt_sha256'):
        if identity[key] != reference[key]:
            raise ValueError(f'Adapter identity mismatch: {key}')
    if identity['smoke']:
        raise ValueError('Do not use a smoke adapter as the final model')
    rows = load_tickets(ROOT/'data/test.jsonl')
    if len(rows) != reference['test_examples']:
        raise ValueError('Test example count changed')
    return reference, adapter, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True, help='Run folder containing training/adapter')
    parser.add_argument('--output-dir', type=Path, required=True, help='A new directory; existing results are never overwritten')
    parser.add_argument('--reference', type=Path, default=ROOT/'training/final_test_reference.json')
    parser.add_argument('--check-only', action='store_true', help='Verify frozen files without loading models or writing results')
    args=parser.parse_args()
    reference, adapter, gold = verify(args.bundle, args.reference)
    print(f"Frozen model: {reference['model_id']} @ {reference['resolved_revision']}",flush=True)
    print(f"Adapter: epoch {reference['selected_epoch']}, {reference['selected_checkpoint']}",flush=True)
    print(f"Held-out examples: {len(gold)}; hashes verified",flush=True)
    if args.check_only: return
    if args.output_dir.exists():
        raise ValueError('Output directory exists; preserve the original final test evidence')
    args.output_dir.mkdir(parents=True)
    manifest={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'frozen_reference':reference,'bundle_path':str(args.bundle)}
    write_json(args.output_dir/'final_test_manifest.json',manifest)
    try:
        results={}
        metadata={}
        for name, adapter_path in (('base-test',None),('adapted-test',adapter)):
            output=args.output_dir/name
            command=[sys.executable,'-u','-m','evaluation.run_model','--input',str(ROOT/'data/test.jsonl'),'--model',reference['model_id'],'--revision',reference['resolved_revision'],'--max-new-tokens',str(reference['generation']['max_new_tokens']),'--device','cuda','--output-dir',str(output)]
            if adapter_path is not None: command += ['--adapter',str(adapter_path)]
            subprocess.run(command,cwd=ROOT,check=True)
            metadata[name]=json.loads((output/'metadata.json').read_text())
            results[name]=score(gold,read_jsonl(output/'predictions.jsonl'))
            write_json(output/'metrics.json',results[name])
        for name, meta in metadata.items():
            for key in ('model_id','resolved_revision','prompt_sha256','generation'):
                if meta[key] != reference[key]:
                    raise ValueError(f'Final run setting mismatch: {name}/{key}')
        for key in ('dtype','packages','generation_config','hardware'):
            if metadata['base-test'][key] != metadata['adapted-test'][key]:
                raise ValueError(f'Unmatched original/adapter conditions: {key}')
        comparison={key:{'base':results['base-test'][key],'adapted':results['adapted-test'][key],'change':results['adapted-test'][key]-results['base-test'][key]} for key in ('schema_validity_rate','category_macro_f1','environment_accuracy','team_accuracy')}
        write_json(args.output_dir/'comparison.json',comparison)
        print(f"{'metric':27} {'base':>10} {'adapted':>10} {'change':>10}")
        for key, values in comparison.items():
            print(f"{key:27} {values['base']:10.3f} {values['adapted']:10.3f} {values['change']:+10.3f}")
        base={r['id']:r['prediction'] for r in read_jsonl(args.output_dir/'base-test/predictions.jsonl')}
        adapted={r['id']:r['prediction'] for r in read_jsonl(args.output_dir/'adapted-test/predictions.jsonl')}
        with (args.output_dir/'human_review.csv').open('w',newline='') as stream:
            fields=['id','ticket','expected','base_prediction','adapted_prediction','base_impact_factual','adapted_impact_factual','base_impact_complete','adapted_impact_complete','review_notes']
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            for row in gold:
                writer.writerow({'id':row['id'],'ticket':row['ticket'],'expected':json.dumps(row['expected'],ensure_ascii=False),'base_prediction':base[row['id']],'adapted_prediction':adapted[row['id']]})
        manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),human_review_status='pending; automated scores do not assess impact factuality')
    except Exception as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write_json(args.output_dir/'final_test_manifest.json',manifest)
    archive_path=args.output_dir.parent/(args.output_dir.name+'.zip')
    if archive_path.exists(): raise ValueError('Archive already exists; preserve existing evidence')
    with ZipFile(archive_path,'w',ZIP_DEFLATED) as archive:
        for path in args.output_dir.rglob('*'):
            if path.is_file(): archive.write(path,path.relative_to(args.output_dir))
    print(f'Download results ZIP: {archive_path}',flush=True)
    print('No training or publishing occurred. Review summaries before making extraction claims.')


if __name__=='__main__':main()
