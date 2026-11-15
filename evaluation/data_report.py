"""Summarize dataset coverage and fingerprint the exact dataset revision."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from evaluation.validate_data import ROOT, read_jsonl, validate


def build_report(data_dir):
    counts, errors = validate(data_dir)
    if errors:
        raise ValueError("Dataset validation failed: " + "; ".join(errors))
    report = {"total_examples": sum(counts.values()), "splits": {}}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        report["splits"][split] = {
            "examples": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "categories": dict(sorted(Counter(r["expected"]["category"] for r in rows).items())),
            "environments": dict(sorted(Counter(r["expected"]["environment"] for r in rows).items())),
            "null_impact_examples": sum(r["expected"]["customer_reported_impact"] is None for r in rows),
            "scenario_families": len({r["scenario_family"] for r in rows}),
            "max_ticket_characters": max(len(r["ticket"]) for r in rows),
        }
    report["limitations"] = [
        "Counts and validation do not establish label correctness or absence of semantic leakage.",
        "Character length is not token length; check the actual tokenizer before training.",
        "This report contains no model performance measurements.",
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = json.dumps(build_report(args.data_dir), indent=2) + "\n"
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
