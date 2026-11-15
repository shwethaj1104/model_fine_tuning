"""Validate dataset structure and basic cross-split leakage checks."""

import argparse
import json
from pathlib import Path

from evaluation.contract import expected_team, output_errors

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    """Read project JSONL, allowing explanatory # headers before the first record.

    External JSONL tools need those headers removed. Comments within the data
    are rejected to avoid silently hiding broken records.
    """
    records = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        if not records and line.lstrip().startswith("#"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    return records


def validate(data_dir):
    errors, counts = [], {}
    ids, tickets, families = {}, {}, {}
    for split in ("train", "validation", "test"):
        records = read_jsonl(data_dir / f"{split}.jsonl")
        counts[split] = len(records)
        if not records:
            errors.append(f"{split}: empty dataset")
        for index, record in enumerate(records, 1):
            label = f"{split}:{index}"
            if not isinstance(record, dict) or set(record) != {"id", "scenario_family", "ticket", "expected"}:
                errors.append(f"{label}: invalid record keys")
                continue
            for field in ("id", "scenario_family", "ticket"):
                if not isinstance(record[field], str) or not record[field].strip():
                    errors.append(f"{label}: {field} must be nonempty text")
            for error in output_errors(record["expected"]):
                errors.append(f"{label}: {error}")
            target = record["expected"]
            if isinstance(target, dict) and target.get("suggested_team") != expected_team(target.get("category")):
                errors.append(f"{label}: inconsistent category/team labels")
            for field, seen in (("id", ids), ("ticket", tickets), ("scenario_family", families)):
                value = record[field]
                if not isinstance(value, str):
                    continue
                key = " ".join(value.casefold().split())
                if key in seen and (field != "scenario_family" or seen[key] != split):
                    errors.append(f"{label}: duplicate or cross-split {field} (previously in {seen[key]})")
                seen[key] = split
    return counts, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    try:
        counts, errors = validate(args.data_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps({"counts": counts, "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
