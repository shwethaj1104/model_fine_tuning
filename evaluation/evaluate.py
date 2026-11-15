"""Score saved predictions against gold labels; never executes a model."""

import argparse
import json
from pathlib import Path

from evaluation.contract import CATEGORIES, output_errors
from evaluation.validate_data import read_jsonl


def index_records(records, label):
    indexed = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError(f"{label}: each row needs a string id")
        if record["id"] in indexed:
            raise ValueError(f"{label}: duplicate id {record['id']}")
        indexed[record["id"]] = record
    return indexed


def score(gold_records, prediction_records):
    gold = index_records(gold_records, "gold")
    predictions = index_records(prediction_records, "predictions")
    if not gold:
        raise ValueError("gold dataset is empty")
    if gold.keys() != predictions.keys():
        raise ValueError(f"prediction IDs must match gold exactly; missing={sorted(gold.keys()-predictions.keys())}, extra={sorted(predictions.keys()-gold.keys())}")
    invalid_label = "invalid_output"
    matrix = {c: {p: 0 for p in (*CATEGORIES, invalid_label)} for c in CATEGORIES}
    valid_json = valid_schema = environment_correct = team_correct = 0
    failures = []
    for identifier, record in gold.items():
        target = record.get("expected")
        if output_errors(target):
            raise ValueError(f"gold {identifier}: invalid expected output")
        if "prediction" not in predictions[identifier]:
            raise ValueError(f"{identifier}: missing prediction key")
        predicted = predictions[identifier]["prediction"]
        parsed = True
        if isinstance(predicted, str):
            try:
                predicted = json.loads(predicted)
            except json.JSONDecodeError:
                parsed = False
        valid_json += int(parsed)
        errors = output_errors(predicted) if parsed else ["invalid JSON text"]
        if errors:
            matrix[target["category"]][invalid_label] += 1
            failures.append({"id": identifier, "errors": errors})
            continue
        valid_schema += 1
        matrix[target["category"]][predicted["category"]] += 1
        environment_correct += int(target["environment"] == predicted["environment"])
        team_correct += int(target["suggested_team"] == predicted["suggested_team"])
    per_class = {}
    for category in CATEGORIES:
        tp = matrix[category][category]
        fp = sum(matrix[c][category] for c in CATEGORIES if c != category)
        fn = sum(matrix[category].values()) - tp
        denominator = 2 * tp + fp + fn
        per_class[category] = {"support": sum(matrix[category].values()), "f1": 2 * tp / denominator if denominator else 0.0}
    n = len(gold)
    return {
        "examples": n,
        "json_parse_rate": valid_json / n,
        "schema_validity_rate": valid_schema / n,
        "category_macro_f1": sum(c["f1"] for c in per_class.values()) / len(CATEGORIES),
        "environment_accuracy": environment_correct / n,
        "team_accuracy": team_correct / n,
        "category_per_class": per_class,
        "category_confusion_matrix": matrix,
        "invalid_predictions": failures,
        "notes": [
            "Invalid-schema outputs count as incorrect for every task metric.",
            "Macro-F1 averages all six fixed categories; absent classes receive zero F1. Inspect support before comparing datasets.",
            "JSON parse rate accepts parsed prediction objects or raw JSON text; preserve raw text in real model runs.",
            "Impact summaries need human semantic/factual review; they are not automatically scored.",
            "Latency, GPU memory, and confidence intervals are not measured by this offline scorer.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path, help="JSONL rows with id and prediction (raw text or JSON object)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        metrics = score(read_jsonl(args.gold), read_jsonl(args.predictions))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    report = json.dumps(metrics, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
