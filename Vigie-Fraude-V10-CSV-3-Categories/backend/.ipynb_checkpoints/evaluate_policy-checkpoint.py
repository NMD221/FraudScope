"""Evaluate the review and decline thresholds for fraud scoring."""

import argparse
import json
from itertools import groupby
from pathlib import Path

from .model import ROOT, decide, load_dataset, stratify
from .train import rank, write_json


def evaluate_policy(validation, test, model):
    ranked = rank(validation, model)
    frauds = [value for value, label in ranked if label]
    review = frauds[9 * len(frauds) // 10]
    minimum_precision = 0.80
    block = None
    positives = count = 0
    for threshold, group in groupby(ranked, key=lambda row: row[0]):
        if threshold <= review:
            break
        for _, label in group:
            positives += label
            count += 1
        if positives / count >= minimum_precision:
            block = threshold
    thresholds = {"review": review, "block": block}

    def summarize(rows):
        totals = {category: [0, 0] for category in ("accepted", "review", "blocked")}
        for value, label in rows:
            totals[decide(value, thresholds)][label] += 1
        return totals

    return {"thresholds": thresholds, "minimumBlockPrecision": minimum_precision,
            "validation": summarize(ranked), "test": summarize(rank(test, model))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    model = json.loads((ROOT / "trained-model.json").read_text(encoding="utf-8"))
    splits = stratify(load_dataset(args.csv))
    output = evaluate_policy(splits["validation"], splits["test"], model)
    write_json(ROOT / "decision-policy.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
