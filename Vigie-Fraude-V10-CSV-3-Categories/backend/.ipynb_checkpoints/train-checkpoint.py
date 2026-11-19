"""Train the regularized logistic regression used by Vigie Fraude."""

import argparse
import json
import math
import os
from itertools import groupby
from pathlib import Path

from .model import CATEGORICAL, NUMERIC_KEYS, ROOT, engineer, load_dataset, score, stratify, vectorize


def quantile(values, probability):
    index = (len(values) - 1) * probability
    lower = math.floor(index)
    return values[lower] + (values[math.ceil(index)] - values[lower]) * (index - lower)


def fit_preprocessing(rows):
    engineered = [engineer(transaction) for transaction, _ in rows]
    numeric = {}
    for key in NUMERIC_KEYS:
        values = sorted(row[key] for row in engineered)
        spread = quantile(values, 0.75) - quantile(values, 0.25)
        numeric[key] = {"center": quantile(values, 0.5), "scale": spread or 1}
    categories = {key: sorted({row[key] for row in engineered}, key=lambda value: value.encode("utf-16-be"))
                  for key in CATEGORICAL}
    return {"numeric": numeric, "categories": categories}


def fit_logistic(rows, preprocessing):
    examples = [(vectorize(transaction, preprocessing), label) for transaction, label in rows]
    dimension = len(examples[0][0])
    weights = [0.0] * dimension
    positives = sum(label for _, label in examples)
    intercept = math.log(positives / (len(examples) - positives))
    momentum = [0.0] * (dimension + 1)
    variance = [0.0] * (dimension + 1)
    for epoch in range(1, 701):
        gradient = [0.0] * (dimension + 1)
        for features, label in examples:
            residual = score(features, weights, intercept) - label
            gradient[dimension] += residual
            for index, feature in enumerate(features):
                gradient[index] += residual * feature
        mean_bias, variance_bias = 1 - 0.9 ** epoch, 1 - 0.999 ** epoch
        for index, component in enumerate(gradient):
            penalty = 0.0001 * weights[index] if index < dimension else 0
            delta = component / len(examples) + penalty
            momentum[index] = 0.9 * momentum[index] + 0.1 * delta
            variance[index] = 0.999 * variance[index] + 0.001 * delta ** 2
            step = 0.02 * (momentum[index] / mean_bias) / (math.sqrt(variance[index] / variance_bias) + 1e-8)
            if index == dimension:
                intercept -= step
            else:
                weights[index] -= step
    return {"weights": weights, "intercept": intercept}


def rank(rows, model):
    return sorted(((score(vectorize(transaction, model["preprocessing"]), model["weights"], model["intercept"]), label)
                   for transaction, label in rows), key=lambda row: row[0], reverse=True)


def ranking_metrics(ranked):
    positives = sum(label for _, label in ranked)
    negatives = len(ranked) - positives
    tp = fp = 0
    ap = auc = 0.0
    best_f1, best_threshold = -1, 0.5
    for threshold, rows in groupby(ranked, key=lambda row: row[0]):
        previous_tp, previous_fp = tp, fp
        for _, label in rows:
            tp += label
            fp += 1 - label
        ap += (tp - previous_tp) / positives * tp / (tp + fp)
        auc += (fp - previous_fp) / negatives * (tp + previous_tp) / (2 * positives)
        f1 = 2 * tp / (positives + tp + fp)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return best_threshold, ap, auc


def evaluate(ranked, threshold):
    confusion = dict.fromkeys(("tp", "fp", "tn", "fn"), 0)
    for value, label in ranked:
        if value >= threshold:
            confusion["tp" if label else "fp"] += 1
        else:
            confusion["fn" if label else "tn"] += 1
    tp, fp, fn = (confusion[key] for key in ("tp", "fp", "fn"))
    _, ap, auc = ranking_metrics(ranked)
    return {"confusion": confusion, "averagePrecision": ap, "rocAuc": auc,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn)}


def write_json(target, data):
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    dataset = load_dataset(args.csv)
    splits = stratify(dataset)
    train, validation, test = (splits[key] for key in ("train", "validation", "test"))
    preprocessing = fit_preprocessing(train)
    model = {"preprocessing": preprocessing, **fit_logistic(train, preprocessing)}
    threshold, _, _ = ranking_metrics(rank(validation, model))
    metrics = {
        "total": len(dataset), "frauds": sum(label for _, label in dataset),
        "train": len(train), "validation": len(validation), "test": len(test),
        "testFrauds": sum(label for _, label in test), **evaluate(rank(test, model), threshold),
    }
    write_json(ROOT / "trained-model.json", {**model, "threshold": threshold, "metrics": metrics})
    print(json.dumps({"threshold": threshold, **metrics}, indent=2))


if __name__ == "__main__":
    main()
