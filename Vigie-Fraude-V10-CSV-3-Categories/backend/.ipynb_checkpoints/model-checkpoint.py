"""Transaction schema, scoring and CSV input shared by the API and training."""

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
FIELDS = tuple(field for group in GROUPS for field in group["fields"])
LABELS = {field["key"]: field["label"] for field in FIELDS}
WEEKDAYS = next(field["options"] for field in FIELDS if field["key"] == "day_of_week")
NUMERIC = tuple(field["key"] for field in FIELDS if field["kind"] in {"number", "integer"})
CATEGORICAL = tuple(field["key"] for field in FIELDS if field["kind"] == "category")
FLAGS = tuple(field["key"] for field in FIELDS if field["kind"] == "boolean")
NUMERIC_KEYS = NUMERIC + (
    "amount_to_balance_ratio", "risk_signal_count", "txn_velocity", "is_night", "is_weekend",
    "card_age_ratio", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
)


def validate(transaction):
    if not isinstance(transaction, dict):
        return {"transaction": "Un objet JSON est attendu."}
    errors = {}
    for field in FIELDS:
        key, kind = field["key"], field["kind"]
        value = transaction.get(key)
        if kind == "category":
            if not isinstance(value, str) or not value.strip() or len(value) > 100:
                errors[key] = "Catégorie invalide."
        elif kind == "boolean":
            if type(value) is not bool:
                errors[key] = "Une valeur booléenne est attendue."
        elif key == "account_balance_usd" and value == -1:
            errors[key] = "Le solde doit être différent de −1 USD."
        elif (type(value) not in (int, float)
              or not field.get("min", -1e12) <= value <= field.get("max", 1e12)
              or (kind == "integer" and (not math.isfinite(value) or value != int(value)))):
            errors[key] = "Valeur numérique hors limites ou manquante."
    return errors


def engineer(transaction):
    hour = 2 * math.pi * transaction["time_of_day_hour"] / 24
    day = 2 * math.pi * transaction["day_of_week"] / 7
    return {
        **transaction,
        "amount_to_balance_ratio": transaction["amount_usd"] / (transaction["account_balance_usd"] + 1),
        "risk_signal_count": sum(transaction[key] for key in FLAGS),
        "txn_velocity": transaction["txn_count_last_24h"] / (transaction["hours_since_last_txn"] + 1),
        "is_night": int(transaction["time_of_day_hour"] <= 6),
        "is_weekend": int(transaction["day_of_week"] >= 5),
        "card_age_ratio": transaction["card_age_months"] / (transaction["customer_age"] * 12 + 1),
        "hour_sin": math.sin(hour), "hour_cos": math.cos(hour),
        "dow_sin": math.sin(day), "dow_cos": math.cos(day),
    }


def vectorize(transaction, preprocessing):
    row = engineer(transaction)
    numeric = preprocessing["numeric"]
    return [
        *((row[key] - numeric[key]["center"]) / numeric[key]["scale"] for key in NUMERIC_KEYS),
        *(int(category == row[key]) for key in CATEGORICAL for category in preprocessing["categories"][key]),
        *(int(row[key]) for key in FLAGS),
    ]


def score(features, weights, intercept):
    logit = intercept
    for feature, weight in zip(features, weights, strict=True):
        logit += feature * weight
    try:
        return 1 / (1 + math.exp(-logit))
    except OverflowError:
        return 0.0


def predict(transaction, model):
    return score(vectorize(transaction, model["preprocessing"]), model["weights"], model["intercept"])


def decide(score_value, thresholds):
    if thresholds["block"] is not None and score_value >= thresholds["block"]:
        return "blocked"
    return "review" if score_value >= thresholds["review"] else "accepted"


def risk_factors(transaction, model):
    preprocessing, weights = model["preprocessing"], model["weights"]
    features = vectorize(transaction, preprocessing)
    contributions = {key: features[i] * weights[i] for i, key in enumerate(NUMERIC_KEYS)}
    offset = len(NUMERIC_KEYS)
    for key in CATEGORICAL:
        categories = preprocessing["categories"][key]
        coefficients = weights[offset:offset + len(categories)]
        active = coefficients[categories.index(transaction[key])] if transaction[key] in categories else 0
        contributions[key] = active - sum(coefficients) / len(coefficients)
        offset += len(categories)
    for index, key in enumerate(FLAGS, offset):
        contributions[key] = features[index] * weights[index]

    extras = {"time_of_day_hour": ["is_night", "hour_sin", "hour_cos"],
              "day_of_week": ["is_weekend", "dow_sin", "dow_cos"]}
    active_flags = [key for key in FLAGS if transaction[key]]
    groups = [([key, *extras.get(key, [])], [key], None, None) for key in LABELS]
    groups += [
        (["amount_to_balance_ratio"], ["amount_usd", "account_balance_usd"], "Montant par rapport au solde", None),
        (["txn_velocity"], ["txn_count_last_24h", "hours_since_last_txn"], "Fréquence récente des opérations", None),
        (["card_age_ratio"], ["card_age_months", "customer_age"], "Ancienneté de la carte par rapport à l’âge du client", None),
        (["risk_signal_count"], active_flags, "Nombre de signaux de risque actifs", len(active_flags)),
    ]
    ranked = []
    for keys, inputs, label, value in groups:
        contribution = sum(contributions[key] for key in keys)
        if contribution <= 0:
            continue
        factor = {"fields": [
            {"key": key, "label": LABELS[key],
             "value": WEEKDAYS[int(transaction[key])] if key == "day_of_week" else transaction[key]}
            for key in inputs
        ]}
        if label is not None:
            factor["label"] = label
        if value is not None:
            factor["value"] = value
        ranked.append((contribution, factor))
    total = math.fsum(contribution for contribution, _ in ranked)
    return [{**factor, "share": contribution / total}
            for contribution, factor in sorted(ranked, key=lambda item: item[0], reverse=True)]


def load_dataset(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        header = next(reader, [])
        if len(set(header)) != len(header) or not {*LABELS, "is_fraud"} <= set(header):
            raise ValueError("Le CSV doit contenir les 24 variables du notebook et la cible is_fraud.")
        rows = []
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            raw = dict(zip(header, (value.strip() for value in row)))
            transaction = {}
            for field in FIELDS:
                key, kind = field["key"], field["kind"]
                value = raw.get(key, "")
                if kind == "boolean":
                    value = {"true": True, "1": True, "false": False, "0": False}.get(value.lower())
                elif kind != "category":
                    try:
                        value = float(value)
                    except ValueError:
                        value = None
                transaction[key] = value
            if len(row) != len(header) or validate(transaction) or raw.get("is_fraud") not in {"0", "1"}:
                raise ValueError(f"Données invalides à la ligne {reader.line_num}.")
            rows.append((transaction, int(raw["is_fraud"])))
        return rows


def stratify(rows):
    seed = 42
    splits = {"train": [], "validation": [], "test": []}
    for label in (0, 1):
        group = [row for row in rows if row[1] == label]
        if len(group) < 5:
            raise ValueError("Au moins cinq opérations de chaque classe sont nécessaires.")
        for index in range(len(group) - 1, 0, -1):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            other = seed * (index + 1) // 2**32
            group[index], group[other] = group[other], group[index]
        size = (len(group) + 2) // 5
        splits["test"].extend(group[:size])
        splits["validation"].extend(group[size:2 * size])
        splits["train"].extend(group[2 * size:])
    return splits
