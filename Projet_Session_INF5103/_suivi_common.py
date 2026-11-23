from html import escape
import json
import numpy as np
from contextlib import contextmanager
import csv
import io
import sqlite3
import logging
import zipfile
from threading import Event, Thread
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from uuid import uuid4
import hashlib
import math
import re
import joblib
import pandas as pd
import hmac
import os
from datetime import datetime
import streamlit as st

ROOT = Path(__file__).resolve().parent

GROUPS = [
    {
        "title": "Transaction",
        "fields": [
            {"key": "amount_usd", "label": "Montant (USD)", "kind": "number", "min": 0},
            {
                "key": "merchant_category",
                "label": "Catégorie du commerçant",
                "kind": "category",
            },
            {"key": "card_type", "label": "Type de carte", "kind": "category"},
            {
                "key": "auth_method",
                "label": "Méthode d’authentification",
                "kind": "category",
            },
            {"key": "channel", "label": "Canal", "kind": "category"},
            {"key": "device_type", "label": "Appareil", "kind": "category"},
            {
                "key": "time_of_day_hour",
                "label": "Heure (0–23)",
                "kind": "integer",
                "min": 0,
                "max": 23,
            },
            {
                "key": "day_of_week",
                "label": "Jour de la semaine",
                "kind": "integer",
                "min": 0,
                "max": 6,
                "options": [
                    "Lundi",
                    "Mardi",
                    "Mercredi",
                    "Jeudi",
                    "Vendredi",
                    "Samedi",
                    "Dimanche",
                ],
            },
        ],
    },
    {
        "title": "Compte et activité",
        "fields": [
            {
                "key": "account_balance_usd",
                "label": "Solde du compte (USD)",
                "kind": "number",
            },
            {
                "key": "customer_age",
                "label": "Âge du client (ans)",
                "kind": "integer",
                "min": 0,
            },
            {
                "key": "card_age_months",
                "label": "Ancienneté de la carte (mois)",
                "kind": "integer",
                "min": 0,
            },
            {
                "key": "distance_from_home_km",
                "label": "Distance du domicile (km)",
                "kind": "number",
                "min": 0,
            },
            {
                "key": "hours_since_last_txn",
                "label": "Heures depuis la dernière opération",
                "kind": "number",
                "min": 0,
            },
            {
                "key": "txn_count_last_24h",
                "label": "Opérations sur les dernières 24 h",
                "kind": "integer",
                "min": 0,
            },
            {
                "key": "prior_disputes",
                "label": "Litiges antérieurs",
                "kind": "integer",
                "min": 0,
            },
            {
                "key": "cvv_retry_count",
                "label": "Nouvelles tentatives de CVV",
                "kind": "integer",
                "min": 0,
            },
            {
                "key": "velocity_score",
                "label": "Score de fréquence",
                "kind": "number",
                "min": 0,
            },
            {
                "key": "merchant_risk_score",
                "label": "Score de risque du commerçant",
                "kind": "number",
                "min": 0,
                "max": 100,
            },
        ],
    },
    {
        "title": "Signaux de risque",
        "fields": [
            {
                "key": "is_foreign_transaction",
                "label": "Opération à l’étranger",
                "kind": "boolean",
            },
            {
                "key": "is_new_merchant",
                "label": "Nouveau commerçant",
                "kind": "boolean",
            },
            {"key": "used_vpn", "label": "VPN utilisé", "kind": "boolean"},
            {
                "key": "ip_country_mismatch",
                "label": "Pays de l’adresse IP différent",
                "kind": "boolean",
            },
            {
                "key": "billing_shipping_mismatch",
                "label": "Adresses de facturation et de livraison différentes",
                "kind": "boolean",
            },
            {
                "key": "is_ai_generated_scam_attempt",
                "label": "Signal d’arnaque générée par IA",
                "kind": "boolean",
            },
        ],
    },
]

DEFAULTS = {
    "amount_usd": 125.0,
    "hours_since_last_txn": 12.0,
    "txn_count_last_24h": 1,
    "distance_from_home_km": 5.0,
    "card_age_months": 47,
    "customer_age": 50,
    "account_balance_usd": 3317.0,
    "cvv_retry_count": 0,
    "velocity_score": 20.0,
    "time_of_day_hour": 12,
    "day_of_week": 3,
    "merchant_risk_score": 38.0,
    "prior_disputes": 0,
    "is_foreign_transaction": False,
    "is_new_merchant": False,
    "used_vpn": False,
    "ip_country_mismatch": False,
    "billing_shipping_mismatch": False,
    "is_ai_generated_scam_attempt": False,
    "merchant_category": "Groceries",
    "card_type": "Visa",
    "auth_method": "OTP",
    "channel": "POS",
    "device_type": "POS Terminal",
}



def engineer_features(df):
    df = df.copy()
    df["amount_to_balance_ratio"] = df["amount_usd"] / (df["account_balance_usd"] + 1)

    risk_flags = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    df["risk_signal_count"] = df[risk_flags].sum(axis=1).astype(float)

    df["txn_velocity"] = df["txn_count_last_24h"] / (df["hours_since_last_txn"] + 1)
    df["is_night"] = (
        (df["time_of_day_hour"] >= 22) | (df["time_of_day_hour"] <= 6)
    ).astype(float)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
    df["card_age_ratio"] = df["card_age_months"] / (df["customer_age"] * 12 + 1)

    df["hour_sin"] = np.sin(2 * np.pi * df["time_of_day_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time_of_day_hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def _predict_from_raw(raw_df, pipe, feat_list, apply_fe):
    df_prep = engineer_features(raw_df) if apply_fe else raw_df.copy()
    return float(pipe.predict_proba(df_prep[feat_list])[0, 1])


def calculer_facteurs_score(
    raw_row_df, base_proba, pipe, feat_list, apply_fe, ui_labels
):
    neutral_defaults = {
        "amount_usd": 125.0,
        "hours_since_last_txn": 12.0,
        "txn_count_last_24h": 1,
        "distance_from_home_km": 5.0,
        "card_age_months": 47,
        "customer_age": 50,
        "account_balance_usd": 3317.0,
        "cvv_retry_count": 0,
        "velocity_score": 20.0,
        "time_of_day_hour": 12,
        "day_of_week": 3,
        "merchant_risk_score": 38.0,
        "prior_disputes": 0,
        "is_foreign_transaction": False,
        "is_new_merchant": False,
        "used_vpn": False,
        "ip_country_mismatch": False,
        "billing_shipping_mismatch": False,
        "is_ai_generated_scam_attempt": False,
        "merchant_category": "Groceries",
        "card_type": "Visa",
        "auth_method": "OTP",
        "channel": "POS",
        "device_type": "POS Terminal",
    }

    flag_labels = {
        "is_foreign_transaction": "Opération à l'étranger",
        "is_new_merchant": "Nouveau commerçant",
        "used_vpn": "VPN utilisé",
        "ip_country_mismatch": "Pays de l'adresse IP différent",
        "billing_shipping_mismatch": "Adresses différentes",
        "is_ai_generated_scam_attempt": "Signal d'arnaque générée par IA",
    }

    single_field_labels = {
        "cvv_retry_count": "Nouvelles tentatives de CVV",
        "amount_usd": "Montant (USD)",
        "merchant_risk_score": "Score de risque du commerçant",
        "velocity_score": "Score de fréquence",
        "auth_method": "Méthode d'authentification",
        "merchant_category": "Catégorie du commerçant",
        "channel": "Canal",
        "device_type": "Appareil",
        "card_type": "Type de carte",
        "distance_from_home_km": "Distance du domicile (km)",
        "txn_count_last_24h": "Opérations sur les dernières 24 h",
        "hours_since_last_txn": "Heures depuis la dernière opération",
        "prior_disputes": "Litiges antérieurs",
        "account_balance_usd": "Solde du compte (USD)",
        "card_age_months": "Ancienneté de la carte (mois)",
        "customer_age": "Âge du client (ans)",
        "time_of_day_hour": "Heure",
        "day_of_week": "Jour de la semaine",
    }

    contributions = []

    try:
        active_flags = [
            k for k in flag_labels if bool(raw_row_df.iloc[0].get(k, False))
        ]
        if active_flags:
            df_no_flags = raw_row_df.copy()
            for k in active_flags:
                df_no_flags[k] = False
            p_without = _predict_from_raw(df_no_flags, pipe, feat_list, apply_fe)
            delta = max(0.0, base_proba - p_without)
            if delta > 1e-6:
                details = " · ".join(f"{flag_labels[k]} : Oui" for k in active_flags)
                label = f"Nombre de signaux de risque actifs : {len(active_flags)} — {details}"
                contributions.append((label, delta))

        for col, lbl in single_field_labels.items():
            if col not in raw_row_df.columns or col not in neutral_defaults:
                continue
            cur_val = raw_row_df.iloc[0][col]
            def_val = neutral_defaults[col]
            if cur_val == def_val:
                continue

            df_mod = raw_row_df.copy()
            df_mod[col] = def_val
            p_without = _predict_from_raw(df_mod, pipe, feat_list, apply_fe)
            delta = max(0.0, base_proba - p_without)
            if delta > 1e-6:
                disp_val = ui_labels.get(col, cur_val)
                contributions.append((f"{lbl} : {disp_val}", delta))

    except Exception:
        contributions = []

    if not contributions:
        active_flags = [
            k for k in flag_labels if bool(raw_row_df.iloc[0].get(k, False))
        ]
        if active_flags:
            details = " · ".join(f"{flag_labels[k]} : Oui" for k in active_flags)
            return json.dumps(
                {
                    "items": [
                        f"Nombre de signaux de risque actifs : {len(active_flags)} — {details}"
                    ],
                    "coverage": "",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "items": ["Aucun facteur particulier au-dessus de la référence."],
                "coverage": "",
            },
            ensure_ascii=False,
        )

    contributions.sort(key=lambda x: x[1], reverse=True)
    total_pos = sum(d for _, d in contributions)
    top_items = contributions[:4]
    top_sum = sum(d for _, d in top_items)

    formatted_items = []
    for lbl, d in top_items:
        part_pct = (d / total_pos * 100.0) if total_pos > 0 else 0.0
        part_str = f"{part_pct:.1f}".replace(".", ",") + " %"
        formatted_items.append(f"{lbl} · {part_str}")

    coverage_pct = (top_sum / total_pos * 100.0) if total_pos > 0 else 0.0
    coverage_str = f"{coverage_pct:.1f}".replace(".", ",") + " %"

    return json.dumps(
        {
            "items": formatted_items,
            "coverage": f"Ces facteurs couvrent {coverage_str} des contributions qui ont augmenté le score.",
        },
        ensure_ascii=False,
    )


CATEGORIES = ("Accepted", "Review", "Blocked")


NAMES = {"Accepted": "courantes", "Review": "douteuses", "Blocked": "frauduleuses"}


def timestamp():
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def safe_csv(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


class Store:
    def __init__(self, path):
        self.path = Path(path)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS operations (
                number INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE, bank_reference TEXT UNIQUE,
                fingerprint TEXT, data TEXT NOT NULL)""")
            db.execute(
                """CREATE TABLE IF NOT EXISTS events (
                number INTEGER PRIMARY KEY AUTOINCREMENT, operation_id TEXT NOT NULL,
                data TEXT NOT NULL, FOREIGN KEY(operation_id) REFERENCES operations(id))"""
            )
            db.execute("""CREATE TABLE IF NOT EXISTS weekly_archives (
                    week_start TEXT PRIMARY KEY, week_end TEXT NOT NULL,
                    created_at TEXT NOT NULL, filename TEXT NOT NULL,
                    payload BLOB NOT NULL)""")
            db.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        return json.loads(row[0]) if row else None

    def create(self, entry, fingerprint=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            reference = entry.get("bankReference")
            if reference:
                row = db.execute(
                    "SELECT data, fingerprint FROM operations WHERE bank_reference=?",
                    (reference,),
                ).fetchone()
                if row:
                    if row[1] != fingerprint:
                        raise ValueError(
                            "Référence déjà associée à des données différentes."
                        )
                    return json.loads(row[0]), False
            identifier = str(uuid4())
            cursor = db.execute(
                "INSERT INTO operations(id,bank_reference,fingerprint,data) VALUES(?,?,?,?)",
                (identifier, reference, fingerprint, "{}"),
            )
            entry = {
                **entry,
                "id": identifier,
                "displayId": f"FRS-{cursor.lastrowid:06d}",
            }
            db.execute(
                "UPDATE operations SET data=? WHERE id=?",
                (json.dumps(entry, ensure_ascii=False), identifier),
            )
            return entry, True

    def get(self, identifier):
        with self.connect() as db:
            entry = self.decode(
                db.execute(
                    "SELECT data FROM operations WHERE id=?", (identifier,)
                ).fetchone()
            )
            if entry:
                entry["events"] = [
                    json.loads(row[0])
                    for row in db.execute(
                        "SELECT data FROM events WHERE operation_id=? ORDER BY number",
                        (identifier,),
                    )
                ]
            return entry

    def all(self, category=None):
        with self.connect() as db:
            rows = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT data FROM operations ORDER BY number DESC"
                )
            ]
        rows = [
            row for row in rows if category is None or row["finalDecision"] == category
        ]
        return sorted(
            rows, key=lambda row: datetime.fromisoformat(row["timestamp"]), reverse=True
        )

    def find(self, query, limit=20):
        query = query.strip().casefold()
        if not query:
            return []
        rows = self.all()

        def visible_keys(entry):
            return [
                str(entry.get(name, "")).casefold()
                for name in ("displayId", "legacyId", "bankReference")
                if entry.get(name)
            ]

        if query.isascii() and query.isdigit():
            number = int(query)
            return [
                row for row in rows
                if row["displayId"].split("-")[-1].isdigit()
                and int(row["displayId"].split("-")[-1]) == number
            ][:limit]

        exact = [
            row for row in rows
            if query in visible_keys(row) or query == row["id"].casefold()
        ]
        matches = [
            row for row in rows
            if row not in exact and (
                any(query in key for key in visible_keys(row))
                or (len(query) >= 8 and row["id"].casefold().startswith(query))
            )
        ]
        return (exact + matches)[:limit]

    def expire_pending(self, current=None):
        current = current or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.astimezone()
        changed = []
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for identifier, raw in db.execute(
                "SELECT id,data FROM operations"
            ).fetchall():
                entry = json.loads(raw)
                if entry["finalDecision"] != "Review":
                    continue
                deadline = datetime.fromisoformat(entry["timestamp"]) + timedelta(
                    days=7
                )
                if current < deadline:
                    continue
                event = {
                    "decision": "Blocked",
                    "previousDecision": "Review",
                    "agent": "Système",
                    "reason": "Défaut de validation",
                    "verified": False,
                    "automatic": True,
                    "timestamp": current.astimezone(timezone.utc).isoformat(),
                    "deadline": deadline.isoformat(),
                }
                db.execute(
                    "INSERT INTO events(operation_id,data) VALUES(?,?)",
                    (identifier, json.dumps(event, ensure_ascii=False)),
                )
                entry["reviewObservation"] = entry.get("observation", "")
                entry["observation"] = (
                    entry.get("observation", "") + " Défaut de validation."
                ).strip()
                entry.update(
                    finalDecision="Blocked",
                    blockReason="validation_timeout",
                    lastDecision=event,
                )
                db.execute(
                    "UPDATE operations SET data=? WHERE id=?",
                    (json.dumps(entry, ensure_ascii=False), identifier),
                )
                changed.append(identifier)
        return changed

    def decide(
        self, identifier, decision, agent, reason, verified=False, *, event_time=None
    ):
        self.expire_pending()
        if decision not in ("Accepted", "Blocked"):
            raise ValueError("Décision invalide.")
        if (
            not isinstance(agent, str)
            or not agent.strip()
            or len(agent) > 80
            or any(ord(c) < 32 for c in agent)
        ):
            raise ValueError(
                "Identifiant de l’agent obligatoire (80 caractères maximum)."
            )
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
            raise ValueError("Motif obligatoire (500 caractères maximum).")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            entry = self.decode(
                db.execute(
                    "SELECT data FROM operations WHERE id=?", (identifier,)
                ).fetchone()
            )
            if entry is None:
                raise ValueError("Opération introuvable.")
            current = entry["finalDecision"]
            if current == "Accepted" or (
                current == "Blocked" and decision != "Accepted"
            ):
                raise ValueError("Cette décision a déjà été prise.")
            if decision == "Accepted" and entry.get("insufficientFunds"):
                raise ValueError(
                    "Solde insuffisant. Une nouvelle opération doit être évaluée avec le solde actualisé."
                )
            event = {
                "decision": decision,
                "previousDecision": current,
                "agent": agent.strip(),
                "reason": reason.strip(),
                "verified": bool(verified),
                "timestamp": event_time or timestamp(),
            }
            db.execute(
                "INSERT INTO events(operation_id,data) VALUES(?,?)",
                (identifier, json.dumps(event, ensure_ascii=False)),
            )
            entry.update(finalDecision=decision, lastDecision=event)
            if decision == "Accepted":
                entry["approval"] = event
            db.execute(
                "UPDATE operations SET data=? WHERE id=?",
                (json.dumps(entry, ensure_ascii=False), identifier),
            )
        return self.get(identifier)

    def export(self, category, reference, period, zone, as_of=None):
        if category not in CATEGORIES or period not in ("day", "week"):
            raise ValueError("Catégorie ou période invalide.")
        start = date.fromisoformat(reference)
        tz = None if zone == "local" else ZoneInfo(zone)
        today = as_of or datetime.now(tz).date()
        if start > today:
            raise ValueError("La date d’exportation ne peut pas dépasser aujourd’hui.")
        if period == "week":
            start -= timedelta(days=(start.weekday() + 1) % 7)
        end = min(
            start + timedelta(days=7 if period == "week" else 1),
            today + timedelta(days=1),
        )
        raw_keys = list(DEFAULTS)
        fields = [
            "ID",
            "Référence bancaire",
            "Date et heure UTC",
            "Montant USD",
            "Montant d’origine",
            "Devise d’origine",
            "Score de fraude",
            "Décision initiale",
            "Décision finale",
            "Observation",
            "Agent de validation",
            "Motif de validation",
            "Date validation UTC",
            "Agent dernière décision",
            "Motif dernière décision",
            "Historique des décisions",
            "Facteurs",
            *raw_keys,
        ]
        buffer = io.StringIO()
        buffer.write("\ufeff")
        writer = csv.writer(buffer, lineterminator="\r\n")
        writer.writerow(fields)
        for row in reversed(self.all(category)):
            if (
                not start
                <= datetime.fromisoformat(row["timestamp"]).astimezone(tz).date()
                < end
            ):
                continue
            entry = self.get(row["id"])
            approval = entry.get("approval", {})
            last = entry.get("lastDecision", {})
            original = entry.get("originalPayment", {})
            values = [
                entry["displayId"],
                entry.get("bankReference", ""),
                entry["timestamp"],
                entry["amount"],
                original.get("amount", ""),
                original.get("currency", ""),
                entry["score"],
                entry["initialDecision"],
                entry["finalDecision"],
                entry.get("observation", ""),
                approval.get("agent", ""),
                approval.get("reason", ""),
                approval.get("timestamp", ""),
                last.get("agent", ""),
                last.get("reason", ""),
                json.dumps(entry["events"], ensure_ascii=False),
                entry.get("factors", ""),
                *(entry.get("transaction", {}).get(key, "") for key in raw_keys),
            ]
            writer.writerow([safe_csv(value) for value in values])
        return (
            f"fraudscope-{NAMES[category]}-{period}-{start}.csv",
            buffer.getvalue().encode("utf-8"),
        )

    def archive_due_weeks(self, current=None):
        current = current or datetime.now()
        sunday = current.date() - timedelta(days=(current.weekday() + 1) % 7)
        if current.weekday() == 6 and current.hour < 1:
            sunday -= timedelta(days=7)
        with self.connect() as db:
            previous = db.execute(
                "SELECT MAX(week_end) FROM weekly_archives"
            ).fetchone()[0]
        if previous:
            start = date.fromisoformat(previous)
        else:
            entries = self.all()
            first = min(
                (
                    datetime.fromisoformat(row["timestamp"]).astimezone().date()
                    for row in entries
                ),
                default=sunday - timedelta(days=7),
            )
            start = first - timedelta(days=(first.weekday() + 1) % 7)
        created = []
        while start + timedelta(days=7) <= sunday:
            end = start + timedelta(days=7)
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                exists = db.execute(
                    "SELECT 1 FROM weekly_archives WHERE week_start=?",
                    (start.isoformat(),),
                ).fetchone()
                if not exists:
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(
                        buffer, "w", compression=zipfile.ZIP_DEFLATED
                    ) as archive:
                        for category in CATEGORIES:
                            filename, payload = self.export(
                                category,
                                start.isoformat(),
                                "week",
                                "local",
                                as_of=current.date(),
                            )
                            archive.writestr(filename, payload)
                    filename = (
                        f"fraudscope-archives-{start}-{end - timedelta(days=1)}.zip"
                    )
                    db.execute(
                        "INSERT INTO weekly_archives VALUES(?,?,?,?,?)",
                        (
                            start.isoformat(),
                            end.isoformat(),
                            timestamp(),
                            filename,
                            buffer.getvalue(),
                        ),
                    )
                    created.append(filename)
            start = end
        return created

    def list_archives(self, query=""):
        with self.connect() as db:
            rows = db.execute(
                "SELECT week_start,week_end,created_at,filename FROM weekly_archives ORDER BY week_start DESC"
            ).fetchall()
        keys = ("week_start", "week_end", "created_at", "filename")
        return [
            dict(zip(keys, row))
            for row in rows
            if query.strip().casefold() in " ".join(row).casefold()
        ]

    def archive_payload(self, week_start):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM weekly_archives WHERE week_start=?", (week_start,)
            ).fetchone()
        return row[0] if row else None

    def import_legacy(self, path):
        """Reprend uniquement les décisions encore présentes dans le CSV source."""
        with Path(path).open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        count = 0
        for row in rows:
            old_id = row["ID"]

            initial = row["Décision"]
            stamp = (
                datetime.strptime(row["Date et heure"], "%d/%m/%Y %H:%M:%S")
                .astimezone()
                .astimezone(timezone.utc)
                .isoformat()
            )
            score = (
                float(row["Score de fraude"].replace("%", "").replace(",", ".").strip())
                / 100
            )
            entry = {
                "timestamp": stamp,
                "source": "legacy",
                "legacyId": old_id,
                "bankReference": "legacy:" + old_id,
                "amount": float(row["Montant (USD)"]),
                "score": score,
                "initialDecision": initial,
                "finalDecision": initial,
                "observation": row.get("Observation", ""),
                "factors": row.get("Facteurs", ""),
                "transaction": {},
            }
            saved, created = self.create(entry, "legacy:" + old_id)
            if created:
                count += 1
                if (
                    row.get("Agent")
                    and row.get("Décision finale") in ("Accepted", "Blocked")
                    and initial != "Accepted"
                ):
                    decision_time = row.get("Date validation")
                    stamp = (
                        datetime.strptime(decision_time, "%d/%m/%Y %H:%M:%S")
                        .astimezone()
                        .astimezone(timezone.utc)
                        .isoformat()
                        if decision_time
                        else entry["timestamp"]
                    )
                    self.decide(
                        saved["id"],
                        row["Décision finale"],
                        row["Agent"],
                        row.get("Motif")
                        or "Motif non enregistré dans l’ancien journal.",
                        event_time=stamp,
                    )
        return count


FIELDS = [field for group in GROUPS for field in group["fields"]]


class Service:
    def __init__(self, database=None):
        self.model = joblib.load(ROOT / "models/final_model.joblib")
        self.pipeline = self.model["pipeline"]
        self.policy = json.loads(
            (ROOT / "decision/decision-policy.json").read_text(encoding="utf-8")
        )

        self.threshold = self.policy["thresholds"]["review"]
        pre = self.pipeline.named_steps["pre"]
        self.categories = {}
        for name, transformer, columns in pre.transformers_:
            if hasattr(transformer, "categories_"):
                self.categories.update(
                    {
                        key: list(values)
                        for key, values in zip(columns, transformer.categories_)
                    }
                )
        self.store = Store(database or ROOT / "fraudscope.sqlite3")

    def start_archiving(self):
        if getattr(self, "_archive_thread", None) is not None:
            return
        self.archive_error = None
        self._archive_stop = Event()

        def check():
            try:
                self.store.expire_pending()
                self.store.archive_due_weeks()
                self.archive_error = None
            except Exception:
                self.archive_error = (
                    "La création des archives a échoué. Elle sera réessayée."
                )
                logging.exception("FraudScope weekly archive failed")

        check()

        def run():
            while not self._archive_stop.wait(60):
                check()

        self._archive_thread = Thread(
            target=run, name="FraudScope archives", daemon=True
        )
        self._archive_thread.start()

    def validate(self, transaction):
        if not isinstance(transaction, dict):
            raise ValueError("Un objet transaction est attendu.")
        result = {}
        for field in FIELDS:
            key, kind = field["key"], field["kind"]
            value = transaction.get(key)
            if kind == "boolean":
                valid = type(value) is bool
            elif kind == "category":
                valid = isinstance(value, str) and value in self.categories[key]
            else:
                valid = (
                    type(value) in (int, float)
                    and field.get("min", -1e12) <= value <= field.get("max", 1e12)
                    and math.isfinite(value)
                    and (kind != "integer" or value == int(value))
                )
                if key == "account_balance_usd" and value == -1:
                    valid = False
            if not valid:
                raise ValueError(f"Valeur invalide ou manquante — {field['label']}")
            result[key] = (
                int(value)
                if kind == "integer"
                else float(value) if kind == "number" else value
            )
        return result

    def analyze(self, transaction, reference=None, original=None, source="manual"):
        transaction = self.validate(transaction)
        if reference is not None and (
            not isinstance(reference, str)
            or not re.fullmatch(r"[A-Za-z0-9._:/-]{1,128}", reference)
        ):
            raise ValueError("Référence bancaire invalide.")
        if original is not None:
            if (
                not isinstance(original, dict)
                or not isinstance(original.get("amount"), str)
                or not re.fullmatch(
                    r"(0|[1-9][0-9]{0,11})(\.[0-9]{1,6})?", original["amount"]
                )
                or not isinstance(original.get("currency"), str)
                or not re.fullmatch(r"[A-Z]{3}", original["currency"])
            ):
                raise ValueError("Montant ou devise d’origine invalide.")
            original = {key: original[key] for key in ("amount", "currency")}
        fingerprint = hashlib.sha256(
            json.dumps(
                [transaction, original], sort_keys=True, ensure_ascii=False
            ).encode()
        ).hexdigest()
        raw = pd.DataFrame([transaction])
        score = _predict_from_raw(
            raw, self.pipeline, self.model["features"], self.model.get("use_fe", False)
        )
        factors = calculer_facteurs_score(
            raw,
            score,
            self.pipeline,
            self.model["features"],
            self.model.get("use_fe", False),
            {},
        )
        insufficient = transaction["amount_usd"] > transaction["account_balance_usd"]
        decision = (
            "Blocked"
            if insufficient
            else "Review" if score >= self.threshold else "Accepted"
        )
        observation = ""
        if insufficient:
            observation = "Solde insuffisant. Le montant dépasse le solde du compte."
        elif decision == "Review":
            observation = f"Mise en attente. Le score de {score:.2%} atteint le seuil de vérification de {self.threshold:.2%}."
        entry = {
            "timestamp": timestamp(),
            "source": source,
            "amount": transaction["amount_usd"],
            "score": score,
            "initialDecision": decision,
            "finalDecision": decision,
            "observation": observation,
            "reviewThreshold": self.threshold,
            "insufficientFunds": insufficient,
            "blockReason": "insufficient_funds" if insufficient else None,
            "transaction": transaction,
            "factors": factors,
        }
        if reference:
            entry["bankReference"] = reference
        if original is not None:
            entry["originalPayment"] = original
        return self.store.create(entry, fingerprint)

    def load_csv(self, source):
        data = pd.read_csv(source)
        rows = []
        for _, row in data.iterrows():
            transaction = {}
            for field in FIELDS:
                key, kind = field["key"], field["kind"]
                value = row.get(key)
                if kind == "boolean":
                    value = {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                        "1.0": True,
                        "0.0": False,
                    }.get(str(value).lower())
                elif kind in ("number", "integer"):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = None
                transaction[key] = value
            tx = self.validate(transaction)
            reference = row.get("bank_reference")
            reference = str(reference) if pd.notna(reference) else None
            original = None
            if "original_amount" in data or "original_currency" in data:
                amount = row.get("original_amount")
                currency = row.get("original_currency")
                if pd.notna(amount) != pd.notna(currency):
                    raise ValueError(
                        "Le montant et la devise d’origine doivent être fournis ensemble."
                    )
                if pd.notna(amount) and pd.notna(currency):
                    original = {"amount": str(amount), "currency": str(currency)}
            rows.append((tx, reference, original))
        return rows


def bank_mode():
    mode = os.getenv("FRAUDSCOPE_MODE", "demo")
    if mode not in ("demo", "bank"):
        raise ValueError("FRAUDSCOPE_MODE must be demo or bank.")
    return mode == "bank"


def secret(name):
    value = os.getenv(name, "")
    if len(value) < 32 or not value.isascii():
        raise ValueError(f"{name} must contain at least 32 ASCII characters.")
    return value


def matches(value, expected):
    return hmac.compare_digest((value or "").encode(), expected.encode())


def gateway_identity(headers):
    if not matches(
        headers.get("x-fraudscope-gateway-secret"), secret("FRAUDSCOPE_GATEWAY_SECRET")
    ):
        raise PermissionError("Connexion bancaire requise.")
    agent = headers.get("x-fraudscope-agent-id", "")
    role = headers.get("x-fraudscope-agent-role", "")
    if not agent.strip() or len(agent) > 80 or role not in ("viewer", "reviewer"):
        raise PermissionError("Identité ou habilitation invalide.")
    return {"id": agent, "role": role, "verified": True}


def streamlit_identity(st):
    if not bank_mode():
        return {"id": "", "role": "reviewer", "verified": False}
    return gateway_identity(
        {key.lower(): value for key, value in st.context.headers.items()}
    )


DECISIONS = {
    "Accepted": "Acceptée",
    "Review": "Douteuse",
    "Blocked": "Fraude confirmée",
}


def t(text):
    return text


def decision_label(value, entry=None):
    if (
        entry
        and entry.get("blockReason") == "validation_timeout"
        and value == "Blocked"
    ):
        return t("Frauduleuse — défaut de validation")
    if entry and entry.get("insufficientFunds") and value == "Blocked":
        return t("Bloquée — solde insuffisant")
    return t(DECISIONS.get(value, value))


def observation_text(entry):
    return entry.get("observation", "")


def automatic_reason(event):
    return event.get("reason", "")


def factor_text(text):
    return text


def show_observation(entry):
    if entry.get("insufficientFunds"):
        st.error(observation_text(entry))
    elif entry["initialDecision"] == "Review":
        st.warning(observation_text(entry))


def factors(entry):
    with st.expander(t("Champs ayant augmenté le score")):
        try:
            data = json.loads(entry.get("factors") or "{}")
        except (ValueError, TypeError):
            data = {}
        for item in data.get("items", []):
            st.text(factor_text(item))
        if data.get("coverage"):
            st.caption(factor_text(data["coverage"]))


def table_rows(entries):
    rows = []
    for entry in entries:
        approval = entry.get("approval", {})
        original = entry.get("originalPayment", {})
        rows.append(
            {
                "ID": entry["displayId"],
                t("Référence bancaire"): entry.get("bankReference", ""),
                t("Date et heure"): datetime.fromisoformat(entry["timestamp"]),
                t("Montant (USD)"): entry["amount"],
                t(
                    "Montant d’origine"
                ): f"{original.get('amount','')} {original.get('currency','')}".strip(),
                t("Score de fraude"): f"{entry['score']:.2%}",
                t("Décision initiale"): decision_label(entry["initialDecision"], entry),
                t("Décision finale"): decision_label(entry["finalDecision"], entry),
                t("Agent de validation"): approval.get("agent", ""),
                t("Motif de validation"): approval.get("reason", ""),
            }
        )
    return pd.DataFrame(rows)


def decision_form(service, entry, identity, *, action=None, form_key="decision"):
    choices = (
        ["Accepted", "Blocked"] if entry["finalDecision"] == "Review" else ["Accepted"]
    )
    with st.form(form_key + "_" + entry["id"], clear_on_submit=False):
        if action is None:
            labels = {
                value: t(
                    "Confirmation de fraude"
                    if value == "Blocked"
                    else (
                        "Validation après réclamation"
                        if entry["finalDecision"] == "Blocked"
                        else "Valider l’opération"
                    )
                )
                for value in choices
            }
            action = st.selectbox(
                t("Décision"),
                choices,
                format_func=labels.get,
                key=form_key + "_action_" + entry["id"],
            )
        else:
            st.caption(
                t(
                    "Confirmation de fraude"
                    if action == "Blocked"
                    else (
                        "Validation après réclamation"
                        if entry["finalDecision"] == "Blocked"
                        else "Valider l’opération"
                    )
                )
            )
        agent_column, reason_column = st.columns(2)
        with agent_column:
            if identity["verified"]:
                st.text(f"{t('Identité vérifiée')} · {identity['id']}")
                agent = identity["id"]
            else:
                agent = st.text_input(
                    t("Identifiant de l’agent"),
                    max_chars=80,
                    placeholder="28GF023",
                    key=form_key + "_agent_" + entry["id"],
                )
        with reason_column:
            reason = st.text_input(
                t("Motif du refus" if action == "Blocked" else "Motif de validation"),
                max_chars=500,
                key=form_key + "_reason_" + entry["id"],
            )
        confirm_column, cancel_column, _ = st.columns([1.2, 1, 3])
        with confirm_column:
            confirmed = st.form_submit_button(t("Confirmer"), type="primary")
        with cancel_column:
            cancelled = st.form_submit_button(t("Annuler"))
    if cancelled:
        return "cancelled"
    if confirmed:
        try:
            service.store.decide(
                entry["id"], action, agent, reason, identity["verified"]
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            return "confirmed"
    return None


def review_detail(service, identity):
    entry = service.store.get(st.session_state.detail)
    if st.button(t("Retour"), key="back_detail"):
        st.session_state.detail = None
        st.session_state.table_epoch = st.session_state.get("table_epoch", 0) + 1
        st.rerun()
    if not entry:
        st.warning(t("Opération introuvable."))
        return
    st.subheader(f"{t('Dossier')} · {entry['displayId']}")
    st.caption(f"{t('Identifiant complet')} · {entry['id']}")
    if entry.get("bankReference"):
        st.text(f"{t('Référence bancaire')} · {entry['bankReference']}")
    st.dataframe(table_rows([entry]), hide_index=True, width="stretch")
    show_observation(entry)
    if entry["initialDecision"] != "Accepted":
        factors(entry)
    with st.expander(t("Données complètes")):
        data = entry.get("transaction", {})
        st.table(
            pd.DataFrame(
                [
                    {
                        t("Données de l’opération"): t(field["label"]),
                        t("Valeur"): data.get(field["key"], "—"),
                    }
                    for field in FIELDS
                ]
            ).astype(str)
        )
    st.subheader(t("Historique des décisions"))
    events = entry.get("events", [])
    if events:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        t("Décision"): decision_label(event["decision"]),
                        t("Agent"): t(event["agent"]) if event.get("automatic") else event["agent"],
                        t("Motif"): automatic_reason(event),
                        t("Date et heure"): event["timestamp"],
                        t("Identité vérifiée"): event["verified"],
                    }
                    for event in events
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption(t("Aucune décision manuelle."))
    if (
        identity["role"] != "reviewer"
        or entry["finalDecision"] == "Accepted"
        or entry.get("insufficientFunds")
    ):
        return
    form_column, _ = st.columns([4.8, 4.2])
    with form_column:
        outcome = decision_form(service, entry, identity)
    if outcome:
        if outcome == "cancelled":
            st.session_state.detail = None
        st.rerun()


def history_page(service, category):
    st.subheader(
        t(
            {
                "Accepted": "Opérations courantes",
                "Review": "Opérations douteuses",
                "Blocked": "Opérations frauduleuses",
            }[category]
        )
    )
    with st.container(width=260):
        period = st.radio(
            t("Exportation"),
            ["day", "week"],
            format_func=lambda value: t(
                "Quotidienne" if value == "day" else "Hebdomadaire"
            ),
            key="export_period_" + category,
        )
        today = date.today()
        date_key = "export_reference_" + category + "_" + period
        if st.session_state.get(date_key, today) > today:
            st.session_state[date_key] = today
        reference = st.date_input(
            t("Journée à exporter" if period == "day" else "Semaine contenant le"),
            value=today,
            max_value=today,
            key=date_key,
        )
        if period == "week":
            start = reference - timedelta(days=(reference.weekday() + 1) % 7)
            end = min(start + timedelta(days=6), today)
            st.caption(f"{start:%d/%m/%Y} — {end:%d/%m/%Y}")
        try:
            filename, data = service.store.export(
                category, reference.isoformat(), period, "local"
            )
        except ValueError as exc:
            st.error(t(str(exc)))
        else:
            st.download_button(
                t("Exporter la journée" if period == "day" else "Exporter la semaine"),
                data,
                file_name=filename,
                mime="text/csv",
                key="export_" + category + period,
            )
    entries = service.store.all(category)
    select_table(
        entries,
        "table_" + category + "_" + str(st.session_state.get("table_epoch", 0)),
        service,
    )


def select_table(entries, key, service):
    if not entries:
        st.info(t("Aucune opération trouvée."))
        return
    show_observation_column = entries[0]["finalDecision"] != "Accepted"
    headers = [
        ("Heure locale / ID", 1.4),
        ("Montant", 1),
        ("Score", 1),
        ("Décision initiale", 1.1),
        ("État final ou validation", 2.3),
        ("Agent de validation", 1.8),
        ("Motif de validation", 1.4),
    ]
    if show_observation_column:
        headers.append(("Observation", 2.4))
    header = "".join(
        f'<div style="flex:{width};">{escape(t(label))}</div>'
        for label, width in headers
    )
    st.markdown(
        '<div style="display:flex;background:rgba(20,184,166,.14);color:#2dd4bf;padding:12px;'
        "font-weight:700;font-size:13px;border-radius:6px 6px 0 0;"
        'border-bottom:2px solid #14b8a6;gap:8px;">' + header + "</div>",
        unsafe_allow_html=True,
    )
    with st.container(height=500, border=False):
        for entry in entries:
            approval = entry.get("approval", {})
            initial = decision_label(entry["initialDecision"], entry)
            final = decision_label(entry["finalDecision"], entry)
            color = (
                "#f59e0b"
                if entry["initialDecision"] == "Review"
                else "#f87171" if entry["initialDecision"] == "Blocked" else "#2dd4bf"
            )
            stamp = (
                datetime.fromisoformat(entry["timestamp"])
                .astimezone()
                .strftime("%d/%m/%Y %H:%M:%S")
            )
            identifier = escape(entry["displayId"])
            cells = [
                f'<div style="opacity:.78;font-size:12px;margin-bottom:4px;">{stamp}</div>'
                f'<span style="background:rgba(20,184,166,.14);border:1px solid rgba(45,212,191,.4);'
                f'padding:2px 8px;border-radius:4px;font-weight:700;color:#2dd4bf;">{identifier}</span>',
                f'{entry["amount"]:.2f} $US',
                f'{entry["score"]:.2%}',
                f'<strong style="color:{color};">{escape(initial)}</strong>',
                escape(final)
                + (
                    "<br><small>" + escape(automatic_reason(entry["lastDecision"])) + "</small>"
                    if entry.get("blockReason") == "validation_timeout"
                    else ""
                ),
                escape(approval.get("agent") or "—"),
                escape(approval.get("reason") or "—"),
            ]
            if entry.get("originalPayment"):
                original = entry["originalPayment"]
                cells[1] += (
                    "<br><small>"
                    + escape(original["amount"] + " " + original["currency"])
                    + "</small>"
                )
            open_key = "factors_" + entry["id"]
            if show_observation_column:
                observation = escape(observation_text(entry))
                if st.session_state.get(open_key):
                    data = json.loads(entry.get("factors") or "{}")
                    items = "".join(
                        "<li>" + escape(factor_text(item)) + "</li>"
                        for item in data.get("items", [])
                    )
                    observation += (
                        '<div style="margin-top:8px;padding:8px 10px;border-left:3px solid #0d9488;'
                        'background:rgba(15,23,42,.06);border-radius:0 6px 6px 0;">'
                        '<strong style="color:#14b8a6;">'
                        + escape(t("Champs ayant augmenté le score"))
                        + "</strong><ul>"
                        + items
                        + "</ul><small>"
                        + escape(factor_text(data.get("coverage", "")))
                        + "</small></div>"
                    )
                cells.append(observation)
            content = "".join(
                f'<div style="flex:{width};min-width:0;overflow-wrap:anywhere;">{value}</div>'
                for value, (_, width) in zip(cells, headers)
            )
            st.markdown(
                '<div style="display:flex;padding:16px 12px;border-bottom:1px solid rgba(148,163,184,.25);'
                'font-size:13px;align-items:flex-start;gap:8px;">' + content + "</div>",
                unsafe_allow_html=True,
            )
            columns = st.columns([1.2, 1.2, 1.8, 2, 2.8])
            if columns[0].button(
                t("Ouvrir le dossier"), key=key + "_open_" + entry["id"]
            ):
                st.session_state.detail = entry["id"]
                st.rerun()
            current_identity = streamlit_identity(st)
            if current_identity["role"] == "reviewer" and not entry.get(
                "insufficientFunds"
            ):
                actions = (
                    [("Accepted", "Valider"), ("Blocked", "Refuser")]
                    if entry["finalDecision"] == "Review"
                    else (
                        [("Accepted", "Valider après réclamation")]
                        if entry["finalDecision"] == "Blocked"
                        else []
                    )
                )
                for column, (decision, label) in zip(columns[1:3], actions):
                    if column.button(
                        t(label), key=key + "_" + decision + "_" + entry["id"]
                    ):
                        st.session_state["inline_focus_" + key] = entry["id"]
                        st.session_state["inline_decision_" + key] = {
                            "id": entry["id"],
                            "action": decision,
                        }
                        st.rerun()
            if show_observation_column and columns[-1].button(
                (
                    t("Masquer les champs")
                    if st.session_state.get(open_key)
                    else t("Champs ayant augmenté le score")
                ),
                key=key + "_factors_" + entry["id"],
            ):
                st.session_state[open_key] = not st.session_state.get(open_key, False)
                st.rerun()
            pending = st.session_state.get("inline_decision_" + key)
            if (
                pending
                and pending["id"] == entry["id"]
                and current_identity["role"] == "reviewer"
                and entry["finalDecision"] != "Accepted"
                and not entry.get("insufficientFunds")
            ):
                _, form_column, _ = st.columns([1.2, 4.8, 3])
                with form_column, st.container(key="inline_panel_" + entry["id"]):
                    outcome = decision_form(
                        service,
                        entry,
                        current_identity,
                        action=pending["action"],
                        form_key="inline_" + key + "_" + pending["action"],
                    )
                if outcome:
                    st.session_state.pop("inline_decision_" + key, None)
                    st.rerun()
                if st.session_state.pop("inline_focus_" + key, None) == entry["id"]:
                    panel = "inline_panel_" + entry["id"]
                    st.html(
                        f"""
                        <script>
                        (() => {{
                            const show = () => document.querySelector('.st-key-{panel}')
                                ?.scrollIntoView({{block: "nearest", behavior: "instant"}});
                            requestAnimationFrame(() => requestAnimationFrame(show));
                            setTimeout(show, 150);
                        }})();
                        </script>
                        """,
                        unsafe_allow_javascript=True,
                    )


def archive_search(service):
    with st.expander(t("Archives hebdomadaires")):
        if service.archive_error:
            st.error(t(service.archive_error))
        query = st.text_input(
            t("Rechercher une archive par date"), key="archive_query", live=True
        )
        archives = service.store.list_archives(query)
        if not archives:
            st.caption(t("Aucune archive disponible."))
            return
        by_start = {entry["week_start"]: entry for entry in archives}
        labels = {
            key: f"{date.fromisoformat(key):%d/%m/%Y} — "
            f"{date.fromisoformat(entry['week_end']) - timedelta(days=1):%d/%m/%Y}"
            for key, entry in by_start.items()
        }
        chosen = st.selectbox(
            t("Semaine archivée"),
            list(by_start),
            format_func=labels.get,
            key="archive_week",
        )
        entry = by_start[chosen]
        payload = service.store.archive_payload(chosen)
        st.download_button(
            t("Télécharger les trois catégories"),
            payload,
            file_name=entry["filename"],
            mime="application/zip",
            key="archive_zip",
        )
        category = st.selectbox(
            t("Catégorie"),
            list(CATEGORIES),
            format_func={
                key: t(label)
                for key, label in {
                    "Accepted": "Opérations courantes",
                    "Review": "Opérations douteuses",
                    "Blocked": "Opérations frauduleuses",
                }.items()
            }.get,
            key="archive_category",
        )
        filename = f"fraudscope-{NAMES[category]}-week-{chosen}.csv"
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            data = archive.read(filename)
        st.download_button(
            t("Télécharger le CSV de cette catégorie"),
            data,
            file_name=filename,
            mime="text/csv",
            key="archive_csv",
        )
        st.caption(
            t("L’archive conserve les décisions connues au moment de sa création.")
        )


def search_page(service):
    with st.container(width=320, key="compact_search", gap="small"):
        query = st.text_input(
            "Rechercher une référence ou un identifiant",
            key="search_query", live="150ms",
            placeholder="Ex. : 10 ou FRS-000010",
            label_visibility="collapsed",
        )
        entries = service.store.find(query, limit=8)
        numeric_query = query.strip()
        if numeric_query.isascii() and numeric_query.isdigit():
            entries = [
                entry for entry in entries
                if entry["displayId"].split("-")[-1].isdigit()
                and int(entry["displayId"].split("-")[-1]) == int(numeric_query)
            ]
        for entry in entries:
            if st.button(
                entry["displayId"], key="suggest_" + entry["id"],
                type="tertiary", icon=":material/receipt_long:",
                help=entry.get("bankReference") or "Ouvrir le dossier",
                width="content",
            ):
                st.session_state.detail = entry["id"]
                st.session_state.active_dialog = "search"
                st.rerun()
        if query and not entries:
            st.caption("Aucune opération trouvée.")
        archive_search(service)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Importer un ancien historique FraudScope."
    )
    parser.add_argument("--import-history", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=ROOT / "fraudscope.sqlite3")
    args = parser.parse_args()
    print(Store(args.database).import_legacy(args.import_history))
