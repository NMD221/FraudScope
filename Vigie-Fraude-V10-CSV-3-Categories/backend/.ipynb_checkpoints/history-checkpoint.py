"""Persistent JSONL history and decisions."""

import csv
import hashlib
import io
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from threading import RLock
from uuid import uuid4
from zoneinfo import ZoneInfo

from .model import FIELDS, decide, predict, risk_factors


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def percent(value):
    return f"{value * 100:.2f}".rstrip("0").rstrip(".").replace(".", ",") + "\u00a0%"


def input_fingerprint(transaction):
    # Preserve the fingerprint of requests previously recorded by JSON.stringify in Node.js.
    values = []
    for field in FIELDS:
        value = transaction[field["key"]]
        if field["kind"] in {"number", "integer"}:
            if value == 0:
                values.append("0")
                continue
            number = Decimal(repr(float(value)))
            if abs(value) >= 1e-6:
                text = format(number, "f")
                values.append(text.rstrip("0").rstrip(".") if "." in text else text)
            else:
                mantissa, exponent = format(number.normalize(), "e").split("e")
                values.append(f"{mantissa}e{int(exponent)}")
        else:
            values.append(json.dumps(value, ensure_ascii=False))
    return hashlib.sha256(("[" + ",".join(values) + "]").encode()).hexdigest()


def csv_text(value):
    if value and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def approval_fields(entry):
    if entry.get("appeal"):
        appeal = entry["appeal"]
        return appeal.get("operatorId"), appeal.get("reason")
    if entry["initialDecision"] == "review" and entry["finalDecision"] == "accepted":
        return entry.get("operatorId"), entry.get("approvalReason")
    return None, None


class History:
    def __init__(self, file, model, thresholds):
        self.file, self.model, self.thresholds = file, model, thresholds
        self.lock = RLock()
        self.entries = {}
        self.bank_ids = {}
        self.display_ids = {}
        self.next_number = 1
        if file.exists():
            for line in file.read_text(encoding="utf-8-sig").split("\n"):
                if line.strip():
                    self._index(json.loads(line))

        for entry in self.entries.values():
            display_id = entry.get("displayId")
            if display_id:
                self.display_ids[display_id] = entry["id"]
                self.next_number = max(self.next_number, int(display_id.split("-")[1]) + 1)
        # Old journals get stable IDs in the order in which their operations were created.
        next_legacy = 1
        for entry in self.entries.values():
            if not entry.get("displayId") and entry["source"] != "demo":
                while f"VIG-{next_legacy:06d}" in self.display_ids:
                    next_legacy += 1
                self._assign_display_id(entry, next_legacy)
                self.next_number = max(self.next_number, next_legacy + 1)

    def _assign_display_id(self, entry, number):
        identifier = f"VIG-{number:06d}"
        entry["displayId"] = identifier
        self.display_ids[identifier] = entry["id"]

    def _index(self, entry):
        self.entries[entry["id"]] = entry
        if entry.get("bankReference"):
            self.bank_ids[entry["bankReference"]] = entry["id"]

    def _save(self, entry):
        if "displayId" not in entry:
            entry["displayId"] = f"VIG-{self.next_number:06d}"
        with self.file.open("a", encoding="utf-8", newline="\n") as journal:
            journal.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        self._index(entry)
        self.display_ids[entry["displayId"]] = entry["id"]
        self.next_number = max(self.next_number, int(entry["displayId"].split("-")[1]) + 1)
        return entry

    def record(self, transaction, source, **bank_details):
        score = predict(transaction, self.model)
        insufficient = transaction["amount_usd"] > transaction["account_balance_usd"]
        decision = "blocked" if insufficient else decide(score, self.thresholds)
        timestamp = now()
        entry = {
            **bank_details, "id": str(uuid4()), "timestamp": timestamp, "source": source,
            "amount": transaction["amount_usd"], "score": score, "initialDecision": decision,
            "finalDecision": None if decision == "review" else decision,
            "observation": "", "resolvedAt": None if decision == "review" else timestamp,
        }
        if insufficient:
            entry["blockReason"] = "insufficient_funds"
            entry["observation"] = "Solde insuffisant. Le montant dépasse le solde du compte."
        elif decision != "accepted":
            threshold = self.thresholds["review" if decision == "review" else "block"]
            entry["decisionThreshold"] = threshold
            entry["factors"] = risk_factors(transaction, self.model)
            action, limit = ("Mise en attente", "vérification") if decision == "review" else ("Blocage automatique", "refus")
            entry["observation"] = f"{action}. Le score de {percent(score)} atteint le seuil de {limit} de {percent(threshold)}."
        with self.lock:
            return self._save(entry)

    def find_bank_transaction(self, reference):
        with self.lock:
            return self.entries.get(self.bank_ids.get(reference))

    def find_transaction_by_id(self, identifier):
        with self.lock:
            entry = self.entries.get(identifier)
            return entry if entry and entry["source"] != "demo" else None

    def lookup_transactions(self, reference):
        with self.lock:
            number = re.fullmatch(r"(?:VIG-)?([0-9]{1,12})", reference.upper())
            short_id = f"VIG-{int(number[1]):06d}" if number else reference.upper()
            candidates = (
                self.bank_ids.get(reference),
                self.display_ids.get(short_id),
                reference.lower(),
            )
            return [entry for identifier in dict.fromkeys(candidates)
                    if (entry := self.find_transaction_by_id(identifier)) is not None]

    def suggest_transactions(self, prefix, limit=8):
        normalized = prefix.casefold()
        number = re.fullmatch(r"(?:vig-)?([0-9]{1,12})", normalized)
        with self.lock:
            results = self.lookup_transactions(prefix)[:limit]
            seen = {entry["id"] for entry in results}
            for entry in reversed(self.entries.values()):
                if len(results) >= limit:
                    break
                if entry["id"] in seen or entry["source"] == "demo":
                    continue
                display = entry["displayId"].casefold()
                bank = entry.get("bankReference", "").casefold()
                digits = display.removeprefix("vig-")
                short = digits.lstrip("0")
                if (display.startswith(normalized) or digits.startswith(normalized) or bank.startswith(normalized)
                        or entry["id"].startswith(normalized)
                        or (number and short.startswith(number[1].lstrip("0") or "0"))):
                    results.append(entry)
            return results

    def record_bank_transaction(self, reference, transaction, original=None):
        fingerprint = input_fingerprint(transaction)
        with self.lock:
            existing = self.find_bank_transaction(reference)
            if existing:
                same_input = existing["inputHash"] == fingerprint and existing.get("originalPayment") == original
                return (existing if same_input else None), False
            details = {"bankReference": reference, "inputHash": fingerprint}
            if original is not None:
                details["originalPayment"] = original
            return self.record(transaction, "bank", **details), True

    def resolve(self, identifier, decision, reason, operator, verified):
        with self.lock:
            entry = self.entries.get(identifier)
            if entry is None:
                return None
            timestamp = now()
            if decision == "approve-appeal":
                if entry.get("blockReason") == "insufficient_funds":
                    return "insufficient_funds"
                if entry["finalDecision"] != "blocked":
                    return False
                return self._save({
                    **entry, "finalDecision": "accepted", "resolvedAt": timestamp,
                    "appeal": {"reason": reason, "operatorId": operator, "operatorVerified": verified,
                               "refusedAt": entry.get("resolvedAt")},
                })
            if entry["initialDecision"] != "review" or entry["finalDecision"]:
                return False
            updated = {**entry, "operatorId": operator, "operatorVerified": verified,
                       "finalDecision": decision, "resolvedAt": timestamp}
            if decision == "accepted":
                updated["approvalReason"] = reason
            else:
                updated["reviewObservation"] = entry["observation"]
                updated["refusalReason"] = reason
                updated["observation"] = f"Refus après examen. {reason}"
            return self._save(updated)

    def list(self, category):
        with self.lock:
            return [entry for entry in reversed(self.entries.values()) if entry["source"] != "demo"
                    and (entry["finalDecision"] or entry["initialDecision"]) == category]

    def export_csv(self, period, reference, zone, category):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", reference) or period not in {"day", "week"}:
            raise ValueError("Invalid date or period")
        start = date.fromisoformat(reference)
        if period == "week":
            start -= timedelta(days=start.weekday())
        end = start + timedelta(days=7 if period == "week" else 1)
        time_zone = ZoneInfo(zone)
        buffer = io.StringIO()
        buffer.write("\ufeff")
        writer = csv.writer(buffer, lineterminator="\r\n")
        writer.writerow(["id", "horodatage_utc", "source", "montant_usd", "score", "decision_initiale",
                         "decision_finale", "decision_finale_utc", "motif_blocage",
                         "agent_validation", "motif_validation"])
        with self.lock:
            entries = list(self.entries.values())
        for entry in entries:
            local_date = datetime.fromisoformat(entry["timestamp"]).astimezone(time_zone).date()
            if (entry["source"] != "demo" and start <= local_date < end
                    and (entry["finalDecision"] or entry["initialDecision"]) == category):
                columns = [entry.get(key) for key in (
                    "id", "timestamp", "source", "amount", "score", "initialDecision",
                    "finalDecision", "resolvedAt", "blockReason",
                )]
                writer.writerow([*columns, *(csv_text(value) for value in approval_fields(entry))])
        return start.isoformat(), buffer.getvalue()
