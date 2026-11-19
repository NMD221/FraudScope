"""Bilingual assets, HTTP routing and CSV playback."""

import csv
import hmac
import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock, Thread
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfoNotFoundError

from .model import GROUPS, load_dataset, validate

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
SECURITY_POLICY = "default-src 'self'; frame-ancestors 'self'; base-uri 'none'; object-src 'none'"
SECRET_PATTERN = re.compile(r"[A-Za-z0-9_-]{32,256}")
MAX_BODY = 8192
EXPORT_CATEGORIES = {"accepted": "courantes", "review": "douteuses", "blocked": "frauduleuses"}


@dataclass(frozen=True)
class BankConfig:
    gateway_secret: str
    payment_token: str

    @classmethod
    def from_env(cls, env):
        mode = env.get("VIGIE_MODE", "demo")
        if mode == "demo":
            return None
        if mode != "bank":
            raise ValueError("VIGIE_MODE must be demo or bank.")
        gateway, payment = (env.get(key, "") for key in ("VIGIE_GATEWAY_SECRET", "VIGIE_PAYMENT_TOKEN"))
        if gateway == payment or not all(SECRET_PATTERN.fullmatch(value) for value in (gateway, payment)):
            raise ValueError("VIGIE_GATEWAY_SECRET and VIGIE_PAYMENT_TOKEN must be distinct secrets of 32 to 256 URL-safe characters.")
        return cls(gateway, payment)


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    headers: dict


def json_response(data, status=200):
    return Response(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                    {"Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store"})


def error(message, status, **extra):
    return json_response({"error": message, **extra}, status)


def read_json(headers, body):
    if headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        return None, error("Le contenu doit être au format application/json.", 415)
    if len(body) > MAX_BODY:
        return None, error("La requête dépasse 8 Kio.", 413)
    try:
        return json.loads(body.decode("utf-8")), None
    except (UnicodeError, ValueError, RecursionError):
        return None, error("Le corps de la requête doit être un JSON valide.", 400)


def load_assets():
    files = {"index.html": "text/html", "style.css": "text/css"}
    files.update({f"{name}.js": "application/javascript" for name in
                  ("app", "live", "factors", "language", "translations")})
    sources = {name: (FRONTEND / name).read_text(encoding="utf-8-sig") for name in files}
    catalog = sources["translations.js"].strip().removeprefix("export const english = ").removesuffix(";")
    translations = json.loads(catalog)
    translate = lambda value: translations.get(value, value)
    page = sources["index.html"]
    english = (page.replace('<html lang="fr">', '<html lang="en">')
               .replace('class="brand" href="/"', 'class="brand" href="/en"')
               .replace('hreflang="fr" aria-current="page"', 'hreflang="fr"')
               .replace('hreflang="en"', 'hreflang="en" aria-current="page"')
               .replace('aria-label="Langue"', 'aria-label="Language"'))
    english = re.sub(r">([^<>]+)<", lambda match: ">" + re.sub(
        r"\S(?:.*\S)?", lambda text: translate(text[0]), match[1]) + "<", english)
    english = re.sub(r'(aria-label|content)="([^"]+)"', lambda m: f'{m[1]}="{translate(m[2])}"', english)
    assets = {}
    for name, mime in files.items():
        path = "/" if name == "index.html" else "/" + name
        assets[path] = Response(200, sources[name].encode(), {
            "Content-Type": mime + "; charset=utf-8", "Content-Security-Policy": SECURITY_POLICY,
        })
    assets["/en"] = Response(200, english.encode(), assets["/"].headers)
    return assets


def matches(value, expected):
    return hmac.compare_digest((value or "").encode("utf-8"), expected.encode("utf-8"))


def valid_operator(value):
    return isinstance(value, str) and 0 < len(value) <= 80 and value == value.strip() and not re.search(r"[\x00-\x1f\x7f,]", value)


class Playback:
    def __init__(self, history, path):
        self.history, self.path = history, path
        self.rows = None
        self.index = 0
        self.running = None
        self.error = ""
        self.lock = RLock()

    def snapshot(self):
        with self.lock:
            return {"running": self.running is not None, "csvAvailable": self.path is not None,
                    "processed": self.index, "total": len(self.rows) if self.rows is not None else None,
                    "error": self.error}

    def stop(self):
        with self.lock:
            if self.running is not None:
                self.running.set()
                self.running = None

    def start(self):
        with self.lock:
            if self.running is not None:
                return
            if self.rows is None:
                self.rows = load_dataset(self.path)
            if not self.rows:
                raise ValueError("No transactions in CSV")
            if self.index == len(self.rows):
                self.index = 0
            self.error = ""
            self.running = token = Event()
            self._tick(token)
            if self.running is not None:
                Thread(target=self._run, args=(token,), daemon=True).start()

    def _run(self, token):
        while not token.wait(1):
            self._tick(token)

    def _tick(self, token):
        with self.lock:
            if self.running is not token:
                return
            try:
                self.history.record(self.rows[self.index][0], "csv")
            except (OSError, ValueError, ArithmeticError):
                self.error = "Impossible d’enregistrer la transaction."
                self.stop()
                return
            self.index += 1
            if self.index == len(self.rows):
                self.stop()


class Application:
    def __init__(self, model, policy, history, csv_path=None, bank=None):
        self.model, self.history, self.bank = model, history, bank
        self.playback = Playback(history, csv_path)
        self.assets = load_assets()
        self.categories = model["preprocessing"]["categories"]
        self.metadata = {
            "groups": [{**group, "fields": [
                {**field, "options": self.categories[field["key"]]} if field["kind"] == "category" else field
                for field in group["fields"]
            ]} for group in GROUPS],
            "metrics": model["metrics"], "threshold": model["threshold"], "policy": policy, "bankMode": bank is not None,
        }

    def handle(self, method, target, headers, body=b""):
        url = urlsplit(target)
        path = url.path
        query = {key: values[0] for key, values in parse_qs(url.query, keep_blank_values=True).items()}
        operator = role = None
        if self.bank:
            if path == "/api/bank/transactions":
                return self._bank_transaction(method, query, headers, body)
            operator = headers.get("x-vigie-agent-id")
            role = headers.get("x-vigie-agent-role")
            if not (matches(headers.get("x-vigie-gateway-secret"), self.bank.gateway_secret)
                    and valid_operator(operator) and role in {"viewer", "reviewer"}):
                return error("Accès non autorisé.", 401)
            if path == "/api/predict" or path.startswith("/api/simulation/"):
                return error("Cette fonction est réservée à la démonstration.", 403)

        if method in {"GET", "HEAD"} and path in self.assets:
            return self.assets[path]
        if path == "/api/model":
            return json_response(self.metadata) if method == "GET" else error("Utilisez GET.", 405)
        if path == "/api/predict":
            return self._predict(method, headers, body)
        if method == "GET" and path == "/api/history":
            category = query.get("category", "accepted")
            if category not in {"accepted", "review", "blocked"}:
                return error("Catégorie inconnue.", 400)
            entries = [self._public_entry(entry) for entry in self.history.list(category)]
            return json_response({**self.playback.snapshot(), "entries": entries,
                                  "authenticated": operator is not None, "canReview": role != "viewer"})
        if method == "GET" and path == "/api/history/suggestions":
            prefix = query.get("reference", "").strip()
            if len(prefix) > 128:
                return error("Renseignez une référence ou un identifiant Vigie.", 422)
            suggestions = self.history.suggest_transactions(prefix) if prefix else []
            return json_response({"entries": [
                {key: entry[key] for key in ("id", "displayId", "bankReference") if key in entry}
                for entry in suggestions
            ]})
        if method == "GET" and path == "/api/history/lookup":
            reference = query.get("reference", "").strip()
            if not reference or len(reference) > 128:
                return error("Renseignez une référence ou un identifiant Vigie.", 422)
            entries = self.history.lookup_transactions(reference)
            return json_response({
                "entries": [self._public_entry(entry) for entry in entries],
                "authenticated": operator is not None,
                "canReview": role != "viewer",
            })
        if method == "GET" and (detail := re.fullmatch(r"/api/history/([a-f0-9-]+)", path)):
            entry = self.history.find_transaction_by_id(detail[1])
            if not entry:
                return error("Opération introuvable.", 404)
            return json_response({
                "entries": [self._public_entry(entry)],
                "authenticated": operator is not None, "canReview": role != "viewer",
            })
        if method == "GET" and path == "/api/history/export":
            period = query.get("period")
            category = query.get("category")
            if category not in EXPORT_CATEGORIES:
                return error("Catégorie inconnue.", 400)
            try:
                start, data = self.history.export_csv(period, query.get("date", ""), query.get("tz", "UTC"), category)
            except (ValueError, ZoneInfoNotFoundError, OverflowError):
                return error("Date, période ou fuseau horaire invalide.", 400)
            return Response(200, data.encode("utf-8"), {
                "Content-Type": "text/csv; charset=utf-8", "Cache-Control": "no-store",
                "Content-Disposition": f'attachment; filename="vigie-{EXPORT_CATEGORIES[category]}-{period}-{start}.csv"',
            })
        if method == "POST" and path == "/api/simulation/start":
            if not self.playback.path:
                return error("Lancez le serveur avec le chemin d’un fichier CSV.", 422)
            try:
                self.playback.start()
            except (OSError, ValueError, csv.Error):
                return error("Impossible de charger le flux. Vérifiez le CSV.", 422)
            return json_response(self.playback.snapshot())
        if method == "POST" and path == "/api/simulation/pause":
            self.playback.stop()
            return json_response(self.playback.snapshot())
        resolution = re.fullmatch(r"/api/history/([a-f0-9-]+)/(accepted|blocked|approve-appeal)", path)
        if method == "POST" and resolution:
            return self._resolve(*resolution.groups(), headers, body, operator, role)
        return error("Page introuvable.", 404)

    @staticmethod
    def _public_entry(entry):
        return {key: value for key, value in entry.items() if key != "inputHash"}

    def _predict(self, method, headers, body):
        if method != "POST":
            return error("Utilisez POST.", 405)
        transaction, failure = read_json(headers, body)
        if failure:
            return failure
        errors = validate(transaction)
        if errors:
            return error("Vérifiez les champs de la transaction.", 422, fields=errors)
        entry = self.history.record(transaction, "manual")
        threshold = self.model["threshold"]
        return json_response({**entry, "fraud_score": entry["score"], "threshold": threshold,
                              "is_fraud": int(entry["score"] >= threshold)})

    def _bank_transaction(self, method, query, headers, body):
        if not matches(headers.get("authorization"), "Bearer " + self.bank.payment_token):
            return error("Accès non autorisé.", 401)
        status = 200
        if method == "GET":
            entry = self.history.find_bank_transaction(query.get("reference"))
            if not entry:
                return error("Opération introuvable.", 404)
        elif method == "POST":
            data, failure = read_json(headers, body)
            if failure:
                return failure
            reference = data.get("reference") if isinstance(data, dict) else None
            if not isinstance(reference, str) or not re.fullmatch(r"[A-Za-z0-9._:/-]{1,128}", reference):
                return error("Référence bancaire invalide.", 422)
            transaction = data.get("transaction")
            errors = validate(transaction)
            if errors:
                return error("Vérifiez les champs de la transaction.", 422, fields=errors)
            for key, categories in self.categories.items():
                if transaction[key] not in categories:
                    return error("Catégorie inconnue.", 422, field=key)
            original = data.get("originalPayment")
            if "originalPayment" in data:
                amount = original.get("amount") if isinstance(original, dict) else None
                currency = original.get("currency") if isinstance(original, dict) else None
                if (not isinstance(amount, str) or not re.fullmatch(r"(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,6})?", amount)
                        or not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency)):
                    return error("Montant ou devise d’origine invalide.", 422)
                original = {"amount": amount, "currency": currency}
            entry, created = self.history.record_bank_transaction(reference, transaction, original)
            if entry is None:
                return error("Cette référence est déjà associée à des données différentes.", 409)
            status = 201 if created else 200
        else:
            return error("Utilisez GET ou POST.", 405)
        return json_response({key: entry[key] for key in ("id", "displayId", "bankReference", "score", "initialDecision",
                                                          "finalDecision", "blockReason", "originalPayment") if key in entry}, status)

    def _resolve(self, identifier, decision, headers, body, operator, role):
        verified = operator is not None
        if verified and (role != "reviewer" or headers.get("x-vigie-action") != "review"):
            return error("Cette décision requiert une habilitation de validation.", 403)
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError):
            return error("Renseignez votre identifiant d’agent.", 400)
        if not isinstance(data, dict):
            data = {}
        if not verified:
            submitted = data.get("operatorId")
            operator = submitted.strip() if isinstance(submitted, str) else ""
        if not valid_operator(operator):
            return error("Renseignez un identifiant d’agent valide (1 à 80 caractères).", 422)
        reason = data.get("reason")
        reason = reason.strip() if isinstance(reason, str) else ""
        if not 1 <= len(reason) <= 500:
            return error("Le motif doit contenir entre 1 et 500 caractères.", 422)
        result = self.history.resolve(identifier, decision, reason, operator, verified)
        if result is None:
            return error("Opération introuvable.", 404)
        if result == "insufficient_funds":
            return error("Un refus pour solde insuffisant ne peut pas être levé après réclamation.", 409)
        if result is False:
            return error("Seule une opération encore refusée peut être validée après réclamation."
                         if decision == "approve-appeal" else "Cette opération a déjà une décision finale.", 409)
        return json_response(self._public_entry(result))
