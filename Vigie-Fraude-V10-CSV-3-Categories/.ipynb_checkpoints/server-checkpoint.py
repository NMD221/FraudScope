"""Run Vigie Fraude locally or behind a bank gateway."""

import argparse
import json
import logging
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from backend.history import History
from backend.web import Application, BankConfig, MAX_BODY, error

ROOT = Path(__file__).resolve().parent


def load_json(name):
    return json.loads((ROOT / "backend" / name).read_text(encoding="utf-8"))


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(15)
        super().setup()

    def log_request(self, code="-", size="-"):
        # A query may include a bank reference. Keep it out of console logs.
        logging.info("%s %s", self.command, code)

    def dispatch(self):
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) > 1:
            self._reply(error("Le corps de la requête doit être un JSON valide.", 400))
            return
        length = lengths[0] if lengths else "0"
        if not re.fullmatch(r"[0-9]{1,10}", length):
            self._reply(error("Le corps de la requête doit être un JSON valide.", 400))
            return
        if int(length) > MAX_BODY:
            self._reply(error("La requête dépasse 8 Kio.", 413))
            return
        # Join duplicates so trusted singleton headers with duplicates fail validation.
        headers = {name.lower(): ", ".join(self.headers.get_all(name)) for name in self.headers.keys()}
        try:
            body = self.rfile.read(int(length))
            response = self.server.application.handle(self.command, self.path, headers, body)
        except Exception:
            logging.exception("Vigie Fraude request failed")
            response = error("Erreur interne du serveur.", 500)
        self._reply(response)

    def _reply(self, response):
        try:
            self.send_response(response.status)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(response.body)))
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(response.body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_PATCH = dispatch


def main():
    parser = argparse.ArgumentParser(description="Vigie Fraude")
    parser.add_argument("csv", nargs="?", type=Path, help="Fichier CSV à rejouer en démonstration")
    args = parser.parse_args()
    try:
        port = int(os.environ.get("PORT", "8083"))
        if not 1 <= port <= 65535:
            raise ValueError("PORT doit être compris entre 1 et 65535.")
        bank = BankConfig.from_env(os.environ)
        if bank and args.csv:
            raise ValueError("La lecture CSV est réservée au mode de démonstration.")
    except ValueError as exc:
        parser.error(str(exc))
    model = load_json("trained-model.json")
    policy = load_json("decision-policy.json")
    journal = ROOT / ("history-bank.jsonl" if bank else "history.jsonl")
    application = Application(model, policy, History(journal, model, policy["thresholds"]), args.csv, bank)
    try:
        with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
            server.application = application
            print(f"Vigie Fraude : http://127.0.0.1:{port}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                application.playback.stop()
    except OSError as exc:
        parser.exit(1, f"Impossible d’écouter sur 127.0.0.1:{port}. {exc}\n")


if __name__ == "__main__":
    main()
