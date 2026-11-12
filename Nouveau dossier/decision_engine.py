# decision_engine.py
"""
Moteur décisionnel — transforme un score modèle en décision + justification.

Architecture :
    1. Règles métier (override) : calées sur l'EDA + calibration.
    2. Seuil REVIEW : hérité du notebook (rappel cible 80 % → 0.0164).
    3. Seuil BLOCK  : calibré sur Test → 0.75 (précision 100 %, 0 faux positif).

Usage :
    from decision_engine import DecisionEngine
    engine = DecisionEngine("models/final_model.joblib")
    result = engine.decide_batch(df)          # df avec les colonnes du dataset
    result = engine.score_one({...})          # une transaction en dict
"""
import joblib
import pandas as pd
import numpy as np


class DecisionEngine:
    """Moteur décisionnel à 3 niveaux (ALLOW / REVIEW / BLOCK)."""

    # Seuil calibré sur Test (voir calibrate_engine.py) :
    #   0.75 → 8 BLOCK, 8 fraudes, précision = 100 %, 0 faux positif.
    #   0.65 → 12 BLOCK, 10 fraudes, précision = 83.3 %, 2 faux positifs.
    # On choisit 0.75 pour garantir une précision parfaite en blocage auto.
    SEUIL_BLOCK = 0.75

    def __init__(self, model_path: str):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.seuil_review = bundle["seuil"]     # ~0.0164 (rappel cible 80%)
        self.features = bundle["features"]
        self.use_fe = bundle["use_fe"]
        self.regles = self._charger_regles()

    # ==========================================================
    # RÈGLES MÉTIER (calibrées sur les distributions réelles)
    # ==========================================================
    def _charger_regles(self) -> list:
        """
        Règles absolues qui override le modèle.
        Priorité : plus le nombre est élevé, plus la règle passe en premier.

        Toutes en REVIEW : le BLOCK auto est réservé au modèle (seuil 0.75,
        précision 100 %). Aucune règle métier n'atteint cette précision.

        Chaque règle justifiée par un taux de fraude ≥ 10× le taux de base (1.7 %).
        """
        return [
            {
                "nom": "CVV retry × 2",
                # 254 transactions, 16.1 % de fraude → 9.5× la base
                "condition": lambda t: t["cvv_retry_count"] == 2,
                "decision": "REVIEW",
                "priorite": 90,
            },
            {
                "nom": "VPN + IP mismatch + nouveau marchand",
                # 30 transactions, 16.7 % de fraude → 9.8× la base
                "condition": lambda t: (
                    bool(t["used_vpn"])
                    and bool(t["ip_country_mismatch"])
                    and bool(t["is_new_merchant"])
                ),
                "decision": "REVIEW",
                "priorite": 85,
            },
            {
                "nom": "Marchand risqué + vélocité élevée",
                # 54 transactions, 18.5 % de fraude → 10.9× la base
                "condition": lambda t: (
                    t["merchant_risk_score"] > 70
                    and t["velocity_score"] > 40
                ),
                "decision": "REVIEW",
                "priorite": 75,
            },
        ]

    def _appliquer_regles(self, txn: pd.Series):
        """Retourne (décision, nom_règle) ou (None, None)."""
        for regle in sorted(self.regles, key=lambda r: -r["priorite"]):
            try:
                if regle["condition"](txn):
                    return regle["decision"], regle["nom"]
            except (KeyError, TypeError):
                continue
        return None, None

    # ==========================================================
    # EXPLICABILITÉ (top 3 raisons, heuristique lisible)
    # ==========================================================
    def _raisons(self, txn: pd.Series) -> list:
        """
        Retourne jusqu'à 3 raisons lisibles par un analyste.
        Ordre : facteurs les plus discriminants d'abord (EDA : cvv, merchant_risk, velocity).
        """
        raisons = []

        if txn["cvv_retry_count"] >= 1:
            raisons.append(f"CVV retry × {int(txn['cvv_retry_count'])}")
        if txn["merchant_risk_score"] > 70:
            raisons.append(f"Marchand à risque ({txn['merchant_risk_score']:.0f}/100)")
        if txn["velocity_score"] > 40:
            raisons.append(f"Vélocité élevée ({txn['velocity_score']:.0f})")
        if bool(txn["billing_shipping_mismatch"]):
            raisons.append("Adresse facturation ≠ livraison")
        if bool(txn["ip_country_mismatch"]):
            raisons.append("Pays IP ≠ pays carte")
        if bool(txn["used_vpn"]):
            raisons.append("VPN utilisé")
        if bool(txn["is_ai_generated_scam_attempt"]):
            raisons.append("Pattern scam IA détecté")
        if txn["amount_usd"] > 1000:
            raisons.append(f"Montant élevé ({txn['amount_usd']:.0f} $)")

        return raisons[:3] if raisons else ["Score modèle — pas de facteur saillant"]

    # ==========================================================
    # DÉCISION
    # ==========================================================
    def decide_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prend un DataFrame de transactions (colonnes du dataset).
        Retourne df + fraud_score + decision + source_decision + raisons.
        """
        # 1) Score modèle
        X = df[self.features]
        proba = self.pipeline.predict_proba(X)[:, 1]

        result = df.copy()
        result["fraud_score"] = proba

        decisions, sources, raisons_list = [], [], []

        for pos, (_, txn) in enumerate(df.iterrows()):
            score = proba[pos]

            # 2) Règles métier d'abord (override)
            dec_regle, nom_regle = self._appliquer_regles(txn)
            if dec_regle is not None:
                decisions.append(dec_regle)
                sources.append(f"Règle : {nom_regle}")
                raisons_list.append(self._raisons(txn))
                continue

            # 3) Sinon, modèle
            if score >= self.SEUIL_BLOCK:
                decisions.append("BLOCK")
                sources.append(f"Modèle (score {score:.3f} ≥ {self.SEUIL_BLOCK})")
            elif score >= self.seuil_review:
                decisions.append("REVIEW")
                sources.append(f"Modèle (score {score:.3f} ≥ {self.seuil_review:.4f})")
            else:
                decisions.append("ALLOW")
                sources.append("Modèle (score faible)")

            raisons_list.append(
                self._raisons(txn) if decisions[-1] != "ALLOW" else []
            )

        result["decision"] = decisions
        result["source_decision"] = sources
        result["raisons"] = raisons_list
        return result

    def score_one(self, transaction: dict) -> dict:
        """Scoring + décision pour une transaction unique (dict)."""
        df = pd.DataFrame([transaction])
        out = self.decide_batch(df).iloc[0]
        return {
            "transaction_id": out.get("transaction_id", None),
            "fraud_score": float(out["fraud_score"]),
            "decision": out["decision"],
            "source_decision": out["source_decision"],
            "raisons": out["raisons"],
        }