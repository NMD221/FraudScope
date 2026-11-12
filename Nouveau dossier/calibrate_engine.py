# calibrate_engine.py
"""
Calibration des seuils (REVIEW, BLOCK) et des règles métier
sur le jeu de Test — jamais utilisé dans le notebook pour ça.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def load_test_split():
    """Reconstruit le même split Test que dans le notebook (random_state=42)."""
    data = pd.read_csv("credit/credit_card_fraud_2026.csv")
    X = data.drop(columns=["transaction_id", "is_fraud"])
    y = data["is_fraud"].astype(int)

    # Même split que cellule 3 du notebook
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    return X_test, y_test, data


def analyser_distribution_scores(X_test, y_test):
    """Distribution des scores sur Test → aide à choisir SEUIL_BLOCK."""
    bundle = joblib.load("models/final_model.joblib")
    pipe = bundle["pipeline"]
    features = bundle["features"]

    proba = pipe.predict_proba(X_test[features])[:, 1]

    print("=" * 60)
    print("A. DISTRIBUTION DES SCORES SUR TEST")
    print("=" * 60)
    print(f"Score min / médian / max : {proba.min():.4f} / {np.median(proba):.4f} / {proba.max():.4f}")
    print(f"Score moyen (fraudes)   : {proba[y_test == 1].mean():.4f}")
    print(f"Score moyen (légitimes) : {proba[y_test == 0].mean():.4f}")

    print(f"\nPercentiles des scores (toutes transactions) :")
    for p in [50, 90, 95, 99, 99.5, 99.9]:
        print(f"  P{p:5.1f} : {np.percentile(proba, p):.4f}")

    print(f"\nPercentiles des scores (fraudes uniquement) :")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p:5.1f} : {np.percentile(proba[y_test == 1], p):.4f}")

    return proba


def tester_seuils_block(X_test, y_test, proba):
    """Teste différents seuils BLOCK et affiche précision/rappel."""
    print("\n" + "=" * 60)
    print("B. CHOIX DU SEUIL BLOCK — performances sur Test")
    print("=" * 60)
    print(f"{'Seuil':>8} | {'# BLOCK':>8} | {'# fraudes':>10} | "
          f"{'Précision':>10} | {'Rappel':>8}")
    print("-" * 60)

    for seuil in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        y_block = (proba >= seuil).astype(int)
        n_block = y_block.sum()
        n_fraudes = (y_block & y_test).sum()
        if n_block == 0:
            print(f"{seuil:>8.2f} | {0:>8} | {0:>10} | {'-':>10} | {'-':>8}")
            continue
        prec = n_fraudes / n_block
        rec = n_fraudes / y_test.sum()
        print(f"{seuil:>8.2f} | {n_block:>8} | {n_fraudes:>10} | "
              f"{prec:>9.1%} | {rec:>7.1%}")


def analyser_regles(df_full):
    """Analyse la distribution et le taux de fraude des features utilisées dans les règles."""
    print("\n" + "=" * 60)
    print("C. ANALYSE DES FEATURES POUR LES RÈGLES MÉTIER")
    print("=" * 60)

    # 1. cvv_retry_count
    print("\n-- cvv_retry_count --")
    print(df_full.groupby("cvv_retry_count")["is_fraud"].agg(["count", "sum", "mean"]).round(4))

    # 2. amount_usd
    print("\n-- amount_usd (quantiles) --")
    print(df_full["amount_usd"].describe(percentiles=[.5, .9, .95, .99, .995]).round(2))

    # 3. Montant + étranger
    print("\n-- Combinaison montant + étranger --")
    for seuil in [1000, 2000, 3000, 5000]:
        mask = (df_full["amount_usd"] > seuil) & df_full["is_foreign_transaction"]
        n = mask.sum()
        fraudes = df_full.loc[mask, "is_fraud"].sum()
        taux = fraudes / n if n > 0 else 0
        print(f"  amount > {seuil:>4} + étranger : {n:>3} transactions, "
              f"{fraudes:>2} fraudes, taux = {taux:.1%}")

    # 4. VPN + IP mismatch + nouveau marchand
    print("\n-- VPN + IP mismatch + nouveau marchand --")
    mask = df_full["used_vpn"] & df_full["ip_country_mismatch"] & df_full["is_new_merchant"]
    n = mask.sum()
    fraudes = df_full.loc[mask, "is_fraud"].sum()
    print(f"  {n} transactions, {fraudes} fraudes, taux = {fraudes/n if n > 0 else 0:.1%}")

    # 5. Marchand à risque + vélocité
    print("\n-- merchant_risk_score + velocity_score --")
    for mrs in [60, 70, 80]:
        for vs in [40, 50, 60]:
            mask = (df_full["merchant_risk_score"] > mrs) & (df_full["velocity_score"] > vs)
            n = mask.sum()
            fraudes = df_full.loc[mask, "is_fraud"].sum()
            taux = fraudes / n if n > 0 else 0
            print(f"  mrs > {mrs} & vélocité > {vs} : {n:>3} tx, {fraudes:>2} fraudes, taux = {taux:.1%}")


if __name__ == "__main__":
    X_test, y_test, df_full = load_test_split()
    proba = analyser_distribution_scores(X_test, y_test)
    tester_seuils_block(X_test, y_test, proba)
    analyser_regles(df_full)