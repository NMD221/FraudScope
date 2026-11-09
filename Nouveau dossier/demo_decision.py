# demo_decision.py
"""
Démonstration de l'outil décisionnel.
Scoring d'un batch de 200 transactions + évaluation de l'OUTIL (règles + modèle).
"""
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from decision_engine import DecisionEngine


def main():
    engine = DecisionEngine("models/final_model.joblib")
    print(f"Seuil REVIEW = {engine.seuil_review:.4f}")
    print(f"Seuil BLOCK  = {engine.SEUIL_BLOCK}\n")

    # Simule le batch quotidien de l'analyste
    df = pd.read_csv("credit/credit_card_fraud_2026.csv").sample(500, random_state=1)
    decisions = engine.decide_batch(df)

    # File d'alertes triée
    file_alertes = decisions[decisions["decision"] != "ALLOW"].sort_values(
        "fraud_score", ascending=False
    )

    print(f"=== File d'alertes : {len(file_alertes)} transactions ===\n")
    print(
        file_alertes[
            ["transaction_id", "amount_usd", "fraud_score",
             "decision", "source_decision", "is_fraud"]
        ].head(10).to_string(index=False)
    )

    # Évaluation de l'OUTIL (pas du modèle)
    y_true = decisions["is_fraud"]
    y_review_or_block = (decisions["decision"] != "ALLOW").astype(int)
    y_block = (decisions["decision"] == "BLOCK").astype(int)
    y_review = (decisions["decision"] == "REVIEW").astype(int)

    print(f"\n=== Performance de l'OUTIL (règles + modèle) ===")
    print(f"Fraudes réelles dans le batch : {y_true.sum()} / {len(y_true)}")
    print(f"BLOCK  : {y_block.sum():3d} transactions | "
          f"dont {(y_block & y_true).sum():2d} fraudes | "
          f"précision = {precision_score(y_true, y_block, zero_division=0):.1%}")
    print(f"REVIEW : {y_review.sum():3d} transactions | "
          f"dont {(y_review & y_true).sum():2d} fraudes | "
          f"précision = {precision_score(y_true, y_review, zero_division=0):.1%}")
    print(f"Rappel global (BLOCK + REVIEW) : "
          f"{recall_score(y_true, y_review_or_block):.1%}")

    # Détail des règles déclenchées
    print("\n=== Règles déclenchées ===")
    regles_declenchees = decisions[
        decisions["source_decision"].str.startswith("Règle")
    ]["source_decision"].value_counts()
    print(regles_declenchees.to_string() if len(regles_declenchees) else "Aucune règle déclenchée")


if __name__ == "__main__":
    main()