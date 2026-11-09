from pathlib import Path
import numpy as np
import pandas as pd


def extract_and_clean_features(df):
  df = df.copy()

  #Création des nouvelles features (voir analyse exploratoire de Yasmine)
  df["amount_to_balance_ratio"] = df["amount_usd"] / (df["account_balance_usd"] + 1)

  risk_flags = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt"
  ]
  df["risk_signal_count"] = df[risk_flags].sum(axis=1)
  df["txn_velocity"] = df["txn_count_last_24h"] / (df["hours_since_last_txn"] + 1)
  df["is_night"] = ((df["time_of_day_hour"] >= 0) & (df["time_of_day_hour"] <= 6)).astype(int)
  df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
  df["card_age_ratio"] = df["card_age_months"] / (df["customer_age"] * 12 + 1)

  #Encodage cyclique
  df["hour_sin"] = np.sin(2 * np.pi * df["time_of_day_hour"] / 24)
  df["hour_cos"] = np.cos(2 * np.pi * df["time_of_day_hour"] / 24)
  df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
  df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

  # Suppression des colonnes sources devenues inutiles + les drapeaux individuels de risque
  cols_to_drop = [
                   "amount_usd", "account_balance_usd",
                   "txn_count_last_24h", "hours_since_last_txn",
                   "time_of_day_hour", "day_of_week",
                   "card_age_months", "customer_age"
                 ] + risk_flags

  df = df.drop(columns=cols_to_drop, errors="ignore")

  return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
  print("Début du Feature Engineering sur les colonnes spécifiques")
  #Création de nouvelle features
  data=extract_and_clean_features(df)

  #Encodage des variables booléennes
  bool_cols = [
      "is_foreign_transaction",
      "is_new_merchant",
      "used_vpn",
      "ip_country_mismatch",
      "billing_shipping_mismatch",
      "is_ai_generated_scam_attempt",
  ]
  for col in bool_cols:
    if col in data.columns:
      data[col] = data[col].astype(int)
  print("Variables booléennes converties en entiers (0/1).")

  #Encodage One-Hot pour les variables catégorielles (merchant_category, card_type, auth_method, channel, device_type)
  cat_cols = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
  existing_cats = [c for c in cat_cols if c in data.columns]

  if existing_cats:
    data = pd.get_dummies(data, columns=existing_cats, drop_first=True, dtype=int)
    print(f"Encodage One-Hot appliqué sur : {existing_cats}")

  #Suppression de l'identifiant unique s'il est présent (inutile pour le modèle)
  if "transaction_id" in data.columns:
    # On peut le stocker temporairement ou le dropper pour l'entraînement
   data = data.drop(columns=['transaction_id'])

  print(f"Feature Engineering terminé. Nombre total de colonnes finales : {data.shape[1]}")
  return data


if __name__ == "__main__":
  from data_loader import load_raw_data

  try:
    df_raw = load_raw_data()
    df_processed = engineer_features(df_raw)

    #Sauvegarde dans data/processed
    base_dir = Path(__file__).resolve().parent.parent
    processed_path = base_dir / "data" / "processed" / "clean_transactions.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(processed_path, index=False)
    print(f"Données transformées sauvegardées avec succès à : {processed_path}")

  except Exception as e:
    print(f"[ERREUR] {e}")