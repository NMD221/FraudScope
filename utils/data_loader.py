from pathlib import Path
import pandas as pd


def load_raw_data(filename: str = "credit_card_fraud_2026.csv") -> pd.DataFrame:

  # Définition dynamique du chemin racine du projet (deux niveaux au-dessus de src/)
  base_dir = Path(__file__).resolve().parent.parent
  raw_data_path = base_dir / "data" / "raw" / filename

  if not raw_data_path.exists():
    raise FileNotFoundError(
        f"Le fichier de données est introuvable à l'emplacement :"
        f" {raw_data_path}\n"
        f"Veuillez y déposer votre fichier CSV."
    )

  # Chargement des données
  df = pd.read_csv(raw_data_path)
  print(f"Données chargées avec succès : {df.shape[0]} lignes et {df.shape[1]} colonnes.")

  return df


if __name__ == "__main__":
  # Test rapide du module directement dans PyCharm
  df_transactions = load_raw_data()
  print("\nAperçu des premières lignes :")
  print(df_transactions.head())
  print("\nTypes de données :")
  print(df_transactions.dtypes)