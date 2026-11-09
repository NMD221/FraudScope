from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
#from src.features import extract_and_clean_features


#Chemins dynamiques avec pathlib
#On remonte à la racine
BASE_DIR = Path(__file__).resolve().parent.parent.parent
#on pointe vers le dossier model puis vers le fichier du modèle
MODEL_PATH = BASE_DIR /"models" / "xgboost_fraud_model.joblib"
#On pointe vers les données nettoyées
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "clean_transactions.csv"


@st.cache_resource
def load_model_and_columns():

  if not MODEL_PATH.exists():
    return None, None

  model = joblib.load(MODEL_PATH)

  #Chargement d'une référence de colonnes depuis le dataset nettoyé si disponible
  feature_columns = None
  if PROCESSED_DATA_PATH.exists():
    df_ref = pd.read_csv(PROCESSED_DATA_PATH, nrows=5)
    if "is_fraud" in df_ref.columns:
      df_ref = df_ref.drop(columns=["is_fraud"])
    if "transaction_id" in df_ref.columns:
      df_ref = df_ref.drop(columns=["transaction_id"])
    feature_columns = list(df_ref.columns)

  return model, feature_columns


model, feature_columns = load_model_and_columns()

# Titre principal
st.title("Simulateur Intelligent de Transaction et de Détection de Fraude à la Carte Bancaire")
st.markdown(
    "Cette interface permet de simuler et d'évaluer les risques de fraude par transaction en"
    " temps réel."
)
st.info("Veuillez renseigner les paramètres de la Transaction ci_dessous et cliquer sur **Analyser la Transaction**. Puis vous rendre en bas de page pour l'évaluation")

if model is None:
  st.error(
      f"Modèle introuvable à l'emplacement : {MODEL_PATH}. Veuillez d'abord"
      " exécuter le script d'entraînement (`utils/modeling.py`)."
  )
else:


        st.markdown("# Paramètres de la Transaction")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        # Formulaire de saisie utilisateur groupé par thématiques
        with col1.form(key="Montant_Contexte"):
            st.subheader("1. Montant et Contexte")
            amount_usd = st.number_input("Montant de la transaction (USD)", min_value=0.0, value=150.0, step=10.0)
            customer_age = st.slider("Âge du client", 18, 90, 34)
            card_age_months = st.number_input("Ancienneté de la carte (mois)", min_value=0, value=24)
            account_balance_usd = st.number_input("Solde du compte (USD)", value=2500.0)

        with col2.form(key="Catégories_Canaux"):
            st.subheader("2. Catégories et Canaux")
            merchant_category = st.selectbox(
                "Catégorie du marchand",
                ["electronics", "grocery", "travel", "luxury", "entertainment", "clothing"],
            )
            card_type = st.selectbox("Type de carte", ["credit", "debit"])
            auth_method = st.selectbox("Méthode d'authentification", ["chip", "magstripe", "online", "biometric"])
            channel = st.selectbox("Canal de paiement", ["web", "mobile_app", "pos", "atm"])
            device_type = st.selectbox("Type d'appareil", ["ios", "android", "windows", "macos", "other"])

        with col3.form(key="indicateurs_comportementaux_risques"):
            st.subheader("3. Indicateurs Comportementaux et Risques")
            hours_since_last_txn = st.number_input("Heures depuis la dernière transaction", value=12.5)
            txn_count_last_24h = st.number_input("Nombre de transactions (dernières 24h)", min_value=0, value=2)
            distance_from_home_km = st.number_input("Distance du domicile (km)", min_value=0.0, value=15.2)
            cvv_retry_count = st.number_input("Nombre de tentatives CVV échouées", min_value=0, value=0)
            velocity_score = st.number_input("Score de vélocité", value=1.2)
            merchant_risk_score = st.slider("Score de risque du marchand", 0.0, 1.0, 0.15)
            prior_disputes = st.number_input("Litiges antérieurs", min_value=0, value=0)

        with col4.form(key="drapeaux_sécurité"):
            st.subheader("4. Drapeaux d'Alerte ou de Sécurité")
            is_foreign_transaction = st.checkbox("Transaction étrangère", value=False)
            is_new_merchant = st.checkbox("Nouveau marchand", value=False)
            used_vpn = st.checkbox("Utilisation d'un VPN", value=False)
            ip_country_mismatch = st.checkbox("Incohérence IP / Pays", value=False)
            billing_shipping_mismatch = st.checkbox("Incohérence Facturation / Livraison", value=False)
            is_ai_generated_scam_attempt = st.checkbox("Tentative suspecte générée par IA", value=False)

            time_of_day_hour = st.slider("Heure de la journée", 0, 23, 14)
            day_of_week = st.selectbox(
                "Jour de la semaine",
                [0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: [
                    "Lundi",
                    "Mardi",
                    "Mercredi",
                    "Jeudi",
                    "Vendredi",
                    "Samedi",
                    "Dimanche",
                ][x],
            )

            submit_button = st.form_submit_button(label="Analyser la Transaction")

        st.markdown('___________')
        # corps principal : Affichage des résultats
        st.markdown("# Evaluation de la Transaction")
        sous_col1, sous_col2 = st.columns([1, 1])
        if submit_button:
            # Construction du dictionnaire de données brutes saisies
            input_data = {
                "amount_usd": amount_usd,
                "merchant_category": merchant_category,
                "card_type": card_type,
                "auth_method": auth_method,
                "channel": channel,
                "device_type": device_type,
                "is_foreign_transaction": is_foreign_transaction,
                "hours_since_last_txn": hours_since_last_txn,
                "txn_count_last_24h": txn_count_last_24h,
                "distance_from_home_km": distance_from_home_km,
                "card_age_months": card_age_months,
                "customer_age": customer_age,
                "account_balance_usd": account_balance_usd,
                "is_new_merchant": is_new_merchant,
                "used_vpn": used_vpn,
                "ip_country_mismatch": ip_country_mismatch,
                "billing_shipping_mismatch": billing_shipping_mismatch,
                "cvv_retry_count": cvv_retry_count,
                "velocity_score": velocity_score,
                "time_of_day_hour": time_of_day_hour,
                "day_of_week": day_of_week,
                "is_ai_generated_scam_attempt": is_ai_generated_scam_attempt,
                "merchant_risk_score": merchant_risk_score,
                "prior_disputes": prior_disputes,
                "log_amount": np.log1p(amount_usd),
            }

            df_input = pd.DataFrame([input_data])

            # Creation de nouvelles features
            # df_input=extract_and_clean_features(df_input)
            # Encodage des booléens en entiers
            bool_cols = [
                "is_foreign_transaction",
                "is_new_merchant",
                "used_vpn",
                "ip_country_mismatch",
                "billing_shipping_mismatch",
                "is_ai_generated_scam_attempt",
            ]
            for b_col in bool_cols:
                if b_col in df_input.columns:
                    df_input[b_col] = df_input[b_col].astype(int)

            # Encodage One-Hot identique à l'entraînement
            cat_cols = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
            df_input = pd.get_dummies(df_input, columns=cat_cols, drop_first=True, dtype=int)

            # Réalignement des colonnes sur celles attendues par le modèle
            if feature_columns:
                for col in feature_columns:
                    if col not in df_input.columns:
                        df_input[col] = 0
                df_input = df_input[feature_columns]

            # Prédiction
            fraud_prob = model.predict_proba(df_input)[0][1]

            with sous_col1:
                st.subheader("Résultat de l'Analyse")
                st.metric(label="Probabilité de Fraude estimée", value=f"{fraud_prob * 100 :.2f}%")

                # Seuils de décision métier
                if fraud_prob >= 0.7:
                    st.error(
                        " **ALERTE ROUGE : Transaction hautement suspecte !** Recommandation"
                        " : Bloquer la transaction et contacter le client."
                    )
                elif fraud_prob >= 0.3:
                    st.warning(
                        "**RISQUE MODÉRÉ :** Transaction atypique. Recommandation :"
                        " Soumettre à une vérification 3-D Secure / SMS."
                    )
                else:
                    st.success(
                        "**TRANSACTION SÛRE :** Aucun indicateur critique majeur détecté."
                        " Autorisation validée."
                    )

            with sous_col2:
                st.subheader("Indicateurs Clés de la Saisie")
                st.write(f"- **Montant :** {amount_usd} USD")
                st.write(f"- **Score de vélocité :** {velocity_score}")
                st.write(f"- **Distance du domicile :** {distance_from_home_km} km")
                st.write(f"- **Tentatives CVV échouées :** {cvv_retry_count}")
        else:
            with sous_col2:
                st.info("Veuillez renseigner les paramètres de la Transaction ci_dessus et cliquer sur **Analyser la Transaction**.")