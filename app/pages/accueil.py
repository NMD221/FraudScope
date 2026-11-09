# accueil
import streamlit as st

st.title("Application intelligente de détection de fraude à la carte bancaire")
st.markdown("### Cette application est un projet académique du cours de Concepts Statistique pour la Science de données à UQO (INF 5103)")

st.info(
    "**Objectif du projet :** Concevoir et évaluer un pipeline de machine learning "
    "capable de détecter les transactions bancaires frauduleuses dans un contexte de données fortement déséquilibrées."
)

st.header("Contexte du Problème")
st.write(
    "La fraude à la carte bancaire représente un défi majeur pour les institutions financières. "
    "Le problème principal réside dans l'extrême rareté des cas de fraude (souvent moins de 1 % des transactions totales), "
    "ce qui rend les modèles traditionnels inefficaces sans une approche rigoureuse de traitement des données et d'évaluation."
)

st.subheader("Donnes Utilisées")
st.write("Le jeu de données utilisé est : Credit Card Fraud Detection 2026. Il provient du site Kaggle."
       " Ces données sont globalement structurée ainsi :" )
col1, col2, col3 = st.columns(3)
col1.metric("Volume analysé", "20 000", "Transactions")
col2.metric("Taux de fraude", "1,7%", "Très déséquilibré")
col3.metric("Modèle Utilisé", "XGBoost", "Optimisés")

st.header("Navigation dans l'application")
st.write(
    "Utilisez le menu latéral pour explorer les différentes sections :"
)
st.markdown(
    """
    - **Analyse des Données :** Visualisation de la distribution des données utilisées dans le cadre de cette étude.
    - **Modélisation :** Présentation des performances de notre modèle de machine learning à partir des métriques (ROC-AUC, PR-AUC).
    - **Simulateur de Fraude :** Testez une transaction en temps réel pour évaluer la décision du modèle.
    """
)