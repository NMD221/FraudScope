
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="FraudScope", layout="wide")

st.title("FraudScope")
st.write("Analyse des transactions bancaires")

df = pd.read_csv("credit_card_fraud_2026.csv")

st.metric("Nombre de transactions", len(df))
st.metric("Transactions frauduleuses", int(df["is_fraud"].sum()))

st.subheader("Répartition des transactions")
st.bar_chart(
    df["is_fraud"]
      .map({0: "Normale", 1: "Fraude"})
      .value_counts()
)
st.subheader("Explorer les transactions")

filtre = st.selectbox(
    "Type de transaction",
    ["Toutes", "Normales", "Frauduleuses"]
)

if filtre == "Normales":
    donnees = df[df["is_fraud"] == 0]
elif filtre == "Frauduleuses":
    donnees = df[df["is_fraud"] == 1]
else:
    donnees = df

st.write(f"{len(donnees)} transactions trouvées")
st.dataframe(donnees.head(100))

st.divider()
st.header("Tableau de bord statistique")

# Calcul des indicateurs
total = len(df)
nb_fraudes = (df["is_fraud"] == 1).sum()
taux_fraude = nb_fraudes / total * 100
montant_moyen = df["amount_usd"].mean()
montant_median = df["amount_usd"].median()

# Affichage
col1, col2, col3 = st.columns(3)

col1.metric("Taux de fraude", f"{taux_fraude:.2f} %")
col2.metric("Montant moyen", f"{montant_moyen:.2f} $")
col3.metric("Montant médian", f"{montant_median:.2f} $")

st.subheader("Distribution des montants")

fig, ax = plt.subplots()

ax.hist(
    df["amount_usd"],
    bins=40,
    edgecolor="black"
)

ax.set_xlabel("Montant de la transaction ($)")
ax.set_ylabel("Nombre de transactions")
ax.set_title("Distribution des montants")

st.pyplot(fig)
plt.close(fig)

st.subheader("Comparaison des montants selon le type de transaction")

montants_normaux = df.loc[
    df["is_fraud"] == 0, "amount_usd"
]

montants_fraudes = df.loc[
    df["is_fraud"] == 1, "amount_usd"
]

fig2, ax2 = plt.subplots()

ax2.boxplot(
    [montants_normaux, montants_fraudes],
    tick_labels=["Normales", "Frauduleuses"]
)

ax2.set_ylabel("Montant ($)")
ax2.set_title("Montants par catégorie de transaction")

st.pyplot(fig2)
plt.close(fig2)

# =====================================
# PARTIE 1 : ANALYSE COMPARATIVE
# =====================================

st.divider()
st.header("Analyse comparative des fraudes")

variable = st.selectbox(
    "Comparer les fraudes selon :",
    ["merchant_category", "channel", "auth_method", "card_type"]
)

comparaison = (
    df.groupby(variable)["is_fraud"]
    .agg(["count", "sum", "mean"])
    .reset_index()
)

comparaison.columns = [
    "Catégorie",
    "Transactions",
    "Fraudes",
    "Taux de fraude"
]

comparaison["Taux de fraude"] *= 100

st.dataframe(comparaison, hide_index=True)

st.bar_chart(
    comparaison.set_index("Catégorie")["Taux de fraude"]
)

# =====================================
# PARTIE 2 : MODÈLE DE CLASSIFICATION
# =====================================

st.divider()
st.header("Modèle de détection de fraude")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Séparer les variables explicatives de la cible
X = df.drop(columns=["is_fraud", "transaction_id"])
y = df["is_fraud"]

# Colonnes numériques et catégorielles
numeriques = X.select_dtypes(include=["number"]).columns.tolist()
categorielles = X.columns.difference(numeriques).tolist()

# Division entraînement/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Prétraitement appris uniquement sur l'entraînement
preparation = ColumnTransformer([
    ("num", Pipeline([
        ("imputation", SimpleImputer(strategy="median")),
        ("standardisation", StandardScaler())
    ]), numeriques),
    ("cat", Pipeline([
        ("imputation", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encodage", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ]), categorielles)
])

# Construction du modèle
modele = Pipeline([
    ("preparation", preparation),
    ("classification", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

@st.cache_resource
def entrainer_modele(_X_train, _y_train):
    modele.fit(_X_train, _y_train)
    return modele

# Entraînement
modele_entraine = entrainer_modele(X_train, y_train)

# Prédictions sur les données de test
probabilites = modele_entraine.predict_proba(X_test)[:, 1]

seuil = st.slider(
    "Seuil de classification",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05
)

predictions = (probabilites >= seuil).astype(int)

st.success("Modèle entraîné et prêt à être évalué.")

# =====================================
# PARTIE 3 : ÉVALUATION DU MODÈLE
# =====================================

st.divider()
st.header("Évaluation du modèle")

precision = precision_score(
    y_test, predictions, zero_division=0
)
rappel = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)

col1, col2, col3 = st.columns(3)

col1.metric("Précision", f"{precision:.2%}")
col2.metric("Rappel", f"{rappel:.2%}")
col3.metric("Score F1", f"{f1:.2%}")

st.metric(
    "Average Precision (PR-AUC)",
    f"{average_precision_score(y_test, probabilites):.3f}"
)

st.subheader("Matrice de confusion")

matrice = confusion_matrix(
    y_test, predictions, labels=[0, 1]
)

matrice_df = pd.DataFrame(
    matrice,
    index=["Réel : Normal", "Réel : Fraude"],
    columns=["Prédit : Normal", "Prédit : Fraude"]
)

st.dataframe(matrice_df)

st.subheader("Rapport de classification")

rapport = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Normal", "Fraude"],
    zero_division=0,
    output_dict=True
)

st.dataframe(
    pd.DataFrame(rapport).transpose()
)

st.info(
    "Ce modèle est une démonstration statistique. "
    "Il ne constitue pas un système bancaire validé."
)
