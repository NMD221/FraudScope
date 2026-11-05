
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

# ====================================================
# 1. CONFIGURATION ET CHARGEMENT
# ====================================================

st.set_page_config(
    page_title="FraudScope",
    page_icon="💳",
    layout="wide",
)

DOSSIER = Path(__file__).resolve().parent
FICHIER_CSV = DOSSIER / "credit_card_fraud_2026.csv"
FICHIER_MODELE = DOSSIER / "fraudscope_modele.joblib"


@st.cache_data(show_spinner=False)
def charger_donnees(chemin, modification):
    return pd.read_csv(chemin)


@st.cache_resource(show_spinner=False)
def charger_modele(chemin, modification):
    return joblib.load(chemin)


# Vérifier que les deux fichiers existent
if not FICHIER_CSV.exists():
    st.error("Fichier credit_card_fraud_2026.csv introuvable.")
    st.stop()

if not FICHIER_MODELE.exists():
    st.error("Fichier fraudscope_modele.joblib introuvable.")
    st.stop()


try:
    df = charger_donnees(
        str(FICHIER_CSV),
        FICHIER_CSV.stat().st_mtime,
    )

    artefact = charger_modele(
        str(FICHIER_MODELE),
        FICHIER_MODELE.stat().st_mtime,
    )

except Exception as erreur:
    st.error(f"Erreur de chargement : {erreur}")
    st.stop()


# Vérifier le contenu du modèle sauvegardé
cles_obligatoires = {"modele", "seuil", "variables", "methode"}

if not isinstance(artefact, dict):
    st.error("Le fichier du modèle n'a pas le format attendu.")
    st.stop()

if not cles_obligatoires.issubset(artefact):
    st.error("Certaines informations manquent dans le modèle.")
    st.stop()


modele = artefact["modele"]
seuil = float(artefact["seuil"])
variables = list(artefact["variables"])
methode = artefact["methode"]

if not 0 <= seuil <= 1:
    st.error("Le seuil sauvegardé est invalide.")
    st.stop()

if "is_fraud" not in df.columns:
    st.error("La colonne is_fraud manque dans le CSV.")
    st.stop()

if df.empty:
    st.error("Le fichier CSV ne contient aucune transaction.")
    st.stop()

manquantes = [v for v in variables if v not in df.columns]

if manquantes:
    st.error(f"Variables absentes du CSV : {manquantes}")
    st.stop()

if not df["is_fraud"].dropna().isin([0, 1]).all():
    st.error("La colonne is_fraud doit contenir 0 ou 1.")
    st.stop()

if df["is_fraud"].isna().any():
    st.error("La colonne is_fraud contient des valeurs manquantes.")
    st.stop()

if 1 not in list(modele.classes_):
    st.error("Le modèle ne contient pas la classe fraude (1).")
    st.stop()


# ====================================================
# 2. FONCTIONS DE PREDICTION
# ====================================================

def calculer_scores(donnees):
    """Calculer le score de fraude des transactions."""

    X = donnees[variables]

    probabilites = modele.predict_proba(X)

    # Trouver la colonne correspondant à is_fraud = 1
    position_fraude = list(modele.classes_).index(1)

    return probabilites[:, position_fraude]


def afficher_prediction(score, classe_reelle=None):
    """Afficher une prédiction et éventuellement la vérité."""

    prediction = 1 if score >= seuil else 0

    st.metric("Score estimé de fraude", f"{score:.2%}")

    st.progress(float(score))

    st.caption(f"Seuil de décision utilisé : {seuil:.0%}")

    if prediction == 1:
        st.error("Fraude suspectée par FraudScope.")
    else:
        st.success("Transaction classée comme normale.")

    # La classe réelle n'existe que pour les transactions
    # historiques dont l'étiquette est connue.
    if classe_reelle is not None:

        classe_reelle = int(classe_reelle)

        st.write(
            "**Classe réelle :**",
            "Frauduleuse" if classe_reelle == 1 else "Normale",
        )

        if prediction == classe_reelle:
            st.success("Bonne classification.")

        elif prediction == 0 and classe_reelle == 1:
            st.warning(
                "Faux négatif : une fraude réelle a été manquée."
            )

        elif prediction == 1 and classe_reelle == 0:
            st.warning(
                "Faux positif : une transaction normale a été "
                "signalée comme suspecte."
            )

    st.caption(
        "Le score du modèle est une estimation, "
        "pas une preuve de fraude."
    )


def valeur_reference(variable):
    """Valeur typique utilisée pour initialiser les champs."""

    serie = df[variable].dropna()

    if serie.empty:
        if pd.api.types.is_numeric_dtype(df[variable]):
            return 0
        return ""

    if pd.api.types.is_numeric_dtype(serie):

        mediane = serie.median()

        if pd.api.types.is_integer_dtype(serie):
            return int(round(float(mediane)))

        return float(mediane)

    return str(serie.mode().iloc[0])


def saisir_variable(variable, cle, valeur):
    """Créer un champ adapté au type de variable."""

    serie = df[variable]

    if pd.api.types.is_numeric_dtype(serie):

        if pd.api.types.is_integer_dtype(serie):
            return st.number_input(
                variable,
                value=int(valeur),
                step=1,
                key=cle,
            )

        return st.number_input(
            variable,
            value=float(valeur),
            format="%.4f",
            key=cle,
        )

    categories = sorted(
        serie.dropna().astype(str).unique().tolist()
    )

    if not categories:
        return st.text_input(
            variable,
            value=str(valeur),
            key=cle,
        )

    index = (
        categories.index(str(valeur))
        if str(valeur) in categories
        else 0
    )

    return st.selectbox(
        variable,
        categories,
        index=index,
        key=cle,
    )


def profil_reference():
    """Construire une transaction avec des valeurs typiques."""

    return {
        variable: valeur_reference(variable)
        for variable in variables
    }


# ====================================================
# 3. INTERFACE PRINCIPALE
# ====================================================

st.title("💳 FraudScope")
st.write("Analyse et détection des fraudes bancaires")

st.caption(
    "Projet pédagogique utilisant la régression logistique."
)

tab_dashboard, tab_evaluation, tab_analyse = st.tabs([
    "📊 Tableau de bord",
    "🧠 Évaluation du modèle",
    "🔎 Analyse des transactions",
])


# ====================================================
# 4. TABLEAU DE BORD
# ====================================================

with tab_dashboard:

    st.header("Tableau de bord statistique")

    total = len(df)
    nb_fraudes = int((df["is_fraud"] == 1).sum())
    nb_normales = int((df["is_fraud"] == 0).sum())
    taux = nb_fraudes / total * 100

    c1, c2, c3 = st.columns(3)

    c1.metric("Transactions", total)
    c2.metric("Fraudes réelles", nb_fraudes)
    c3.metric("Taux de fraude", f"{taux:.2f}%")

    st.subheader("Répartition des transactions")

    repartition = pd.Series({
        "Normales": nb_normales,
        "Frauduleuses": nb_fraudes,
    })

    st.bar_chart(repartition)

    st.divider()

    st.subheader("Explorer les transactions")

    filtre = st.selectbox(
        "Afficher",
        ["Toutes", "Normales", "Frauduleuses"],
        key="filtre_dashboard",
    )

    if filtre == "Normales":
        donnees = df[df["is_fraud"] == 0]

    elif filtre == "Frauduleuses":
        donnees = df[df["is_fraud"] == 1]

    else:
        donnees = df

    st.write(f"{len(donnees)} transactions trouvées")

    st.dataframe(
        donnees.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    if "amount_usd" in df.columns:

        st.header("Analyse des montants")

        montants = pd.to_numeric(
            df["amount_usd"],
            errors="coerce",
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Montant moyen",
            f"{montants.mean():.2f} $",
        )

        c2.metric(
            "Montant médian",
            f"{montants.median():.2f} $",
        )

        st.subheader("Distribution des montants")

        fig, ax = plt.subplots()

        ax.hist(
            montants.dropna(),
            bins=40,
            edgecolor="black",
        )

        ax.set_xlabel("Montant ($)")
        ax.set_ylabel("Transactions")
        ax.set_title("Distribution des montants")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Montants selon le type de transaction")

        normaux = montants[df["is_fraud"] == 0].dropna()
        fraudes = montants[df["is_fraud"] == 1].dropna()

        if not normaux.empty and not fraudes.empty:

            fig2, ax2 = plt.subplots()

            ax2.boxplot(
                [normaux, fraudes],
                tick_labels=["Normales", "Frauduleuses"],
            )

            ax2.set_ylabel("Montant ($)")
            ax2.set_title("Comparaison des montants")

            st.pyplot(fig2)
            plt.close(fig2)

    st.divider()

    st.header("Analyse comparative des fraudes")

    choix_variables = [
        nom for nom in [
            "merchant_category",
            "channel",
            "auth_method",
            "card_type",
        ]
        if nom in df.columns
    ]

    if choix_variables:

        variable = st.selectbox(
            "Comparer selon",
            choix_variables,
            key="variable_comparaison",
        )

        comparaison = (
            df.groupby(variable, dropna=False)["is_fraud"]
            .agg(["count", "sum", "mean"])
            .reset_index()
        )

        comparaison.columns = [
            "Catégorie",
            "Transactions",
            "Fraudes",
            "Taux de fraude",
        ]

        comparaison["Taux de fraude"] *= 100

        st.dataframe(
            comparaison,
            hide_index=True,
            use_container_width=True,
        )

        st.bar_chart(
            comparaison.set_index("Catégorie")["Taux de fraude"]
        )

    else:
        st.info("Aucune variable comparative disponible.")


# ====================================================
# 5. MODELE ET EVALUATION
# ====================================================

with tab_evaluation:

    st.header("Modèle de classification")

    st.success("Modèle sauvegardé chargé avec succès.")

    st.write("**Algorithme :** Régression logistique")
    st.write(f"**Stratégie :** {methode}")
    st.write(f"**Nombre de variables :** {len(variables)}")
    st.write(f"**Seuil sélectionné :** {seuil:.2f}")

    st.info(
        "Le modèle a déjà été entraîné dans le notebook. "
        "Streamlit le charge sans le réentraîner."
    )

    st.divider()

    st.subheader("Diagnostic indicatif du modèle")

    st.warning(
        "Ce diagnostic utilise l'ancien découpage de test "
        "du projet. Ces données ont déjà été examinées "
        "pendant la modélisation : les résultats ne "
        "constituent pas une évaluation indépendante."
    )

    try:

        X = df[variables]
        y = df["is_fraud"]

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        scores_test = calculer_scores(X_test)

        predictions_test = (
            scores_test >= seuil
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions_test,
            zero_division=0,
        )

        rappel = recall_score(
            y_test,
            predictions_test,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions_test,
            zero_division=0,
        )

        average_precision = average_precision_score(
            y_test,
            scores_test,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Précision", f"{precision:.2%}")
        c2.metric("Rappel", f"{rappel:.2%}")
        c3.metric("F1-score", f"{f1:.2%}")

        st.metric(
            "Average Precision (PR-AUC)",
            f"{average_precision:.3f}",
        )

        st.subheader("Matrice de confusion")

        matrice = confusion_matrix(
            y_test,
            predictions_test,
            labels=[0, 1],
        )

        matrice_df = pd.DataFrame(
            matrice,
            index=["Réel : Normal", "Réel : Fraude"],
            columns=["Prédit : Normal", "Prédit : Fraude"],
        )

        st.dataframe(matrice_df)

        st.subheader("Rapport de classification")

        rapport = classification_report(
            y_test,
            predictions_test,
            labels=[0, 1],
            target_names=["Normal", "Fraude"],
            zero_division=0,
            output_dict=True,
        )

        st.dataframe(
            pd.DataFrame(rapport).transpose()
        )

    except Exception as erreur:
        st.error(f"Diagnostic impossible : {erreur}")

    st.caption(
        "FraudScope est un prototype pédagogique, "
        "pas un système bancaire validé."
    )


# ====================================================
# 6. ANALYSE DES TRANSACTIONS
# ====================================================

with tab_analyse:

    st.header("Analyser les transactions")

    st.write(
        "Choisissez comment fournir les informations "
        "à notre modèle de détection."
    )

    (
        onglet_existant,
        onglet_rapide,
        onglet_complet,
        onglet_csv,
    ) = st.tabs([
        "Transaction existante",
        "Analyse rapide",
        "Analyse complète",
        "Importation CSV",
    ])

    # --------------------------------------------
    # A. TESTER UNE TRANSACTION HISTORIQUE
    # --------------------------------------------

    with onglet_existant:

        st.subheader("Tester une transaction connue")

        numero = st.number_input(
            "Numéro de ligne",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1,
            key="numero_existant",
        )

        transaction = df.iloc[[int(numero)]]

        st.dataframe(
            transaction,
            hide_index=True,
            use_container_width=True,
        )

        if st.button(
            "Analyser cette transaction",
            key="bouton_existant",
        ):

            try:

                score = float(
                    calculer_scores(transaction)[0]
                )

                vraie_classe = int(
                    transaction["is_fraud"].iloc[0]
                )

                afficher_prediction(
                    score,
                    classe_reelle=vraie_classe,
                )

            except Exception as erreur:
                st.error(f"Erreur de prédiction : {erreur}")

        st.caption(
            "La classe réelle est connue ici car il "
            "s'agit d'une transaction historique."
        )

    # --------------------------------------------
    # B. ANALYSE RAPIDE
    # --------------------------------------------

    with onglet_rapide:

        st.subheader("Analyse rapide")

        st.warning(
            "Mode de simulation : seules quelques "
            "variables sont saisies. Les autres "
            "utilisent des valeurs médianes ou "
            "fréquentes de notre base. Le résultat "
            "ne doit pas être utilisé pour prendre "
            "une décision bancaire réelle."
        )

        champs_rapides = [
            nom for nom in [
                "amount_usd",
                "merchant_category",
                "channel",
                "auth_method",
                "card_type",
            ]
            if nom in variables
        ]

        valeurs_rapides = profil_reference()

        with st.form("formulaire_rapide"):

            colonnes = st.columns(2)

            for i, variable in enumerate(champs_rapides):

                with colonnes[i % 2]:

                    valeurs_rapides[variable] = saisir_variable(
                        variable,
                        f"rapide_{variable}",
                        valeurs_rapides[variable],
                    )

            bouton_rapide = st.form_submit_button(
                "Analyser rapidement",
                type="primary",
            )

        if bouton_rapide:

            try:

                nouvelle = pd.DataFrame(
                    [valeurs_rapides],
                    columns=variables,
                )

                score = float(
                    calculer_scores(nouvelle)[0]
                )

                afficher_prediction(score)

                avec_standards = {
                    v: valeurs_rapides[v]
                    for v in variables
                    if v not in champs_rapides
                }

                with st.expander(
                    "Voir les valeurs utilisées automatiquement"
                ):
                    st.dataframe(
                        pd.DataFrame([avec_standards]),
                        hide_index=True,
                    )

            except Exception as erreur:
                st.error(f"Erreur d'analyse rapide : {erreur}")

    # --------------------------------------------
    # C. ANALYSE COMPLETE DES 24 VARIABLES
    # --------------------------------------------

    with onglet_complet:

        st.subheader("Analyse complète")

        st.write(
            "Renseignez toutes les caractéristiques "
            "de la nouvelle transaction."
        )

        valeurs_completes = profil_reference()

        with st.form("formulaire_complet"):

            colonnes = st.columns(2)

            for i, variable in enumerate(variables):

                with colonnes[i % 2]:

                    valeurs_completes[variable] = saisir_variable(
                        variable,
                        f"complet_{variable}",
                        valeurs_completes[variable],
                    )

            bouton_complet = st.form_submit_button(
                "Analyser la nouvelle transaction",
                type="primary",
            )

        if bouton_complet:

            try:

                nouvelle = pd.DataFrame(
                    [valeurs_completes],
                    columns=variables,
                )

                score = float(
                    calculer_scores(nouvelle)[0]
                )

                afficher_prediction(score)

                with st.expander(
                    "Voir les données envoyées au modèle"
                ):
                    st.dataframe(
                        nouvelle,
                        hide_index=True,
                        use_container_width=True,
                    )

            except Exception as erreur:
                st.error(f"Erreur d'analyse : {erreur}")

    # --------------------------------------------
    # D. ANALYSE AUTOMATIQUE D'UN FICHIER CSV
    # --------------------------------------------

    with onglet_csv:

        st.subheader("Analyser plusieurs transactions")

        st.write(
            "Importez un CSV contenant les 24 variables "
            "utilisées par le modèle. La colonne is_fraud "
            "n'est pas nécessaire."
        )

        fichier_importe = st.file_uploader(
            "Choisir un fichier CSV",
            type=["csv"],
            key="fichier_transactions",
        )

        if fichier_importe is not None:

            try:
                fichier_importe.seek(0)
                donnees_csv = pd.read_csv(fichier_importe)

            except Exception as erreur:
                st.error(f"Lecture du CSV impossible : {erreur}")

            else:

                if donnees_csv.empty:
                    st.error("Le CSV est vide.")

                else:

                    colonnes_absentes = [
                        v for v in variables
                        if v not in donnees_csv.columns
                    ]

                    if colonnes_absentes:

                        st.error(
                            "Colonnes manquantes : "
                            + ", ".join(colonnes_absentes)
                        )

                    else:

                        X_csv = donnees_csv[variables].copy()

                        colonnes_invalides = []

                        for variable in variables:

                            if pd.api.types.is_numeric_dtype(
                                df[variable]
                            ):

                                original = X_csv[variable]

                                convertie = pd.to_numeric(
                                    original,
                                    errors="coerce",
                                )

                                invalides = (
                                    original.notna()
                                    & convertie.isna()
                                )

                                if invalides.any():
                                    colonnes_invalides.append(
                                        variable
                                    )

                                X_csv[variable] = convertie

                            else:

                               X_csv[variable] = X_csv[variable].where(
                                    X_csv[variable].notna(), None
                                    )

                        if colonnes_invalides:

                            st.error(
                                "Valeurs numériques invalides dans : "
                                + ", ".join(colonnes_invalides)
                            )

                        elif X_csv.isna().any().any():

                            st.error(
                                "Certaines caractéristiques sont "
                                "manquantes. Complétez le CSV "
                                "avant de l'analyser."
                            )

                        else:

                            try:

                                scores = calculer_scores(X_csv)

                                predictions = (
                                    scores >= seuil
                                ).astype(int)

                                resultats = donnees_csv.copy()

                                resultats["score_fraude"] = scores

                                resultats["classe_predite"] = [
                                    "Fraude suspectée"
                                    if p == 1 else "Normale"
                                    for p in predictions
                                ]

                                nb_suspectes = int(
                                    predictions.sum()
                                )

                                nb_normaux = (
                                    len(predictions) - nb_suspectes
                                )

                                st.success(
                                    "Analyse du fichier terminée !"
                                )

                                c1, c2, c3 = st.columns(3)

                                c1.metric(
                                    "Transactions analysées",
                                    len(resultats),
                                )

                                c2.metric(
                                    "Prédites normales",
                                    nb_normaux,
                                )

                                c3.metric(
                                    "Fraudes suspectées",
                                    nb_suspectes,
                                )

                                filtre_csv = st.selectbox(
                                    "Filtrer les résultats",
                                    [
                                        "Toutes",
                                        "Normale",
                                        "Fraude suspectée",
                                    ],
                                    key="filtre_csv",
                                )

                                if filtre_csv == "Toutes":
                                    affichage = resultats
                                else:
                                    affichage = resultats[
                                        resultats["classe_predite"]
                                        == filtre_csv
                                    ]

                                st.dataframe(
                                    affichage,
                                    hide_index=True,
                                    use_container_width=True,
                                )

                                contenu = resultats.to_csv(
                                    index=False
                                ).encode("utf-8-sig")

                                st.download_button(
                                    "Télécharger les résultats CSV",
                                    data=contenu,
                                    file_name="fraudscope_resultats.csv",
                                    mime="text/csv",
                                    key="telecharger_resultats",
                                )

                                st.caption(
                                    "Ces classifications sont "
                                    "des prédictions. Elles ne "
                                    "constituent pas des fraudes "
                                    "réellement confirmées."
                                )

                            except Exception as erreur:

                                st.error(
                                    f"Erreur de prédiction : {erreur}"
                                )

# ====================================================
# FIN DE FRAUDSCOPE
# ====================================================
