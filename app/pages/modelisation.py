#Modélisation
import streamlit as st
import pandas as pd

import streamlit as st
import os

st.title("Analyse du Modèle")

#Tableau des métriques
st.subheader("Tableau synthétique des performances du modèle")
metrics_data = {
    "Seuil de Décision": [0.5],
    "ROC-AUC": [0.9373],
    "PR-AUC (Précision-Rappel)": [0.3658],
    "Rappel (Recall)": [0.85],
    "F1-Score": [0.17]
}
df_metrics = pd.DataFrame(metrics_data)
st.dataframe(df_metrics.set_index("Seuil de Décision"), use_container_width=True)
st.markdown("----")

st.subheader("Analyse des choix techniques")
st.markdown(
    """
    - **Pourquoi le PR-AUC plutôt que le ROC-AUC ?** Sur des données hautement déséquilibrées, le score PR-AUC est beaucoup plus strict et reflète fidèlement la capacité du modèle à détecter la fraude sans générer trop de fausses alertes.
    - **Optimisation des seuils :** Plutôt que de fixer le seuil de décision à 0.5 par défaut, un ajustement du seuil (ex: 0.3) permet d'optimiser le compromis entre le coût bloquant des faux positifs et le risque financier des faux négatifs.
    """
)
st.markdown("----")

st.subheader("Visualisations graphiques des indicateurs de performance du modèle.")
#On remonte de 2 niveaux puis on entre dans 'utils/image_modele'
dossier_images = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "utils", "image_modele"))

if os.path.exists(dossier_images):
    fichiers_images = [f for f in os.listdir(dossier_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if fichiers_images:
        fichiers_images.sort()

        for fichier in fichiers_images:
            chemin_complet = os.path.join(dossier_images, fichier)
            #nom_propre = fichier.split('.')[0].replace('_', ' ').capitalize()
            if fichier.startswith('01'):
                st.subheader('Matrice de Confusion')
                st.markdown(
                    "L'analyse de la matrice de confusion du modèle XGBoost optimisé au seuil de 0,5 met en lumière des points saillants "
                    "sur la performance de classification. Il ressort que si le modèle identifie correctement la grande majorité des transactions légitimes (3404 vrais négatifs) "
                    "et détecte une part importante des fraudes réelles (58 vrais positifs), il génère également 528 faux positifs et laisse passer 10 faux négatifs. "
                    "Cela nous conduit au constat majeur que ce compromis entre faux positifs et faux négatifs nécessite une analyse fine du seuil de décision "
                    "afin d'ajuster le modèle aux contraintes opérationnelles du système de détection de fraude."
                )
            elif fichier.startswith('02'):
                st.subheader('Courbe de la Precision du Recall')
                st.markdown(
                    "L'analyse de la courbe Précision-Rappel du modèle, montre une précision moyenne (AP) de 0,37. "
                    "Toutefois, on observe une évolution contradictoire entre la précision et le rappel. En effet, le modèle parvient à maintenir une précision "
                    "élevée pour de faibles niveaux de rappel avant de décroître progressivement à mesure que l'on cherche à capturer une plus grande proportion de fraudes. "
                    "Ce qui conduit au constat majeur que cette courbe est essentielle pour ajuster finement le seuil de décision "
                    "en fonction de la tolérance opérationnelle de l'organisation face aux faux positifs et aux faux négatifs."
                )
            elif fichier.startswith('03'):
                st.subheader('Courbe ROC')
                st.markdown(
                    "L'analyse de la courbe ROC (Receiver Operating Characteristic) du modèle montre une aire sous la courbe (AUC) élevée de 0,94. "
                    "Ce taux révèle que ce le modèle maintient un taux de vrais positifs très élevé tout en conservant un taux de faux positifs particulièrement bas sur une large plage de seuils, "
                    "Cela  nous permet de constater que le modèle présente une excellente performance intrinsèque pour séparer les transactions légitimes des fraudes, "
                    "bien que son exploitation concrète doit être complétée par l'analyse de précision-rappel en raison du déséquilibre des classes."
                )

            st.image(chemin_complet, use_container_width=True)
            st.markdown("---")
    else:
        st.warning(f"Aucune image trouvée dans `{dossier_images}`.")
else:
    st.error(f"Le dossier `{dossier_images}` est introuvable. Vérifiez son emplacement.")




