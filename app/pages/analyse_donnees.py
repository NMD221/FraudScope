import streamlit as st
import os

st.title("Analyse Exploratoire des Données")
st.markdown("---")

#On remonte de 2 niveaux puis entre dans 'utils/image_analyse'
dossier_images = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image_analyse"))

if os.path.exists(dossier_images):
    fichiers_images = [f for f in os.listdir(dossier_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if fichiers_images:
        fichiers_images.sort()

        for fichier in fichiers_images:
            chemin_complet = os.path.join(dossier_images, fichier)
            #nom_propre = fichier.split('.')[0].replace('_', ' ').capitalize()

            if fichier.startswith('01'):
                st.subheader("Analyse de la distribution des fraudes")
                st.markdown(
                    "Le diagramme circulaire intitulé « Répartition des transactions » met en évidence "
                    "un déséquilibre prononcé entre les classes, révélant que les transactions légitimes "
                    "constituent l'essentiel du volume avec 98,31 % contre seulement 1,70 % pour les transactions "
                    "frauduleuses, ce qui conduit au constat majeur qu'un modèle naïf maximisant l'accuracy "
                    "serait inefficace et qu'il est impératif d'utiliser des métriques et des algorithmes adaptés à ce déséquilibre."
                )

            elif fichier.startswith('02'):
                st.subheader("Analyse de la distribution des variables numériques par rapport à la variable d'interêt")
                st.markdown(
                    "L'analyse comparative des histogrammes entre transactions légitimes (en vert) et frauduleuses (en rouge) "
                    "met en lumière des points saillants sur plusieurs variables clés, révélant que des indicateurs tels que "
                    "le score de vélocité (`velocity_score`), le risque marchand (`merchant_risk_score`) et le nombre de "
                    "tentatives CVV (`cvv_retry_count`) présentent un décalage net des cas de fraude vers des valeurs plus élevées, "
                    "ce qui conduit au constat majeur que ces comportements anormaux constituent des signaux discriminants forts "
                    "et des variables prédictives essentielles pour alimenter et optimiser la performance des modèles de détection."
                )
            elif fichier.startswith('03'):
                st.subheader("Analyse de la distribution des variables numériques")
                st.markdown(
                    "L'analyse des diagrammes en boîtes dédiée à la détection visuelle des valeurs aberrantes "
                    "met en lumière des points saillants sur plusieurs variables continues, révélant que des attributs tels que "
                    "le montant des transactions (`amount_usd`), la distance par rapport au domicile (`distance_from_home_km`) "
                    "et le solde du compte (`account_balance_usd`) présentent une forte concentration de valeurs extrêmes au-delà "
                    "de la moustache supérieure, ce qui conduit au constat majeur que ces asymétries et ces comportements "
                    "atypiques nécessitent un traitement rigoureux afin d'éviter qu'ils ne biaisent l'apprentissage des modèles "
                    "de prédiction de fraude."
                )
            elif fichier.startswith('04'):
                st.subheader("Analyse de la distribution des variables booléennes par rapport à la variable d'interêt")
                st.markdown(
                    "L'analyse des indicateurs booléens par rapport au taux de fraude "
                    "met en lumière des points saillants sur l'ensemble des variables binaires, "
                    "révélant que des caractéristiques telles que l'incompatibilité d'adresse IP et de pays (`ip_country_mismatch`), "
                    "la non-correspondance entre facturation et livraison (`billing_shipping_mismatch`) ainsi que les tentatives d'escroquerie par IA (`is_ai_generated_scam_attempt`) "
                    "affichent un taux de fraude nettement supérieur à la moyenne globale lorsqu'elles prennent la valeur Oui (1)[cite: 4], "
                    "ce qui conduit au constat majeur que ces indicateurs contextuels constituent des facteurs de risque discriminants et de puissants leviers "
                    "pour alimenter les règles de décision et les modèles prédictifs."
                )
            elif fichier.startswith('05'):
                st.subheader("Analyse de la distribution des variables catégorielles par rapport à la variable d'intêret")
                st.markdown(
                    "L'analyse des variables catégorielles par rapport au taux de fraude "
                    "met en lumière des points saillants sur l'ensemble des attributs qualitatifs, "
                    "révélant que des catégories spécifiques telles que les plateformes de change crypto (`Crypto Exchange`) "
                    "pour les marchands, l'absence d'authentification (`No Authentication`) pour la méthode d'authentification "
                    "ou encore le canal en ligne (`Online`) affichent un taux de fraude nettement supérieur à la moyenne globale, "
                    "ce qui conduit au constat majeur que ces caractéristiques contextuelles constituent des facteurs de risque discriminants "
                    "et des leviers prédictifs incontournables pour optimiser la détection des fraudes."
                )
            elif fichier.startswith('06'):
                st.subheader("Matrice de corrélation")
                st.markdown(
                    "L'analyse de la matrice de corrélation des variables numériques met en lumière des points saillants "
                    "sur les relations linéaires entre les attributs, révélant que si la plupart des corrélations directes avec "
                    "la cible (`is_fraud`) restent faibles, des indicateurs tels que le nombre de tentatives CVV (`cvv_retry_count`), "
                    "le score de risque marchand (`merchant_risk_score`) et le score de vélocité (`velocity_score`) "
                    "affichent les liaisons les plus perceptibles, aux côtés d'une colinéarité marquée entre le nombre de transactions sur 24h et la vélocité (0,62), "
                    "ce qui conduit au constat majeur que les associations linéaires simples sont insuffisantes et qu'il est indispensable de s'appuyer "
                    "sur des modèles non linéaires robustes (tels que XGBoost) pour exploiter pleinement l'ensemble des interactions prédictives."
                )

            st.image(chemin_complet, use_container_width=True)
            st.markdown("---")


    else:
        st.warning(f"Aucune image trouvée dans `{dossier_images}`.")
else:
    st.error(f"Le dossier `{dossier_images}` est introuvable. Vérifiez son emplacement.")