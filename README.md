
# FraudScope 

## Presentation

**FraudScope** est une application python d'apprentissage automatique et d'aide à la décision conçue pour détecter, expliquer et superviser les transactions financières à risque. Cette application a été réalisée dans le cadre du cours sur les concepts statistiques pour la science de donnée(INF5103) à l'Université du Québec en Outaouais. Elle combine un pipeline de classification fondé sur la **Régression Logistique** (entraîné sur **20 000 transactions** avec **1,70 % de fraudes**) avec une interface opérationnelle multi-vues destinée aux analystes de fraude.

---
## Fonctionnalités de l'application

L'interface interactive (**Streamlit** dans `app.py`) s'articule autour de **5 modules dédiés** pour superviser les transactions :

### 1. Tableau de bord d'analyse des opérations 
Il permet de :
* Simuler une transaction en temps réel.
* Visualiser le résultat de la simulation d'une transaction.
* Visualiser les indicateurs de performance du modèle.

### 2. Suivi des opérations courantes 
Cet onglet permet de :
* Consulter et filtrer des opérations exécutées normalement sans alerte de sécurité.
* Suivre le flux nominal des transactions.

### 3. Suivi des opérations douteuses

* Regroupe les transactions suspectes nécessitant une vérification humaine.
* **Raisons de classement :** Affiche pour chaque opération les **principales raisons** expliquant son classement comme douteuse.
* **Validation / Refus :** Permet à l'analyste de **valider** ou de **refuser** chaque opération, avec mise à jour dans `historique.csv`.

### 4. Registre des opérations frauduleuses 
* Centralise les transactions classées comme frauduleuses afin de faciliter le suivi et l'audit des cas critiques.

### 5. Recherche d'opération par code unique (*Traçabilité & Audit*)
* Recherche directe d'une transaction à partir de son **code/identifiant unique** (`transaction_id`) pour afficher ses détails, son score de risque, son statut et les raisons associées.

---
## Installation

Installer les dépendences : `pip install -r requirements.txt`

---
## Lancement

Placez vous dans le dossier `Projet_Session_INF5103` puis lancer : `Streamlit run app.py`

---
## Données, Méthodologie & Choix du Modèle

* **Jeu de données (`credit_card_fraud_2026.csv`, Kaggle) :** **20 000 transactions** et **26 colonnes** (`transaction_id` exclu dès la séparation pour éviter toute fuite, `is_fraud` en cible, et **24 variables explicatives** : 13 numériques, 5 catégorielles et 6 indicateurs booléens). **Aucune valeur manquante** et **aucun doublon** n'ont été relevés.
* **Échantillonnage stratifié (avant tout prétraitement) :**
  * **Entraînement (64 %) :** 12 800 transactions (217 fraudes — 1,70 %)
  * **Validation (16 %) :** 3 200 transactions (54 fraudes — 1,69 %)
  * **Test (20 %) :** 4 000 transactions (68 fraudes — 1,70 %)
* **Prétraitement (`ColumnTransformer` dans un `Pipeline` Scikit-learn) :**
  * Variables numériques : `StandardScaler` retenu (PR-AUC de **0,4283** vs **0,4259** avec `RobustScaler`).
  * Variables catégorielles : `OneHotEncoder(handle_unknown="ignore", min_frequency=20)`.
  * Variables booléennes : conservées en `0/1`.
  * Variables dérivées : 10 variables testées (ratio montant/solde, compteur de signaux de risque, encodages cycliques heure/jour), non retenues faute de gain significatif.
* **Gestion du déséquilibre :** Comparaison en validation croisée (5 plis) sur la régression logistique entre *(a) Rien*, *(b) `class_weight='balanced'`*, *(c) `SMOTE`* et *(d) `SMOTE + class_weight`*. L'option **(a) Rien** a été retenue.
* **Comparaison des modèles (CV 5 plis) & Choix de la Régression Logistique :**
  * **Régression Logistique (`max_iter=2000`) :** **PR-AUC = 0,4283** (lift de **25,3×** vs hasard à 0,0170), **ROC-AUC = 0,9453**, **F1 (0,5) = 0,3401**, **MCC = 0,3873**, **Bal. Acc = 0,6141**, temps **0,08 s**.
  * Comparée à *Gradient Boosting* (PR-AUC **0,3512**), *Random Forest régularisée* (PR-AUC **0,3016**), *HistGradientBoosting* (PR-AUC **0,2359**) et *Dummy Classifier* (PR-AUC **0,0170**), la **Régression Logistique** a été retenue comme modèle final.
* **Choix du seuil (`0,0164`) :** Déterminé sur l'ensemble de validation selon la règle préétablie visant un rappel cible $\ge 0,80$ lorsque le rappel au seuil F1-max est inférieur à 70 % (sur validation : rappel de **81,48 %**, précision de **9,89 %**, 445 alertes dont 44 fraudes détectées et 10 fraudes manquées).

---

## Résultats sur le Jeu de Test & Stabilité

L'évaluation finale se fait sur le jeu de test (**4 000 transactions**, **68 fraudes**, seuil **0,0164**):

* **PR-AUC :** **0,4598** (vs **0,3395** en validation)
* **ROC-AUC :** **0,9516**
* **F1-Score :** **0,1971**
* **Précision :** **11,07 %** (vs **9,89 %** en validation)
* **Rappel (Recall) :** **89,71 %** (vs **81,48 %** en validation)
* **MCC :** **0,2897** | **Exactitude équilibrée :** **88,62 %**

### Matrice de confusion sur le jeu de test

| Réel \ Prédit | Légitime | Fraude |
| :--- | :---: | :---: |
| **Légitime** | **3 442 (VN)** | **490 (FP)** |
| **Fraude** | **7 (FN)** | **61 (VP)** |

---

## Explicabilité & Variables Clés

L'interprétabilité du modèle a été analysée via l'**importance par permutation** (10 répétitions, mesurant la baisse moyenne de PR-AUC). Les variables les plus influentes sur le jeu de test incluent notamment `cvv_retry_count`, `merchant_risk_score`, `velocity_score`, `billing_shipping_mismatch`, `ip_country_mismatch` et `auth_method`.

---

## Architecture du Projet

La structure de l'application est la suivante:

```text
FraudScope/
├── Projet_Session_INF5103/
│   ├── credit/                 
│   ├── decision/               
│   ├── figures/               
│   ├── models/                 
│   ├── _suivi_common.py       
│   ├── app.py                 
│   ├── historique.csv         
│   ├── Notebook.ipynb          
│   └── Notebook.py                
├── README.md                  
└── requirements.txt            