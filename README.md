
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

### 5. Recherche d'opération par code unique 
* Recherche directe d'une transaction à partir de son **code/identifiant unique** (`transaction_id`) pour afficher ses détails, son score de risque, son statut et les raisons associées.

---
## Installation

Installer les dépendences : `pip install -r requirements.txt`

---
## Lancement

Placez vous dans le dossier `Projet_Session_INF5103` puis lancer : `Streamlit run app.py`

---
## Données, Méthodologie et Choix du Modèle

* ** Le jeu de données utilisé est `credit_card_fraud_2026.csv`. Il provient de Kaggle. Ce jeu de donnée est constitué de : ** **20 000 transactions** et **26 colonnes** donc `transaction_id` (exclu dès la séparation pour éviter toute fuite),  `is_fraud` qui est la variable cible et **24 variables explicatives**. les variables explicatives sont repartie comme suit:  13 numériques, 5 catégorielles et 6 indicateurs booléens. Il est à noté que **Aucune valeur manquante** et **aucun doublon** n'ont été relevés.


* **L'échantillonnage stratifié suivant a été utilisé:**
  * **Entraînement (64 %) :** 12 800 transactions (217 fraudes — 1,70 %)
  * **Validation (16 %) :** 3 200 transactions (54 fraudes — 1,69 %)
  * **Test (20 %) :** 4 000 transactions (68 fraudes — 1,70 %)

* **Nous avons effectué le prétraitement suivant  :**
  * Variables numériques : `StandardScaler` .
  * Variables catégorielles : `OneHotEncoder`.
  * Variables booléennes : conservées en `0/1`.
  
* **Pour la gestion du déséquilibre, nous avons fait:** une omparaison en validation croisée avec 5 plis sur la régression logistique entre *(a) Rien*, *(b) `class_weight='balanced'`*, *(c) `SMOTE`* et *(d) `SMOTE + class_weight`*. L'option **(a) Rien** a été retenue.
* **Comparaison des modèles (CV 5 plis) & Choix de la Régression Logistique :**
  * **La Régression Logistique (`max_iter=2000`) a été retenue comme model final.** 
* **Choix du seuil  :** Il a été déterminé sur l'ensemble de validation selon la règle préétablie visant un rappel cible $\ge 0,80$.

---

## Résultats sur le Jeu de Test et Stabilité

L'évaluation finale fait sur le jeu de test (**4 000 transactions**, **68 fraudes**, seuil **0,0164**) a donné les résultats:

* **PR-AUC :** **0,4598** 
* **ROC-AUC :** **0,9516**
* **F1-Score :** **0,1971**
* **Précision :** **11,07 %** 
* **Rappel (Recall) :** **89,71 %** 


### Matrice de confusion sur le jeu de test

| Réel \ Prédit | Légitime | Fraude |
| :--- | :---: | :---: |
| **Légitime** | **3 442 (VN)** | **490 (FP)** |
| **Fraude** | **7 (FN)** | **61 (VP)** |

---

## Explicabilité et Variables Clés

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
  
