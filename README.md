
# FraudScope

## Présentation
FraudScope est une application Python permettant
d'explorer des transactions bancaires et d'étudier
les risques de fraude.

Projet réalisé dans le cadre du cours INF5103
à l'Université du Québec en Outaouais.

## Fonctionnalités
- Statistiques descriptives
- Visualisation des transactions
- Analyse comparative des fraudes
- Classification par régression logistique
- Évaluation des performances du modèle

## Installation
Installer les dépendances :

pip install -r requirements.txt

## Lancement
Placer le fichier credit_card_fraud_2026.csv
dans le dossier du projet, puis exécuter :

streamlit run app.py

## Données
Source : Kaggle, Credit Card Fraud Detection 2026.

## Statut
Prototype universitaire en développement.

## Partie 1 — Prétraitement et analyse exploratoire

L'analyse des données est disponible dans le fichier
`analyse_exploratoire.ipynb`.

### Données

Le jeu de données utilisé est :
Credit Card Fraud Detection 2026 (Kaggle).

Télécharger le fichier `credit_card_fraud_2026.csv`
et le placer dans le dossier principal du projet.

### Exécution

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Ouvrir ensuite `analyse_exploratoire.ipynb`
dans VS Code avec les extensions Python et Jupyter.

Exécuter toutes les cellules dans l'ordre.

### Fichiers générés

- `train_equilibre.csv` : données d'entraînement
  rééquilibrées par suréchantillonnage aléatoire.
- `test_original.csv` : données de test non rééquilibrées.

Ces fichiers sont générés localement et ne sont
pas inclus dans le dépôt GitHub.
