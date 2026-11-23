# FraudScope

FraudScope analyse les transactions par carte et permet de suivre leur examen. Cette livraison comprend l’application Python, le modèle entraîné et les fichiers de l’étude.

## Installation

Utiliser Python **3.14**. Extraire l’archive et ouvrir le dossier dans PyCharm ou dans un terminal.

Depuis le dossier contenant ce README, exécuter les commandes suivantes.

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -m streamlit run ".\Projet_Session_INF5103\app.py"
```

Ouvrir l’adresse affichée dans le terminal, généralement [http://localhost:8501](http://localhost:8501). Pour arrêter l’application, utiliser **Ctrl + C**.

## Utilisation

- **Analyser une opération** avec les 24 champs du formulaire. Les valeurs préremplies sont destinées aux essais.
- **Consulter les opérations** courantes, douteuses ou frauduleuses depuis la navigation.
- **Rechercher une opération** par son numéro court, son identifiant complet ou sa référence externe. Cliquer sur un résultat pour ouvrir le dossier.
- **Valider ou confirmer une fraude** en renseignant l’identifiant de l’agent et le motif. Le dossier conserve les décisions successives avec leur date et leur auteur.
- **Exporter la journée ou la semaine** depuis la catégorie souhaitée, avec un choix de date distinct pour chaque période. L’export comprend les colonnes distinctes de l’agent et du motif de validation.

Le montant supérieur au solde entraîne un blocage avec le motif **Solde insuffisant**. Les alertes du modèle sont classées comme douteuses en attendant leur examen. Après 7 jours sans décision, elles passent automatiquement dans les opérations frauduleuses avec le motif **Défaut de validation**. Cette action est datée et attribuée au système dans l’historique.

Les montants analysés sont exprimés en USD. Les facteurs affichés expliquent les principales variations du score selon la méthode fournie avec le modèle.

## Charger un fichier de transactions

Dans **Chargement CSV**, sélectionner un fichier, cliquer sur **Charger un CSV**, puis sur **Démarrer**. Le traitement peut être interrompu avec **Pause**.

Le CSV doit contenir les 24 champs du formulaire sous leurs noms techniques. Les champs booléens acceptent `true` / `false` ou `1` / `0`. Les colonnes `bank_reference`, `original_amount` et `original_currency` sont facultatives. Le montant et la devise d’origine doivent être fournis ensemble.

## Archives hebdomadaires

La semaine d’exportation va du dimanche au samedi. La période sélectionnée est affichée avant le téléchargement.

Chaque dimanche à 1 h, selon l’heure locale de la machine, l’application crée une archive ZIP de la semaine terminée, contenant un CSV pour chaque catégorie. L’application doit rester démarrée pour réaliser cette tâche à l’heure prévue. Si elle était arrêtée, les semaines manquantes sont archivées à la prochaine ouverture, notamment au premier démarrage du lundi.

Dans la recherche accessible par la loupe, ouvrir **Archives hebdomadaires** pour choisir une semaine et télécharger son ZIP ou le CSV d’une catégorie. Ces archives sont conservées dans la base de données et reflètent les décisions connues au moment de leur création.

## Conserver les opérations

L’application crée automatiquement la base `Projet_Session_INF5103/fraudscope.sqlite3`. Les données restent disponibles après redémarrage. Pour sauvegarder la base par copie, arrêter d’abord l’application.

Pour reprendre un ancien fichier `historique.csv`, conserver une sauvegarde du fichier, puis lancer la commande suivante depuis le dossier de ce README.

```powershell
& ".\.venv\Scripts\python.exe" ".\Projet_Session_INF5103\_suivi_common.py" --import-history "C:/chemin/historique.csv"
```

L’import reprend les informations présentes dans le CSV. Les décisions déjà écrasées dans l’ancien fichier ne peuvent pas être reconstituées.

## Fichiers du projet

| Fichier ou dossier | Usage |
|---|---|
| `Projet_Session_INF5103/app.py` | Formulaire et navigation |
| `Projet_Session_INF5103/_suivi_common.py` | Analyse, dossiers, historique et exports |
| `Projet_Session_INF5103/models/` | Modèle entraîné et résultats d’évaluation |
| `Projet_Session_INF5103/decision/` | Seuil de vérification |
| `Projet_Session_INF5103/Notebook.ipynb` et `Notebook.py` | Étude et entraînement |
| `Projet_Session_INF5103/figures/` | Graphiques de l’étude |
| `requirements.txt` | Dépendances Python |

