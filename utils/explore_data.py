#Analyse Exploratoire des Données
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_raw_data

#création du dossier "image_analyse"
dossier_sortie="image_analyse"
os.makedirs(dossier_sortie,exist_ok=True)

df = load_raw_data()
#Analysons la distribution de la variable d'interêt
fig, ax = plt.subplots(figsize=(6, 5))
df["is_fraud"].value_counts().plot(
    kind="pie",
    labels=["Légitime (98,3 %)", "Fraude (1,7 %)"],
    colors=["#2ecc71", "#e74c3c"],
    autopct="%1.2f%%",
    ax=ax,
)
ax.set_title("Répartition des transactions")
ax.set_ylabel("")
plt.tight_layout()
#Sauvegardons l'image du déséquilibre
chemin_image=os.path.join(dossier_sortie,"01_desequilibre.png")
plt.savefig(chemin_image, dpi=120, bbox_inches="tight")
#On libère la mémoire
plt.close()

#Analysons la distribution des variables numériques par rapport à la variable cible
# On récupère toutes les colonnes numériques sauf cible et id
num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c not in ["is_fraud", "transaction_id"]]
print(f"{len(num_cols)} variables numériques :", num_cols)

n = len(num_cols)
ncols = 4
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(
        data=df, x=col, hue="is_fraud",
        bins=40, stat="density", common_norm=False,
        palette={0: "#2ecc71", 1: "#e74c3c"},
        alpha=0.6, ax=axes[i], legend=(i == 0)
    )
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_xlabel("")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.suptitle("Histogrammes (seaborn) — Légitime vs Fraude", fontsize=15, y=1.02)
plt.tight_layout()
#Sauvegardons l'image sur la distribution des variables numériques
chemin_image=os.path.join(dossier_sortie,"02_histogrammes.png")
plt.savefig(chemin_image, dpi=120, bbox_inches="tight")
#On libère la mémoire
plt.close()

#On observe la repartition des variables numériques continues
# Boxplots des variables numériques continues
continuous_cols = [
    "amount_usd", "distance_from_home_km", "account_balance_usd",
    "velocity_score", "hours_since_last_txn", "merchant_risk_score"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(continuous_cols):
    sns.boxplot(y=df[col], ax=axes[i], color="#3498db")
    axes[i].set_title(col, fontsize=10)
    axes[i].set_ylabel("")

plt.suptitle("Détection visuelle des outliers", fontsize=14)
plt.tight_layout()
chemin_image=os.path.join(dossier_sortie,"03_outliers_boxplots.png")
plt.savefig(chemin_image, dpi=120, bbox_inches="tight")
#On libère la mémoire
plt.close()

#Explorons les variables booléens par rapport à la variable cible
bool_cols = df.select_dtypes(include="bool").columns.tolist()
print(f"{len(bool_cols)} booléens :", bool_cols)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(bool_cols):
    taux = df.groupby(col)["is_fraud"].mean() * 100
    effectif = df.groupby(col)["is_fraud"].count()
    axes[i].bar(["Non (0)", "Oui (1)"], taux.values, color=["#2ecc71", "#e74c3c"])
    axes[i].axhline(df["is_fraud"].mean() * 100, color="blue", ls="--", label="Taux global")
    axes[i].set_title(col, fontsize=11)
    axes[i].set_ylabel("Taux de fraude (%)")
    for j, v in enumerate(taux.values):
        axes[i].text(j, v + 0.2, f"{v:.1f}%\n(n={effectif.iloc[j]})", ha="center", fontsize=8)
    axes[i].legend(fontsize=8)

plt.suptitle("Indicateurs booléens", fontsize=15)
plt.tight_layout()
#Sauvegardons l'image sur la distribution des variables booléens
chemin_image=os.path.join(dossier_sortie,"04_booléen.png")
plt.savefig(chemin_image, dpi=120, bbox_inches="tight")
#On libère la mémoire
plt.close()

#Explorons les variables catégorielles par rapport à la variable d'interêt
cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
print(f"{len(cat_cols)} catégorielles :", cat_cols)

fig, axes = plt.subplots(1, 5, figsize=(22, 6))

for i, col in enumerate(cat_cols):
    taux = df.groupby(col)["is_fraud"].mean().sort_values() * 100
    effectif = df.groupby(col)["is_fraud"].count()
    colors = ["#e74c3c" if v > df["is_fraud"].mean()*100 else "#2ecc71" for v in taux.values]
    axes[i].barh(taux.index.astype(str), taux.values, color=colors)
    axes[i].axvline(df["is_fraud"].mean() * 100, color="blue", ls="--", label="Taux global")
    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel("Taux de fraude (%)")
    for j, (idx, v) in enumerate(taux.items()):
        axes[i].text(v + 0.05, j, f"{v:.1f}% (n={effectif[idx]})", va="center", fontsize=7)
    axes[i].legend(fontsize=7)

plt.suptitle("Variables catégorielles", fontsize=15)
plt.tight_layout()
#Sauvegardons l'image sur la distribution des variables categorielles
chemin_image=os.path.join(dossier_sortie,"05_categorielles.png")
plt.savefig(chemin_image, dpi=120, bbox_inches="tight")
#On libère la mémoire
plt.close()

#Matrice de corrélation des variables numériques
num_cols_all = df.select_dtypes(include=np.number).columns.tolist()
num_cols_all = [c for c in num_cols_all if c != "transaction_id"]

corr_matrix = df[num_cols_all].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, square=True, linewidths=0.5,
    cbar_kws={"shrink": 0.8}, annot_kws={"size": 7}
)
plt.title("Matrice de corrélation des variables numériques", fontsize=14)
plt.tight_layout()
#Sauvegardons l'image sur la matrice de corrélation
chemin_image=os.path.join(dossier_sortie,"06_correlation_matrix.png")
plt.savefig(chemin_image, dpi=120, bbox_inches="tight")
#On libère la mémoire
plt.close()