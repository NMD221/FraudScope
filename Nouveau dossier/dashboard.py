# dashboard.py
"""
Dashboard analyste fraude — Streamlit.
Lance avec : streamlit run dashboard.py

Icônes : Font Awesome 6 (CDN) + Material Icons (boutons Streamlit).
Audit  : les décisions de l'analyste sont enregistrées dans decisions_log.jsonl
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from decision_engine import DecisionEngine


# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Fraude — Poste analyste",
    page_icon=":material/security:",
    layout="wide",
)

# Charger Font Awesome (une seule fois)
st.markdown(
    """
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    """,
    unsafe_allow_html=True,
)


def icon(name: str, color: str = "#2c3e50", size: str = "1em") -> str:
    """Retourne un <i> Font Awesome prêt à insérer dans du HTML."""
    return (
        f'<i class="fa-solid fa-{name}" '
        f'style="color:{color};font-size:{size};"></i>'
    )


@st.cache_resource
def load_engine():
    return DecisionEngine("models/final_model.joblib")


@st.cache_data
def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Score n'importe quel DataFrame de transactions (avec ou sans is_fraud)."""
    engine = load_engine()
    return engine.decide_batch(df)


def get_demo_data() -> pd.DataFrame:
    """Échantillon de démo du dataset du projet."""
    return pd.read_csv("credit/credit_card_fraud_2026.csv").sample(500, random_state=1)


# ============================================================
# AUDIT — enregistrement des décisions analyste
# ============================================================
LOG_FILE = Path("decisions_log.jsonl")


def log_decision(tid, action, score, source, analyste="demo"):
    """Enregistre une décision dans decisions_log.jsonl (une ligne JSON par décision)."""
    entry = {
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "transaction_id": tid,
        "action":         action,          # "confirme_fraude" / "faux_positif" / "escalade"
        "fraud_score":    float(score),
        "source_decision": source,
        "analyste_id":    analyste,
        "modele_version": "v1",
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_decisions_log() -> pd.DataFrame:
    """Recharge les décisions déjà enregistrées (ou DataFrame vide)."""
    if not LOG_FILE.exists():
        return pd.DataFrame(
            columns=[
                "timestamp", "transaction_id", "action",
                "fraud_score", "source_decision", "analyste_id", "modele_version",
            ]
        )
    return pd.read_json(LOG_FILE, lines=True)


engine = load_engine()


# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
    <h1>{icon("magnifying-glass", "#2c3e50", "0.9em")}
        Détection de fraude — Poste analyste</h1>
    """,
    unsafe_allow_html=True,
)
st.caption(
    f"Seuil REVIEW = {engine.seuil_review:.4f} · "
    f"Seuil BLOCK = {engine.SEUIL_BLOCK}"
)


# ============================================================
# SOURCE DE DONNÉES
# ============================================================
st.markdown(
    f"### {icon('folder-open', '#3498db')} Source de données",
    unsafe_allow_html=True,
)

col_demo, col_upload = st.columns([1, 1])

with col_demo:
    use_demo = st.checkbox(
        "Utiliser l'échantillon de démo (500 transactions)",
        value=True,
    )

with col_upload:
    uploaded_file = st.file_uploader(
        "Ou téléversez votre fichier CSV de transactions",
        type=["csv"],
        help=(
            "Le CSV doit contenir les mêmes colonnes que le dataset "
            "(sauf `is_fraud`, qui est le label, et `transaction_id`, optionnel)."
        ),
    )

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.markdown(
            f"{icon('circle-check', '#27ae60')} "
            f"Fichier chargé : **{len(df_input)} transactions**",
            unsafe_allow_html=True,
        )
        scored = score_dataframe(df_input)
    except Exception as e:
        st.markdown(
            f"{icon('circle-xmark', '#e74c3c')} "
            f"Erreur de lecture du fichier : {e}",
            unsafe_allow_html=True,
        )
        st.stop()
elif use_demo:
    df_input = get_demo_data()
    scored = score_dataframe(df_input)
else:
    st.info(
        "Veuillez téléverser un fichier CSV "
        "ou cocher « Utiliser l'échantillon de démo »."
    )
    st.stop()

has_label = "is_fraud" in scored.columns


# ============================================================
# SIDEBAR — FILTRES + EXEMPLE + AUDIT
# ============================================================
with st.sidebar:
    st.markdown(
        f"## {icon('sliders', '#2c3e50')} Filtres",
        unsafe_allow_html=True,
    )

    decisions_choisies = st.multiselect(
        "Décisions à afficher",
        options=["BLOCK", "REVIEW", "ALLOW"],
        default=["BLOCK", "REVIEW"],
    )

    score_min = st.slider(
        "Score minimum",
        min_value=0.0, max_value=1.0, value=0.0, step=0.01,
    )

    sources = st.multiselect(
        "Source de décision",
        options=["Modèle", "Règle"],
        default=["Modèle", "Règle"],
    )

    st.divider()
    st.markdown(
        f"**{icon('circle-info', '#3498db')} Architecture**",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"{icon('circle', '#e74c3c', '0.6em')} "
        f"BLOCK : modèle seul (précision calibrée)",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"{icon('circle', '#f39c12', '0.6em')} "
        f"REVIEW : modèle + 3 règles métier",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"{icon('circle', '#2ecc71', '0.6em')} "
        f"ALLOW : aucune action",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        f"### {icon('file-arrow-down', '#3498db')} Fichier d'exemple",
        unsafe_allow_html=True,
    )
    example = (
        get_demo_data()
        .drop(columns=["is_fraud"], errors="ignore")
        .head(50)
    )
    st.download_button(
        "Télécharger 50 transactions (sans label)",
        example.to_csv(index=False).encode("utf-8"),
        "exemple_transactions.csv",
        "text/csv",
        help="Vous pouvez ensuite re-téléverser ce fichier pour tester.",
    )

    # --- Audit ---
    st.divider()
    st.markdown(
        f"### {icon('clipboard-check', '#27ae60')} Journal d'audit",
        unsafe_allow_html=True,
    )
    log_df = load_decisions_log()
    st.markdown(
        f"{icon('database', '#6c757d')} {len(log_df)} décision(s) enregistrée(s)",
        unsafe_allow_html=True,
    )
    if len(log_df) > 0:
        st.download_button(
            "Télécharger decisions_log.jsonl",
            LOG_FILE.read_bytes(),
            "decisions_log.jsonl",
            "application/json",
            icon=":material/download:",
        )
        if st.button("Effacer le journal", icon=":material/delete:"):
            LOG_FILE.unlink()
            st.rerun()


# ============================================================
# FILTRAGE
# ============================================================
filtered = scored[
    scored["decision"].isin(decisions_choisies)
    & (scored["fraud_score"] >= score_min)
].copy()

if sources:
    if "Modèle" not in sources:
        filtered = filtered[~filtered["source_decision"].str.startswith("Modèle")]
    if "Règle" not in sources:
        filtered = filtered[~filtered["source_decision"].str.startswith("Règle")]

filtered = filtered.sort_values("fraud_score", ascending=False)


# ============================================================
# KPIs
# ============================================================
def kpi(col, icon_name, icon_color, label, value):
    col.markdown(
        f"""
        <div style="background:#f8f9fa;border-radius:8px;padding:12px;
                    border-left:4px solid {icon_color};">
            <div style="font-size:0.85em;color:#6c757d;">
                {icon(icon_name, icon_color)} {label}
            </div>
            <div style="font-size:1.8em;font-weight:bold;color:#2c3e50;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


col1, col2, col3, col4, col5 = st.columns(5)
kpi(col1, "database",            "#3498db", "Scorées", len(scored))
kpi(col2, "ban",                 "#e74c3c", "BLOCK",   (scored["decision"] == "BLOCK").sum())
kpi(col3, "triangle-exclamation","#f39c12", "REVIEW",  (scored["decision"] == "REVIEW").sum())
kpi(col4, "circle-check",        "#2ecc71", "ALLOW",   (scored["decision"] == "ALLOW").sum())
kpi(col5, "clipboard-list",      "#2c3e50", "Alertes", len(filtered))

st.divider()


# ============================================================
# FILE D'ALERTES
# ============================================================
st.markdown(
    f"## {icon('clipboard-list', '#2c3e50')} "
    f"File d'alertes — {len(filtered)} transaction(s)",
    unsafe_allow_html=True,
)

if len(filtered) == 0:
    st.info("Aucune transaction ne correspond aux filtres sélectionnés.")
else:
    MAX_DISPLAY = 50
    if len(filtered) > MAX_DISPLAY:
        st.warning(
            f"Affichage des {MAX_DISPLAY} transactions les plus risquées "
            f"sur {len(filtered)}. Affinez les filtres pour voir plus."
        )

    DECISION_ICONS = {
        "BLOCK":  ("ban",                  "#e74c3c"),
        "REVIEW": ("triangle-exclamation", "#f39c12"),
        "ALLOW":  ("circle-check",         "#2ecc71"),
    }

    for _, row in filtered.head(MAX_DISPLAY).iterrows():
        icon_name, icon_color = DECISION_ICONS[row["decision"]]

        if "transaction_id" in row.index:
            tid_display = f"#{int(row['transaction_id'])}"
            tid_key = int(row["transaction_id"])
        else:
            tid_display = f"txn_{_}"
            tid_key = str(_)

        label = (
            f"{row['decision']} · {tid_display} · "
            f"{row['amount_usd']:.2f} $ · "
            f"score {row['fraud_score']:.3f}"
        )

        with st.expander(label):
            col_left, col_right = st.columns([2, 1])

            with col_left:
                src = row["source_decision"]
                src_icon = "gear" if src.startswith("Modèle") else "scale-balanced"
                st.markdown(
                    f"**Source :** {icon(src_icon, '#3498db')} {src}",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"**{icon('triangle-exclamation', '#f39c12')} Facteurs de risque**",
                    unsafe_allow_html=True,
                )
                for raison in row["raisons"]:
                    st.markdown(
                        f"- {icon('circle-dot', '#e74c3c', '0.5em')} {raison}",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"**{icon('circle-info', '#3498db')} Contexte transaction**",
                    unsafe_allow_html=True,
                )
                for icon_ctx, label_ctx, value_ctx in [
                    ("store", "Marchand", row["merchant_category"]),
                    ("globe", "Canal",     row["channel"]),
                    ("key",   "Auth",      row["auth_method"]),
                    ("clock", "Heure",     f"{row['time_of_day_hour']}h"),
                ]:
                    st.markdown(
                        f"- {icon(icon_ctx, '#6c757d')} {label_ctx} : {value_ctx}",
                        unsafe_allow_html=True,
                    )

                if has_label:
                    if row["is_fraud"] == 1:
                        st.markdown(
                            f"**Label réel :** "
                            f"{icon('skull-crossbones', '#e74c3c')} FRAUDE",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"**Label réel :** "
                            f"{icon('shield-halved', '#2ecc71')} légitime",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        f"**Label réel :** "
                        f"{icon('question', '#95a5a6')} "
                        f"non disponible (données de production)",
                        unsafe_allow_html=True,
                    )

            with col_right:
                st.markdown(
                    f"**{icon('gavel', '#2c3e50')} Décision analyste**",
                    unsafe_allow_html=True,
                )

                # Boutons avec enregistrement d'audit
                if st.button(
                    "Confirmer fraude",
                    key=f"confirm_{tid_key}",
                    icon=":material/check_circle:",
                ):
                    log_decision(
                        tid_key, "confirme_fraude",
                        row["fraud_score"], row["source_decision"],
                    )
                    st.success(f"{tid_display} marquée fraude (enregistré)")

                if st.button(
                    "Faux positif",
                    key=f"fp_{tid_key}",
                    icon=":material/cancel:",
                ):
                    log_decision(
                        tid_key, "faux_positif",
                        row["fraud_score"], row["source_decision"],
                    )
                    st.info(f"{tid_display} marquée légitime (enregistré)")

                if st.button(
                    "Escalader",
                    key=f"esc_{tid_key}",
                    icon=":material/arrow_upward:",
                ):
                    log_decision(
                        tid_key, "escalade",
                        row["fraud_score"], row["source_decision"],
                    )
                    st.warning(f"{tid_display} escaladée (enregistré)")


# ============================================================
# EXPORT — file d'alertes + journal d'audit
# ============================================================
st.divider()
st.markdown(
    f"## {icon('file-csv', '#2c3e50')} Export des décisions",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs([
    ":material/list: File d'alertes",
    ":material/history: Journal d'audit",
])

# --- Onglet 1 : file d'alertes (ce que le modèle a produit) ---
with tab1:
    export_cols = [
        "amount_usd", "fraud_score", "decision",
        "source_decision", "raisons",
    ]
    if "transaction_id" in filtered.columns:
        export_cols.insert(0, "transaction_id")
    if has_label:
        export_cols.append("is_fraud")

    export_df = filtered[export_cols].copy()
    export_df["raisons"] = export_df["raisons"].apply(lambda r: " | ".join(r))
    export_df["timestamp_export"] = datetime.now().isoformat(timespec="seconds")

    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger la file d'alertes (CSV)",
        csv,
        "file_alertes.csv",
        "text/csv",
        icon=":material/download:",
    )

# --- Onglet 2 : journal d'audit (ce que l'analyste a décidé) ---
with tab2:
    log_df = load_decisions_log()
    if len(log_df) == 0:
        st.info("Aucune décision analyste enregistrée pour l'instant.")
    else:
        st.markdown(
            f"{icon('database', '#6c757d')} "
            f"**{len(log_df)} décision(s)** enregistrée(s)",
            unsafe_allow_html=True,
        )
        st.dataframe(log_df, use_container_width=True)

        # Résumé par action
        st.markdown("**Résumé par action :**")
        st.dataframe(
            log_df["action"].value_counts().rename("nombre").to_frame(),
            use_container_width=True,
        )

        # Export JSONL + CSV
        col_json, col_csv = st.columns(2)
        with col_json:
            st.download_button(
                "Télécharger decisions_log.jsonl",
                LOG_FILE.read_bytes(),
                "decisions_log.jsonl",
                "application/json",
                icon=":material/download:",
            )
        with col_csv:
            st.download_button(
                "Télécharger decisions_log.csv",
                log_df.to_csv(index=False).encode("utf-8"),
                "decisions_log.csv",
                "text/csv",
                icon=":material/download:",
            )