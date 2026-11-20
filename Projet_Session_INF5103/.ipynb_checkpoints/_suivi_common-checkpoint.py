import json
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

HISTORY_FILE = Path(__file__).resolve().parent / "historique.csv"

# --- CSS : en-tête collé, boutons et bloc facteurs (palette émeraude / sarcelle, compatible dark) ---
st.markdown("""
<style>
    div[data-testid="stDialog"] div[role="dialog"] > div > div > div[data-testid="stHorizontalBlock"]:nth-of-type(1) {
        position: sticky;
        top: 0;
        z-index: 999;
        padding-bottom: 10px;
        border-bottom: 2px solid #14b8a6;
    }
    div[data-testid="stDialog"] .stButton > button,
    div[data-testid="stDialog"] .stDownloadButton > button {
        border: 1px solid rgba(148, 163, 184, 0.45) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease;
    }
    div[data-testid="stDialog"] .stButton > button:hover,
    div[data-testid="stDialog"] .stDownloadButton > button:hover {
        background-color: rgba(20, 184, 166, 0.16) !important;
        color: #5eead4 !important;
        border-color: #14b8a6 !important;
    }
    div[data-testid="stDialog"] .stButton > button[kind="primary"] {
        background-color: #0f766e !important;
        color: #ffffff !important;
        border: 1px solid #14b8a6 !important;
    }
    div[data-testid="stDialog"] .stButton > button[kind="primary"]:hover {
        background-color: #115e59 !important;
        border-color: #5eead4 !important;
    }
    .factor-inline {
        margin-top: 8px;
        padding: 8px 10px;
        background-color: rgba(15, 23, 42, 0.06);
        border-left: 3px solid #0d9488;
        border-top: 1px solid rgba(148, 163, 184, 0.3);
        border-right: 1px solid rgba(148, 163, 184, 0.3);
        border-bottom: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 0 6px 6px 0;
        font-size: 12px;
        color: inherit;
    }
    .factor-inline-title {
        font-weight: 700;
        color: #14b8a6;
        margin-bottom: 4px;
    }
    .factor-inline ul {
        margin: 4px 0 6px 16px;
        padding: 0;
    }
    .factor-inline li {
        margin-bottom: 3px;
    }
    .factor-inline-cov {
        opacity: 0.85;
        font-size: 11.5px;
        font-weight: 500;
    }
    div[data-testid="stDialog"] [data-testid="stWidgetLabel"],
    div[data-testid="stDialog"] [data-testid="stWidgetLabel"] p {
        color: inherit !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)


def _format_factors(raw_factors):
    """Parse une liste JSON de facteurs si présente dans le CSV."""
    if not raw_factors:
        return None
    try:
        data = json.loads(raw_factors) if isinstance(raw_factors, str) else raw_factors
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def _render_observation_cell(row, is_open):
    """Affiche l'observation et, quand déplié, les champs ayant augmenté le score dans la colonne Observation."""
    t_obs = row.get("Observation", "")
    dec_init = row.get("Décision", "")
    prefix_label = (
        "Motif du blocage automatique — " if dec_init == "Blocked"
        else "Motif de mise en attente — " if dec_init == "Review" and t_obs
        else ""
    )
    obs_html = ""
    if t_obs:
        if prefix_label and not t_obs.startswith(prefix_label):
            obs_html = f"<div><strong>{prefix_label}</strong>{t_obs}</div>"
        else:
            obs_html = f"<div>{t_obs}</div>"

    if not is_open:
        return obs_html

    factors_data = _format_factors(row.get("Facteurs", ""))
    if factors_data and factors_data.get("items"):
        items_li = "".join(f"<li>{item}</li>" for item in factors_data["items"])
        cov_div = (
            f"<div class='factor-inline-cov'>{factors_data['coverage']}</div>"
            if factors_data.get("coverage") else ""
        )
    else:
        items_li = "<li>Détail des contributions non enregistré pour cette opération.</li>"
        cov_div = ""

    factors_html = (
        "<div class='factor-inline'>"
        "<div class='factor-inline-title'>Champs ayant augmenté le score</div>"
        f"<ul>{items_li}</ul>{cov_div}"
        "</div>"
    )
    return obs_html + factors_html


def render_transaction_row(row, category_key):
    """Affiche une opération sous forme de ligne avec boutons d'action directs."""
    show_obs = category_key != "Accepted"
    t_id = row.get("ID", "-")
    t_time = row.get("Date et heure", "-")
    try:
        montant_val = float(row.get("Montant (USD)", 0))
    except Exception:
        montant_val = 0.0
    t_amount = f"{montant_val:.2f}".replace(".", ",") + " $US"
    t_score = row.get("Score de fraude", "-")

    t_agent = row.get("Agent", "")
    t_motif = row.get("Motif", "")
    t_date_val = row.get("Date validation", "")

    dec_init = row.get("Décision", "-")
    dec_fin = row.get("Décision finale", "-")

    badge_color = (
        "#f59e0b" if dec_init == "Review"
        else "#f87171" if dec_init == "Blocked"
        else "#2dd4bf"
    )
    initial_dec_text = (
        "Douteuse" if dec_init == "Review"
        else "Refusée" if dec_init == "Blocked"
        else "Acceptée"
    )

    if dec_fin == "Review":
        etat_final_html = "<span style='opacity: 0.8; font-weight: 500;'>En attente</span>"
    elif dec_init == "Blocked" and dec_fin == "Accepted":
        etat_final_html = (
            "<span style='font-weight: 700;'>Validée après réclamation</span>"
            + (f"<br><span style='font-size:12px; opacity:0.8;'>Motif : {t_motif}</span>" if t_motif else "")
        )
    elif dec_init == "Review" and dec_fin == "Blocked":
        etat_final_html = (
            "<span style='font-weight: 700;'>Refusée après examen</span><br>"
            "<span style='font-size: 12px; opacity: 0.8;'>Refus manuel"
            + (f" · Agent déclaré : {t_agent}" if t_agent else "")
            + (f" · {t_date_val}" if t_date_val else "")
            + "</span>"
            + (f"<br><span style='font-size:12px; font-weight:600;'>Motif du refus — {t_motif}</span>" if t_motif else "")
        )
    elif dec_fin == "Accepted":
        etat_final_html = "<span style='color: #2dd4bf; font-weight: 600;'>Acceptée</span>"
    else:
        etat_final_html = "<span style='font-weight: 600;'>Refusée</span>"

    if dec_fin == "Accepted" and dec_init == "Blocked" and t_agent:
        agent_html = (
            "Validation après réclamation ·<br>"
            f"<span style='font-size: 12px; opacity: 0.85;'>Agent déclaré : {t_agent}</span>"
        )
    elif t_agent:
        agent_html = (
            "Validation manuelle ·<br>"
            f"<span style='font-size: 12px; opacity: 0.85;'>Agent déclaré : {t_agent}</span>"
        )
    else:
        agent_html = "—"

    factor_key = f"factors_open_{category_key}_{t_id}"
    is_open = show_obs and st.session_state.get(factor_key, False)

    if show_obs:
        obs_cell_html = _render_observation_cell(row, is_open)
        obs_col_html = f'<div style="flex: 2.4;">{obs_cell_html}</div>'
    else:
        obs_col_html = ""

    row_html = (
        '<div style="display: flex; padding: 16px 12px; border-bottom: 1px solid rgba(148, 163, 184, 0.25); font-size: 13px; align-items: flex-start; gap: 8px;">'
        f'<div style="flex: 1.4;"><div style="opacity: 0.78; font-size: 12px; margin-bottom: 4px;">{t_time}</div>'
        f'<div style="background-color: rgba(20, 184, 166, 0.14); border: 1px solid rgba(45, 212, 191, 0.4); display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 700; color: #2dd4bf;">{t_id}</div></div>'
        f'<div style="flex: 1; font-weight: 500;">{t_amount}</div>'
        f'<div style="flex: 1;">{t_score}</div>'
        f'<div style="flex: 1.1; color: {badge_color}; font-weight: 700;">{initial_dec_text}</div>'
        f'<div style="flex: 2.3;">{etat_final_html}</div>'
        f'<div style="flex: 1.8; opacity: 0.85;">{agent_html}</div>'
        f'<div style="flex: 1.4; opacity: 0.85;">{t_motif if t_motif else "—"}</div>'
        f'{obs_col_html}'
        '</div>'
    )
    st.markdown(row_html, unsafe_allow_html=True)

    can_review_act = (dec_fin == "Review")
    can_claim_act = (dec_fin == "Blocked")

    if can_review_act or can_claim_act or show_obs:
        cols = st.columns([1.2, 1.2, 1.8, 2.0, 2.8])
        if can_review_act:
            with cols[0]:
                if st.button("Valider", key=f"quick_val_{category_key}_{t_id}"):
                    st.session_state[f"auth_mode_{t_id}"] = "Accepted"
                    st.session_state[f"show_auth_{t_id}"] = True
                    st.rerun(scope="fragment")
            with cols[1]:
                if st.button("Refuser", key=f"quick_ref_{category_key}_{t_id}"):
                    st.session_state[f"auth_mode_{t_id}"] = "Blocked"
                    st.session_state[f"show_auth_{t_id}"] = True
                    st.rerun(scope="fragment")
        elif can_claim_act:
            with cols[0]:
                if st.button("Valider après réclamation", key=f"quick_rec_{category_key}_{t_id}", use_container_width=True):
                    st.session_state[f"auth_mode_{t_id}"] = "Accepted"
                    st.session_state[f"show_auth_{t_id}"] = True
                    st.rerun(scope="fragment")

        if show_obs:
            with cols[-1]:
                btn_label = (
                    "Masquer les champs (-)" if is_open
                    else "Champs ayant augmenté le score (+)"
                )
                if st.button(btn_label, key=f"btn_factors_{category_key}_{t_id}"):
                    st.session_state[factor_key] = not is_open
                    st.rerun(scope="fragment")

    if st.session_state.get(f"show_auth_{t_id}", False):
        target_decision = st.session_state.get(f"auth_mode_{t_id}", "Accepted")
        action_title = (
            "Valider après réclamation" if (dec_fin == "Blocked" and target_decision == "Accepted")
            else "Valider l'opération" if target_decision == "Accepted"
            else "Refuser l'opération"
        )
        with st.container(border=True):
            st.markdown(f"**Authentification requise — {action_title} ({t_id})**")
            c_ag, c_mo = st.columns(2)
            with c_ag:
                agent_id = st.text_input(
                    "Identifiant de l'agent",
                    key=f"agent_{category_key}_{t_id}",
                    max_chars=20,
                    placeholder="ex: 28GF023",
                )
            with c_mo:
                motif = st.text_input(
                    "Motif de validation",
                    key=f"motif_{category_key}_{t_id}",
                    max_chars=60,
                    placeholder="ex: FRAUDE",
                )
            c_ok, c_cancel, _ = st.columns([1.2, 1, 3])
            with c_ok:
                if st.button("Confirmer", key=f"confirm_{category_key}_{t_id}", type="primary"):
                    if agent_id.strip() and motif.strip():
                        _update_decision(t_id, target_decision, agent_id.strip(), motif.strip())
                    else:
                        st.error("Identifiant de l'agent et motif obligatoires.")
            with c_cancel:
                if st.button("Annuler", key=f"cancel_{category_key}_{t_id}"):
                    st.session_state[f"show_auth_{t_id}"] = False
                    st.rerun(scope="fragment")


def render_table_header(show_obs=True):
    """Affiche la ligne d'en-tête du tableau."""
    obs_header = '<div style="flex: 2.4;">Observation</div>' if show_obs else ""
    header_html = (
        '<div style="display: flex; background-color: rgba(20, 184, 166, 0.14); color: #2dd4bf; padding: 12px; font-weight: 700; font-size: 13px; border-radius: 6px 6px 0 0; border-bottom: 2px solid #14b8a6; gap: 8px;">'
        '<div style="flex: 1.4;">Heure locale / ID</div>'
        '<div style="flex: 1;">Montant</div>'
        '<div style="flex: 1;">Score</div>'
        '<div style="flex: 1.1;">Décision initiale</div>'
        '<div style="flex: 2.3;">État final ou validation</div>'
        '<div style="flex: 1.8;">Agent de validation</div>'
        '<div style="flex: 1.4;">Motif de validation</div>'
        f'{obs_header}'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)


def _load_history_df():
    """Charge et normalise l'historique CSV."""
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_FILE, encoding="utf-8", dtype=str).fillna("")
    if "Décision finale" not in df.columns:
        df["Décision finale"] = df.get("Décision", "")
    for col in ("Agent", "Motif", "Observation", "Date validation", "Facteurs"):
        if col not in df.columns:
            df[col] = ""
    return df


@st.fragment
def _history_body(category_key: str):
    """Contenu rafraîchi localement sans fermer la modale."""
    show_obs = category_key != "Accepted"

    c_lbl, c_dt, c_ex1, c_ex2, _ = st.columns([1.3, 1.5, 1.5, 1.5, 2.2])
    with c_lbl:
        st.markdown(
            '<div style="padding-top:8px; font-size:14px; font-weight:600;">Date de référence</div>',
            unsafe_allow_html=True,
        )
    with c_dt:
        ref_date = st.date_input("Date de référence", datetime.today(), label_visibility="collapsed", key=f"ref_date_{category_key}")

    df = _load_history_df()
    if not df.empty:
        df_cat = df[df["Décision finale"] == category_key].copy()
        ref_str = ref_date.strftime("%d/%m/%Y")
        df_day = df_cat[df_cat["Date et heure"].str.startswith(ref_str)]
    else:
        df_cat = pd.DataFrame()
        df_day = pd.DataFrame()

    with c_ex1:
        st.download_button(
            "Exporter la journée",
            data=df_day.to_csv(index=False).encode("utf-8"),
            file_name=f"export_jour_{ref_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"exp_jour_{category_key}",
        )
    with c_ex2:
        st.download_button(
            "Exporter la semaine",
            data=df_cat.to_csv(index=False).encode("utf-8"),
            file_name=f"export_semaine_{category_key.lower()}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"exp_sem_{category_key}",
        )

    if df_cat.empty:
        st.info("Aucune opération dans cette catégorie.")
        return

    df_cat = df_cat.sort_values("Date et heure", ascending=False)

    render_table_header(show_obs=show_obs)
    with st.container(height=500, border=False):
        for _, row in df_cat.iterrows():
            render_transaction_row(row, category_key)


@st.dialog("Rechercher une opération", width="large")
def render_search_dialog():
    """Dialogue de recherche d'une opération par identifiant."""
    st.markdown('<h2 style="margin:0 0 10px 0; font-size:24px;">Rechercher une opération</h2>', unsafe_allow_html=True)

    search_query = st.text_input("Identifiant FraudeScope", placeholder="Ex: FRS-000005")

    if st.button("Rechercher", type="primary", use_container_width=False):
        if not search_query.strip():
            st.info("Veuillez renseigner un identifiant.")
            return
        try:
            df = _load_history_df()
            result = df[df["ID"].str.contains(search_query.strip(), case=False, na=False)]
            if not result.empty:
                render_table_header(show_obs=True)
                with st.container(height=420, border=False):
                    for _, row in result.iterrows():
                        render_transaction_row(row, "search")
            else:
                st.warning("Aucune opération trouvée.")
        except Exception as e:
            st.error(f"Erreur lors de la recherche : {e}")


@st.dialog("Suivi des opérations", width="large")
def render_history(category_key: str) -> None:
    """Dialogue listant les opérations d'une catégorie."""
    config = {
        "Accepted": ("Opérations courantes", ""),
        "Review": ("Opérations douteuses", "Ces opérations attendent une décision après examen."),
        "Blocked": (
            "Opérations frauduleuses",
            "En cas de réclamation, vérifiez l'authenticité de l'opération avant de la valider.",
        ),
    }
    title, note = config.get(category_key, ("Historique", ""))

    st.markdown(f'<h2 style="margin:0; font-size:24px;">{title}</h2>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div style="opacity:0.8; margin-top:4px; font-size:14px;">{note}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    _history_body(category_key)


def _update_decision(transaction_id, new_decision, agent_id, motif):
    """Met à jour la décision finale, l'agent, le motif et la date de validation."""
    df = _load_history_df()
    if df.empty:
        return

    mask = df["ID"] == transaction_id
    df.loc[mask, "Décision finale"] = new_decision
    df.loc[mask, "Agent"] = agent_id
    df.loc[mask, "Motif"] = motif
    df.loc[mask, "Date validation"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8")

    st.session_state[f"show_auth_{transaction_id}"] = False
    st.rerun(scope="fragment")