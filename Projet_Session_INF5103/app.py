import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime

# Module de suivi (tableaux et recherche dans des dialogues)
from _suivi_common import render_history, render_search_dialog

st.set_page_config(page_title="FraudeScope", layout="wide", initial_sidebar_state="collapsed")

ROOT = Path(__file__).resolve().parent
HISTORY_FILE = ROOT / "historique.csv"
METRICS_FILE = ROOT / "models" / "metrics.json"
POLICY_FILE = ROOT / "decision" / "decision-policy.json"

# --- CSS GLOBAL (palette émeraude / sarcelle) ---
st.markdown("""
<style>
    /* On masque la navigation latérale de Streamlit */
    [data-testid="collapsedControl"], [data-testid="stSidebarNav"] {display: none;}
    .stApp { background-color: #f8fafc; }

    /* Barre supérieure avec le logo */
    .top-bar {
        background-color: #0f172a; padding: 12px 24px; display: flex;
        align-items: center; justify-content: space-between;
        border-bottom: 4px solid #14b8a6; margin-top: -60px;
        margin-left: -40px; margin-right: -40px; margin-bottom: 40px;
    }
    .top-bar-logo { color: white; font-size: 22px; font-weight: 800; display: flex; align-items: center; gap: 10px;}
    .top-bar-logo .icon { background-color: #0f766e; border-radius: 8px; padding: 4px 10px; color: #ffffff; border: 1px solid #5eead4; }

    /* Boutons de navigation : pleine largeur, sur une seule ligne */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background-color: #ffffff !important; color: #1e293b !important;
        border: 1px solid #cbd5e1 !important; border-radius: 6px !important;
        padding: 8px 12px !important; font-weight: 600 !important;
        white-space: nowrap !important; font-size: 14px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important;
        width: 100% !important;
        transition: all 0.15s ease;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background-color: #f0fdfa !important;
        color: #0f766e !important;
        border-color: #14b8a6 !important;
    }

    /* Bouton principal (Analyser) */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #0f766e !important;
        color: #ffffff !important;
        border: 1px solid #0f766e !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #115e59 !important;
        border-color: #115e59 !important;
    }

    /* Cartes blanches autour des blocs */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff; border-radius: 12px !important; padding: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05); border: 1px solid #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- FONCTIONS UTILITAIRES ---
def next_display_id():
    """Renvoie le prochain identifiant FRS-xxxxxx disponible."""
    if not os.path.exists(HISTORY_FILE):
        return "FRS-000001"
    try:
        df = pd.read_csv(HISTORY_FILE, encoding="utf-8")
        nums = df["ID"].dropna().astype(str).str.extract(r"FRS-(\d+)")[0].dropna().astype(int)
        return f"FRS-{(nums.max() + 1) if not nums.empty else 1:06d}"
    except Exception:
        return "FRS-000001"


def sauvegarder_historique(montant, score, decision, display_id, observation="", facteurs_json=""):
    """Ajoute une opération au fichier historique."""
    ligne = pd.DataFrame([{
        "Date et heure": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "ID": display_id,
        "Montant (USD)": montant,
        "Score de fraude": f"{score * 100:.2f} %",
        "Décision": decision,
        "Décision finale": decision,
        "Observation": observation,
        "Agent": "",
        "Motif": "",
        "Date validation": "",
        "Facteurs": facteurs_json,
    }])
    if not os.path.exists(HISTORY_FILE):
        ligne.to_csv(HISTORY_FILE, index=False, encoding="utf-8")
    else:
        existing = pd.read_csv(HISTORY_FILE, encoding="utf-8", dtype=str).fillna("")
        for col in ligne.columns:
            if col not in existing.columns:
                existing[col] = ""
        combined = pd.concat([existing, ligne.astype(str)], ignore_index=True)
        combined.to_csv(HISTORY_FILE, index=False, encoding="utf-8")


def engineer_features(df):
    """Crée les 10 features dérivées (ratios, vélocité, encodage cyclique)."""
    df = df.copy()
    df["amount_to_balance_ratio"] = df["amount_usd"] / (df["account_balance_usd"] + 1)

    risk_flags = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    df["risk_signal_count"] = df[risk_flags].sum(axis=1).astype(float)

    df["txn_velocity"] = df["txn_count_last_24h"] / (df["hours_since_last_txn"] + 1)
    df["is_night"] = ((df["time_of_day_hour"] >= 22) | (df["time_of_day_hour"] <= 6)).astype(float)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
    df["card_age_ratio"] = df["card_age_months"] / (df["customer_age"] * 12 + 1)

    df["hour_sin"] = np.sin(2 * np.pi * df["time_of_day_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time_of_day_hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def _predict_from_raw(raw_df, pipe, feat_list, apply_fe):
    """Calcule la probabilité de fraude à partir d'un DataFrame brut (1 ligne)."""
    df_prep = engineer_features(raw_df) if apply_fe else raw_df.copy()
    return float(pipe.predict_proba(df_prep[feat_list])[0, 1])


def calculer_facteurs_score(raw_row_df, base_proba, pipe, feat_list, apply_fe, ui_labels):
    """Estime les champs ayant augmenté le score de fraude et leur part relative."""
    neutral_defaults = {
        "amount_usd": 125.0,
        "hours_since_last_txn": 12.0,
        "txn_count_last_24h": 1,
        "distance_from_home_km": 5.0,
        "card_age_months": 47,
        "customer_age": 50,
        "account_balance_usd": 3317.0,
        "cvv_retry_count": 0,
        "velocity_score": 20.0,
        "time_of_day_hour": 12,
        "day_of_week": 3,
        "merchant_risk_score": 38.0,
        "prior_disputes": 0,
        "is_foreign_transaction": False,
        "is_new_merchant": False,
        "used_vpn": False,
        "ip_country_mismatch": False,
        "billing_shipping_mismatch": False,
        "is_ai_generated_scam_attempt": False,
        "merchant_category": "Groceries",
        "card_type": "Visa",
        "auth_method": "OTP",
        "channel": "POS",
        "device_type": "POS Terminal",
    }

    flag_labels = {
        "is_foreign_transaction": "Opération à l'étranger",
        "is_new_merchant": "Nouveau commerçant",
        "used_vpn": "VPN utilisé",
        "ip_country_mismatch": "Pays de l'adresse IP différent",
        "billing_shipping_mismatch": "Adresses différentes",
        "is_ai_generated_scam_attempt": "Signal d'arnaque générée par IA",
    }

    single_field_labels = {
        "cvv_retry_count": "Nouvelles tentatives de CVV",
        "amount_usd": "Montant (USD)",
        "merchant_risk_score": "Score de risque du commerçant",
        "velocity_score": "Score de fréquence",
        "auth_method": "Méthode d'authentification",
        "merchant_category": "Catégorie du commerçant",
        "channel": "Canal",
        "device_type": "Appareil",
        "card_type": "Type de carte",
        "distance_from_home_km": "Distance du domicile (km)",
        "txn_count_last_24h": "Opérations sur les dernières 24 h",
        "hours_since_last_txn": "Heures depuis la dernière opération",
        "prior_disputes": "Litiges antérieurs",
        "account_balance_usd": "Solde du compte (USD)",
        "card_age_months": "Ancienneté de la carte (mois)",
        "customer_age": "Âge du client (ans)",
        "time_of_day_hour": "Heure",
        "day_of_week": "Jour de la semaine",
    }

    contributions = []

    try:
        active_flags = [k for k in flag_labels if bool(raw_row_df.iloc[0].get(k, False))]
        if active_flags:
            df_no_flags = raw_row_df.copy()
            for k in active_flags:
                df_no_flags[k] = False
            p_without = _predict_from_raw(df_no_flags, pipe, feat_list, apply_fe)
            delta = max(0.0, base_proba - p_without)
            if delta > 1e-6:
                details = " · ".join(f"{flag_labels[k]} : Oui" for k in active_flags)
                label = f"Nombre de signaux de risque actifs : {len(active_flags)} — {details}"
                contributions.append((label, delta))

        for col, lbl in single_field_labels.items():
            if col not in raw_row_df.columns or col not in neutral_defaults:
                continue
            cur_val = raw_row_df.iloc[0][col]
            def_val = neutral_defaults[col]
            if cur_val == def_val:
                continue

            df_mod = raw_row_df.copy()
            df_mod[col] = def_val
            p_without = _predict_from_raw(df_mod, pipe, feat_list, apply_fe)
            delta = max(0.0, base_proba - p_without)
            if delta > 1e-6:
                disp_val = ui_labels.get(col, cur_val)
                contributions.append((f"{lbl} : {disp_val}", delta))

    except Exception:
        contributions = []

    if not contributions:
        active_flags = [k for k in flag_labels if bool(raw_row_df.iloc[0].get(k, False))]
        if active_flags:
            details = " · ".join(f"{flag_labels[k]} : Oui" for k in active_flags)
            return json.dumps({
                "items": [f"Nombre de signaux de risque actifs : {len(active_flags)} — {details}"],
                "coverage": "",
            }, ensure_ascii=False)
        return json.dumps({
            "items": ["Aucun facteur particulier au-dessus de la référence."],
            "coverage": "",
        }, ensure_ascii=False)

    contributions.sort(key=lambda x: x[1], reverse=True)
    total_pos = sum(d for _, d in contributions)
    top_items = contributions[:4]
    top_sum = sum(d for _, d in top_items)

    formatted_items = []
    for lbl, d in top_items:
        part_pct = (d / total_pos * 100.0) if total_pos > 0 else 0.0
        part_str = f"{part_pct:.1f}".replace(".", ",") + " %"
        formatted_items.append(f"{lbl} · {part_str}")

    coverage_pct = (top_sum / total_pos * 100.0) if total_pos > 0 else 0.0
    coverage_str = f"{coverage_pct:.1f}".replace(".", ",") + " %"

    return json.dumps({
        "items": formatted_items,
        "coverage": f"Ces facteurs couvrent {coverage_str} des contributions qui ont augmenté le score.",
    }, ensure_ascii=False)


@st.cache_resource
def load_model():
    """Charge le modèle une seule fois pour toute la session."""
    return joblib.load(ROOT / "models" / "final_model.joblib")


@st.cache_data
def load_metrics():
    """Charge les métriques d'évaluation."""
    try:
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data
def load_policy():
    """Charge la politique de décision (seuils)."""
    try:
        return json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def percent(v):
    """Formate un taux en pourcentage à la française."""
    return f"{v * 100:.2f}".replace(".", ",") + " %"


# --- CHARGEMENT DU MODÈLE ---
try:
    model = load_model()
    pipeline = model["pipeline"]
    seuil_defaut = model["seuil"]
    features = model["features"]
    use_fe = bool(model.get("use_fe", False))
    metrics = load_metrics()
    policy = load_policy()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()


# --- EN-TÊTE ---
st.markdown(
    '<div class="top-bar"><div class="top-bar-logo">'
    '<span class="icon">F</span> Fraude'
    '<span style="font-weight: 300; color: #5eead4;">Scope</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

# Navigation : quatre boutons sur une seule ligne, pleine largeur
c1, c2, c3, c4 = st.columns(4, gap="small")
with c1:
    if st.button("Opérations courantes", use_container_width=True):
        render_history("Accepted")
with c2:
    if st.button("Opérations douteuses", use_container_width=True):
        render_history("Review")
with c3:
    if st.button("Opérations frauduleuses", use_container_width=True):
        render_history("Blocked")
with c4:
    if st.button("Rechercher une opération", use_container_width=True):
        render_search_dialog()


# --- BANDEAU D'ANALYSE ---
st.markdown("""
<div style="background: linear-gradient(135deg, #0f172a 0%, #115e59 100%); padding: 40px; border-radius: 15px; color: white; margin-bottom: 30px; border-bottom: 4px solid #14b8a6;">
    <h1 style="font-size: 48px; font-weight: 800; margin: 0;">Analyse d'une opération</h1>
</div>
""", unsafe_allow_html=True)

if "risk_fields" not in st.session_state:
    st.session_state["risk_fields"] = []

col_left, col_right = st.columns([1.5, 1], gap="large")


# --- FORMULAIRE ---
with col_left:
    with st.container(border=True):
        st.markdown(
            "<h2 style='margin-top: 0; font-size: 24px; margin-bottom: 20px; color: #0f172a;'>"
            "Données de l'opération</h2>",
            unsafe_allow_html=True,
        )
        with st.form("form_transaction"):
            c1, c2 = st.columns(2)
            with c1:
                amount = st.number_input("Montant (USD)", min_value=0.0, value=125.0)
                auth = st.selectbox(
                    "Méthode d'authentification",
                    ["Code à usage unique (OTP)", "Code PIN", "Biométrie", "Aucune"],
                )
                ai_scam = st.checkbox("Signal d'arnaque générée par IA")
            with c2:
                cvv_retry = st.number_input("Nouvelles tentatives de CVV", min_value=0, value=0)
                merchant_risk = st.number_input("Score de risque du commerçant", min_value=0.0, value=38.0)
                velocity = st.number_input("Score de fréquence", min_value=0.0, value=20.0)

            with st.expander("Autres champs"):
                c3, c4 = st.columns(2)
                with c3:
                    merchant_cat = st.selectbox(
                        "Catégorie du commerçant", ["Alimentation", "Électronique", "Voyage"]
                    )
                    channel = st.selectbox("Canal", ["Point de vente", "En ligne", "Distributeur"])
                    hour = st.number_input("Heure (0-23)", 0, 23, 12)
                with c4:
                    card_type = st.selectbox("Type de carte", ["Visa", "Mastercard"])
                    device = st.selectbox("Appareil", ["Terminal de paiement", "Téléphone"])
                    dow = st.selectbox(
                        "Jour de la semaine",
                        range(7),
                        index=3,
                        format_func=lambda x: ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"][x],
                    )

                c5, c6 = st.columns(2)
                with c5:
                    balance = st.number_input("Solde du compte (USD)", min_value=0.0, value=3317.0)
                    card_age = st.number_input("Ancienneté de la carte (mois)", min_value=0, value=47)
                    hours = st.number_input("Heures depuis la dernière opération", min_value=0.0, value=12.0)
                    disputes = st.number_input("Litiges antérieurs", min_value=0, value=0)
                with c6:
                    customer_age = st.number_input("Âge du client (ans)", min_value=18, value=50)
                    distance = st.number_input("Distance du domicile (km)", min_value=0.0, value=5.0)
                    txn_count = st.number_input("Opérations sur les dernières 24 h", min_value=0, value=1)

                f1, f2 = st.columns(2)
                with f1:
                    is_foreign = st.checkbox("Opération à l'étranger")
                    vpn = st.checkbox("VPN utilisé")
                    billing_mismatch = st.checkbox("Adresses différentes")
                with f2:
                    is_new = st.checkbox("Nouveau commerçant")
                    ip_mismatch = st.checkbox("Pays de l'adresse IP différent")

            submitted = st.form_submit_button("Analyser", type="primary")


# --- RÉSULTATS ---
with col_right:
    if submitted:
        dict_auth = {
            "Code à usage unique (OTP)": "OTP",
            "Code PIN": "PIN",
            "Biométrie": "Biometric",
            "Aucune": "No Authentication",
        }
        dict_cat = {"Alimentation": "Groceries", "Électronique": "Electronics", "Voyage": "Travel"}
        dict_chan = {"Point de vente": "POS", "En ligne": "Online", "Distributeur": "ATM"}
        dict_dev = {"Terminal de paiement": "POS Terminal", "Téléphone": "Android Phone"}

        raw_transaction = pd.DataFrame([{
            "amount_usd": amount,
            "hours_since_last_txn": hours,
            "txn_count_last_24h": txn_count,
            "distance_from_home_km": distance,
            "card_age_months": card_age,
            "customer_age": customer_age,
            "account_balance_usd": balance,
            "cvv_retry_count": cvv_retry,
            "velocity_score": velocity,
            "time_of_day_hour": hour,
            "day_of_week": dow,
            "merchant_risk_score": merchant_risk,
            "prior_disputes": disputes,
            "is_foreign_transaction": is_foreign,
            "is_new_merchant": is_new,
            "used_vpn": vpn,
            "ip_country_mismatch": ip_mismatch,
            "billing_shipping_mismatch": billing_mismatch,
            "is_ai_generated_scam_attempt": ai_scam,
            "merchant_category": dict_cat[merchant_cat],
            "card_type": card_type,
            "auth_method": dict_auth[auth],
            "channel": dict_chan[channel],
            "device_type": dict_dev[device],
        }])

        proba = _predict_from_raw(raw_transaction, pipeline, features, use_fe)

        ui_labels = {
            "auth_method": auth,
            "merchant_category": merchant_cat,
            "channel": channel,
            "device_type": device,
            "day_of_week": ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"][dow],
        }
        facteurs_json = calculer_facteurs_score(
            raw_transaction, proba, pipeline, features, use_fe, ui_labels
        )

        if policy and "thresholds" in policy:
            seuil_validation = policy["thresholds"].get("review", seuil_defaut)
            seuil_refus = policy["thresholds"].get("block")
        else:
            seuil_validation = seuil_defaut
            seuil_refus = None

        if proba >= seuil_validation:
            st.session_state["risk_fields"] = ["cvv_retry_count"] if cvv_retry > 2 else []
        else:
            st.session_state["risk_fields"] = []

        is_blocked = (seuil_refus is not None) and (proba >= seuil_refus)
        is_review = (not is_blocked) and (proba >= seuil_validation)

        if is_blocked:
            obs = f"Blocage automatique. Le score de {percent(proba)} atteint le seuil de refus de {percent(seuil_refus)}."
            sauvegarder_historique(amount, proba, "Blocked", next_display_id(), obs, facteurs_json)
        elif is_review:
            obs = f"Mise en attente. Le score de {percent(proba)} atteint le seuil de vérification de {percent(seuil_validation)}."
            sauvegarder_historique(amount, proba, "Review", next_display_id(), obs, facteurs_json)
        else:
            sauvegarder_historique(amount, proba, "Accepted", next_display_id(), "", facteurs_json)

        with st.container(border=True):
            st.markdown(
                "<h2 style='margin-top: 0; font-size: 24px; margin-bottom: 20px; color: #0f172a;'>"
                "Résultat</h2>",
                unsafe_allow_html=True,
            )

            if is_blocked:
                bg, txt, icon, title = "#fee2e2", "#991b1b", "!", "Transaction refusée"
            elif is_review:
                bg, txt, icon, title = "#fef3c7", "#92400e", "!", "Transaction douteuse"
            else:
                bg, txt, icon, title = "#f0fdfa", "#0f766e", "OK", "Transaction acceptée"

            st.markdown(f"""
            <div style="background-color: {bg}; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; align-items: center; gap: 15px;">
                <div style="font-size: 20px; font-weight: bold; color: {txt};">{icon}</div>
                <div style="color: {txt}; font-weight: bold; font-size: 16px;">{title}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                "<div style='display: flex; justify-content: space-between; font-size: 16px; margin-bottom: 10px; color: #0f172a;'>"
                f"<span>Score de fraude</span><strong>{percent(proba)}</strong></div>",
                unsafe_allow_html=True,
            )

            color_bar = "#ef4444" if is_blocked else "#f59e0b" if is_review else "#0d9488"
            st.markdown(f"""
            <div style="width: 100%; background-color: #e2e8f0; border-radius: 4px; height: 8px; margin-bottom: 15px;">
                <div style="width: {min(proba * 100, 100)}%; background-color: {color_bar}; height: 8px; border-radius: 4px;"></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                "<div style='display: flex; justify-content: space-between; color: #475569; font-size: 14px; margin-bottom: 5px;'>"
                f"<span>Seuil de validation</span><span>{percent(seuil_validation)}</span></div>",
                unsafe_allow_html=True,
            )
            if seuil_refus is not None:
                st.markdown(
                    "<div style='display: flex; justify-content: space-between; color: #475569; font-size: 14px;'>"
                    f"<span>Seuil de refus</span><span>{percent(seuil_refus)}</span></div>",
                    unsafe_allow_html=True,
                )

        if metrics:
            with st.container(border=True):
                st.markdown(
                    "<h2 style='margin-top: 0; font-size: 20px; margin-bottom: 15px; color: #0f172a;'>Performance du modèle</h2>",
                    unsafe_allow_html=True,
                )

                c1, c2, c3 = st.columns(3)
                for col_m, lbl_m, key_m in [
                    (c1, "Précision", "precision"),
                    (c2, "Rappel", "rappel"),
                    (c3, "Score F1", "f1"),
                ]:
                    col_m.markdown(
                        "<div style='background-color: #f8fafc; border: 1px solid #e2e8f0; border-top: 3px solid #14b8a6; padding: 10px; border-radius: 8px; text-align: center;'>"
                        f"<div style='color: #475569; font-size: 12px; margin-bottom: 5px;'>{lbl_m}</div>"
                        f"<div style='font-size: 20px; font-weight: bold; color: #0f172a;'>{percent(metrics[key_m])}</div></div>",
                        unsafe_allow_html=True,
                    )