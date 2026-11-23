import io
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from Projet_Session_INF5103._suivi_common import (
    DEFAULTS,
    FIELDS,
    Service,
    bank_mode,
    streamlit_identity,
    t,
    history_page,
    search_page,
    review_detail,
    show_observation,
)

st.set_page_config(
    page_title="FraudScope", layout="wide", initial_sidebar_state="collapsed"
)
ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_service():
    import os

    database = ROOT / (
        "fraudscope-bank.sqlite3" if bank_mode() else "fraudscope.sqlite3"
    )
    service = Service(os.getenv("FRAUDSCOPE_DATABASE") or database)
    service.start_archiving()
    return service


st.session_state["language"] = "fr"
st.session_state.setdefault("detail", None)
st.session_state.setdefault("result", None)
st.session_state.setdefault("csv_running", False)
st.session_state.setdefault("csv_index", 0)
service = load_service()
service.store.expire_pending()
try:
    identity = streamlit_identity(st)
except (ValueError, PermissionError) as exc:
    st.error(str(exc))
    st.stop()

STYLE = """

<style>
    
    [data-testid="collapsedControl"], [data-testid="stSidebarNav"] {display: none;}

    
    .top-bar {
        background-color: #0f172a; padding: 12px 24px; display: flex;
        align-items: center; justify-content: space-between;
        border-bottom: 4px solid #14b8a6; margin-top: -60px;
        margin-left: -40px; margin-right: -40px; margin-bottom: 40px;
    }
    .top-bar-logo { color: white; font-size: 22px; font-weight: 800; display: flex; align-items: center; gap: 10px;}
    .top-bar-logo .icon { background-color: #0f766e; border-radius: 8px; padding: 4px 10px; color: #ffffff; border: 1px solid #5eead4; }

    
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background-color: #1e293b !important; color: #f8fafc !important;
        border: 1px solid #334155 !important; border-radius: 6px !important;
        padding: 8px 12px !important; font-weight: 600 !important;
        white-space: nowrap !important; font-size: 14px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
        width: 100% !important;
        transition: all 0.15s ease;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background-color: #0f766e !important;
        color: #ffffff !important;
        border-color: #14b8a6 !important;
    }

    
    .st-key-analyze_button button {
        background-color: #0f766e !important;
        color: #ffffff !important;
        border: 1px solid #14b8a6 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
    }
    .st-key-analyze_button button:hover {
        background-color: #115e59 !important;
        border-color: #5eead4 !important;
    }

    
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        padding: 24px !important;
        border: 1px solid rgba(148, 163, 184, 0.35) !important;
    }

    
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] label p,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        color: inherit !important;
        opacity: 1 !important;
    }

    [class*="st-key-risk_"] {
        border: 2px solid #d38b08;
        border-radius: 10px;
        padding: 12px;
        background-color: #fff4d5;
        box-shadow: 0 0 0 2px rgba(211,139,8,.12);
        color: #4b3307;
        margin-bottom: 8px;
    }
    [class*="st-key-risk_"] [data-testid="stWidgetLabel"],
    [class*="st-key-risk_"] [data-testid="stWidgetLabel"] p,
    [class*="st-key-risk_"] label p {
        color: #4b3307 !important;
        font-weight: 700;
    }
    .risk-marker {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #6b4200;
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .risk-marker span {
        font-size: 28px;
        line-height: 1;
        color: #8b4b00;
    }
</style>

"""

st.markdown(STYLE, unsafe_allow_html=True)
st.markdown(
    '<div class="top-bar"><div class="top-bar-logo">'
    '<span class="icon">F</span> Fraude'
    '<span style="font-weight: 300; color: #5eead4;">Scope</span>'
    "</div></div>",
    unsafe_allow_html=True,
)
def close_dialog():
    st.session_state.active_dialog = None
    st.session_state.detail = None


st.session_state.setdefault("active_dialog", None)
navigation = [
    ("Accepted", "Opérations courantes"),
    ("Review", "Opérations douteuses"),
    ("Blocked", "Opérations frauduleuses"),
    ("search", "Rechercher une opération"),
]
for column, (category, label) in zip(
    st.columns([1, 1, 1, 0.2], gap="small"), navigation
):
    if category == "search":
        with column.popover(
            ":material/search:", help="Rechercher une opération", key="nav_search",
        ):
            search_page(service)
    elif column.button(t(label), key="nav_" + category, width="stretch"):
        st.session_state.active_dialog = category
        st.session_state.detail = None


for field in FIELDS:
    st.session_state.setdefault("value_" + field["key"], DEFAULTS[field["key"]])
    widget = "field_" + field["key"]
    if widget not in st.session_state:
        st.session_state[widget] = st.session_state["value_" + field["key"]]

FR_CATEGORY = {
    "Groceries": "Alimentation",
    "Electronics": "Électronique",
    "Travel": "Voyage",
    "Crypto Exchange": "Plateforme crypto",
    "Fuel": "Carburant",
    "Gaming": "Jeux vidéo",
    "Gift Cards": "Cartes-cadeaux",
    "Healthcare": "Santé",
    "Online Retail": "Commerce en ligne",
    "Utilities": "Services publics",
    "Biometric": "Biométrie",
    "No Authentication": "Aucune authentification",
    "OTP": "Code à usage unique (OTP)",
    "PIN": "Code PIN",
    "ATM": "Distributeur automatique",
    "Contactless": "Sans contact",
    "In-App": "Dans l’application",
    "Online": "En ligne",
    "POS": "Point de vente",
    "ATM Machine": "Distributeur automatique",
    "Android Phone": "Téléphone Android",
    "POS Terminal": "Terminal de paiement",
    "Smart Watch": "Montre connectée",
    "Tablet": "Tablette",
    "Windows PC": "PC Windows",
}


def category_label(value):
    return FR_CATEGORY.get(value, value)


def remember():
    for field in FIELDS:
        key = field["key"]
        if "field_" + key in st.session_state:
            st.session_state["value_" + key] = st.session_state["field_" + key]
    st.session_state.result = None


def render_field_control(field, alert=False):
    key, kind = field["key"], field["kind"]
    widget = "field_" + key
    label = ("⚠ " if alert else "") + t(field["label"])
    if kind == "category":
        options = service.categories[key]
        st.selectbox(
            label, options, key=widget, format_func=category_label, on_change=remember
        )
    elif kind == "boolean":
        st.checkbox(label, key=widget, on_change=remember)
    elif kind == "integer" and field.get("options"):
        st.selectbox(
            label,
            list(range(len(field["options"]))),
            key=widget,
            format_func={
                value: t(label) for value, label in enumerate(field["options"])
            }.get,
            on_change=remember,
        )
    elif kind == "integer":
        st.number_input(
            label,
            min_value=int(field.get("min", 0)),
            max_value=field.get("max"),
            step=1,
            key=widget,
            on_change=remember,
        )
    else:
        st.number_input(
            label,
            min_value=float(field["min"]) if "min" in field else None,
            max_value=float(field["max"]) if "max" in field else None,
            key=widget,
            on_change=remember,
        )


def field_widget(field, alert=False):
    if alert:
        with st.container(key="risk_" + field["key"]):
            st.markdown(
                '<div class="risk-marker"><span aria-hidden="true">⚠</span> '
                + t("Facteur de risque")
                + "</div>",
                unsafe_allow_html=True,
            )
            render_field_control(field, True)
    else:
        render_field_control(field)


def risk_keys(entry):
    if not entry or (
        entry["initialDecision"] != "Review" and not entry.get("insufficientFunds")
    ):
        return set()
    items = json.loads(entry.get("factors") or "{}").get("items", [])
    text = " ".join(items)
    labels = {
        "amount_usd": "Montant (USD)",
        "cvv_retry_count": "Nouvelles tentatives de CVV",
        "merchant_risk_score": "Score de risque du commerçant",
        "velocity_score": "Score de fréquence",
        "auth_method": "Méthode d'authentification",
        "merchant_category": "Catégorie du commerçant",
        "channel": "Canal",
        "device_type": "Appareil",
        "card_type": "Type de carte",
        "distance_from_home_km": "Distance du domicile",
        "txn_count_last_24h": "Opérations sur les dernières 24 h",
        "hours_since_last_txn": "Heures depuis la dernière opération",
        "prior_disputes": "Litiges antérieurs",
        "account_balance_usd": "Solde du compte",
        "card_age_months": "Ancienneté de la carte",
        "customer_age": "Âge du client",
        "time_of_day_hour": "Heure :",
        "day_of_week": "Jour de la semaine",
        "is_foreign_transaction": "Opération à l'étranger",
        "is_new_merchant": "Nouveau commerçant",
        "used_vpn": "VPN utilisé",
        "ip_country_mismatch": "Pays de l'adresse IP différent",
        "billing_shipping_mismatch": "Adresses différentes",
        "is_ai_generated_scam_attempt": "Signal d'arnaque générée par IA",
    }
    keys = {key for key, label in labels.items() if label in text}
    if entry.get("insufficientFunds"):
        keys.update(("amount_usd", "account_balance_usd"))
    return keys


def reset_values():
    st.session_state.result = None
    st.session_state.original_amount = ""
    st.session_state.original_currency = ""
    for field in FIELDS:
        key = field["key"]
        st.session_state["value_" + key] = DEFAULTS[key]
        st.session_state["field_" + key] = DEFAULTS[key]


@st.fragment(run_every="1s")
def playback():
    rows = st.session_state.get("csv_rows", [])
    cols = st.columns(3)
    if cols[0].button(t("Démarrer"), disabled=not rows or st.session_state.csv_running):
        st.session_state.csv_running = True
    if cols[1].button(t("Pause"), disabled=not st.session_state.csv_running):
        st.session_state.csv_running = False
    if st.session_state.csv_running:
        index = st.session_state.csv_index
        if index < len(rows):
            tx, reference, original = rows[index]
            try:
                service.analyze(tx, reference, original, "csv")
            except ValueError as exc:
                st.error(str(exc))
                st.session_state.csv_running = False
            else:
                st.session_state.csv_index += 1
        if st.session_state.csv_index >= len(rows):
            st.session_state.csv_running = False
    st.caption(f"{t('Opérations traitées')} · {st.session_state.csv_index}/{len(rows)}")


def analyze_page():
    st.markdown(
        '<div style="background: linear-gradient(135deg, #0f172a 0%, #115e59 100%); '
        "padding: 40px; border-radius: 15px; color: white; margin-bottom: 30px; "
        'border-bottom: 4px solid #14b8a6;">'
        '<h1 style="font-size: 48px; font-weight: 800; margin: 0;">'
        + t("Analyse d’une opération")
        + "</h1></div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.5, 1])
    entry = st.session_state.result
    highlighted = risk_keys(entry)
    primary = [
        "amount_usd",
        "cvv_retry_count",
        "auth_method",
        "merchant_risk_score",
        "is_ai_generated_scam_attempt",
        "velocity_score",
    ]
    by_key = {field["key"]: field for field in FIELDS}
    with left, st.container(border=True):
        st.markdown(
            "<h2 style='margin-top:0;font-size:24px;margin-bottom:20px;'>"
            + t("Données de l’opération")
            + "</h2>",
            unsafe_allow_html=True,
        )
        for offset in range(0, len(primary), 2):
            columns = st.columns(2)
            for column, key in zip(columns, primary[offset : offset + 2]):
                with column:
                    field_widget(by_key[key], key in highlighted)
        promoted = [
            field
            for field in FIELDS
            if field["key"] in highlighted and field["key"] not in primary
        ]
        for offset in range(0, len(promoted), 2):
            for column, field in zip(st.columns(2), promoted[offset : offset + 2]):
                with column:
                    field_widget(field, True)
        with st.expander(t("Autres champs")):
            other_fields = [
                field
                for field in FIELDS
                if field["key"] not in primary and field not in promoted
            ]
            for offset in range(0, len(other_fields), 2):
                for column, field in zip(
                    st.columns(2), other_fields[offset : offset + 2]
                ):
                    with column:
                        field_widget(field)
            amount_column, currency_column = st.columns(2)
            with amount_column:
                st.text_input(
                    t("Montant d’origine"), key="original_amount", on_change=remember
                )
            with currency_column:
                st.text_input(
                    t("Devise d’origine"),
                    key="original_currency",
                    max_chars=3,
                    on_change=remember,
                )
        submitted = st.button(t("Analyser"), type="primary", key="analyze_button")
        st.button(t("Restaurer les valeurs par défaut"), on_click=reset_values)
        if submitted:
            st.session_state.scroll_to_top = uuid4().hex
            transaction = {
                field["key"]: st.session_state["field_" + field["key"]]
                for field in FIELDS
            }
            try:
                amount = st.session_state.get("original_amount", "").strip()
                currency = st.session_state.get("original_currency", "").strip().upper()
                original = (
                    {"amount": amount, "currency": currency}
                    if amount or currency
                    else None
                )
                result, _ = service.analyze(transaction, original=original)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.result = result
                st.rerun()
    with right:
        if entry:
            with st.container(border=True):
                st.markdown(
                    "<h2 style='margin-top:0;font-size:24px;margin-bottom:20px;'>"
                    + t("Résultat")
                    + "</h2>",
                    unsafe_allow_html=True,
                )
                st.caption(entry["displayId"])
                if entry.get("insufficientFunds"):
                    background, foreground, title = (
                        "#fee2e2",
                        "#991b1b",
                        "Bloquée — solde insuffisant",
                    )
                elif entry["initialDecision"] == "Accepted":
                    background, foreground, title = (
                        "#f0fdfa",
                        "#0f766e",
                        "Transaction acceptée",
                    )
                else:
                    background, foreground, title = (
                        "#fef3c7",
                        "#92400e",
                        "Transaction douteuse",
                    )
                st.markdown(
                    f'<div style="background:{background};color:{foreground};padding:15px;border-radius:8px;'
                    f'margin:15px 0;font-weight:700;">{t(title)}</div>',
                    unsafe_allow_html=True,
                )
                score_label = t("Score de fraude")
                st.markdown(
                    '<div style="display:flex;justify-content:space-between;font-size:16px;margin-bottom:10px;">'
                    f'<span>{score_label}</span><strong>{entry["score"]:.2%}</strong></div>',
                    unsafe_allow_html=True,
                )
                show_observation(entry)
                color = (
                    "#ef4444"
                    if entry.get("insufficientFunds")
                    else (
                        "#f59e0b" if entry["initialDecision"] == "Review" else "#0d9488"
                    )
                )
                st.markdown(
                    '<div style="background:rgba(148,163,184,.3);border-radius:4px;height:8px;margin-bottom:15px;">'
                    f'<div style="width:{min(entry["score"]*100,100)}%;background:{color};height:8px;border-radius:4px;"></div></div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"{t('Seuil de validation')} · {service.threshold:.2%}")
                if st.button(t("Ouvrir le dossier"), key="open_result"):
                    st.session_state.detail = entry["id"]
                    st.session_state.active_dialog = "search"
                    st.rerun()
    with st.expander(t("Chargement CSV")):
        uploaded = st.file_uploader(
            t("CSV de transactions"), type=["csv"], key="csv_upload"
        )
        if uploaded and st.button(t("Charger un CSV")):
            try:
                rows = service.load_csv(io.BytesIO(uploaded.getvalue()))
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(str(exc))
            else:
                st.session_state.csv_rows = rows
                st.session_state.csv_index = 0
                st.session_state.csv_running = False
        playback()


@st.dialog(t("Suivi des opérations"), width="large", on_dismiss=close_dialog)
def show_operations():
    if st.session_state.detail:
        review_detail(service, identity)
    elif st.session_state.active_dialog == "search":
        search_page(service)
    else:
        history_page(service, st.session_state.active_dialog)


if not bank_mode():
    analyze_page()
if st.session_state.active_dialog:
    show_operations()


scroll_event = st.session_state.pop("scroll_to_top", None)
if scroll_event:
    st.html(
        f"""
        <script data-scroll-event="{scroll_event}">
            (() => {{
            const scrollToTop = () => {{
                const main = document.querySelector('[data-testid="stMain"]');
                if (main) main.scrollTo({{top: 0, behavior: "instant"}});
                window.scrollTo({{top: 0, behavior: "instant"}});
            }};
            requestAnimationFrame(() => requestAnimationFrame(scrollToTop));
            setTimeout(scrollToTop, 150);
            }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )
