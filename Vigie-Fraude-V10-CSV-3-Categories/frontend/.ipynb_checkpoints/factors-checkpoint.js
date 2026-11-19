import { locale, t } from "./language.js";

const translated = {
  "Crypto Exchange": "Plateforme crypto", Electronics: "Électronique", Fuel: "Carburant",
  Gaming: "Jeux vidéo", "Gift Cards": "Cartes-cadeaux", Groceries: "Alimentation",
  Healthcare: "Santé", "Online Retail": "Commerce en ligne", Travel: "Voyage",
  Utilities: "Services publics", Biometric: "Biométrie", "No Authentication": "Aucune authentification",
  OTP: "Code à usage unique (OTP)", PIN: "Code PIN", ATM: "Distributeur automatique",
  Contactless: "Sans contact", "In-App": "Dans l’application", Online: "En ligne",
  POS: "Point de vente", "ATM Machine": "Distributeur automatique", "Android Phone": "Téléphone Android",
  "POS Terminal": "Terminal de paiement", "Smart Watch": "Montre connectée", Tablet: "Tablette",
  "Windows PC": "PC Windows",
};

export function translate(value) {
  return locale === "fr" && Object.hasOwn(translated, value) ? translated[value] : value;
}

function formatValue(value) {
  if (typeof value === "boolean") return value ? t("Oui") : t("Non");
  if (typeof value === "number") return new Intl.NumberFormat(locale, { maximumSignificantDigits: 15 }).format(value);
  return translate(t(value));
}

const targetCoverage = 0.9;
const coverageTolerance = 1e-12;
const percent = new Intl.NumberFormat(locale, { style: "percent", maximumFractionDigits: 1 });

export function selectRiskFactors(factors = []) {
  if (!factors.length || !factors.every(factor => Number.isFinite(factor.share) && factor.share > 0)) {
    return { selected: factors, coverage: null };
  }
  let coverage = 0;
  const selected = [];
  for (const factor of factors) {
    selected.push(factor);
    coverage += factor.share;
    if (coverage + coverageTolerance >= targetCoverage) break;
  }
  return { selected, coverage: Math.min(1, coverage) };
}

export function renderFactors(container, factors, { open = false } = {}) {
  if (factors === undefined || !factors.length) {
    const note = document.createElement("p");
    note.textContent = factors === undefined ? t("Champs à l’origine du score indisponibles pour cette ancienne opération.")
      : t("Aucun champ n’a relevé le score par rapport aux valeurs de référence du modèle.");
    container.replaceChildren(note);
    return;
  }
  const details = document.createElement("details");
  details.className = "risk-factors";
  details.open = open;
  const summary = document.createElement("summary");
  summary.textContent = t("Champs ayant augmenté le score");
  const { selected, coverage } = selectRiskFactors(factors);
  const list = document.createElement("ul");
  const separator = locale === "en" ? ": " : " : ";
  for (const factor of selected) {
    const item = document.createElement("li");
    const fields = factor.fields.map(field => t(field.label) + separator + formatValue(field.value));
    const label = factor.label && (factor.value === undefined ? t(factor.label) : t(factor.label) + separator + formatValue(factor.value));
    item.textContent = label && fields.length ? label + " — " + fields.join(" · ") : label || fields[0];
    if (coverage !== null) item.append(" · " + percent.format(factor.share));
    list.append(item);
  }
  const note = document.createElement("p");
  note.textContent = coverage === null
    ? t("Contributions du modèle, de la plus forte à la plus faible par rapport à ses valeurs de référence.")
    : t("Ces facteurs couvrent {coverage} des contributions qui ont augmenté le score.")
      .replace("{coverage}", percent.format(coverage));
  details.append(summary, list, note);
  container.replaceChildren(details);
}
