import { locale, observationText, t } from "./language.js";
import { selectRiskFactors, translate } from "./factors.js";

const byId = id => document.getElementById(id);
const form = byId("transaction");
const fieldMenu = byId("field-menu");
const submit = byId("submit");
const reset = byId("reset");
const message = byId("form-message");
const controls = new Map();
const riskBadges = new Map();
const movedFields = new Map();
const promotedFields = byId("promoted-fields");
const promotedGrid = byId("promoted-grid");
const primaryFieldKeys = [
  "amount_usd",
  "cvv_retry_count",
  "auth_method",
  "merchant_risk_score",
  "is_ai_generated_scam_attempt",
  "velocity_score",
];
const percent = new Intl.NumberFormat(locale, { style: "percent", maximumFractionDigits: 2 });
const integer = new Intl.NumberFormat(locale);
const defaults = {
  amount_usd: 125,
  hours_since_last_txn: 12,
  txn_count_last_24h: 1,
  distance_from_home_km: 5,
  card_age_months: 47,
  customer_age: 50,
  account_balance_usd: 3317,
  cvv_retry_count: 0,
  velocity_score: 20,
  time_of_day_hour: 12,
  day_of_week: 3,
  merchant_risk_score: 38,
  prior_disputes: 0,
  is_foreign_transaction: false,
  is_new_merchant: false,
  used_vpn: false,
  ip_country_mismatch: false,
  billing_shipping_mismatch: false,
  is_ai_generated_scam_attempt: false,
  merchant_category: "Groceries",
  card_type: "Visa",
  auth_method: "OTP",
  channel: "POS",
  device_type: "POS Terminal",
};
let metadata;
let revision = 0;

function showMessage(text, info = false) {
  message.textContent = text;
  message.classList.toggle("info", info);
}

function showResult(data) {
  const results = {
    accepted: ["clear", t("Transaction acceptée"), t("Le score est inférieur au seuil de validation.")],
    review: ["review", t("Transaction douteuse"), t("Validez ou refusez cette opération depuis le suivi.")],
    blocked: ["fraud", t("Transaction refusée"), t("Le score atteint le seuil de refus.")],
  };
  const [state, title, note] = results[data?.initialDecision] ??
    ["idle", t("En attente d’analyse"), t("Vérifiez les données de l’opération, puis lancez l’analyse.")];
  byId("verdict").dataset.state = state;
  byId("decision").textContent = title;
  byId("decision-note").textContent = (data && observationText(data)) || note;
  byId("score").textContent = data ? percent.format(data.fraud_score) : "—";
  byId("track").hidden = !data;
  if (data) {
    byId("fill").style.width = (data.fraud_score * 100) + "%";
    byId("track").setAttribute("aria-valuenow", String(data.fraud_score * 100));
    byId("track").setAttribute("aria-valuetext", percent.format(data.fraud_score));
  }
}

function createField(field) {
  const label = document.createElement("label");
  label.className = field.kind === "boolean" ? "field flag" : "field";
  const caption = document.createElement("span");
  caption.textContent = t(field.label);
  const control = document.createElement(field.options ? "select" : "input");
  control.name = field.key;
  if (field.options) {
    control.required = true;
    control.add(new Option(t("Sélectionner"), ""));
    field.options.forEach((option, index) => {
      const value = field.kind === "integer" ? index : option;
      const selected = value === defaults[field.key];
      control.add(new Option(translate(t(option)), value, selected, selected));
    });
  } else if (field.kind === "boolean") {
    control.type = "checkbox";
    control.defaultChecked = defaults[field.key] ?? false;
  } else {
    control.type = "number";
    control.required = true;
    control.step = field.kind === "integer" ? "1" : "any";
    control.min = field.min ?? -1e12;
    control.max = field.max ?? 1e12;
    control.defaultValue = defaults[field.key] ?? "";
  }
  const badge = document.createElement("span");
  badge.className = "risk-badge";
  badge.id = "risk-" + field.key;
  badge.hidden = true;
  const icon = document.createElement("span");
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = "⚠";
  badge.append(icon, " " + t("Facteur de risque"));
  controls.set(field.key, control);
  riskBadges.set(field.key, badge);
  if (field.kind === "boolean") label.append(control, caption);
  else label.append(caption, control);
  label.append(badge);
  return label;
}

function renderFields(groups) {
  const fieldsByKey = new Map(groups.flatMap(group => group.fields.map(field => [field.key, field])));
  byId("primary-fields").replaceChildren(...primaryFieldKeys.map(key => createField(fieldsByKey.get(key))));
  const otherGroups = groups.map(group => ({
    ...group,
    fields: group.fields.filter(field => !primaryFieldKeys.includes(field.key)),
  }));
  byId("fields").replaceChildren(...otherGroups.filter(group => group.fields.length).map((group, index) => {
    const section = document.createElement("details");
    section.open = index === 0;
    const summary = document.createElement("summary");
    const count = document.createElement("span");
    count.className = "field-count";
    count.textContent = group.fields.length;
    summary.append(t(group.title), count);
    const grid = document.createElement("div");
    grid.className = "field-grid";
    grid.append(...group.fields.map(createField));
    section.append(summary, grid);
    return section;
  }));
}

function clearRiskBadges() {
  for (const [key, badge] of riskBadges) {
    badge.hidden = true;
    badge.closest("label").classList.remove("risk-field");
    controls.get(key).removeAttribute("aria-describedby");
  }
}

function updateFieldCounts() {
  for (const grid of byId("fields").querySelectorAll(".field-grid")) {
    const count = grid.childElementCount;
    grid.previousElementSibling.querySelector(".field-count").textContent = count;
    grid.closest("details").hidden = count === 0;
  }
}

function restoreMovedFields() {
  for (const [key, position] of movedFields) position.replaceWith(controls.get(key).closest("label"));
  movedFields.clear();
  promotedFields.hidden = true;
  updateFieldCounts();
}

function showRiskFields(data) {
  clearRiskBadges();
  restoreMovedFields();
  if (data.initialDecision !== "review" && data.initialDecision !== "blocked") return;
  const { selected } = selectRiskFactors(data.factors);
  const keys = [...new Set(selected.flatMap(factor => factor.fields.map(field => field.key)))]
    .filter(key => controls.has(key));
  for (const key of keys) {
    const control = controls.get(key);
    const label = control.closest("label");
    const badge = riskBadges.get(key);
    badge.hidden = false;
    label.classList.add("risk-field");
    control.setAttribute("aria-describedby", badge.id);
    if (fieldMenu.contains(label)) {
      const position = document.createComment("field position");
      label.before(position);
      movedFields.set(key, position);
      promotedGrid.append(label);
    }
  }
  promotedFields.hidden = movedFields.size === 0;
  updateFieldCounts();
}

function revealField(control) {
  const section = control.closest("details");
  if (section) {
    fieldMenu.open = true;
    section.open = true;
  }
}

function clearResult() {
  revision++;
  clearRiskBadges();
  showMessage("");
  showResult();
}

form.noValidate = true;
form.addEventListener("input", clearResult);
reset.addEventListener("click", () => {
  form.reset();
  clearResult();
  restoreMovedFields();
});

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (!metadata) return;
  const invalid = form.querySelector(":invalid:is(input, select)");
  if (invalid) {
    revealField(invalid);
    invalid.reportValidity();
    showMessage(t("Complétez les champs obligatoires et vérifiez leurs valeurs."));
    return;
  }
  const transaction = Object.fromEntries(metadata.groups.flatMap(group => group.fields.map(field => {
    const control = controls.get(field.key);
    if (field.kind === "boolean") return [field.key, control.checked];
    if (field.kind === "category") return [field.key, control.value];
    return [field.key, Number(control.value)];
  })));
  const started = ++revision;
  submit.disabled = true;
  showMessage(t("Analyse en cours…"), true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(transaction),
    });
    const data = await response.json();
    if (started !== revision) return;
    if (!response.ok) {
      const key = Object.keys(data.fields ?? {})[0];
      const control = controls.get(key);
      if (control) {
        revealField(control);
        control.focus();
      }
      showMessage(t(data.fields?.[key] ?? data.error));
      return;
    }
    showMessage("");
    showResult(data);
    showRiskFields(data);
    window.dispatchEvent(new Event("history-changed"));
    if (matchMedia("(max-width: 725px)").matches) byId("result-title").scrollIntoView({ block: "start" });
  } catch {
    if (started === revision) showMessage(t("Le serveur ne répond pas. Réessayez."));
  } finally {
    submit.disabled = false;
  }
});

try {
  const response = await fetch("/api/model");
  if (!response.ok) throw new Error(t("Métadonnées indisponibles."));
  metadata = await response.json();
  if (metadata.bankMode) {
    document.body.classList.add("bank-mode");
    byId("bank-mode-note").hidden = false;
  } else {
    renderFields(metadata.groups);
  }
  const { review, block } = metadata.policy.thresholds;
  byId("threshold").textContent = percent.format(review);
  byId("block-threshold").textContent = block === null ? t("Désactivé") : percent.format(block);
  byId("marker").style.left = (review * 100) + "%";
  const { test, testFrauds } = metadata.metrics;
  const counts = metadata.policy.test;
  const tp = counts.review[1] + counts.blocked[1];
  const fp = counts.review[0] + counts.blocked[0];
  const fn = counts.accepted[1];
  const precision = tp / (tp + fp);
  const recall = tp / (tp + fn);
  const f1 = 2 * tp / (2 * tp + fp + fn);
  byId("test-summary").textContent = locale === "en" ? `${integer.format(test)} test transactions, including ${integer.format(testFrauds)} fraud cases.`
    : `${integer.format(test)} opérations de test, dont ${integer.format(testFrauds)} fraudes.`;
  byId("precision").textContent = percent.format(precision);
  byId("recall").textContent = percent.format(recall);
  byId("f1").textContent = percent.format(f1);
  byId("errors-summary").textContent = locale === "en" ? `${tp} fraud cases flagged, ${fn} accepted, and ${counts.blocked[0]} legitimate transactions declined.`
    : `${tp} fraudes signalées, ${fn} acceptées et ${counts.blocked[0]} opérations légitimes refusées.`;
  for (const button of [submit, reset]) button.disabled = false;
} catch {
  fieldMenu.open = true;
  byId("fields").textContent = t("Le formulaire est indisponible. Vérifiez que le serveur est lancé, puis rechargez la page.");
}
