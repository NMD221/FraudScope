import { locale, observationText, reviewObservationText, t } from "./language.js";
import { renderFactors } from "./factors.js";

const byId = id => document.getElementById(id);
const dialog = byId("suivi");
const categoryNote = byId("category-note");
const history = byId("history");
const play = byId("play");
const pause = byId("pause");
const date = byId("export-date");
const lookup = byId("transaction-lookup");
const lookupInput = byId("transaction-reference");
const suggestions = byId("transaction-suggestions");
const suggestionsStatus = byId("suggestions-status");
const exportControls = date.closest(".exports");
const historyScroll = byId("history-scroll");
const detailPanel = byId("transaction-detail");
const details = byId("transaction-details");
const back = byId("back-to-transactions");
const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
date.value = new Intl.DateTimeFormat("en-CA", { year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date());
const dateTime = new Intl.DateTimeFormat(locale, { dateStyle: "short", timeStyle: "medium" });
const currency = new Intl.NumberFormat(locale, { style: "currency", currency: "USD" });
const percent = new Intl.NumberFormat(locale, { style: "percent", maximumFractionDigits: 2 });
const labels = { accepted: t("Acceptée"), review: t("Douteuse"), blocked: t("Refusée") };
let previous = "";
const titles = { lookup: t("Rechercher une opération"), accepted: t("Opérations courantes"), review: t("Opérations douteuses"), blocked: t("Opérations frauduleuses") };
const notes = {
  accepted: "",
  review: t("Ces opérations attendent une décision après examen."),
  blocked: t("En cas de réclamation, vérifiez l’authenticité de l’opération avant de la valider."),
};
let category;
let activeReference = "";
let activeDetail = "";
let authenticated = false;
let requestId = 0;
let suggestionRequest = 0;
let suggestionTimer;

async function request(path, method = "GET", body) {
  const options = { method };
  if (body) {
    options.headers = { "content-type": "application/json", "x-vigie-action": "review" };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw Error(t(data.error));
  return data;
}

function operatorNote(operatorId, timestamp, verified, action) {
  const note = document.createElement("div");
  note.className = "operator-note";
  note.textContent = (operatorId ? t(verified ? "Agent authentifié" : "Agent déclaré") + (locale === "en" ? ": " : " : ") + operatorId : t("Identifiant d’agent indisponible"))
    + (timestamp ? " · " + dateTime.format(new Date(timestamp)) : "");
  note.prepend(t(action) + " · ");
  return note;
}

function reasonNote(label, value) {
  const note = document.createElement("div");
  note.className = "operator-note";
  const heading = document.createElement("strong");
  heading.textContent = t(label);
  note.append(heading, " — " + (value || t("Motif non renseigné.")));
  return note;
}

function decisionButton(entry, decision, label) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "quiet";
  button.textContent = label;
  button.dataset.entry = entry.id;
  button.dataset.resolution = decision;
  return button;
}

function render(snapshot) {
  const { entries, running, csvAvailable, processed, total, error, canReview } = snapshot;
  authenticated = snapshot.authenticated;
  byId("stream-controls").hidden = !csvAvailable;
  play.disabled = !csvAvailable || running;
  pause.disabled = !running;
  const state = running ? t("en cours") : total && processed === total ? t("terminé") : processed ? t("en pause") : t("prêt");
  const description = t("Lecture du CSV") + " · " + state + (total ? " · " + processed + "/" + total : "");
  if (byId("stream-state").textContent !== description) byId("stream-state").textContent = description;
  byId("stream-error").textContent = error ? t(error) : "";
  const signature = JSON.stringify({ entries, canReview, authenticated });
  if (signature === previous) return;
  previous = signature;
  const focused = document.activeElement;
  const focusSelector = focused?.dataset.entry ? '[data-entry="' + focused.dataset.entry +
    '"][data-resolution="' + focused.dataset.resolution + '"]' : null;
  const openFactors = new Set([...history.querySelectorAll("tr:has(.risk-factors[open])")].map(row => row.dataset.id));
  if (activeDetail && details.querySelector(".risk-factors[open]")) openFactors.add(activeDetail);
  const rows = entries.map(entry => {
    const row = document.createElement("tr");
    row.dataset.id = entry.id;
    for (const value of [dateTime.format(new Date(entry.timestamp)), currency.format(entry.amount),
      percent.format(entry.score), labels[entry.initialDecision]]) {
      row.insertCell().textContent = value;
    }
    if (!activeDetail) {
      const link = document.createElement("button");
      link.type = "button";
      link.className = "quiet transaction-link";
      link.textContent = entry.displayId ?? entry.id;
      link.setAttribute("aria-label", t("Ouvrir le dossier") + " " + link.textContent);
      link.dataset.openTransaction = entry.id;
      row.cells[0].append(link);
    }
    if (entry.originalPayment) {
      const original = document.createElement("small");
      original.className = "operator-note original-amount";
      const { amount, currency: code } = entry.originalPayment;
      const formatted = new Intl.NumberFormat(locale, {
        style: "currency", currency: code, currencyDisplay: "code", maximumFractionDigits: 6,
      }).format(amount);
      original.textContent = t("Montant d’origine") + " · " + formatted;
      row.cells[1].append(original);
    }
    if (entry.bankReference) {
      const reference = document.createElement("div");
      reference.className = "operator-note transaction-id";
      reference.textContent = t("Référence bancaire") + " · " + entry.bankReference;
      row.cells[0].append(reference);
    }
    row.cells[3].dataset.decision = entry.initialDecision;
    const final = row.insertCell();
    if (entry.appeal) {
      final.className = "observation";
      final.textContent = t("Validée après réclamation");
      const refusal = document.createElement("p");
      refusal.textContent = observationText(entry) || t("Refus antérieur. Motif indisponible dans l’historique.");
      const factors = document.createElement("div");
      renderFactors(factors, entry.factors, { open: openFactors.has(entry.id) });
      final.append(refusal);
      if (entry.initialDecision === "review") final.append(operatorNote(entry.operatorId, entry.appeal.refusedAt, entry.operatorVerified, "Refus manuel"));
      final.append(factors);
    } else if (entry.finalDecision === "blocked") {
      const status = document.createElement("div");
      status.textContent = entry.initialDecision === "review" ? t("Refusée après examen") : t("Refusée");
      const action = decisionButton(entry, "approve-appeal", t("Valider après réclamation"));
      action.classList.add("appeal-button");
      final.append(status);
      if (entry.initialDecision === "review") {
        final.append(operatorNote(entry.operatorId, entry.resolvedAt, entry.operatorVerified, "Refus manuel"),
          reasonNote("Motif du refus", entry.refusalReason || observationText(entry)));
      }
      if (canReview && entry.blockReason !== "insufficient_funds") final.append(action);
    } else if (entry.finalDecision === "accepted") {
      final.textContent = entry.initialDecision === "review" ? t("Acceptée après examen") : t("Acceptée");
    } else if (canReview) {
      final.append(decisionButton(entry, "accepted", t("Valider")), decisionButton(entry, "blocked", t("Refuser")));
    } else {
      final.textContent = t("En attente de décision");
    }
    const approval = entry.appeal ?? (entry.initialDecision === "review" && entry.finalDecision === "accepted" ? {
      operatorId: entry.operatorId, operatorVerified: entry.operatorVerified, reason: entry.approvalReason,
    } : null);
    const agentCell = row.insertCell();
    agentCell.className = "validation-cell";
    const reasonCell = row.insertCell();
    reasonCell.className = "validation-cell";
    if (approval) {
      agentCell.append(operatorNote(approval.operatorId, entry.resolvedAt, approval.operatorVerified,
        entry.appeal ? "Validation après réclamation" : "Validation manuelle"));
      reasonCell.textContent = approval.reason || t("Motif non renseigné.");
    } else {
      agentCell.textContent = "—";
      reasonCell.textContent = "—";
    }
    if (activeDetail || category !== "accepted") {
      const observation = row.insertCell();
      observation.className = "observation";
      if (entry.initialDecision === "accepted" || (entry.appeal && entry.initialDecision !== "review")) {
        observation.textContent = "—";
      } else {
        const message = entry.initialDecision === "review"
          ? reasonNote("Motif de mise en attente", reviewObservationText(entry))
          : reasonNote("Motif du blocage automatique", observationText(entry));
        observation.append(message);
        if (entry.blockReason !== "insufficient_funds" && !entry.appeal) {
          const factors = document.createElement("div");
          renderFactors(factors, entry.factors, { open: openFactors.has(entry.id) });
          observation.append(factors);
        }
      }
    }
    return row;
  });
  if (activeDetail && rows.length) {
    const entry = entries[0];
    byId("live-title").textContent = t("Dossier de l’opération") + " · " + (entry.displayId ?? entry.id);
    const columns = ["Heure locale", "Montant", "Score", "Décision initiale", "État final ou validation", "Agent de validation", "Motif de validation", "Observation"];
    details.replaceChildren(...Array.from(rows[0].cells, (cell, index) => {
      const group = document.createElement("div");
      const label = document.createElement("dt");
      label.textContent = t(columns[index]);
      const value = document.createElement("dd");
      value.append(...cell.childNodes);
      group.append(label, value);
      return group;
    }));
  } else if (rows.length) history.replaceChildren(...rows);
  else showTableMessage(category === "lookup" ? "Opération introuvable." : "Aucune opération enregistrée.");
  if (focusSelector) (activeDetail ? details : history).querySelector(focusSelector)?.focus({ preventScroll: true });
}

function showTableMessage(message) {
  const row = document.createElement("tr");
  const cell = row.insertCell();
  cell.colSpan = category === "accepted" ? 7 : 8;
  cell.textContent = t(message);
  history.replaceChildren(row);
}

async function refresh() {
  if (!dialog.open || (!activeDetail && category === "lookup" && !activeReference)) return;
  let path = "/api/history?category=" + category;
  if (activeDetail) path = "/api/history/" + activeDetail;
  else if (category === "lookup") path = "/api/history/lookup?" + new URLSearchParams({ reference: activeReference });
  const id = ++requestId;
  try {
    const snapshot = await request(path);
    if (id === requestId && dialog.open) {
      byId("lookup-status").textContent = !activeDetail && category === "lookup" ? t(snapshot.entries.length ? "Opération trouvée." : "Opération introuvable.") : "";
      render(snapshot);
    }
  } catch (error) {
    if (id === requestId && dialog.open) byId("live-error").textContent = t(error.message);
  }
}

function hideSuggestions() {
  clearTimeout(suggestionTimer);
  suggestionRequest++;
  suggestions.hidden = true;
  suggestions.replaceChildren();
  suggestionsStatus.textContent = "";
}

function renderSuggestions(entries) {
  const items = entries.map(entry => {
    const item = document.createElement("li");
    const action = document.createElement("button");
    action.type = "button";
    action.className = "quiet";
    action.dataset.openTransaction = entry.id;
    action.dataset.lookupReference = entry.displayId;
    action.append(entry.displayId);
    if (entry.bankReference) {
      const reference = document.createElement("small");
      reference.textContent = entry.bankReference;
      action.append(reference);
    }
    item.append(action);
    return item;
  });
  suggestions.replaceChildren(...items);
  suggestions.hidden = !items.length;
  suggestionsStatus.textContent = t(items.length ? "Des opérations correspondent à la saisie." : "Aucune suggestion.");
}

async function refreshSuggestions(prefix, id) {
  try {
    const path = "/api/history/suggestions?" + new URLSearchParams({ reference: prefix });
    const snapshot = await request(path);
    if (id === suggestionRequest && dialog.open && category === "lookup" && !activeDetail) {
      renderSuggestions(snapshot.entries);
    }
  } catch (error) {
    if (id === suggestionRequest && dialog.open) byId("live-error").textContent = t(error.message);
  }
}

async function act(path, button, body) {
  button.disabled = true;
  byId("live-error").textContent = "";
  try {
    await request(path, "POST", body);
    await refresh();
  } catch (error) {
    byId("live-error").textContent = t(error.message);
    button.disabled = false;
  } finally {
    if (button.dataset.entry) button.disabled = false;
  }
}

for (const button of document.querySelectorAll("[data-category]")) {
  button.addEventListener("click", () => {
    hideSuggestions();
    category = button.dataset.category;
    activeReference = "";
    activeDetail = "";
    detailPanel.hidden = true;
    historyScroll.hidden = false;
    previous = "";
    const searching = category === "lookup";
    lookup.hidden = !searching;
    exportControls.hidden = searching;
    byId("lookup-status").textContent = "";
    byId("live-title").textContent = titles[category];
    categoryNote.textContent = notes[category];
    categoryNote.hidden = !notes[category];
    byId("observation-heading").hidden = category === "accepted";
    byId("live-error").textContent = "";
    byId("stream-error").textContent = "";
    byId("stream-controls").hidden = true;
    showTableMessage(searching ? "Renseignez une référence ou un identifiant Vigie." : "Chargement des opérations…");
    dialog.showModal();
    if (searching) lookupInput.focus();
    else void refresh();
  });
}
lookupInput.addEventListener("input", () => {
  hideSuggestions();
  byId("live-error").textContent = "";
  const prefix = lookupInput.value.trim();
  if (!prefix) return;
  const id = suggestionRequest;
  suggestionTimer = setTimeout(() => refreshSuggestions(prefix, id), 140);
});
dialog.addEventListener("keydown", event => {
  if (suggestions.hidden || (event.target !== lookupInput && !suggestions.contains(event.target))) return;
  if (event.key === "Escape") {
    event.preventDefault();
    hideSuggestions();
    lookupInput.focus();
  } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const buttons = [...suggestions.querySelectorAll("button")];
    const index = buttons.indexOf(document.activeElement);
    const next = event.key === "ArrowDown" ? index + 1 : index - 1;
    if (next < 0) lookupInput.focus();
    else buttons[Math.min(next, buttons.length - 1)]?.focus();
  }
});
lookup.addEventListener("submit", event => {
  event.preventDefault();
  hideSuggestions();
  activeReference = lookupInput.value.trim();
  requestId++;
  previous = "";
  byId("live-error").textContent = "";
  byId("lookup-status").textContent = "";
  if (!activeReference) {
    showTableMessage("Renseignez une référence ou un identifiant Vigie.");
    return;
  }
  showTableMessage("Chargement des opérations…");
  void refresh();
});
back.addEventListener("click", () => {
  hideSuggestions();
  activeDetail = "";
  requestId++;
  previous = "";
  detailPanel.hidden = true;
  historyScroll.hidden = false;
  lookup.hidden = category !== "lookup";
  exportControls.hidden = category === "lookup";
  categoryNote.hidden = !notes[category];
  byId("live-title").textContent = titles[category];
  byId("live-error").textContent = "";
  showTableMessage("Chargement des opérations…");
  if (category === "lookup") lookupInput.focus();
  else historyScroll.focus();
  void refresh();
});
dialog.addEventListener("close", () => { requestId++; hideSuggestions(); });

play.addEventListener("click", () => act("/api/simulation/start", play));
pause.addEventListener("click", () => act("/api/simulation/pause", pause));
dialog.addEventListener("click", event => {
  const action = event.target.closest("button");
  if (!action) return;
  const { entry, resolution, openTransaction, lookupReference } = action.dataset;
  if (openTransaction) {
    if (lookupReference) {
      activeReference = lookupReference;
      lookupInput.value = lookupReference;
    }
    hideSuggestions();
    activeDetail = openTransaction;
    requestId++;
    previous = "";
    historyScroll.hidden = true;
    detailPanel.hidden = false;
    lookup.hidden = true;
    exportControls.hidden = true;
    categoryNote.hidden = true;
    byId("stream-controls").hidden = true;
    byId("live-error").textContent = "";
    byId("live-title").textContent = t("Chargement du dossier…");
    details.replaceChildren();
    back.focus();
    void refresh();
    return;
  }
  if (!entry || !resolution) return;
  const path = "/api/history/" + entry + "/" + resolution;
  const questions = {
    "accepted": "Après examen, indiquez le motif de validation (500 caractères maximum).",
    "blocked": "Motif du refus (500 caractères maximum)",
    "approve-appeal": "Après vérification de l’authenticité, indiquez le motif de validation (500 caractères maximum).",
  };
  const reason = window.prompt(t(questions[resolution]));
  if (reason === null) return;
  if (!reason.trim() || reason.trim().length > 500) {
    byId("live-error").textContent = t("Le motif doit contenir entre 1 et 500 caractères.");
    return;
  }
  if (authenticated) {
    void act(path, action, { reason });
    return;
  }
  const operatorId = window.prompt(t("Votre identifiant d’agent (80 caractères maximum)"));
  if (operatorId === null) return;
  if (!operatorId.trim() || operatorId.trim().length > 80) {
    byId("live-error").textContent = t("Renseignez votre identifiant d’agent (1 à 80 caractères).");
    return;
  }
  void act(path, action, { reason, operatorId });
});
for (const button of document.querySelectorAll("[data-export]")) {
  button.addEventListener("click", () => {
    if (!date.reportValidity()) return;
    const params = new URLSearchParams({ date: date.value, period: button.dataset.export, tz: timeZone, category });
    window.location.assign("/api/history/export?" + params);
  });
}
window.addEventListener("history-changed", () => refresh());
setInterval(refresh, 1000);
