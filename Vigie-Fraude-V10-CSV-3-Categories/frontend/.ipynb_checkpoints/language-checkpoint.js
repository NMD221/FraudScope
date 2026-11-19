import { english } from "./translations.js";

export const locale = document.documentElement.lang === "en" ? "en" : "fr";

export function t(text) {
  return locale === "en" && Object.hasOwn(english, text) ? english[text] : text;
}

const percent = new Intl.NumberFormat(locale, { style: "percent", maximumFractionDigits: 2 });
const manualPrefix = "Refus après examen. ";

export function observationText(entry) {
  if (entry.blockReason === "insufficient_funds") return t("Solde insuffisant. Le montant dépasse le solde du compte.");
  if (entry.initialDecision === "accepted") return "";
  const manualDecline = entry.initialDecision === "review" && (entry.finalDecision === "blocked" || entry.appeal);
  if (locale === "fr" && entry.observation) return entry.observation;
  if (manualDecline) {
    const archivedReason = entry.observation?.startsWith(manualPrefix) ? entry.observation.slice(manualPrefix.length) : "";
    const reason = entry.refusalReason || (archivedReason === "Motif non renseigné." ? "" : archivedReason);
    return t("Refus après examen.") + " " + (reason || t("Motif non renseigné."));
  }
  const blocked = entry.initialDecision === "blocked";
  const archivedThreshold = entry.observation?.match(/de ([\d,]+)\s*%\.$/)?.[1];
  const threshold = entry.decisionThreshold ?? (archivedThreshold && Number(archivedThreshold.replace(",", ".")) / 100);
  if (locale === "en" && Number.isFinite(threshold)) {
    const action = blocked ? "Automatically declined" : "Flagged for review";
    return action + ". The score of " + percent.format(entry.score)
      + " reached the " + (blocked ? "decline" : "review") + " threshold of " + percent.format(threshold) + ".";
  }
  return t(blocked ? "Le score a atteint le seuil de refus." : "Le score a atteint le seuil de vérification.");
}


export function reviewObservationText(entry) {
  if (entry.initialDecision !== "review") return "";
  const reviewedAndDeclined = entry.finalDecision === "blocked" || entry.appeal;
  return observationText({
    ...entry, finalDecision: null, appeal: null,
    observation: reviewedAndDeclined ? entry.reviewObservation : entry.observation,
  });
}
