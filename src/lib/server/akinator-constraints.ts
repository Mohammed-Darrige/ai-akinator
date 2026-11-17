import type { AnimalProfile, TraitValue } from "./akinator-kb";
export type DirectAnswer = {
    questionId: string;
    answer: "yes" | "no" | "unknown";
};
/** Biological uncertainty is not a player error. Only categorical opposites conflict. */
export function contradicts(value: TraitValue | undefined, answer: DirectAnswer["answer"]): boolean {
    return (value === "yes" && answer === "no") || (value === "no" && answer === "yes");
}
export function matchesAnswers(animal: AnimalProfile, answers: readonly DirectAnswer[]): boolean {
    return answers.every(({ questionId, answer }) => !contradicts(animal.traits[questionId], answer));
}
/** Only directly observed facts are retained. No speculative taxonomic closure. */
export function observedFacts(answers: readonly DirectAnswer[]): Record<string, "yes" | "no"> {
    return Object.fromEntries(answers.filter(a => a.answer !== "unknown").map(a => [a.questionId, a.answer])) as Record<string, "yes" | "no">;
}
