import { questionFamily, type AnimalProfile, type Question } from "./akinator-kb";
export interface ScoringCandidate {
    animal: AnimalProfile;
    probability: number;
}
export type PolicyOptions = {
    lookahead: number;
    ambiguity: number;
    repetition: number;
    variety: number;
};
export const DEFAULT_POLICY: PolicyOptions = { lookahead: 0.5, ambiguity: 0.25, repetition: 0.025, variety: 0.035 };
export function answerLikelihood(animal: AnimalProfile, questionId: string, answer: "yes" | "no"): number {
    const value = animal.traits[questionId];
    const yes = value === "yes" ? 1 : value === "no" ? 0 : value === "likely" ? 0.78 : value === "unlikely" ? 0.22 : 0.5;
    return answer === "yes" ? yes : 1 - yes;
}
export function normalize(candidates: readonly ScoringCandidate[]): ScoringCandidate[] {
    const total = candidates.reduce((s, c) => s + c.probability, 0);
    return total > 0 ? candidates.filter(c => c.probability > 0).map(c => ({ ...c, probability: c.probability / total })).sort((a, b) => b.probability - a.probability || a.animal.id.localeCompare(b.animal.id)) : [];
}
export function entropy(candidates: readonly ScoringCandidate[]): number {
    return candidates.reduce((sum, c) => sum - (c.probability > 0 ? c.probability * Math.log2(c.probability) : 0), 0);
}
export function simulatePosterior(candidates: readonly ScoringCandidate[], questionId: string, outcome: "yes" | "no"): ScoringCandidate[] {
    return normalize(candidates.map(c => ({ animal: c.animal, probability: c.probability * answerLikelihood(c.animal, questionId, outcome) })));
}
export function informationGain(candidates: readonly ScoringCandidate[], questionId: string): number {
    const pYes = candidates.reduce((s, c) => s + c.probability * answerLikelihood(c.animal, questionId, "yes"), 0);
    return Math.max(0, entropy(candidates) - pYes * entropy(simulatePosterior(candidates, questionId, "yes")) - (1 - pYes) * entropy(simulatePosterior(candidates, questionId, "no")));
}
function seededTie(seed: number, id: string): number {
    let hash = seed >>> 0;
    for (const ch of id) {
        hash ^= ch.charCodeAt(0);
        hash = Math.imul(hash, 16777619);
    }
    hash ^= hash >>> 16;
    hash = Math.imul(hash, 0x45d9f3b);
    hash ^= hash >>> 16;
    return (hash >>> 0) / 4294967296;
}
/** Two actual question levels; random variation is restricted to near-equal utilities. */
export function selectBestQuestion(candidates: readonly ScoringCandidate[], questions: readonly Question[], familyCounts: Record<string, number> = {}, recentFamilies: string[] = [], seed = 0, options: PolicyOptions = DEFAULT_POLICY): string | null {
    if (candidates.length < 2)
        return null;
    const base = questions.map(q => {
        const gain = informationGain(candidates, q.id);
        const vague = candidates.reduce((s, c) => s + (["variable", "na"].includes(c.animal.traits[q.id] ?? "na") ? c.probability : 0), 0);
        const family = questionFamily(q.id);
        return { q, gain, score: gain - options.ambiguity * vague - 0.15 * (1 - q.clarity) - options.repetition * Math.min(familyCounts[family] ?? 0, 3) - (recentFamilies[0] === family ? options.repetition : 0) };
    }).filter(x => x.gain > 0.0001).sort((a, b) => b.score - a.score);
    if (!base.length)
        return null;
    const shortlist = base.slice(0, 6);
    for (const entry of shortlist) {
        if (options.lookahead > 0 && candidates.length > 2) {
            let future = 0;
            for (const outcome of ["yes", "no"] as const) {
                const mass = candidates.reduce((s, c) => s + c.probability * answerLikelihood(c.animal, entry.q.id, outcome), 0);
                if (!mass)
                    continue;
                const branch = simulatePosterior(candidates, entry.q.id, outcome);
                let best = 0;
                for (const next of base)
                    if (next.q.id !== entry.q.id)
                        best = Math.max(best, informationGain(branch, next.q.id));
                future += mass * best;
            }
            entry.score += options.lookahead * future;
        }
    }
    shortlist.sort((a, b) => b.score - a.score);
    const best = shortlist[0].score;
    const near = shortlist.filter(x => best - x.score <= options.variety);
    near.sort((a, b) => seededTie(seed, a.q.id) - seededTie(seed, b.q.id));
    return near[0].q.id;
}
