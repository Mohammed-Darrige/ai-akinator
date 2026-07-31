import { AnimalProfile, Question, questionFamily, pYes } from "./akinator-kb";

export interface ScoringCandidate {
  animal: AnimalProfile;
  probability: number;
}

/**
 * Inverse Simpson index = 1 / Σ(Pi²)
 * Etkin aday sayısının ölçüsü.
 */
export function inverseSimpson(candidates: readonly ScoringCandidate[]): number {
  const sumSquared = candidates.reduce((sum, c) => sum + c.probability * c.probability, 0);
  return sumSquared > 0 ? 1 / sumSquared : candidates.length;
}

/**
 * Bir yanıt sonrası tahmini posterior dağılımını hesaplar.
 * Gerçek state güncellemesi DEĞİL, sadece simülasyon.
 */
export function simulatePosterior(
  candidates: readonly ScoringCandidate[],
  questionId: string,
  outcome: "yes" | "no",
): ScoringCandidate[] {
  const updated = candidates.map((c) => {
    const p = pYes(c.animal, questionId);
    const likelihood = outcome === "yes" ? p : 1 - p;
    return { animal: c.animal, probability: c.probability * likelihood };
  });
  const total = updated.reduce((sum, c) => sum + c.probability, 0);
  if (total < 1e-15) return updated;
  for (const c of updated) c.probability /= total;
  updated.sort((a, b) => b.probability - a.probability);
  return updated;
}

export function edgeCutScore(
  questionId: string,
  candidates: readonly ScoringCandidate[],
): number {
  let score = 0;
  for (let i = 0; i < candidates.length; i++) {
    for (let j = i + 1; j < candidates.length; j++) {
      const pi = candidates[i].probability;
      const pj = candidates[j].probability;
      const diff = Math.abs(
        pYes(candidates[i].animal, questionId) -
        pYes(candidates[j].animal, questionId)
      );
      score += pi * pj * diff;
    }
  }
  return score;
}

export function marginGainScore(
  questionId: string,
  candidates: readonly ScoringCandidate[],
): number {
  const currentMargin = (candidates[0]?.probability ?? 0) - (candidates[1]?.probability ?? 0);

  let expectedMarginAfter = 0;
  for (const outcome of ["yes", "no"] as const) {
    const pOutcome = candidates.reduce(
      (sum, c) => sum + c.probability * (outcome === "yes" ? pYes(c.animal, questionId) : 1 - pYes(c.animal, questionId)),
      0,
    );
    if (pOutcome < 1e-9) continue;

    const simulated = simulatePosterior(candidates, questionId, outcome);
    const margin = (simulated[0]?.probability ?? 0) - (simulated[1]?.probability ?? 0);
    expectedMarginAfter += pOutcome * margin;
  }

  return Math.max(0, expectedMarginAfter - currentMargin);
}

export function effectiveReductionScore(
  questionId: string,
  candidates: readonly ScoringCandidate[],
): number {
  const currentEffective = inverseSimpson(candidates);

  let expectedEffective = 0;
  for (const outcome of ["yes", "no"] as const) {
    const pOutcome = candidates.reduce(
      (sum, c) => sum + c.probability * (outcome === "yes" ? pYes(c.animal, questionId) : 1 - pYes(c.animal, questionId)),
      0,
    );
    if (pOutcome < 1e-9) continue;
    const simulated = simulatePosterior(candidates, questionId, outcome);
    expectedEffective += pOutcome * inverseSimpson(simulated);
  }

  return currentEffective - expectedEffective;
}

export function redundancyPenalty(
  questionId: string,
  askedFamilies: Record<string, number>,
): number {
  const family = questionFamily(questionId);
  const count = askedFamilies[family] ?? 0;
  return count * 0.35;
}

export function ambiguityPenalty(
  questionId: string,
  candidates: readonly ScoringCandidate[],
): number {
  let vagueWeight = 0;
  for (const c of candidates) {
    const tv = c.animal.traits[questionId];
    if (tv === "variable" || tv === "na") {
      vagueWeight += c.probability;
    }
  }
  return vagueWeight; // 0.0 – 1.0 arası
}

export const SCORING_WEIGHTS = {
  edgeCut:            1.8,
  marginGain:         1.3,
  effectiveReduction: 1.0,
  lookahead:          0.7,  // depth-2 lookahead bonusu
  clarity:            0.4,
  redundancy:        -1.2,
  ambiguity:         -1.0,
  familyRepetition:  -0.8,
} as const;

export function compositeScore(
  questionId: string,
  candidates: readonly ScoringCandidate[],
  askedFamilies: Record<string, number>,
  recentFamilies: string[],
  question: Question,
): number {
  const ec = edgeCutScore(questionId, candidates);
  const mg = marginGainScore(questionId, candidates);
  const er = effectiveReductionScore(questionId, candidates);
  const rd = redundancyPenalty(questionId, askedFamilies);
  const am = ambiguityPenalty(questionId, candidates);

  let seqPenalty = 0;
  const family = questionFamily(questionId);
  if (recentFamilies[0] === family) seqPenalty += 0.6;
  if (recentFamilies[1] === family) seqPenalty += 0.3;

  return (
    SCORING_WEIGHTS.edgeCut * ec +
    SCORING_WEIGHTS.marginGain * mg +
    SCORING_WEIGHTS.effectiveReduction * er +
    SCORING_WEIGHTS.clarity * question.clarity +
    SCORING_WEIGHTS.redundancy * rd +
    SCORING_WEIGHTS.ambiguity * am +
    SCORING_WEIGHTS.familyRepetition * seqPenalty
  );
}

export function selectBestQuestion(
  candidates: readonly ScoringCandidate[],
  eligibleQuestions: readonly Question[],
  askedFamilies: Record<string, number>,
  recentFamilies: string[],
): string {
  const scored = eligibleQuestions.map((q) => ({
    question: q,
    myopicScore: compositeScore(q.id, candidates, askedFamilies, recentFamilies, q),
  }));

  scored.sort((a, b) => b.myopicScore - a.myopicScore);
  const shortlist = scored.slice(0, 8);

  if (candidates.length <= 3 || eligibleQuestions.length <= 1) {
    return shortlist[0].question.id;
  }

  let bestId = shortlist[0].question.id;
  let bestExpectedCost = Infinity;

  for (const entry of shortlist) {
    let expectedCost = 1;

    for (const outcome of ["yes", "no"] as const) {
      const pOutcome = candidates.reduce(
        (sum, c) => sum + c.probability * (outcome === "yes" ? pYes(c.animal, entry.question.id) : 1 - pYes(c.animal, entry.question.id)),
        0,
      );
      if (pOutcome < 1e-9) continue;

      const nextCandidates = simulatePosterior(candidates, entry.question.id, outcome);
      const nextEligible = eligibleQuestions.filter((q) => q.id !== entry.question.id);

      if (nextCandidates[0] && nextCandidates[0].probability >= 0.80) {
        expectedCost += pOutcome * 0;
        continue;
      }

      const updatedFamilies = { ...askedFamilies };
      const family = questionFamily(entry.question.id);
      updatedFamilies[family] = (updatedFamilies[family] ?? 0) + 1;
      const updatedRecent = [family, recentFamilies[0] ?? ""];

      let bestSecondScore = 0;
      for (const q2 of nextEligible.slice(0, 12)) {
        const s = compositeScore(q2.id, nextCandidates, updatedFamilies, updatedRecent, q2);
        if (s > bestSecondScore) bestSecondScore = s;
      }

      const effectiveCount = inverseSimpson(nextCandidates);
      const heuristicRemaining = Math.max(0, Math.log2(effectiveCount));

      expectedCost += pOutcome * (1 + heuristicRemaining * 0.7);
    }

    if (expectedCost < bestExpectedCost) {
      bestExpectedCost = expectedCost;
      bestId = entry.question.id;
    }
  }

  return bestId;
}
