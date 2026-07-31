import { AnimalProfile, EXCLUSIVE_GROUPS, IMPLICATIONS } from "./akinator-kb";

export interface ConstraintState {
  inferred: Record<string, "yes" | "no">;
  eligible: Record<string, boolean>;
}

export function propagateConstraints(
  questionId: string,
  answer: "yes" | "no",
  currentConstraints: ConstraintState,
): ConstraintState {
  const result: ConstraintState = {
    inferred: { ...currentConstraints.inferred },
    eligible: { ...currentConstraints.eligible },
  };

  const queue: Array<{ qId: string; ans: "yes" | "no" }> = [
    { qId: questionId, ans: answer },
  ];

  const visited = new Set<string>();

  while (queue.length > 0) {
    const current = queue.shift()!;
    const key = `${current.qId}:${current.ans}`;
    if (visited.has(key)) continue;
    visited.add(key);

    result.inferred[current.qId] = current.ans;
    result.eligible[current.qId] = false;

    for (const implication of IMPLICATIONS) {
      if (
        implication.when.questionId === current.qId &&
        implication.when.answer === current.ans
      ) {
        for (const consequent of implication.then) {
          const existing = result.inferred[consequent.questionId];
          if (existing && existing !== consequent.answer) {
            continue;
          }
          if (!existing) {
            queue.push({ qId: consequent.questionId, ans: consequent.answer });
          }
        }
      }
    }

    if (current.ans === "yes") {
      for (const group of EXCLUSIVE_GROUPS) {
        if (group.questionIds.includes(current.qId)) {
          for (const otherId of group.questionIds) {
            if (otherId !== current.qId && !result.inferred[otherId]) {
              queue.push({ qId: otherId, ans: "no" });
            }
          }
        }
      }
    }
  }

  return result;
}

export type CandidateTier = "active" | "rescue" | "dead";

export interface TieredCandidate {
  animalIndex: number;
  tier: CandidateTier;
  contradictions: number;
  logPosterior: number;
}

export function tierCandidates(
  animals: readonly AnimalProfile[],
  directAnswers: Array<{ questionId: string; answer: "yes" | "no" | "unknown" }>,
  constraints: ConstraintState,
): TieredCandidate[] {
  const allKnown: Record<string, "yes" | "no"> = { ...constraints.inferred };
  for (const da of directAnswers) {
    if (da.answer !== "unknown") {
      allKnown[da.questionId] = da.answer;
    }
  }

  return animals.map((animal, index) => {
    let contradictions = 0;

    for (const [qId, knownAnswer] of Object.entries(allKnown)) {
      const traitValue = animal.traits[qId];
      if (!traitValue || traitValue === "variable" || traitValue === "na") continue;

      const animalSaysYes = traitValue === "yes" || traitValue === "likely";
      const animalSaysNo = traitValue === "no" || traitValue === "unlikely";

      if (knownAnswer === "yes" && animalSaysNo) contradictions++;
      if (knownAnswer === "no" && animalSaysYes) contradictions++;
    }

    let tier: CandidateTier;
    if (contradictions === 0) tier = "active";
    else if (contradictions === 1) tier = "rescue";
    else tier = "dead";

    return { animalIndex: index, tier, contradictions, logPosterior: 0 };
  });
}
