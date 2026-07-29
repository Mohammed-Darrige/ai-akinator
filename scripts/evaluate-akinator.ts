import {
  createGameState,
  rankAnimals,
  recordAnswer,
  selectNextTurn,
} from "../src/lib/server/akinator-engine";
import {
  ANIMALS,
  traitLikelihood,
} from "../src/lib/server/akinator-knowledge-base";

const scenarios = [
  { variant: "baseline", difficulty: "easy", targetId: "dog", overrides: {} },
  { variant: "baseline", difficulty: "easy", targetId: "duck", overrides: {} },
  { variant: "baseline", difficulty: "medium", targetId: "platypus", overrides: {} },
  { variant: "baseline", difficulty: "medium", targetId: "chameleon", overrides: {} },
  { variant: "baseline", difficulty: "hard", targetId: "axolotl", overrides: {} },
  { variant: "human-uncertainty", difficulty: "easy", targetId: "dog", overrides: { omnivore: "unknown" } },
  { variant: "human-uncertainty", difficulty: "easy", targetId: "duck", overrides: { domesticated: "unknown" } },
  { variant: "human-uncertainty", difficulty: "medium", targetId: "platypus", overrides: { venomous: "unknown" } },
  { variant: "human-uncertainty", difficulty: "medium", targetId: "chameleon", overrides: { tiny: "unknown" } },
  { variant: "human-uncertainty", difficulty: "hard", targetId: "axolotl", overrides: { regenerates_limbs: "unknown" } },
  { variant: "one-wrong-answer", difficulty: "easy", targetId: "dog", overrides: { domesticated: "no" } },
  { variant: "one-wrong-answer", difficulty: "easy", targetId: "duck", overrides: { webbed_feet: "no" } },
  { variant: "one-wrong-answer", difficulty: "medium", targetId: "platypus", overrides: { lays_eggs: "no" } },
  { variant: "one-wrong-answer", difficulty: "medium", targetId: "chameleon", overrides: { color_change: "no" } },
  { variant: "one-wrong-answer", difficulty: "hard", targetId: "axolotl", overrides: { mainly_aquatic: "no" } },
] as const;

const results = scenarios.map(({ variant, difficulty, targetId, overrides }) => {
  const target = ANIMALS.find((animal) => animal.id === targetId);
  if (!target) throw new Error(`Target not found: ${targetId}`);

  const state = createGameState("en");
  const transcript: Array<Record<string, unknown>> = [];
  const guesses: string[] = [];

  for (let eventIndex = 0; eventIndex < 40; eventIndex += 1) {
    const turn = selectNextTurn(state);
    if (turn.action === "give_up") {
      transcript.push({ event: "give_up", confidence: turn.confidence });
      break;
    }

    if (turn.action === "guess") {
      const guessed = ANIMALS.find((animal) => animal.name.en === turn.guess);
      guesses.push(turn.guess);
      transcript.push({
        event: "guess",
        guess: turn.guess,
        confidence: Number(turn.confidence.toFixed(4)),
        correct: guessed?.id === targetId,
      });
      if (guessed?.id === targetId) break;
      if (guessed) state.rejected_guesses.push(guessed.id);
      state.turn += 1;
      continue;
    }

    const likelihood = traitLikelihood(target, turn.trait_key);
    const override = (overrides as Record<string, string>)[turn.trait_key];
    const answer = override === "yes"
      ? true
      : override === "no"
        ? false
        : override === "unknown"
          ? null
          : likelihood >= 0.85
            ? true
            : likelihood <= 0.15
              ? false
              : null;
    state.last_trait_key = turn.trait_key;
    state.last_question = turn.question;
    state.asked_traits.push(turn.trait_key);
    recordAnswer(state, answer);
    const top = rankAnimals(state).slice(0, 3).map((candidate) => ({
      animal: candidate.animal.name.en,
      probability: Number(candidate.probability.toFixed(4)),
    }));
    transcript.push({
      event: "question",
      number: state.ledger.evidence.length,
      trait: turn.trait_key,
      question: turn.question,
      answer: answer === true ? "yes" : answer === false ? "no" : "unknown",
      informationGain: Number((turn.information_gain ?? 0).toFixed(4)),
      top,
    });
  }

  return {
    variant,
    difficulty,
    target: target.name.en,
    questions: state.ledger.evidence.length,
    guesses,
    firstGuessCorrect: guesses[0]?.toLocaleLowerCase() === target.name.en.toLocaleLowerCase(),
    wrongGuesses: guesses.filter((guess) => guess.toLocaleLowerCase() !== target.name.en.toLocaleLowerCase()).length,
    success: guesses.some((guess) => guess.toLocaleLowerCase() === target.name.en.toLocaleLowerCase()),
    transcript,
  };
});

console.log(JSON.stringify(results, null, 2));
