import { describe, expect, it } from "vitest";
import {
  ANIMAL_SIGNATURES,
  createGameState,
  rankAnimals,
  recordAnswer,
  selectNextTurn,
  type GameState,
} from "../src/lib/server/akinator-engine";
import {
  ANIMALS,
  QUESTIONS,
  traitLikelihood,
} from "../src/lib/server/akinator-knowledge-base";

function playIdealGame(targetId: string, mutate?: (answer: boolean | null, turn: number) => boolean | null) {
  const target = ANIMALS.find((animal) => animal.id === targetId);
  if (!target) throw new Error(`Unknown fixture: ${targetId}`);
  const state = createGameState("en");

  for (let step = 0; step < 30; step += 1) {
    const turn = selectNextTurn(state);
    if (turn.action === "guess") {
      const guessed = ANIMALS.find((animal) => animal.name.en === turn.guess);
      if (guessed?.id === targetId) return { state, questions: state.ledger.evidence.length };
      if (guessed) state.rejected_guesses.push(guessed.id);
      state.turn += 1;
      continue;
    }
    if (turn.action === "give_up") throw new Error(`Engine gave up on ${targetId}`);

    state.last_trait_key = turn.trait_key;
    state.last_question = turn.question;
    state.asked_traits.push(turn.trait_key);
    const probability = traitLikelihood(target, turn.trait_key);
    const idealAnswer = probability >= 0.85 ? true : probability <= 0.15 ? false : null;
    recordAnswer(state, mutate ? mutate(idealAnswer, step) : idealAnswer);
  }
  throw new Error(`Engine did not identify ${targetId}`);
}

describe("Akinator knowledge base", () => {
  it("contains a substantial, unique, animal-only catalog", () => {
    expect(ANIMALS.length).toBeGreaterThanOrEqual(110);
    expect(new Set(ANIMALS.map((animal) => animal.id)).size).toBe(ANIMALS.length);
    expect(new Set(ANIMALS.map((animal) => animal.name.en.toLowerCase())).size).toBe(ANIMALS.length);
    expect(ANIMALS.some((animal) => animal.tags.has("mythical"))).toBe(false);
  });

  it("defines every trait as a localized, unique question", () => {
    expect(new Set(QUESTIONS.map((question) => question.id)).size).toBe(QUESTIONS.length);
    for (const question of QUESTIONS) {
      expect(question.text.en.endsWith("?")).toBe(true);
      expect(question.text.tr.endsWith("?")).toBe(true);
      expect(question.text.ar.endsWith("؟")).toBe(true);
      expect(question.clarity).toBeGreaterThanOrEqual(0.9);
    }
  });

  it("models the duck's defining facts", () => {
    const duck = ANIMALS.find((animal) => animal.id === "duck");
    expect(duck).toBeDefined();
    for (const trait of ["bird", "feathers", "flies", "semi_aquatic", "webbed_feet", "lays_eggs"]) {
      expect(traitLikelihood(duck!, trait)).toBeGreaterThan(0.9);
    }
  });

  it("keeps the compatibility signatures exhaustive", () => {
    expect(ANIMAL_SIGNATURES).toHaveLength(ANIMALS.length);
    expect(Object.keys(ANIMAL_SIGNATURES[0].traits)).toHaveLength(QUESTIONS.length);
  });

  it("does not contain indistinguishable animal profiles", () => {
    const fingerprints = ANIMALS.map((animal) =>
      QUESTIONS.map((question) => traitLikelihood(animal, question.id)).join(","),
    );
    expect(new Set(fingerprints).size).toBe(ANIMALS.length);
  });

  it("keeps size and venom semantics aligned with their question wording", () => {
    const cat = ANIMALS.find((animal) => animal.id === "cat")!;
    const goose = ANIMALS.find((animal) => animal.id === "goose")!;
    const pufferfish = ANIMALS.find((animal) => animal.id === "pufferfish")!;
    expect(traitLikelihood(cat, "tiny")).toBeLessThan(0.1);
    expect(traitLikelihood(goose, "large")).toBeLessThan(0.1);
    expect(traitLikelihood(pufferfish, "venomous")).toBeLessThan(0.1);
  });
});

describe("Akinator probabilistic engine", () => {
  it("maintains a normalized posterior without hard elimination", () => {
    const state = createGameState("en");
    state.ledger.evidence = [
      { trait: "bird", answer: "yes" },
      { trait: "feathers", answer: "no" },
      { trait: "mainly_aquatic", answer: "yes" },
    ];
    const ranked = rankAnimals(state);
    expect(ranked.reduce((sum, item) => sum + item.probability, 0)).toBeCloseTo(1, 10);
    expect(ranked.every((item) => item.probability > 0)).toBe(true);
  });

  it.each([
    "duck",
    "dog",
    "elephant",
    "platypus",
    "axolotl",
    "octopus",
    "chameleon",
    "penguin",
    "tyrannosaurus",
    "firefly",
  ])("identifies %s from truthful answers", (targetId) => {
    const result = playIdealGame(targetId);
    expect(result.questions).toBeLessThanOrEqual(20);
  });

  it("can identify every catalog animal without exhausting the game", () => {
    const questionCounts = ANIMALS.map((animal) => playIdealGame(animal.id).questions);
    expect(Math.max(...questionCounts)).toBeLessThanOrEqual(24);
    expect(questionCounts.reduce((sum, count) => sum + count, 0) / questionCounts.length).toBeLessThan(16);
  });

  it("tolerates an unknown answer and one mistaken answer", () => {
    const result = playIdealGame("duck", (answer, turn) => {
      if (turn === 2 && answer !== null) return !answer;
      if (turn === 4) return null;
      return answer;
    });
    expect(result.questions).toBeLessThanOrEqual(24);
  });

  it("does not ask the same canonical trait twice", () => {
    const state: GameState = createGameState("tr");
    const seen = new Set<string>();
    for (let index = 0; index < 12; index += 1) {
      const turn = selectNextTurn(state);
      if (turn.action !== "ask_question") break;
      expect(seen.has(turn.trait_key)).toBe(false);
      seen.add(turn.trait_key);
      state.last_trait_key = turn.trait_key;
      state.last_question = turn.question;
      state.asked_traits.push(turn.trait_key);
      recordAnswer(state, null);
    }
  });
});
