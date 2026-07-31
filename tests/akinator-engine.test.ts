process.env.AKINATOR_SESSION_SECRET = "0123456789abcdef0123456789abcdef";
import { describe, expect, it } from "vitest";
import {
  ANIMAL_SIGNATURES,
  processAkinatorTurn,
  type EngineTurn,
} from "../src/lib/server/akinator-engine";
import {
  ANIMALS,
  QUESTIONS,
  pYes,
} from "../src/lib/server/akinator-kb";

async function playIdealGame(targetId: string, mutate?: (answer: boolean | null, turn: number) => boolean | null) {
  const target = ANIMALS.find((animal) => animal.id === targetId);
  if (!target) throw new Error(`Unknown fixture: ${targetId}`);
  
  let sessionToken: string | undefined = undefined;
  let questions = 0;
  let userAnswer: string | undefined = undefined;

  for (let step = 0; step < 30; step += 1) {
    let turnResult: EngineTurn | undefined = undefined;
    const iterator = processAkinatorTurn(sessionToken, userAnswer, "en");
    for await (const chunk of iterator) {
      if (!chunk.startsWith("data: ")) continue;
      const parsed = JSON.parse(chunk.replace("data: ", ""));
      if (parsed.type === "session_id") {
        sessionToken = parsed.session_id;
      } else if (parsed.type === "result") {
        turnResult = parsed;
      } else if (parsed.type === "error") {
        throw new Error(parsed.content);
      }
    }

    if (!turnResult) throw new Error("No turn result yielded");

    if (turnResult.action === "guess") {
      const guessed = ANIMALS.find((animal) => animal.name.en === turnResult?.guess);
      if (guessed?.id === targetId) return { questions };
      userAnswer = "no"; 
      continue;
    }
    
    if (turnResult.action === "guess_correct") {
       return { questions };
    }

    if (turnResult.action === "give_up") throw new Error(`Engine gave up on ${targetId}`);

    questions++;
    const probability = pYes(target, turnResult.trait_key);
    let idealAnswer: boolean | null = probability >= 0.85 ? true : probability <= 0.15 ? false : null;
    if (mutate) {
      idealAnswer = mutate(idealAnswer, step);
    }
    userAnswer = idealAnswer === true ? "yes" : idealAnswer === false ? "no" : "unknown";
  }
  throw new Error(`Engine did not identify ${targetId} within 30 turns`);
}

describe("Akinator knowledge base", () => {
  it("contains a substantial, unique, animal-only catalog", () => {
    expect(ANIMALS.length).toBeGreaterThanOrEqual(110);
    expect(new Set(ANIMALS.map((animal) => animal.id)).size).toBe(ANIMALS.length);
    expect(new Set(ANIMALS.map((animal) => animal.name.en.toLowerCase())).size).toBe(ANIMALS.length);
    expect(ANIMALS.some((animal) => animal.tags?.has("mythical"))).toBe(false);
  });

  it("defines every trait as a localized, unique question", () => {
    expect(new Set(QUESTIONS.map((question) => question.id)).size).toBe(QUESTIONS.length);
    for (const question of QUESTIONS) {
      expect(question.text.en.endsWith("?")).toBe(true);
      expect(question.text.tr.endsWith("?")).toBe(true);
      expect(question.text.ar.endsWith("؟")).toBe(true);
    }
  });

  it("keeps the compatibility signatures exhaustive", () => {
    expect(ANIMAL_SIGNATURES).toHaveLength(ANIMALS.length);
    expect(Object.keys(ANIMAL_SIGNATURES[0].traits)).toHaveLength(QUESTIONS.length);
  });

  it("does not contain indistinguishable animal profiles", () => {
    const fingerprints = ANIMALS.map((animal) =>
      QUESTIONS.map((question) => pYes(animal, question.id)).join(","),
    );
    expect(new Set(fingerprints).size).toBe(ANIMALS.length);
  });

  it("keeps size and venom semantics aligned with their question wording", () => {
    const cat = ANIMALS.find((animal) => animal.id === "cat")!;
    const goose = ANIMALS.find((animal) => animal.id === "goose")!;
    const pufferfish = ANIMALS.find((animal) => animal.id === "pufferfish")!;
    expect(pYes(cat, "tiny")).toBeLessThan(0.1);
    expect(pYes(goose, "large")).toBeLessThan(0.1);
    expect(pYes(pufferfish, "venomous")).toBeLessThan(0.1);
  });
});

describe("Akinator probabilistic engine", () => {
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
  ])("identifies %s from truthful answers", async (targetId) => {
    const result = await playIdealGame(targetId);
    expect(result.questions).toBeLessThanOrEqual(20);
  });

  it("can identify almost every catalog animal without exhausting the game", async () => {
    let maxQuestions = 0;
    let sumQuestions = 0;
    let failures = 0;
    for (const animal of ANIMALS) {
       try {
         const result = await playIdealGame(animal.id);
         maxQuestions = Math.max(maxQuestions, result.questions);
         sumQuestions += result.questions;
       } catch (e) {
         failures++;
       }
    }
    expect(failures).toBeLessThanOrEqual(3);
    expect(maxQuestions).toBeLessThanOrEqual(24);
    expect(sumQuestions / (ANIMALS.length - failures)).toBeLessThan(16);
  }, 30_000);

  it("tolerates an unknown answer and one mistaken answer", async () => {
    const result = await playIdealGame("duck", (answer, turn) => {
      if (turn === 2 && answer !== null) return !answer;
      if (turn === 4) return null;
      return answer;
    });
    expect(result.questions).toBeLessThanOrEqual(24);
  });
});
