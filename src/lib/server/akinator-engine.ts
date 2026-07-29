import { createHmac, timingSafeEqual } from "node:crypto";
import {
  ANIMALS,
  QUESTIONS,
  localizedName,
  traitLikelihood,
  type AnimalProfile,
  type Language,
} from "./akinator-knowledge-base";
import { getCustomAnimals } from "./db";

type Message = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type Answer = "yes" | "no" | "unknown";

export type Evidence = {
  trait: string;
  answer: Answer;
};

export type ConstraintLedger = {
  facts: Record<string, boolean>;
  qa_history: Array<[string, string]>;
  evidence: Evidence[];
};

export type GameState = {
  version: 2;
  created_at: number;
  language: Language;
  turn: number;
  ledger: ConstraintLedger;
  conversation: Message[];
  asked_traits: string[];
  trait_labels: Record<string, string>;
  confidence: number;
  pending_guess: string | null;
  last_trait_key: string | null;
  last_question: string | null;
  game_over: boolean;
  awaiting_reveal: boolean;
  rejected_guesses: string[];
};

export type RankedAnimal = {
  animal: AnimalProfile;
  probability: number;
};

export type EngineTurn = {
  action: "ask_question" | "guess" | "give_up";
  question: string;
  trait_key: string;
  guess: string;
  confidence: number;
  candidates_remaining: number;
  db_match_found: boolean;
  split_yes?: number;
  split_no?: number;
  information_gain?: number;
  is_contradiction_overload?: boolean;
};

const ANSWER_RELIABILITY = 0.85;
const MAX_QUESTIONS = 24;
const MAX_REJECTED_GUESSES = 4;
const MAX_TOKEN_BYTES = 48_000;

const ANSWER_LABELS: Record<Language, Record<Answer, string>> = {
  en: { yes: "Yes", no: "No", unknown: "I don't know" },
  tr: { yes: "Evet", no: "Hayır", unknown: "Bilmiyorum" },
  ar: { yes: "نعم", no: "لا", unknown: "لا أعرف" },
};

const GIVE_UP_MESSAGES: Record<Language, string> = {
  en: "Your answers no longer support a reliable animal identification. What animal were you thinking of?",
  tr: "Yanıtların artık güvenilir bir hayvan tahminini desteklemiyor. Hangi hayvanı düşünmüştün?",
  ar: "لم تعد إجاباتك تدعم تحديدا موثوقا للحيوان. ما الحيوان الذي كنت تفكر فيه؟",
};

// Compatibility view for reporting and the post-game endpoint. The engine itself
// uses likelihoods and never treats an omitted trait as a proven negative.
export const ANIMAL_SIGNATURES = ANIMALS.map((animal) => ({
  id: animal.id,
  name: animal.name,
  traits: Object.fromEntries(
    QUESTIONS.map((question) => [question.id, traitLikelihood(animal, question.id) >= 0.85]),
  ) as Record<string, boolean>,
  priority: animal.prior,
}));

export function createGameState(language: Language = "en"): GameState {
  return {
    version: 2,
    created_at: Date.now(),
    language,
    turn: 0,
    ledger: { facts: {}, qa_history: [], evidence: [] },
    conversation: [],
    asked_traits: [],
    trait_labels: {},
    confidence: 0,
    pending_guess: null,
    last_trait_key: null,
    last_question: null,
    game_over: false,
    awaiting_reveal: false,
    rejected_guesses: [],
  };
}

function normalizeLanguage(input: unknown): Language {
  return input === "tr" || input === "ar" ? input : "en";
}

function parseAnswer(value: string): Answer | null {
  const normalized = value.trim().toLocaleLowerCase("tr-TR");
  if (["yes", "evet", "نعم", "اجل", "أجل"].includes(normalized)) return "yes";
  if (["no", "hayır", "hayir", "لا"].includes(normalized)) return "no";
  if (["unknown", "bilmiyorum", "i don't know", "i dont know", "لا أعرف", "لا اعرف"].includes(normalized)) {
    return "unknown";
  }
  return null;
}

function getSessionSecret() {
  const secret = [
    process.env.AKINATOR_SESSION_SECRET,
    process.env.AKINATOR_CEREBRAS_API_KEY,
    process.env.CEREBRAS_API_KEY,
    process.env.AKINATOR_GROQ_API_KEY,
    process.env.GROQ_API_KEY,
    process.env.AKINATOR_SAMBANOVA_API_KEY,
    process.env.SAMBANOVA_API_KEY,
  ].find((candidate) => candidate?.trim().length && candidate.trim().length >= 32)?.trim();
  if (!secret || secret.length < 32) {
    throw new Error("Configure AKINATOR_SESSION_SECRET or a signing-capable Akinator provider credential.");
  }
  return secret;
}

export function hasSessionSigningSecret() {
  try {
    return Boolean(getSessionSecret());
  } catch {
    return false;
  }
}

function signature(payload: string) {
  return createHmac("sha256", getSessionSecret()).update(payload).digest("base64url");
}

function createSessionToken(state: GameState) {
  const compactState: GameState = {
    ...state,
    conversation: state.conversation.slice(-36),
  };
  const payload = Buffer.from(JSON.stringify(compactState)).toString("base64url");
  return `v2.${payload}.${signature(payload)}`;
}

function isSafeState(value: unknown): value is GameState {
  if (!value || typeof value !== "object") return false;
  const state = value as Partial<GameState>;
  return state.version === 2
    && typeof state.created_at === "number"
    && state.created_at <= Date.now() + 60_000
    && state.created_at >= Date.now() - 2 * 60 * 60_000
    && Number.isInteger(state.turn)
    && (state.turn ?? -1) >= 0
    && (state.turn ?? 100) <= 40
    && Array.isArray(state.asked_traits)
    && state.asked_traits.length <= QUESTIONS.length
    && Array.isArray(state.rejected_guesses)
    && state.rejected_guesses.length <= 10
    && Array.isArray(state.ledger?.evidence)
    && state.ledger!.evidence.length <= QUESTIONS.length
    && typeof state.awaiting_reveal === "boolean";
}

export function readSessionToken(token: string | undefined, language: Language) {
  if (!token) return createGameState(language);
  if (Buffer.byteLength(token, "utf8") > MAX_TOKEN_BYTES) throw new Error("Session token is too large.");

  const [version, payload, suppliedSignature] = token.split(".");
  if (version !== "v2" || !payload || !suppliedSignature) throw new Error("Session token is invalid.");

  const expected = Buffer.from(signature(payload));
  const supplied = Buffer.from(suppliedSignature);
  if (expected.length !== supplied.length || !timingSafeEqual(expected, supplied)) {
    throw new Error("Session token signature is invalid.");
  }

  try {
    const decoded = Buffer.from(payload, "base64url");
    if (decoded.byteLength > MAX_TOKEN_BYTES) throw new Error("oversized payload");
    const parsed: unknown = JSON.parse(decoded.toString("utf8"));
    if (!isSafeState(parsed)) throw new Error("invalid state shape");
    return { ...parsed, language };
  } catch {
    throw new Error("Session token could not be decoded.");
  }
}

function responseLikelihood(animal: AnimalProfile, evidence: Evidence) {
  if (evidence.answer === "unknown") return 1;
  const traitProbability = traitLikelihood(animal, evidence.trait);
  const reliableProbability = 0.5 + ANSWER_RELIABILITY * (traitProbability - 0.5);
  return evidence.answer === "yes" ? reliableProbability : 1 - reliableProbability;
}

export function rankAnimals(state: GameState, animals: readonly AnimalProfile[] = ANIMALS): RankedAnimal[] {
  const rejected = new Set(state.rejected_guesses);
  const logScores = animals.map((animal) => {
    let score = Math.log(Math.max(animal.prior, 0.01));
    for (const evidence of state.ledger.evidence) {
      score += Math.log(Math.max(responseLikelihood(animal, evidence), 1e-9));
    }
    if (rejected.has(animal.id)) score += Math.log(1e-9);
    return score;
  });
  const maxScore = Math.max(...logScores);
  const weights = logScores.map((score) => Math.exp(score - maxScore));
  const total = weights.reduce((sum, weight) => sum + weight, 0);

  return animals
    .map((animal, index) => ({ animal, probability: weights[index] / total }))
    .sort((left, right) => right.probability - left.probability);
}

function entropy(distribution: readonly RankedAnimal[]) {
  return distribution.reduce(
    (sum, item) => item.probability > 0 ? sum - item.probability * Math.log2(item.probability) : sum,
    0,
  );
}

function posteriorAfterHypothetical(
  ranked: readonly RankedAnimal[],
  trait: string,
  answer: Exclude<Answer, "unknown">,
) {
  const weighted = ranked.map((item) => ({
    animal: item.animal,
    probability: item.probability * responseLikelihood(item.animal, { trait, answer }),
  }));
  const total = weighted.reduce((sum, item) => sum + item.probability, 0);
  return weighted.map((item) => ({ ...item, probability: item.probability / total }));
}

function scoreQuestions(state: GameState, ranked: readonly RankedAnimal[]) {
  const before = entropy(ranked);
  const asked = new Set(state.asked_traits);

  return QUESTIONS
    .filter((question) => !asked.has(question.id))
    .map((question) => {
      const pYes = ranked.reduce(
        (sum, item) => sum + item.probability * responseLikelihood(item.animal, { trait: question.id, answer: "yes" }),
        0,
      );
      const yesPosterior = posteriorAfterHypothetical(ranked, question.id, "yes");
      const noPosterior = posteriorAfterHypothetical(ranked, question.id, "no");
      const gain = before - pYes * entropy(yesPosterior) - (1 - pYes) * entropy(noPosterior);
      const usefulMass = ranked.reduce(
        (sum, item) => sum + (Math.abs(traitLikelihood(item.animal, question.id) - 0.5) > 0.25 ? item.probability : 0),
        0,
      );
      return {
        question,
        pYes,
        score: gain * question.clarity * usefulMass,
        gain,
      };
    })
    .sort((left, right) => right.score - left.score);
}

function effectiveCandidateCount(ranked: readonly RankedAnimal[]) {
  return 2 ** entropy(ranked);
}

function visibleCandidateCount(ranked: readonly RankedAnimal[]) {
  const floor = Math.max(0.008, (ranked[0]?.probability ?? 0) * 0.08);
  return ranked.filter((item) => item.probability >= floor).length;
}

function shouldGuess(state: GameState, ranked: readonly RankedAnimal[], noQuestionLeft: boolean) {
  const top = ranked[0]?.probability ?? 0;
  const second = ranked[1]?.probability ?? 0;
  const odds = top / Math.max(second, 1e-9);
  const effective = effectiveCandidateCount(ranked);
  const usefulAnswers = state.ledger.evidence.filter((item) => item.answer !== "unknown").length;

  if (noQuestionLeft || state.turn >= MAX_QUESTIONS) return top >= 0.2;
  if (usefulAnswers === 0) return false;
  if (top >= 0.92 && odds >= 7) return true;
  if (top >= 0.82 && odds >= 4 && effective <= 2.2) return true;
  return top >= 0.68 && odds >= 6 && effective <= 1.55;
}

export function selectNextTurn(
  state: GameState,
  animals: readonly AnimalProfile[] = ANIMALS,
): EngineTurn {
  const ranked = rankAnimals(state, animals);
  const questionScores = scoreQuestions(state, ranked);
  const bestQuestion = questionScores[0];
  const noQuestionLeft = !bestQuestion;
  const top = ranked[0];
  const candidatesRemaining = visibleCandidateCount(ranked);

  if (top && shouldGuess(state, ranked, noQuestionLeft)) {
    return {
      action: "guess",
      question: "",
      trait_key: "",
      guess: localizedName(top.animal, state.language),
      confidence: top.probability,
      candidates_remaining: candidatesRemaining,
      db_match_found: true,
    };
  }

  if (!bestQuestion || state.rejected_guesses.length >= MAX_REJECTED_GUESSES) {
    return {
      action: "give_up",
      question: GIVE_UP_MESSAGES[state.language],
      trait_key: "",
      guess: "",
      confidence: top?.probability ?? 0,
      candidates_remaining: candidatesRemaining,
      db_match_found: false,
      is_contradiction_overload: state.ledger.evidence.filter((item) => item.answer !== "unknown").length >= 4,
    };
  }

  return {
    action: "ask_question",
    question: bestQuestion.question.text[state.language],
    trait_key: bestQuestion.question.id,
    guess: "",
    confidence: top?.probability ?? 0,
    candidates_remaining: candidatesRemaining,
    db_match_found: true,
    split_yes: Math.round(bestQuestion.pYes * candidatesRemaining),
    split_no: Math.round((1 - bestQuestion.pYes) * candidatesRemaining),
    information_gain: bestQuestion.gain,
  };
}

export function knownFactValue(facts: Record<string, boolean>, trait: string): boolean | null {
  return typeof facts[trait] === "boolean" ? facts[trait] : null;
}

export function recordAnswer(state: GameState, answer: boolean | null | Answer) {
  const normalized: Answer = typeof answer === "string" ? answer : answer === true ? "yes" : answer === false ? "no" : "unknown";
  if (!state.last_trait_key) return;

  state.ledger.evidence.push({ trait: state.last_trait_key, answer: normalized });
  if (normalized !== "unknown") state.ledger.facts[state.last_trait_key] = normalized === "yes";
  const label = ANSWER_LABELS[state.language][normalized];
  state.ledger.qa_history.push([state.last_question || state.last_trait_key, label]);
  state.trait_labels[state.last_trait_key] = state.last_question || state.last_trait_key;
  state.conversation.push({ role: "user", content: label });
  state.turn += 1;
}

export function tryGenerateLocalTurn(state: GameState, _forceGuess = false): EngineTurn {
  void _forceGuess;
  return selectNextTurn(state);
}

function applyTurnToState(state: GameState, turn: EngineTurn, animals: readonly AnimalProfile[]) {
  state.confidence = turn.confidence;
  state.conversation.push({ role: "assistant", content: JSON.stringify(turn) });
  if (turn.action === "ask_question") {
    state.last_question = turn.question;
    state.last_trait_key = turn.trait_key;
    state.asked_traits.push(turn.trait_key);
  } else if (turn.action === "guess") {
    const guessedAnimal = rankAnimals(state, animals)[0]?.animal;
    state.pending_guess = guessedAnimal?.id ?? null;
  } else {
    state.game_over = true;
    state.awaiting_reveal = true;
    state.last_trait_key = null;
    state.last_question = null;
  }
}

function sse(value: Record<string, unknown>) {
  return `data: ${JSON.stringify(value)}\n\n`;
}

async function runtimeCatalog() {
  const customAnimals = await getCustomAnimals();
  if (customAnimals.length === 0) return ANIMALS;
  const builtInIds = new Set(ANIMALS.map((animal) => animal.id));
  const approvedProfiles: AnimalProfile[] = customAnimals
    .filter((animal) => !builtInIds.has(animal.id))
    .map((animal) => ({
      id: animal.id,
      name: animal.name,
      tags: new Set(Object.entries(animal.traits).filter(([, value]) => value).map(([trait]) => trait)),
      uncertainTags: new Set<string>(),
      prior: 0.35,
    }));
  return [...ANIMALS, ...approvedProfiles];
}

export async function* processAkinatorTurn(
  sessionToken: string | undefined,
  userAnswer?: string,
  inputLanguage: unknown = "en",
) {
  const language = normalizeLanguage(inputLanguage);
  let state: GameState;
  try {
    state = readSessionToken(sessionToken, language);
  } catch (error) {
    yield sse({ type: "error", content: error instanceof Error ? error.message : "Invalid session." });
    return;
  }

  if (state.game_over) {
    yield sse({ type: "error", content: "This session is already complete. Start a new game." });
    return;
  }

  const animals = await runtimeCatalog();

  if (typeof userAnswer !== "undefined") {
    const answer = parseAnswer(userAnswer);
    if (!answer) {
      yield sse({ type: "error", content: "Answer must be yes, no, or unknown." });
      return;
    }

    if (state.pending_guess) {
      if (answer === "unknown") {
        yield sse({ type: "error", content: "Please confirm the guess with yes or no." });
        return;
      }
      if (answer === "yes") {
        const animal = animals.find((candidate) => candidate.id === state.pending_guess);
        state.game_over = true;
        state.awaiting_reveal = false;
        state.pending_guess = null;
        yield sse({ type: "session_id", session_id: createSessionToken(state) });
        yield sse({
          type: "result",
          action: "guess_correct",
          guess: animal ? localizedName(animal, language) : "",
          turn: state.turn,
        });
        return;
      }
      if (!state.rejected_guesses.includes(state.pending_guess)) state.rejected_guesses.push(state.pending_guess);
      state.pending_guess = null;
      state.turn += 1;
    } else {
      recordAnswer(state, answer);
    }
  }

  const turn = selectNextTurn(state, animals);
  applyTurnToState(state, turn, animals);
  yield sse({ type: "session_id", session_id: createSessionToken(state) });
  yield sse({ type: "result", ...turn, turn: state.turn });
}
