import { createHmac, timingSafeEqual } from "node:crypto";
import {
  ANIMALS,
  QUESTIONS,
  localizedName,
  pYes,
  traitConfidence,
  questionFamily,
  OPENING_BOOK,
  type AnimalProfile,
  type Language,
  type Question,
  type TraitValue,
} from "./akinator-kb";
import {
  type ConstraintState,
  propagateConstraints,
  tierCandidates,
} from "./akinator-constraints";
import {
  type ScoringCandidate,
  selectBestQuestion,
  inverseSimpson,
} from "./akinator-scoring";
import { getCustomAnimals } from "./db";

export type Answer = "yes" | "no" | "unknown";

export type Evidence = {
  trait: string;
  answer: Answer;
};

export type GameState = {
  version: 3;
  created_at: number;
  language: Language;
  turn: number;
  directAnswers: Array<{ questionId: string; answer: "yes" | "no" | "unknown" }>;
  inferredAnswers: Record<string, "yes" | "no">;
  askedQuestions: string[];
  familyCounts: Record<string, number>;
  recentFamilies: [string, string];
  qaHistory: Array<[string, string]>;
  mode: "EXPLORE" | "CONFIRM" | "GUESS";
  confidence: number;
  pendingGuess: string | null;
  lastTraitKey: string | null;
  lastQuestion: string | null;
  gameOver: boolean;
  awaitingReveal: boolean;
  rejectedGuesses: string[];
  sessionReliability: number;
  conversation: Array<{ role: string; content: string }>;
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
  top_candidates?: Array<{ id: string; name: string; prob: number }>;
};

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

const MAX_QUESTIONS = 24;
const MAX_REJECTED_GUESSES = 3;
const MAX_TOKEN_BYTES = 48_000;
const INITIAL_SESSION_RELIABILITY = 0.92;
const CONTRADICTION_PENALTY = 0.08;
const MIN_SESSION_RELIABILITY = 0.55;
const MAX_SESSION_RELIABILITY = 0.95;

export const ANIMAL_SIGNATURES = ANIMALS.map((animal) => ({
  id: animal.id,
  name: animal.name,
  traits: Object.fromEntries(
    Object.entries(animal.traits).map(([key, val]) => [key, val === "yes" || val === "likely"])
  ) as Record<string, boolean>,
  priority: animal.prior,
}));

export function createGameState(language: Language = "en"): GameState {
  return {
    version: 3,
    created_at: Date.now(),
    language,
    turn: 0,
    directAnswers: [],
    inferredAnswers: {},
    askedQuestions: [],
    familyCounts: {},
    recentFamilies: ["", ""],
    qaHistory: [],
    mode: "EXPLORE",
    confidence: 0,
    pendingGuess: null,
    lastTraitKey: null,
    lastQuestion: null,
    gameOver: false,
    awaitingReveal: false,
    rejectedGuesses: [],
    sessionReliability: INITIAL_SESSION_RELIABILITY,
    conversation: [],
  };
}

function normalizeLanguage(input: unknown): Language {
  return input === "tr" || input === "ar" ? input : "en";
}

function parseAnswer(value: string): Answer | null {
  const normalized = value.trim().toLocaleLowerCase("tr-TR");
  if (["yes", "evet", "نعم", "اجل", "أجل"].includes(normalized)) return "yes";
  if (["no", "hayır", "hayir", "لا"].includes(normalized)) return "no";
  if (["unknown", "bilmiyorum", "i don't know", "i dont know", "لا أعرف", "لا اعرف"].includes(normalized)) return "unknown";
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
  return `v3.${payload}.${signature(payload)}`;
}

function isSafeState(value: unknown): value is GameState {
  if (!value || typeof value !== "object") return false;
  const state = value as Partial<GameState>;
  return state.version === 3
    && typeof state.created_at === "number"
    && typeof state.turn === "number";
}

export function readSessionToken(token: string | undefined, language: Language) {
  if (!token) return createGameState(language);
  if (Buffer.byteLength(token, "utf8") > MAX_TOKEN_BYTES) throw new Error("Session token is too large.");

  const [version, payload, suppliedSignature] = token.split(".");
  if (version !== "v3" || !payload || !suppliedSignature) throw new Error("Session token is invalid.");

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

export interface EvidenceItem {
  questionId: string;
  answer: "yes" | "no" | "unknown";
  questionReliability: number;
}

export function updatePosterior(
  animals: readonly AnimalProfile[],
  evidence: readonly EvidenceItem[],
  familyCounts: Record<string, number>,
): ScoringCandidate[] {
  const logScores = animals.map((animal) => {
    let logScore = Math.log(Math.max(animal.prior, 0.01));

    for (const ev of evidence) {
      if (ev.answer === "unknown") continue;

      const p = pYes(animal, ev.questionId);
      const confidence = traitConfidence(animal, ev.questionId);

      const reliability = ev.questionReliability * confidence;
      const adjustedP = 0.5 + reliability * (p - 0.5);
      const likelihood = ev.answer === "yes" ? adjustedP : 1 - adjustedP;

      const family = questionFamily(ev.questionId);
      const familyCount = familyCounts[family] ?? 0;
      const discount = 1 / (1 + 0.4 * familyCount);

      logScore += discount * Math.log(Math.max(likelihood, 1e-12));
    }

    return logScore;
  });

  const maxLog = Math.max(...logScores);
  const weights = logScores.map((s) => Math.exp(s - maxLog));
  const total = weights.reduce((sum, w) => sum + w, 0);

  const result = animals.map((animal, i) => ({
    animal,
    probability: weights[i] / total,
  }));

  result.sort((a, b) => b.probability - a.probability);
  return result;
}

export type EngineMode = "EXPLORE" | "CONFIRM" | "GUESS";

export function determineMode(
  candidates: readonly ScoringCandidate[],
  turn: number,
  rejectedGuesses: number,
): EngineMode {
  if (candidates.length === 0) return "GUESS"; 

  const p1 = candidates[0]?.probability ?? 0;
  const p2 = candidates[1]?.probability ?? 0;
  const margin = p1 - p2;
  const odds = p1 / Math.max(p2, 1e-9);
  const effective = inverseSimpson(candidates);

  if (p1 >= 0.88 && odds >= 6) return "GUESS";
  if (p1 >= 0.75 && odds >= 4 && effective <= 2.5) return "GUESS";
  if (p1 >= 0.62 && odds >= 5 && effective <= 1.8) return "GUESS";
  if (turn >= 10 && p1 >= 0.55 && margin >= 0.25) return "GUESS";
  if (turn >= 15 && p1 >= 0.40) return "GUESS";

  if (p1 >= 0.45 && odds >= 2.5) return "CONFIRM";
  if (effective <= 5 && p1 >= 0.30) return "CONFIRM";

  return "EXPLORE";
}

export function selectConfirmationQuestion(
  leader: ScoringCandidate,
  rivals: readonly ScoringCandidate[],
  eligibleQuestions: readonly Question[],
): string | null {
  const topRivals = rivals.slice(0, 5);

  let bestId: string | null = null;
  let bestScore = -1;

  for (const q of eligibleQuestions) {
    let score = 0;
    for (const rival of topRivals) {
      const diff = Math.abs(
        pYes(leader.animal, q.id) - pYes(rival.animal, q.id)
      );
      score += rival.probability * diff;
    }

    const leaderTrait = leader.animal.traits[q.id];
    if (leaderTrait === "variable" || leaderTrait === "na") {
      score *= 0.3; 
    }

    if (score > bestScore) {
      bestScore = score;
      bestId = q.id;
    }
  }

  return bestScore > 0.05 ? bestId : null;
}

export function findSilverBullet(
  leader: ScoringCandidate,
  otherActives: readonly ScoringCandidate[],
  eligibleQuestions: readonly Question[],
): string | null {
  for (const q of eligibleQuestions) {
    const leaderP = pYes(leader.animal, q.id);
    if (leaderP < 0.85 && leaderP > 0.15) continue; 

    const leaderSaysYes = leaderP >= 0.85;

    let isBullet = true;
    for (const other of otherActives) {
      const otherP = pYes(other.animal, q.id);
      if (leaderSaysYes && otherP >= 0.40) { isBullet = false; break; }
      if (!leaderSaysYes && otherP <= 0.60) { isBullet = false; break; }
    }

    if (isBullet) return q.id;
  }
  return null;
}

export function openingBookQuestion(state: GameState): string | null {
  if (state.turn === 0) return OPENING_BOOK["start"]?.[0] ?? null;

  if (state.turn === 1 && state.directAnswers.length === 1) {
    const firstAnswer = state.directAnswers[0];
    const key = `${firstAnswer.questionId}:${firstAnswer.answer}`;
    return OPENING_BOOK[key]?.[0] ?? null;
  }

  return null; 
}

function detectContradiction(
  newInferred: Record<string, "yes" | "no">,
  oldInferred: Record<string, "yes" | "no">,
): boolean {
  for (const [key, value] of Object.entries(newInferred)) {
    if (oldInferred[key] && oldInferred[key] !== value) return true;
  }
  return false;
}

function updateSessionReliability(state: GameState, hadContradiction: boolean): void {
  if (hadContradiction) {
    state.sessionReliability = Math.max(
      MIN_SESSION_RELIABILITY,
      state.sessionReliability - CONTRADICTION_PENALTY,
    );
  } else {
    state.sessionReliability = Math.min(
      MAX_SESSION_RELIABILITY,
      state.sessionReliability + 0.01,
    );
  }
}

async function runtimeCatalog(): Promise<AnimalProfile[]> {
  const customAnimals = await getCustomAnimals();
  if (customAnimals.length === 0) return ANIMALS as unknown as AnimalProfile[];
  const builtInIds = new Set(ANIMALS.map((animal) => animal.id));
  const approvedProfiles: AnimalProfile[] = customAnimals
    .filter((animal) => !builtInIds.has(animal.id))
    .map((animal) => {
      const traits: Record<string, TraitValue> = {};
      for (const q of QUESTIONS) {
        traits[q.id] = animal.traits[q.id] ? "yes" : "no";
      }
      return {
        id: animal.id,
        name: animal.name,
        traits,
        prior: 0.35,
      };
    });
  return [...ANIMALS, ...approvedProfiles];
}

function sse(value: Record<string, unknown>) {
  return `data: ${JSON.stringify(value)}\n\n`;
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

  if (state.gameOver) {
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

    if (state.pendingGuess) {
      if (answer === "unknown") {
        yield sse({ type: "error", content: "Please confirm the guess with yes or no." });
        return;
      }
      if (answer === "yes") {
        const animal = animals.find((candidate) => candidate.id === state.pendingGuess);
        state.gameOver = true;
        state.awaitingReveal = false;
        state.pendingGuess = null;
        yield sse({ type: "session_id", session_id: createSessionToken(state) });
        yield sse({
          type: "result",
          action: "guess_correct",
          guess: animal ? localizedName(animal, language) : "",
          turn: state.turn,
        });
        return;
      }
      if (!state.rejectedGuesses.includes(state.pendingGuess)) {
        state.rejectedGuesses.push(state.pendingGuess);
      }
      state.pendingGuess = null;
      state.turn += 1;
    } else {
      if (state.lastTraitKey) {
        state.directAnswers.push({ questionId: state.lastTraitKey, answer });
        const label = ANSWER_LABELS[state.language][answer];
        state.qaHistory.push([state.lastQuestion || state.lastTraitKey, label]);
        state.conversation.push({ role: "user", content: label });
        
        if (answer !== "unknown") {
          const oldInferred = { ...state.inferredAnswers };
          const cState: ConstraintState = { inferred: state.inferredAnswers, eligible: {} };
          const newState = propagateConstraints(state.lastTraitKey, answer, cState);
          state.inferredAnswers = newState.inferred;
          const hadContradiction = detectContradiction(newState.inferred, oldInferred);
          updateSessionReliability(state, hadContradiction);
        }
        
        state.turn += 1;
      }
    }
  }

  const evidence: EvidenceItem[] = [];
  for (const da of state.directAnswers) {
    const q = QUESTIONS.find(q => q.id === da.questionId);
    evidence.push({
      questionId: da.questionId,
      answer: da.answer,
      questionReliability: q ? (q.phase === "broad" ? 0.95 : 0.90) : 0.90,
    });
  }
  for (const [qId, ans] of Object.entries(state.inferredAnswers)) {
    if (!state.directAnswers.find(da => da.questionId === qId)) {
      evidence.push({
        questionId: qId,
        answer: ans,
        questionReliability: 0.98,
      });
    }
  }

  const posterior = updatePosterior(animals, evidence, state.familyCounts);
  
  const cState: ConstraintState = { inferred: state.inferredAnswers, eligible: {} };
  const tiered = tierCandidates(animals, state.directAnswers, cState);
  
  let activeCandidates = posterior.filter(p => {
    const t = tiered.find(t => animals[t.animalIndex].id === p.animal.id);
    return t && t.tier === "active";
  });
  
  if (activeCandidates.length <= 2) {
    const rescueCandidates = posterior.filter(p => {
      const t = tiered.find(t => animals[t.animalIndex].id === p.animal.id);
      return t && t.tier === "rescue";
    });
    activeCandidates = [...activeCandidates, ...rescueCandidates].sort((a, b) => b.probability - a.probability);
  }
  
  if (activeCandidates.length === 0) {
    activeCandidates = posterior;
  }

  state.mode = determineMode(activeCandidates, state.turn, state.rejectedGuesses.length);
  const candidatesRemaining = activeCandidates.length;

  let turnResult: EngineTurn;
  const top = activeCandidates[0];

  if (state.mode === "GUESS" && top && state.rejectedGuesses.length < MAX_REJECTED_GUESSES) {
    turnResult = {
      action: "guess",
      question: "",
      trait_key: "",
      guess: localizedName(top.animal, state.language),
      confidence: top.probability,
      candidates_remaining: candidatesRemaining,
      db_match_found: true,
    };
  } else {
    let nextQuestionId: string | null = null;

    if (state.mode === "CONFIRM" && activeCandidates.length > 1) {
      nextQuestionId = findSilverBullet(top, activeCandidates.slice(1), QUESTIONS.filter(q => !state.askedQuestions.includes(q.id) && !state.inferredAnswers[q.id]));
      if (!nextQuestionId) {
        nextQuestionId = selectConfirmationQuestion(top, activeCandidates.slice(1), QUESTIONS.filter(q => !state.askedQuestions.includes(q.id) && !state.inferredAnswers[q.id]));
      }
    }
    
    if (!nextQuestionId) {
      nextQuestionId = openingBookQuestion(state);
    }
    
    if (!nextQuestionId) {
      const eligible = QUESTIONS.filter(q => !state.askedQuestions.includes(q.id) && !state.inferredAnswers[q.id]);
      if (eligible.length > 0) {
        nextQuestionId = selectBestQuestion(activeCandidates, eligible, state.familyCounts, state.recentFamilies);
      }
    }

    if (!nextQuestionId || state.turn >= MAX_QUESTIONS || state.rejectedGuesses.length >= MAX_REJECTED_GUESSES) {
      turnResult = {
        action: "give_up",
        question: GIVE_UP_MESSAGES[state.language],
        trait_key: "",
        guess: "",
        confidence: top?.probability ?? 0,
        candidates_remaining: candidatesRemaining,
        db_match_found: false,
        is_contradiction_overload: state.sessionReliability < 0.6,
      };
    } else {
      const q = QUESTIONS.find(q => q.id === nextQuestionId);
      turnResult = {
        action: "ask_question",
        question: q!.text[state.language],
        trait_key: q!.id,
        guess: "",
        confidence: top?.probability ?? 0,
        candidates_remaining: candidatesRemaining,
        db_match_found: true,
      };
    }
  }

  state.confidence = turnResult.confidence;
  state.conversation.push({ role: "assistant", content: JSON.stringify(turnResult) });
  
  if (turnResult.action === "ask_question") {
    state.lastQuestion = turnResult.question;
    state.lastTraitKey = turnResult.trait_key;
    state.askedQuestions.push(turnResult.trait_key);
    
    const family = questionFamily(turnResult.trait_key);
    state.familyCounts[family] = (state.familyCounts[family] || 0) + 1;
    state.recentFamilies = [family, state.recentFamilies[0]];
  } else if (turnResult.action === "guess") {
    state.pendingGuess = top?.animal.id ?? null;
  } else {
    state.gameOver = true;
    state.awaitingReveal = true;
    state.lastTraitKey = null;
    state.lastQuestion = null;
  }

  turnResult.top_candidates = activeCandidates.slice(0, 5).map(c => ({
    id: c.animal.id,
    name: localizedName(c.animal, state.language),
    prob: Math.round(c.probability * 100)
  }));

  yield sse({ type: "session_id", session_id: createSessionToken(state) });
  yield sse({ type: "result", ...turnResult, turn: state.turn });
}
