import { createHash, createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { ANIMALS, QUESTIONS, localizedName, questionFamily, type AnimalProfile, type Language } from "./akinator-kb";
import { matchesAnswers, observedFacts, type DirectAnswer } from "./akinator-constraints";
import { answerLikelihood, normalize, selectBestQuestion, type ScoringCandidate } from "./akinator-scoring";
import { getCustomAnimals } from "./db";
import { isExcludedAnimalName } from "./akinator-scope";
export type Answer = "yes" | "no" | "unknown";
export type GameState = {
    version: 4;
    created_at: number;
    id: string;
    seed: number;
    language: Language;
    turn: number;
    directAnswers: DirectAnswer[];
    inferredAnswers: Record<string, "yes" | "no">;
    askedQuestions: string[];
    familyCounts: Record<string, number>;
    recentFamilies: [
        string,
        string
    ];
    qaHistory: Array<[
        string,
        string
    ]>;
    mode: "EXPLORE" | "CONFIRM" | "GUESS";
    confidence: number;
    pendingGuess: string | null;
    lastTraitKey: string | null;
    lastQuestion: string | null;
    gameOver: boolean;
    awaitingReveal: boolean;
    rejectedGuesses: string[];
    conversation: Array<{
        role: string;
        content: string;
    }>;
    catalogRevision: string;
};
export type EngineTurn = {
    action: "ask_question" | "guess" | "guess_correct" | "give_up";
    question: string;
    trait_key: string;
    guess: string;
    confidence: number;
    candidates_remaining: number;
    catalog_size: number;
    db_match_found: boolean;
    stop_reason?: "no_match" | "question_limit" | "guess_limit" | "no_questions";
    top_candidates?: Array<{
        id: string;
        name: string;
        prob: number;
    }>;
};
export const MAX_QUESTIONS = 24;
export const MAX_REJECTED_GUESSES = 3;
export const SESSION_TTL_MS = 2 * 60 * 60000;
export const ANSWER_LABELS: Record<Language, Record<Answer, string>> = {
    en: { yes: "Yes", no: "No", unknown: "I don't know" },
    tr: { yes: "Evet", no: "Hayır", unknown: "Bilmiyorum" },
    ar: { yes: "نعم", no: "لا", unknown: "لا أعرف" },
};
export const ANIMAL_SIGNATURES = ANIMALS.map(a => ({ id: a.id, name: a.name, traits: Object.fromEntries(Object.entries(a.traits).map(([k, v]) => [k, v === "yes" || v === "likely"])), priority: a.prior }));
export function createGameState(language: Language = "en", seed = randomBytes(4).readUInt32BE()): GameState {
    return { version: 4, created_at: Date.now(), id: randomBytes(16).toString("hex"), seed, language, turn: 0, directAnswers: [], inferredAnswers: {}, askedQuestions: [], familyCounts: {}, recentFamilies: ["", ""], qaHistory: [], mode: "EXPLORE", confidence: 0, pendingGuess: null, lastTraitKey: null, lastQuestion: null, gameOver: false, awaitingReveal: false, rejectedGuesses: [], conversation: [], catalogRevision: "" };
}
function secret(): string {
    const value = process.env.AKINATOR_SESSION_SECRET?.trim();
    if (!value || value.length < 32)
        throw new Error("Configure AKINATOR_SESSION_SECRET (>= 32 characters).");
    return value;
}
export function hasSessionSigningSecret(): boolean { try {
    return Boolean(secret());
}
catch {
    return false;
} }
function sign(payload: string): string { return createHmac("sha256", secret()).update(`v4.${payload}`).digest("base64url"); }
export function createSessionToken(state: GameState): string {
    const payload = Buffer.from(JSON.stringify(state)).toString("base64url");
    if (payload.length > 32000)
        throw new Error("Session exceeds its size limit.");
    return `v4.${payload}.${sign(payload)}`;
}
function safeState(input: unknown): input is GameState {
    if (!input || typeof input !== "object")
        return false;
    const s = input as GameState;
    const ids = new Set(QUESTIONS.map(q => q.id));
    return s.version === 4 && Number.isFinite(s.created_at) && Number.isInteger(s.seed) && s.seed >= 0 && s.seed <= 0xffffffff
        && typeof s.id === "string" && /^[a-f0-9]{32}$/.test(s.id)
        && Number.isInteger(s.turn) && s.turn >= 0 && s.turn <= MAX_QUESTIONS
        && Array.isArray(s.directAnswers) && s.directAnswers.length === s.turn
        && s.directAnswers.every(a => a && ids.has(a.questionId) && ["yes", "no", "unknown"].includes(a.answer))
        && new Set(s.directAnswers.map(a => a.questionId)).size === s.directAnswers.length
        && Array.isArray(s.askedQuestions) && s.askedQuestions.length <= MAX_QUESTIONS && s.askedQuestions.every(id => ids.has(id))
        && new Set(s.askedQuestions).size === s.askedQuestions.length
        && Array.isArray(s.rejectedGuesses) && s.rejectedGuesses.length <= MAX_REJECTED_GUESSES && s.rejectedGuesses.every(id => typeof id === "string" && id.length <= 80)
        && Array.isArray(s.qaHistory) && s.qaHistory.length <= MAX_QUESTIONS && s.qaHistory.every(pair => Array.isArray(pair) && pair.length === 2 && pair.every(x => typeof x === "string" && x.length <= 500))
        && Array.isArray(s.recentFamilies) && s.recentFamilies.length === 2 && s.recentFamilies.every(x => typeof x === "string")
        && Boolean(s.familyCounts) && typeof s.familyCounts === "object" && Object.values(s.familyCounts).every(n => Number.isInteger(n) && n >= 0 && n <= MAX_QUESTIONS)
        && ["EXPLORE", "CONFIRM", "GUESS"].includes(s.mode) && typeof s.gameOver === "boolean" && typeof s.awaitingReveal === "boolean"
        && (s.pendingGuess === null || typeof s.pendingGuess === "string")
        && (s.lastTraitKey === null || ids.has(s.lastTraitKey)) && (s.lastQuestion === null || typeof s.lastQuestion === "string")
        && typeof s.catalogRevision === "string" && (s.catalogRevision === "" || /^[a-f0-9]{64}$/.test(s.catalogRevision));
}
export function readSessionToken(token: string | undefined, language: Language): GameState {
    if (token === undefined)
        return createGameState(language);
    if (typeof token !== "string" || token.length > 33000)
        throw new Error("Invalid session token.");
    const parts = token.split(".");
    if (parts.length !== 3 || parts[0] !== "v4" || !/^[A-Za-z0-9_-]+$/.test(parts[1]) || !/^[A-Za-z0-9_-]{43}$/.test(parts[2]))
        throw new Error("Invalid or outdated session. Start a new game.");
    const expected = Buffer.from(sign(parts[1]));
    const supplied = Buffer.from(parts[2]);
    if (expected.length !== supplied.length || !timingSafeEqual(expected, supplied))
        throw new Error("Invalid session signature.");
    let state: unknown;
    try {
        state = JSON.parse(Buffer.from(parts[1], "base64url").toString("utf8"));
    }
    catch {
        throw new Error("Invalid session payload.");
    }
    if (!safeState(state))
        throw new Error("Invalid session state.");
    if (state.created_at > Date.now() + 60000 || Date.now() - state.created_at > SESSION_TTL_MS)
        throw new Error("Session expired. Start a new game.");
    return { ...state, language };
}
export interface EvidenceItem extends DirectAnswer {
    questionReliability?: number;
}
/** Direct observations only. Unknown is an exact no-op; incompatible animals never return. */
export function updatePosterior(animals: readonly AnimalProfile[], evidence: readonly EvidenceItem[]): ScoringCandidate[] {
    const eligible = animals.filter(a => matchesAnswers(a, evidence));
    if (!eligible.length)
        return [];
    const logs = eligible.map(animal => ({ animal, log: evidence.reduce((log, e) => e.answer === "unknown" ? log : log + Math.log(answerLikelihood(animal, e.questionId, e.answer)), Math.log(Math.max(animal.prior, 0.01))) }));
    const max = Math.max(...logs.map(x => x.log));
    return normalize(logs.map(x => ({ animal: x.animal, probability: Math.exp(x.log - max) })));
}
async function catalog(): Promise<AnimalProfile[]> {
    const builtInIds = new Set(ANIMALS.map(a => a.id));
    const custom = (await getCustomAnimals()).filter(a => !builtInIds.has(a.id) && !Object.values(a.name).some(isExcludedAnimalName) && QUESTIONS.every(q => typeof a.traits[q.id] === "boolean"));
    // A partial community profile cannot masquerade as a complete negative profile.
    return [...ANIMALS, ...custom.map(a => ({ id: a.id, name: a.name, prior: 1, traits: Object.fromEntries(QUESTIONS.map(q => [q.id, a.traits[q.id] ? "yes" as const : "no" as const])) }))].sort((a, b) => a.id.localeCompare(b.id));
}
function encode(data: object): string { return `data: ${JSON.stringify(data)}\n\n`; }
function parseAnswer(input: unknown): Answer | null {
    if (typeof input !== "string")
        return null;
    const s = input.trim().toLowerCase();
    if (["yes", "evet", "نعم"].includes(s))
        return "yes";
    if (["no", "hayır", "hayir", "لا"].includes(s))
        return "no";
    if (["unknown", "bilmiyorum", "i don't know", "لا أعرف", "لا اعرف"].includes(s))
        return "unknown";
    return null;
}
const FINISH: Record<Language, string> = { en: "I cannot identify an animal from these answers. Which animal did you choose?", tr: "Bu yanıtlarla bir hayvan belirleyemedim. Hangi hayvanı seçmiştin?", ar: "لم أتمكن من تحديد حيوان بهذه الإجابات. ما الحيوان الذي اخترته؟" };
export async function* processAkinatorTurn(sessionToken: string | undefined, userAnswer?: string, inputLanguage: unknown = "en") {
    const language: Language = inputLanguage === "tr" || inputLanguage === "ar" ? inputLanguage : "en";
    let state: GameState;
    try {
        state = readSessionToken(sessionToken, language);
    }
    catch (error) {
        yield encode({ type: "error", content: error instanceof Error ? error.message : "Invalid session." });
        return;
    }
    if (state.gameOver) {
        yield encode({ type: "error", content: "This session is already complete. Start a new game." });
        return;
    }
    const animals = await catalog();
    const revision = createHash("sha256").update(JSON.stringify([QUESTIONS, animals])).digest("hex");
    if (state.catalogRevision && state.catalogRevision !== revision) {
        yield encode({ type: "error", content: "The animal catalog changed. Start a new game." });
        return;
    }
    state.catalogRevision = revision;
    if (sessionToken && state.askedQuestions.length > 0 && userAnswer === undefined) {
        yield encode({ type: "error", content: "An answer is required." });
        return;
    }
    if (userAnswer !== undefined) {
        const answer = parseAnswer(userAnswer);
        if (!answer) {
            yield encode({ type: "error", content: "Answer must be yes, no, or unknown." });
            return;
        }
        if (state.pendingGuess) {
            if (answer === "unknown") {
                yield encode({ type: "error", content: "Please confirm the guess with yes or no." });
                return;
            }
            if (answer === "yes") {
                const guessed = animals.find(a => a.id === state.pendingGuess);
                state.gameOver = true;
                state.pendingGuess = null;
                yield encode({ type: "session_id", session_id: createSessionToken(state) });
                yield encode({ type: "result", action: "guess_correct", guess: guessed ? localizedName(guessed, language) : "", turn: state.turn, candidates_remaining: 1, catalog_size: animals.length });
                return;
            }
            state.rejectedGuesses.push(state.pendingGuess);
            state.pendingGuess = null;
        }
        else if (state.lastTraitKey) {
            state.directAnswers.push({ questionId: state.lastTraitKey, answer });
            state.qaHistory.push([state.lastQuestion ?? state.lastTraitKey, ANSWER_LABELS[language][answer]]);
            state.turn = state.directAnswers.length;
            state.inferredAnswers = observedFacts(state.directAnswers);
            state.lastTraitKey = null;
            state.lastQuestion = null;
        }
        else {
            yield encode({ type: "error", content: "There is no pending question." });
            return;
        }
    }
    const ranked = updatePosterior(animals.filter(a => !state.rejectedGuesses.includes(a.id)), state.directAnswers);
    const eligible = QUESTIONS.filter(q => !state.askedQuestions.includes(q.id));
    const next = ranked.length > 1 && state.turn < MAX_QUESTIONS && state.rejectedGuesses.length < MAX_REJECTED_GUESSES ? selectBestQuestion(ranked, eligible, state.familyCounts, state.recentFamilies, state.seed ^ state.turn) : null;
    const result: EngineTurn = { action: "give_up", question: FINISH[language], trait_key: "", guess: "", confidence: ranked[0]?.probability ?? 0, candidates_remaining: ranked.length, catalog_size: animals.length, db_match_found: ranked.length > 0 };
    if (!ranked.length)
        result.stop_reason = "no_match";
    else if (state.rejectedGuesses.length >= MAX_REJECTED_GUESSES)
        result.stop_reason = "guess_limit";
    else if (ranked.length === 1) {
        result.action = "guess";
        result.question = "";
        result.guess = localizedName(ranked[0].animal, language);
        state.pendingGuess = ranked[0].animal.id;
        state.mode = "GUESS";
    }
    else if (state.turn >= MAX_QUESTIONS)
        result.stop_reason = "question_limit";
    else if (next) {
        const q = QUESTIONS.find(q => q.id === next)!;
        result.action = "ask_question";
        result.question = q.text[language];
        result.trait_key = q.id;
        state.lastQuestion = result.question;
        state.lastTraitKey = q.id;
        state.askedQuestions.push(q.id);
        const family = questionFamily(q.id);
        state.familyCounts[family] = (state.familyCounts[family] ?? 0) + 1;
        state.recentFamilies = [family, state.recentFamilies[0]];
        state.mode = ranked.length <= 5 ? "CONFIRM" : "EXPLORE";
    }
    else
        result.stop_reason = "no_questions";
    if (result.action === "give_up") {
        state.gameOver = true;
        state.awaitingReveal = true;
        state.lastQuestion = null;
        state.lastTraitKey = null;
    }
    state.confidence = result.confidence;
    result.top_candidates = ranked.slice(0, 5).map(c => ({ id: c.animal.id, name: localizedName(c.animal, language), prob: Math.round(c.probability * 100) }));
    yield encode({ type: "session_id", session_id: createSessionToken(state) });
    yield encode({ type: "result", ...result, turn: state.turn });
}
