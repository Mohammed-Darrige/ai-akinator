import { beforeAll, describe, expect, it, vi } from "vitest";
vi.mock("../src/lib/server/db", () => ({ getCustomAnimals: async () => [] }));
import { ANIMALS, QUESTIONS, type AnimalProfile } from "../src/lib/server/akinator-kb";
import { createGameState, createSessionToken, readSessionToken, processAkinatorTurn, updatePosterior, SESSION_TTL_MS, type EngineTurn } from "../src/lib/server/akinator-engine";
import { informationGain, selectBestQuestion, simulatePosterior } from "../src/lib/server/akinator-scoring";
beforeAll(() => { process.env.AKINATOR_SESSION_SECRET = "test-only-0123456789abcdef0123456789abcdef"; });
async function turn(token?: string, answer?: string) {
    let session = "";
    let result: EngineTurn | undefined;
    let error = "";
    for await (const chunk of processAkinatorTurn(token, answer, "en")) {
        const event = JSON.parse(chunk.slice(6));
        if (event.type === "session_id")
            session = event.session_id;
        if (event.type === "result")
            result = event;
        if (event.type === "error")
            error = event.content;
    }
    return { session, result, error };
}
function truthful(animal: AnimalProfile, id: string): string {
    const v = animal.traits[id];
    return v === "yes" ? "yes" : v === "no" ? "no" : "unknown";
}
async function play(animal: AnimalProfile, seed: number, unknownFirst = false) {
    let token = createSessionToken(createGameState("en", seed));
    let answers = 0;
    const seen = new Set<string>();
    let guessed = false;
    let current = await turn(token);
    token = current.session;
    for (let step = 0; step < 30; step++) {
        const r = current.result;
        expect(current.error).toBe("");
        expect(r).toBeDefined();
        if (r!.action === "guess_correct")
            return { questions: answers, guessed };
        if (r!.action === "guess") {
            expect(r!.guess, animal.id).toBe(animal.name.en);
            guessed = true;
            current = await turn(token, "yes");
            token = current.session;
            continue;
        }
        expect(r!.action, `${animal.id}: ${r!.stop_reason}`).toBe("ask_question");
        expect(seen.has(r!.trait_key), animal.id).toBe(false);
        seen.add(r!.trait_key);
        answers++;
        current = await turn(token, unknownFirst && answers === 1 ? "unknown" : truthful(animal, r!.trait_key));
        token = current.session;
    }
    throw new Error(`No confirmed result for ${animal.id}`);
}
describe("curated living catalog", () => {
    it.each([
        ["dog", { mammal: "yes", four_legs: "yes", feathers: "no" }],
        ["duck", { bird: "yes", feathers: "yes", webbed_feet: "yes" }],
        ["penguin", { bird: "yes", flies: "no", two_legs: "yes" }],
        ["bat", { mammal: "yes", flies: "yes", feathers: "no" }],
        ["platypus", { mammal: "yes", lays_eggs: "yes" }],
        ["elephant", { mammal: "yes", trunk: "yes", four_legs: "yes" }],
        ["butterfly", { insect: "yes", six_legs: "yes", exoskeleton: "yes" }],
        ["octopus", { mollusk: "yes", eight_arms: "yes", feathers: "no" }]
    ] as const)("matches independently specified basic observations for %s", (id, facts) => {
        // Hand-authored anatomy observations, not pYes-generated expected answers.
        const target = ANIMALS.find(a => a.id === id)!;
        for (const [trait, value] of Object.entries(facts)) expect(target.traits[trait], `${id}/${trait}`).toBe(value);
    });
    it("has complete unique localized profiles and no removed concepts", () => {
        expect(new Set(ANIMALS.map(a => a.id)).size).toBe(ANIMALS.length);
        expect(new Set(QUESTIONS.map(q => q.id)).size).toBe(QUESTIONS.length);
        for (const key of ["primate", "ape", "extinct"])
            expect(QUESTIONS.some(q => q.id === key)).toBe(false);
        for (const animal of ANIMALS) {
            expect(Object.keys(animal.traits).sort()).toEqual(QUESTIONS.map(q => q.id).sort());
            for (const lang of ["en", "tr", "ar"] as const)
                expect(animal.name[lang].trim().length).toBeGreaterThan(0);
        }
        for (const q of QUESTIONS)
            for (const lang of ["en", "tr", "ar"] as const)
                expect(q.text[lang].endsWith(lang === "ar" ? "؟" : "?")).toBe(true);
    });
    it("distinguishes every pair with at least one categorical observable", () => {
        const weak: string[] = [];
        for (let i = 0; i < ANIMALS.length; i++)
            for (let j = i + 1; j < ANIMALS.length; j++)
                if (!QUESTIONS.some(q => {
                    const a = ANIMALS[i].traits[q.id], b = ANIMALS[j].traits[q.id];
                    return a === "yes" && b === "no" || a === "no" && b === "yes";
                }))
                    weak.push(`${ANIMALS[i].id}/${ANIMALS[j].id}`);
        expect(weak).toEqual([]);
    });
    it.each([0, 17])("confirms every target without wrong guesses, seed %i", async (seed) => {
        for (const animal of ANIMALS)
            expect((await play(animal, seed)).questions).toBeLessThanOrEqual(24);
    }, 120000);
    it("survives a missing observation without treating it as a contradiction", async () => {
        for (const id of ["dog", "duck", "owl", "cockroach", "elephant", "snake"]) {
            const animal = ANIMALS.find(a => a.id === id);
            if (animal)
                await play(animal, 41, true);
        }
    }, 30000);
});
describe("strict accumulated observations", () => {
    it("never rescues a target that contradicts a direct categorical answer", () => {
        const dog = ANIMALS.find(a => a.id === "dog")!;
        expect(updatePosterior([dog], [{ questionId: "mammal", answer: "no" }])).toEqual([]);
    });
    it("unknown changes neither weights nor surviving candidates", () => {
        const first = updatePosterior(ANIMALS, [{ questionId: "mammal", answer: "yes" }]);
        const next = updatePosterior(ANIMALS, [{ questionId: "mammal", answer: "yes" }, { questionId: "bird", answer: "unknown" }]);
        expect(next).toEqual(first);
        expect(next.reduce((s, c) => s + c.probability, 0)).toBeCloseTo(1, 12);
    });
    it("rejects wrong guesses permanently and does not count them as factual answers", async () => {
        const a = ANIMALS.find(a => a.id === "dog")!;
        let current = await turn();
        for (let i = 0; i < 24 && current.result?.action === "ask_question"; i++)
            current = await turn(current.session, truthful(a, current.result.trait_key));
        expect(current.result?.action).toBe("guess");
        const before = readSessionToken(current.session, "en");
        const rejected = await turn(current.session, "no");
        expect(rejected.result?.action).toBe("give_up");
        expect(rejected.result?.stop_reason).toBe("no_match");
        const after = readSessionToken(rejected.session, "en");
        expect(after.turn).toBe(before.turn);
        expect(after.rejectedGuesses).toContain(a.id);
    });
    it("handles empty question lists and zero candidate distributions", () => {
        expect(selectBestQuestion([], [])).toBeNull();
        expect(simulatePosterior([], "mammal", "yes")).toEqual([]);
        expect(informationGain([], "mammal")).toBe(0);
    });
    it("varies only bounded question choices across seeds", () => {
        const candidates = updatePosterior(ANIMALS, []);
        const openings = new Set(Array.from({ length: 24 }, (_, seed) => selectBestQuestion(candidates, QUESTIONS, {}, [], seed)));
        expect(openings.size).toBeGreaterThan(1);
        for (const id of openings)
            expect(QUESTIONS.some(q => q.id === id)).toBe(true);
    });
});
describe("signed bounded sessions", () => {
    it("round trips and rejects appended segments or tampering", () => {
        const state = createGameState("tr", 7), token = createSessionToken(state);
        expect(readSessionToken(token, "tr")).toEqual(state);
        expect(() => readSessionToken(token + ".suffix", "tr")).toThrow();
        expect(() => readSessionToken(token.slice(0, -1) + (token.endsWith("a") ? "b" : "a"), "tr")).toThrow();
    });
    it("rejects expired, future, malformed and old-version states", () => {
        const state = createGameState();
        state.created_at = Date.now() - SESSION_TTL_MS - 1000;
        expect(() => readSessionToken(createSessionToken(state), "en")).toThrow(/expired/);
        state.created_at = Date.now() + 120000;
        expect(() => readSessionToken(createSessionToken(state), "en")).toThrow(/expired/);
        state.created_at = Date.now();
        state.turn = 99;
        expect(() => readSessionToken(createSessionToken(state), "en")).toThrow();
        expect(() => readSessionToken("v3.a.b", "en")).toThrow();
    });
    it("duplicate submissions return identical transitions", async () => {
        const first = await turn();
        const [a, b] = await Promise.all([turn(first.session, "yes"), turn(first.session, "yes")]);
        expect(a).toEqual(b);
    });
});
