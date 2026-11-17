import { writeFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { ANIMALS, QUESTIONS, questionFamily } from "../src/lib/server/akinator-kb";
import { updatePosterior } from "../src/lib/server/akinator-engine";
import { DEFAULT_POLICY, selectBestQuestion, type PolicyOptions } from "../src/lib/server/akinator-scoring";
import type { DirectAnswer } from "../src/lib/server/akinator-constraints";
// Pure policy evaluation: no sessions, provider or database calls.
const optimize = process.argv.includes("--optimize");
const policies: Array<{
    name: string;
    options: PolicyOptions;
}> = optimize ? [
    { name: "greedy", options: { ...DEFAULT_POLICY, lookahead: 0 } },
    { name: "lookahead-0.25", options: { ...DEFAULT_POLICY, lookahead: 0.25 } },
    { name: "lookahead-0.5", options: { ...DEFAULT_POLICY, lookahead: 0.5 } },
    { name: "lookahead-0.75", options: { ...DEFAULT_POLICY, lookahead: 0.75 } },
] : [{ name: "current", options: DEFAULT_POLICY }];
const seeds = optimize ? [0, 17] : [0, 17, 71];
const summaries = [];
for (const policy of policies) {
    const games: Array<{
        id: string;
        seed: number;
        unknownCount: number;
        success: boolean;
        questions: number;
    }> = [];
    const timings: number[] = [];
    const start = performance.now();
    for (const seed of seeds)
        for (const unknownCount of [0, 2])
            for (const target of ANIMALS) {
                const answers: DirectAnswer[] = [];
                const asked = new Set<string>();
                const families: Record<string, number> = {};
                let recent: string[] = [];
                let success = false;
                while (answers.length <= 24) {
                    const candidates = updatePosterior(ANIMALS, answers);
                    if (candidates.length === 1) {
                        success = candidates[0].animal.id === target.id;
                        break;
                    }
                    if (!candidates.length || answers.length === 24)
                        break;
                    const t = performance.now();
                    const next = selectBestQuestion(candidates, QUESTIONS.filter(q => !asked.has(q.id)), families, recent, seed ^ answers.length, policy.options);
                    timings.push(performance.now() - t);
                    if (!next)
                        break;
                    const v = target.traits[next];
                    const answer = answers.length < unknownCount ? "unknown" : v === "yes" ? "yes" : v === "no" ? "no" : "unknown";
                    answers.push({ questionId: next, answer });
                    asked.add(next);
                    const f = questionFamily(next);
                    families[f] = (families[f] ?? 0) + 1;
                    recent = [f, recent[0] ?? ""];
                }
                games.push({ id: target.id, seed, unknownCount, success, questions: answers.length });
            }
    const result = { name: policy.name, options: policy.options, elapsedMs: performance.now() - start, scenarios: [0, 2].map(unknownCount => {
            const rows = games.filter(g => g.unknownCount === unknownCount);
            const counts = rows.filter(g => g.success).map(g => g.questions).sort((a, b) => a - b);
            return { unknownCount, total: rows.length, success: counts.length, mean: counts.reduce((s, x) => s + x, 0) / counts.length, p90: counts[Math.ceil(counts.length * .9) - 1], max: counts.at(-1), failures: rows.filter(g => !g.success) };
        }), selectorTiming: { meanMs: timings.reduce((s, x) => s + x, 0) / timings.length, p90Ms: timings.sort((a, b) => a - b)[Math.ceil(timings.length * .9) - 1] } };
    console.log(JSON.stringify(result));
    summaries.push(result);
}
const output = { catalogSize: ANIMALS.length, questionCount: QUESTIONS.length, seeds, provenance: "Synthetic answers from current catalog; policy comparisons are in-sample engineering measurements, not independent biological accuracy or human performance.", policies: summaries };
const outIndex = process.argv.indexOf("--out");
if (outIndex >= 0 && process.argv[outIndex + 1])
    writeFileSync(process.argv[outIndex + 1], JSON.stringify(output, null, 2));
