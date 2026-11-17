# Neural Akinator v4

A deterministic animal guessing engine with strict direct-answer consistency, a curated living-animal catalogue, and bounded two-question information-gain lookahead. This standalone mirror shares its canonical seven modules with the portfolio application; it does not include the Next.js transport or UI.

## Current behavior

- 110 catalogue targets and 90 questions in English, Turkish, and Arabic.
- A categorical yes/no contradiction permanently removes that candidate. There is no error rescue. A mistaken categorical answer can exclude the intended animal.
- Biological uncertainty remains explicit: likely, unlikely, variable, not applicable, and missing traits permit both outcomes with different likelihoods where specified.
- Unknown answers leave the posterior unchanged and consume a question; they cannot be repeated.
- Only direct observations update probability. Taxonomic implications are not independent evidence.
- A guess is issued only when one compatible candidate remains. Otherwise the game asks another useful question or finishes with an explicit reason.
- No LLM or provider call occurs during normal game turns. Optional post-game verification and moderated storage are separate adapters.

## Run locally

```bash
npm install
npm test
npm run typecheck
npm run evaluate
npm run evaluate -- --optimize --out verification/akinator/policy-comparison.json
```

Tests mock storage; the evaluator calls only the pure posterior and question selector. Neither needs provider credentials. Use a dedicated server-only `AKINATOR_SESSION_SECRET` of at least 32 characters when exercising signed sessions outside the tests. Optional Upstash storage uses `KV_REST_API_URL` and `KV_REST_API_TOKEN`; absent storage yields the built-in catalogue. Do not configure production credentials for local evaluation.

## Modules

`src/lib/server/` contains `akinator-kb.ts`, `akinator-constraints.ts`, `akinator-scoring.ts`, `akinator-engine.ts`, `db.ts`, `akinator-open-world.ts`, and `akinator-scope.ts`. The two files directly under `src/` are compatibility reexports, not separate engines.

See [architecture](docs/ai-akinator.md), [catalogue provenance](docs/akinator-catalog-provenance.md), and [limitations](docs/ai-akinator-errors-ledger.md).

## Evaluation boundary

The current evaluator uses answers generated from this same catalogue: yes/no for hard traits and unknown for uncertainty. It reports three fixed seeds, truthful and two-initial-unknown scenarios, success, mean/p90/max question counts, and local selector timings. These are reproducible in-sample engineering checks, not independent biological validation, human accuracy, or evidence of out-of-catalogue recognition. Prior historical five-animal reports are explicitly marked historical and do not describe v4.

No publish or deployment is performed by these commands. MIT license; see LICENSE.
