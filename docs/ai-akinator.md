# AI Akinator Engine

## Scope

The engine identifies real living or extinct animals only. Plants, fungi, objects, people, fictional characters, and mythical creatures are outside the domain.

## Modules

- `src/lib/server/akinator-knowledge-base.ts`: canonical multilingual questions and curated animal profiles.
- `src/lib/server/akinator-engine.ts`: posterior inference, question policy, guess policy, and signed session state.
- `src/lib/server/akinator-open-world.ts`: optional provider adapter for post-game unknown-animal review.
- `src/lib/server/db.ts`: approved and pending custom profiles in Upstash Redis.
- `src/app/api/v1/start/route.ts` and `src/app/api/v1/ask/route.ts`: SSE transport.
- `src/app/api/v1/akinator/post-game/route.ts`: isolated open-world review flow.

## Inference Model

The game starts with a normalized prior over the catalog. For each canonical answer, the engine updates every candidate using Bayes' rule:

`P(animal | answers) proportional to P(animal) * product(P(answer | animal))`

Trait values are likelihoods rather than hard filters. Confirmed traits use `0.97`, confirmed negatives use `0.03`, and biologically variable traits use `0.68`. A calibrated `0.85` response-reliability channel prevents one mistaken human answer from reducing the correct animal to zero probability. `unknown` records the question but does not alter the posterior.

## Question Policy

Every question has one canonical trait ID and synchronized English, Turkish, and Arabic text. On each turn the engine evaluates every unasked question and selects the largest expected entropy reduction:

`IG = H(A) - P(yes)H(A | yes) - P(no)H(A | no)`

The score also accounts for question clarity and useful posterior coverage. Canonical IDs make semantic duplicates impossible without multilingual regex or LLM validation. Low information gain alone never forces a low-confidence guess; while canonical questions remain, the engine continues gathering evidence until a normal posterior threshold or the hard safety limit is reached.

## Guess Policy

There is no fixed minimum question count. Guessing depends on posterior probability, top-two odds, and effective candidate count (`2^entropy`). Rejected guesses receive negligible posterior mass and cannot repeat. The engine asks at most 24 factual questions and permits at most four rejected guesses before requesting the answer.

## Open World

The live turn loop never calls an LLM. Unknown animals can be submitted after a game. The optional provider adapter validates Animalia scope and emits only canonical trait IDs. New profiles enter `PENDING_REVIEW`; only `APPROVED` profiles are loaded into the runtime catalog.

## Security

- Answers are parsed to `yes`, `no`, or `unknown` before inference.
- Stateless session payloads use HMAC-SHA256 and constant-time signature comparison.
- Session shape, turn counts, collection sizes, and payload bytes are bounded.
- Claimed post-game names are length-limited and passed to the provider as untrusted JSON data.
- `AKINATOR_SESSION_SECRET` must contain at least 32 characters.

## Verification

`npm test` simulates every catalog animal, difficult targets, contradictory evidence, duplicate-question prevention, and noisy answers. Release checks are `npm run lint` and `npm run build`.
