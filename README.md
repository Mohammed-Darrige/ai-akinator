# Neural Akinator

A provider-independent, noise-tolerant animal deduction engine built with Bayesian inference, Shannon entropy, and expected information gain.

The production engine powers the interactive AI Akinator experience in Mohammed Darrige's portfolio. This repository contains the same inference core, curated knowledge base, regression tests, and reproducible evaluation harness used by the live application.

## Results

- 115 curated living and extinct animal profiles
- 70+ canonical questions with English, Turkish, and Arabic text
- 115/115 catalog animals identified in deterministic simulation
- 5/5 first-guess accuracy across the published easy/medium/hard evaluation
- 5/5 first-guess accuracy when one plausible answer is `unknown`
- 20 automated regression and knowledge-base invariant tests
- No LLM or external provider in the live turn loop

## Architecture

### Probabilistic belief state

Every candidate keeps non-zero posterior mass. User answers update the distribution through Bayes' rule:

```text
P(animal | answers) proportional to
P(animal) * product(P(answer | animal))
```

Profiles use likelihoods rather than brittle hard elimination. Canonical positive traits use `0.97`, negatives use `0.03`, biologically variable traits use `0.68`, and the calibrated human-response channel uses `0.85` reliability. One mistaken answer therefore cannot permanently eliminate the correct animal.

### Question selection

For every unasked canonical question, the engine computes expected entropy reduction:

```text
IG = H(A) - P(yes) H(A | yes) - P(no) H(A | no)
```

The highest-scoring question is selected after clarity and useful-posterior-coverage adjustments. Canonical IDs prevent repeated or semantically duplicated questions across all three languages.

### Dynamic guessing

There is no arbitrary minimum question count. Guess timing depends on:

- normalized top posterior probability
- top-two posterior odds
- effective candidate count, `2^entropy`
- rejected canonical guesses
- a 24-question safety bound

Low information gain alone never authorizes a low-confidence guess. Rejected guesses receive negligible posterior mass and cannot repeat.

### Open-world boundary

The core game is deterministic and provider-independent. In the portfolio integration, an optional post-game adapter can validate an unknown animal and submit a canonical profile for moderation. New profiles remain `PENDING_REVIEW`; only approved profiles can enter the runtime catalog.

## Repository Structure

```text
src/lib/server/
  akinator-engine.ts          Bayesian inference and game policy
  akinator-knowledge-base.ts Curated profiles and localized questions
  akinator-open-world.ts     Optional bounded post-game provider adapter
  db.ts                       Optional approved-profile persistence adapter
tests/
  akinator-engine.test.ts    Catalog-wide and noisy-answer regression suite
scripts/
  evaluate-akinator.ts       Reproducible five-target evaluation harness
docs/
  ai-akinator.md             Architecture and security model
  ai-akinator-errors-ledger.md Retired defects and residual risks
verification/akinator/
  five-animal-evaluation.md  Full question-by-question audit
```

## Quick Start

```bash
npm install
npm test
npm run evaluate
npm run typecheck
```

No API key is required for inference or evaluation. When embedding the engine with signed stateless sessions, use a dedicated `AKINATOR_SESSION_SECRET` (32+ characters). Existing server-only Akinator provider credentials are accepted as a migration fallback, but a dedicated secret takes precedence. Optional Upstash credentials are used only when embedding the engine with approved custom profiles:

```env
KV_REST_API_URL=https://your-upstash-instance
KV_REST_API_TOKEN=your-token
```

## Evaluation Summary

| Difficulty | Target | Questions | First guess | Wrong guesses |
| --- | --- | ---: | --- | ---: |
| Easy | Dog | 8 | Dog | 0 |
| Easy | Duck | 7 | Duck | 0 |
| Medium | Platypus | 8 | Platypus | 0 |
| Medium | Chameleon | 8 | Chameleon | 0 |
| Hard | Axolotl | 7 | Axolotl | 0 |

The committed report preserves the original audit transcript. Subsequent calibration retained 5/5 first-guess accuracy for truthful and unknown-answer runs while reducing wrong-answer stress-test guesses from five to two.

## Design Principles

- Animalia-only domain boundary
- deterministic core behavior
- explicit uncertainty instead of missing-data guesses
- normalized, inspectable probabilities
- multilingual questions backed by one semantic ID
- moderation before learned profiles become active
- reproducible evaluation rather than anecdotal demos

## License

MIT. See `LICENSE`.

Engineered by Mohammed Darrige, Fırat University AI & Data Engineering.
