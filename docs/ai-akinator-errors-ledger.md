# AI Akinator Error Ledger

## Retired Architecture

The July 2026 rebuild removed the 2,848-line mixed-responsibility engine. The retired design combined sparse booleans, regex synonym matching, hard exclusions, prompt-authored questions, provider failover, guessed confidence formulas, and session handling in one module.

## Closed Defects

| Defect | Root cause | Current control |
| --- | --- | --- |
| Correct animal eliminated after one bad answer | Hard boolean filtering | Noise-tolerant likelihood updates; posterior mass never becomes zero |
| Missing trait interpreted inconsistently | Sparse ad hoc signatures | Canonical question catalog and explicit per-profile likelihood |
| Repeated or compound questions | LLM text generation and synonym regex | One canonical ID and curated localized text per atomic question |
| Arbitrary confidence percentages | Match-count heuristic | Normalized Bayesian posterior |
| Weak question ordering | Trait vote heuristic | Expected Shannon entropy reduction over the current posterior |
| Premature or delayed guesses | Fixed turn/confidence gates | Posterior, odds ratio, and effective-candidate policy |
| Game failure during provider outage | LLM in the turn loop | Provider-independent local inference |
| Same wrong guess repeated | Guess stored only as text | Rejected canonical IDs receive negligible prior mass |
| Prompt injection through answers | Free text forwarded to provider | Strict answer enum; no turn-time provider call |
| Session token abuse | Weak shape validation | HMAC-SHA256, constant-time comparison, byte and collection bounds |
| Unreviewed learned profiles active immediately | Missing moderation boundary | `PENDING_REVIEW` storage; runtime loads only `APPROVED` records |

## Known Limits

- The built-in catalog contains 115 curated profiles, not every animal species. Approved custom profiles are the expansion path.
- Some colloquial answers remain biologically ambiguous. The `unknown` response is preferable when the question does not fit the user's intended species or individual.
- Runtime custom-profile updates currently use an array-backed Redis record. A high-volume moderation system should move to per-profile keys plus an index to avoid concurrent write races.
- Post-game unknown-animal verification is optional and depends on configured provider quota. Provider absence returns `AMBIGUOUS_REVIEW_REQUIRED` without interrupting the game.

## Regression Gate

- `npm test`: all-catalog simulation, hard targets, one-wrong-plus-unknown noise case, posterior normalization, data invariants.
- `npm run lint`: static quality check.
- `npm run build`: Next.js production compile, TypeScript, and route generation.
