# Akinator v4 controls and remaining limits

Updated 2026-11-17. This document supersedes the historical noise-tolerant/rescue descriptions.

| Previous issue | Current control |
| --- | --- |
| Derived traits counted as independent evidence | Direct observations only; no implication-based posterior evidence |
| Tier filtering left unnormalized probability mass | A single compatible pool is normalized after direct-answer likelihood updates |
| Single user error silently rescued | Explicit strict semantics: hard contradictions exclude; uncertainty stays compatible |
| Advertised second-step score was unused | A real branch-weighted second-question information-gain term participates in selection |
| Unknown retroactively discounted prior evidence | Unknown is an exact posterior no-op |
| Stale duplicate root engines | Compatibility reexports of the canonical src/lib/server modules |
| Unsupported catalogue benchmarks | Pure evaluator states its synthetic, in-sample answer provenance |
| Corrupt new localized strings | Tests check localized question text and catalogue consistency |

## Deliberate limits

- A wrong categorical answer can eliminate the correct target; this is the requested product policy, not noise recovery.
- The 110 targets are curated living-animal game categories. Broad labels and biological variation remain simplifications; every inherited trait has not been independently sourced.
- At least one encoded hard discriminator per pair does not guarantee a human knows it or the policy reaches it within 24 questions.
- Unknown consumes a question, and the current selector models yes/no branches rather than a calibrated human unknown channel.
- Uniform priors are an engineering baseline, not measured target popularity.
- High relative posterior is not out-of-catalogue detection. No-match and give-up are explicit terminal outcomes, not scientific species verification.
- Storage-free limits/claims are per process; multi-instance deployments need shared infrastructure and application-level controls.
- Optional providers and Redis moderation are integrations with availability and operational constraints; local tests/evaluation do not verify live credentials or real external services.

## Local checks

npm test, npm run typecheck, npm run evaluate. This standalone package does not provide Next.js lint/build commands. No live deployment or public release is implied by the v4 metadata.
