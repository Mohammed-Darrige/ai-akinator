# AI Akinator v4 architecture

The built-in catalogue contains 110 recognizable living-animal targets and 90 canonical localized questions. It is a bounded game vocabulary, not an exhaustive animal inventory. See the provenance document for narrower target meanings, removed targets, sources, and unresolved variability.

## Direct-answer inference

1. Start with uniform positive priors over the built-in catalogue.
2. Remove candidates whose categorical yes/no trait is opposite a direct answer. No rescue or inferred-taxonomy cascade restores them.
3. Multiply remaining weights by direct-answer likelihoods and normalize. Hard yes/no uses 1/0; likely/unlikely uses 0.78/0.22; variable, not-applicable and missing values use 0.5.
4. Unknown performs no probability update. It records the question so the same question cannot repeat.

The compatibility pYes helper retains its historical soft mapping; the active engine and selector share answerLikelihood instead. Only direct observations are retained in the legacy inferredAnswers field. Biological uncertainty is not a player error.

## Question and guess policy

The selector computes expected entropy reduction, ambiguity/clarity penalties and bounded family repetition penalties for eligible questions. A six-question shortlist receives a real second-question expected information-gain term. Near-equal utilities are ordered with a seed stored in the session. This is bounded depth-2 utility lookahead, not globally optimal total-question planning. DEFAULT_POLICY in the scoring module is the configuration authority; the evaluator compares it against alternatives.

Exactly one compatible candidate permits a guess. More than one candidate requires further information; zero candidates produces no_match. Limits are 24 answered factual questions and three rejected guesses. Other terminal reasons are question_limit, guess_limit, and no_questions. A catalogue-relative posterior is not the probability that the intended animal exists in the catalogue.

## Sessions and adapters

Sessions are version 4 HMAC-SHA256 tokens with a dedicated secret, timing-safe signature checks, bounded shape/collections, two-hour lifetime, and catalogue revision binding. A changed runtime catalogue requires a new game. Current v4 tokens are not interchangeable with historical v3 tokens.

The db adapter loads validated APPROVED records; the engine additionally requires complete boolean coverage of all current questions before a custom profile becomes playable. Partial submissions can remain in moderation without becoming negative-filled profiles. Approval requires complete current-question coverage, and the profile editing helper lets moderation supply missing boolean traits before approval. Hash-based writes preserve individual records; a legacy array is read for migration. Without configured storage, the built-in catalogue works locally.

The optional open-world module calls configured providers only when invoked by a separate post-game integration. This mirror has no HTTP route layer and does not claim to implement application-level authorization or moderation UI. Ordinary inference and the evaluator do not call providers. The shared akinator-scope module applies the product name exclusions to built-in/community integration. Its finite name list is a scope guard, not biological entity verification. Post-game replay claims can bind to a validated canonical game ID. No probabilistic open-set animal detector is implemented.

## Verification

Run npm test and npm run typecheck. Run npm run evaluate for in-sample catalogue coverage and local policy timings; --optimize compares configured lookahead alternatives. Unknown or variable answers are simulated conventions, not measured human behavior. Correctness under source-data mistakes, real answer uncertainty, or unseen targets requires separate evaluation.
