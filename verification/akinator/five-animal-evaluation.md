# AI Akinator Five-Animal Evaluation

Date: 2026-07-29

## Method

The production inference functions were exercised directly through `scripts/evaluate-akinator.ts`. Each answer was generated from the target's curated profile, then the posterior top three candidates were captured after every question. A guess was accepted only when its canonical animal ID matched the target. Wrong guesses were rejected exactly as a user would reject them.

Targets:

| Difficulty label | Target |
| --- | --- |
| Easy | Dog |
| Easy | Duck |
| Medium | Platypus |
| Medium | Chameleon |
| Hard | Axolotl |

The difficulty labels represent expected human familiarity. They do not necessarily represent statistical difficulty inside the catalog.

## Executive Results

| Target | Questions | Guess sequence | First guess correct | Wrong guesses | Result |
| --- | ---: | --- | --- | ---: | --- |
| Dog | 8 | Dog | Yes | 0 | Correct |
| Duck | 7 | Duck | Yes | 0 | Correct |
| Platypus | 8 | Platypus | Yes | 0 | Correct |
| Chameleon | 8 | Chameleon | Yes | 0 | Correct |
| Axolotl | 7 | Axolotl | Yes | 0 | Correct |

Baseline totals:

- Success: 5/5.
- First-guess accuracy: 5/5.
- Total wrong guesses: 0.
- Average question count: 7.6.
- Minimum / maximum: 7 / 8 questions.
- Give-ups: 0.
- Repeated questions: 0.

## Game 1: Dog (Easy)

| # | Question | Answer | Posterior leader after answer |
| ---: | --- | --- | --- |
| 1 | Does it normally have four legs? | Yes | Dog / Cat / Lion tied at 1.6% |
| 2 | Is its body visibly covered in fur or hair? | Yes | Dog / Cat / Lion tied at 3.1% |
| 3 | Is a typical adult larger than a person? | No | Dog / Cat / Fox tied at 5.3% |
| 4 | Is its natural diet mostly plants? | No | Dog / Cat / Fox tied at 9.4% |
| 5 | Does it naturally eat both plants and animals? | Yes | Dog / Fox / Mouse tied at 17.2% |
| 6 | Is it a rodent? | No | Dog / Fox tied at 32.9% |
| 7 | Has it been widely domesticated by humans? | Yes | Dog 73.8% |
| 8 | Is it a member of the dog family? | Yes | Dog 90.4% |

First and only guess: **Dog, 90.4%**, correct.

Assessment: The path is biologically coherent. The engine uses broad morphology first, separates dog from herbivores and rodents, then closes with domestication and taxonomy. `omnivore=yes` is defensible for domestic dogs, although users may answer this inconsistently.

## Game 2: Duck (Easy)

| # | Question | Answer | Posterior leader after answer |
| ---: | --- | --- | --- |
| 1 | Does it normally have four legs? | No | Kangaroo / Bat / Seal tied at 1.6% |
| 2 | Does it normally hunt other animals for food? | No | Kangaroo / Blue whale / Chimpanzee tied at 2.9% |
| 3 | Does it lay eggs? | Yes | Ostrich / Kiwi / Chicken tied at 5.8% |
| 4 | Does it naturally eat both plants and animals? | Yes | Ostrich / Kiwi / Chicken tied at 9.6% |
| 5 | Has it been widely domesticated by humans? | Yes | Chicken / Duck / Goldfish tied at 25.8% |
| 6 | Does it normally live in fresh water? | Yes | Duck / Goldfish tied at 44.7% |
| 7 | Does it have webbed feet? | Yes | Duck 83.9% |

First and only guess: **Duck, 83.9%**, correct.

Assessment: The final freshwater plus webbed-feet split is strong. The generic `duck` profile combines domestic and wild ducks. A user thinking specifically of a wild duck may answer `domesticated` differently; this ambiguity was tested separately and did not cause a wrong guess.

## Game 3: Platypus (Medium)

| # | Question | Answer | Posterior leader after answer |
| ---: | --- | --- | --- |
| 1 | Does it normally have four legs? | Yes | Dog / Cat / Lion tied at 1.6% |
| 2 | Is its body visibly covered in fur or hair? | Yes | Dog / Cat / Lion tied at 3.1% |
| 3 | Is a typical adult larger than a person? | No | Dog / Cat / Fox tied at 5.3% |
| 4 | Is its natural diet mostly plants? | No | Dog / Cat / Fox tied at 9.4% |
| 5 | Does it naturally eat both plants and animals? | No | Cat / Platypus / Otter tied at 17.3% |
| 6 | Is it equally at home on land and in water? | Yes | Platypus / Otter tied at 36.4% |
| 7 | Does it lay eggs? | Yes | Platypus 73.6% |
| 8 | Can it inject venom by biting or stinging? | Yes | Platypus 97.0% |

First and only guess: **Platypus, 97.0%**, correct.

Assessment: This is the strongest path in the sample. Egg-laying separates the monotreme from otter-like mammals. The venom question produces near certainty, but users may not know that male platypuses have venomous ankle spurs.

## Game 4: Chameleon (Medium)

| # | Question | Answer | Posterior leader after answer |
| ---: | --- | --- | --- |
| 1 | Does it normally have four legs? | Yes | Dog / Cat / Lion tied at 1.6% |
| 2 | Is its body visibly covered in fur or hair? | No | Elephant / Giraffe / Zebra tied at 2.8% |
| 3 | Is it a mammal? | No | Crocodile / Alligator / Turtle tied at 5.1% |
| 4 | Is it a reptile? | Yes | Crocodile / Alligator / Turtle tied at 9.0% |
| 5 | Is its natural diet mostly plants? | No | Crocodile / Alligator / Lizard tied at 18.3% |
| 6 | Does it normally live in fresh water? | No | Lizard / Chameleon tied at 30.4% |
| 7 | Could a typical adult fit in one hand? | Yes | Lizard / Chameleon tied at 44.8% |
| 8 | Can it deliberately change its skin color for camouflage or signalling? | Yes | Chameleon 90.6% |

First and only guess: **Chameleon, 90.6%**, correct.

Assessment: Taxonomy and color change form a logical path. The size answer is the weakest biological assertion in the five baseline games because chameleon species vary substantially. This trait should be changed from a hard positive to uncertain, or the profile should be split into a generic group and named species.

## Game 5: Axolotl (Hard)

| # | Question | Answer | Posterior leader after answer |
| ---: | --- | --- | --- |
| 1 | Does it normally have four legs? | Yes | Dog / Cat / Lion tied at 1.6% |
| 2 | Is its body visibly covered in fur or hair? | No | Elephant / Giraffe / Zebra tied at 2.8% |
| 3 | Is it a mammal? | No | Crocodile / Alligator / Turtle tied at 5.1% |
| 4 | Is it a reptile? | No | Frog / Toad / Salamander tied at 10.1% |
| 5 | Does it normally live in fresh water? | Yes | Frog / Toad / Salamander tied at 19.9% |
| 6 | Can it regrow complete lost limbs? | Yes | Salamander / Axolotl tied at 44.3% |
| 7 | Does it spend most of its life in water? | Yes | Axolotl 88.4% |

First and only guess: **Axolotl, 88.4%**, correct.

Assessment: The path is concise and biologically strong. Axolotl was labelled hard because it is less familiar to people, but it is statistically easy in this catalog: freshwater, full limb regeneration, and permanent aquatic life form a highly distinctive signature.

## Human-Uncertainty Stress Test

One plausible `unknown` answer was injected into each game.

| Target | Unknown answer | Questions | Guess sequence | First guess correct |
| --- | --- | ---: | --- | --- |
| Dog | Omnivore | 9 | Dog | Yes |
| Duck | Domesticated | 9 | Duck | Yes |
| Platypus | Venomous | 9 | Platypus | Yes |
| Chameleon | Fits in one hand | 9 | Chameleon | Yes |
| Axolotl | Regenerates limbs | 9 | Axolotl | Yes |

Result: 5/5 success, 5/5 first-guess accuracy, zero wrong guesses. Every uncertain game took exactly one or two additional questions. The `unknown` implementation behaves correctly: it records and retires the question without distorting the posterior.

Notable recovery questions:

- Dog used `predator=no`, `canine_family=yes`, then `domesticated=yes`.
- Duck replaced domestication evidence with size, farm-animal, and webbed-feet evidence.
- Platypus used `mammal=yes` after venom was unknown.
- Chameleon used `color_change=yes` and `arboreal=yes` after size was unknown.
- Axolotl used `semi_aquatic=no`, `mainly_aquatic=yes`, and `amphibian=yes` after regeneration was unknown.

## One-Wrong-Answer Stress Test

One defining answer was deliberately inverted in each game.

| Target | Deliberately wrong answer | Questions | Guess sequence | Wrong guesses | Final result |
| --- | --- | ---: | --- | ---: | --- |
| Dog | Domesticated = No | 12 | Fox -> Dog | 1 | Correct |
| Duck | Webbed feet = No | 10 | Duck | 0 | Correct |
| Platypus | Lays eggs = No | 17 | Otter -> Platypus | 1 | Correct |
| Chameleon | Changes color = No | 10 | Lizard -> Chameleon | 1 | Correct |
| Axolotl | Mainly aquatic = No | 11 | Toad -> Salamander -> Axolotl | 2 | Correct |

Result: eventual success remained 5/5, but first-guess accuracy fell to 1/5 and the engine made five wrong guesses in total.

Interpretation:

- The Bayesian noise channel works: no target is permanently eliminated and every game recovers.
- A false answer to a highly distinctive feature naturally dominates several truthful broad answers.
- The main policy weakness is not posterior recovery; it is guessing too readily when no remaining question clears `MIN_QUESTION_GAIN`. This produced guesses at only 32.3%, 47.9%, and 49.4% confidence in the axolotl and platypus stress games.
- The axolotl sequence demonstrates the worst observed behavior: two low-confidence wrong guesses before recovery.

## Findings

### Strong behavior

1. Information-gain selection produced balanced early splits around 0.60 bits and no duplicate questions.
2. All baseline guesses were correct on the first attempt in seven or eight questions.
3. Posterior confidence increased sharply only after genuinely distinctive facts.
4. `unknown` answers caused no posterior corruption and only modest extra length.
5. Wrong answers never assigned zero probability to the correct target.
6. Rejected guesses were not repeated.

### Problems and improvement potential

1. **Low-confidence forced guesses:** When useful information gain falls below the floor, `shouldGuess` permits a guess at posterior `>= 0.20`. Under contradictory evidence this is too aggressive. A safer policy should ask a contradiction-resolution question, request confirmation of the most influential answer, or require a higher confidence floor before a normal guess.
2. **No explicit answer revision:** The engine can recover only by making and rejecting guesses. It cannot say, "Your aquatic answer conflicts with the rest; should we revisit it?" Storing evidence influence would enable targeted correction.
3. **Generic species-group ambiguity:** `duck`, `dog`, `snake`, and `chameleon` represent broad groups where domestication, diet, venom, habitat, and size vary. These traits should use uncertainty likelihoods or profiles should be split into common species/forms.
4. **Chameleon size is overconfident:** `tiny=true` is not defensible for all chameleons. Marking it uncertain would better match real users; the uncertainty stress test already proves the engine remains accurate without it.
5. **Difficulty labels need quantitative meaning:** Axolotl was easier than dog because catalog separability, not name familiarity, controls game length. Future reports should classify difficulty by expected posterior entropy, nearest-profile distance, and simulated question count.
6. **First questions feel mechanical:** `four_legs` was selected first in all five games because it maximized expected information gain. Mathematically sound does not always mean conversationally varied. A small diversity or naturalness prior could improve replay feel without sacrificing much information.
7. **Self-consistency limitation:** Baseline answers come from the same curated knowledge base used by inference. The transcript was manually reviewed for biological plausibility, but a separate independently authored truth set is required to measure factual database accuracy without circularity.

## Verdict

The engine is reliable for truthful and uncertain users in this sample: 100% success, 100% first-guess accuracy, and no wasted guesses. Its fuzzy posterior also eventually survives one serious false answer in every tested game. The remaining weakness is user-facing contradiction handling: under a wrong defining answer, it recovers through low-confidence guesses rather than explicitly repairing the suspect evidence. That is the highest-value next improvement.

## Post-Evaluation Integration

The deployment candidate incorporates two direct changes from this report:

- Generic chameleon size is now an uncertain likelihood rather than a hard `tiny=true` fact.
- Falling below the information-gain preference no longer authorizes a low-confidence forced guess. The engine keeps asking remaining canonical questions until normal posterior thresholds or the 24-question safety limit apply.
- User-answer reliability was calibrated from `0.90` to `0.85`. Post-change results remained 5/5 on the first guess for truthful and unknown-answer runs. In the deliberately wrong-answer suite, first-guess accuracy improved from 1/5 to 3/5 and total wrong guesses fell from five to two; all five games still converged.

The original transcripts above remain unchanged as evaluation evidence. The post-change regression and stress results are recorded by the reproducible evaluator before release.
