# Akinator catalogue scope and provenance

Reviewed: 2026-11-17. Runtime data: `src/lib/server/akinator-kb.ts`.

## Product scope

The catalogue contains 110 recognizable living-animal targets and 90 localized questions. Scope is a product choice, not a scientific claim: the previous historical targets and the four monkey/ape targets were removed, together with the three associated question identifiers requested by the owner. Further removals reduce overlapping targets and specialist coverage: cobra, toad, axolotl, anglerfish, electric eel, wombat, echidna, capybara, kiwi, puffin, piranha, swordfish, and tick.

Stable internal IDs `turtle`, `falcon`, and `lizard` now mean **tortoise**, **peregrine falcon**, and **wall lizard**, respectively, in all three display languages. They are deliberately narrower than the former generic labels. A cobra belongs within the remaining general snake target; the catalogue does not offer a separate cobra guess. A general frog target remains without a separate overlapping toad target. These are game categories, not an exhaustive species inventory.

## Evidence and simplifications

Turkey is included as a familiar game category covering wild and domestic forms. Flight and domestic status are variable; the fan-display question explicitly concerns adult males and is shared with peafowl rather than pretending that a red wattle is unique to turkey. The profile retains a commonly-farmed positive value while avoiding a universal flightless assertion. [Cornell Lab: wild turkey life history](https://www.allaboutbirds.org/guide/Wild_Turkey/lifehistory/), [Smithsonian National Zoo: wild turkey flight](https://www.nationalzoo.si.edu/animals/wild-turkey), [Indiana DNR: male turkey fan display](https://secure.in.gov/dnr/fish-and-wildlife/files/fw-turkey-ID-graphic.pdf), and [San Diego Zoo Wildlife Alliance: peafowl](https://animals.sandiegozoo.org/animals/peafowl) support this focused addition.

All priors are uniform. No popularity estimate is presented as measured user frequency. The opening book and implication list are empty compatibility exports. The remaining taxonomic/limb exclusive groups are legacy structural metadata; they must not manufacture independent likelihood observations.

The data distinguishes hard yes/no from uncertain values. Known sex, breed, or species variation is represented as `variable`, or occasionally `unlikely`; these values must allow both user outcomes. Examples include cattle/goat/sheep horns, dog size and coat patterns, snake reproduction and habitat, shark reproduction/filter feeding, rays' venom, ant flight/stinging, duck/goose domestic status and flight, butterfly feeding tubes, and squirrel cheek pouches. The hair question explicitly includes short hair and wool: common hairy hoofed mammals no longer have arbitrary hard-negative fur values.

The previous `na` values were removed from these boolean predicates: a non-rodent is not a rodent, and an animal without antlers does not have palm-shaped antlers. This does not assert that missing knowledge is false; the public helpers still use neutral likelihood/confidence fallbacks for a genuinely absent trait key. Every built-in profile has every registered question key.

The snout question explicitly excludes beaks; bird beaks no longer receive positive snout values. Body-size comparisons exclude wingspan. Squid color change is positive; mosquito blood feeding is a separate question rather than being classified as hunting prey. Camel/llama feet are left uncertain under the informal hoof wording because padded two-toed feet do not fit the everyday hoof comparison cleanly.

Broad labels still summarize typical adult animals. Geographic variation, unusual breeds, rare species, age, and seasonal change cannot all be resolved by this finite table. A hard fact about a named catalogue target is not a universal statement about every member of a broad clade. This revision is a focused source-backed correction and discrimination pass, **not an exhaustive zoological review of every inherited cell**. The new negative discriminator cells outside the cited positive target groups are curated catalogue judgments rather than individually sourced observations. No empirical human-answer accuracy or question-count improvement is claimed by catalogue integrity checks.

## Sources supporting the focused changes

Sources were accessed on 2026-11-17. These are institutional animal-care, research, museum, university-extension, and government sources. Links support the specific topic in the first column, not every trait on a profile.

| Topic / affected questions | Primary or institutional source |
| --- | --- |
| Owl facial disc | [Cornell Lab: American barn owl identification](https://www.allaboutbirds.org/guide/American_Barn_Owl/id), [Idaho Fish and Game: owl identification](https://idfg.idaho.gov/old-web/docs/wildlife/nongame/leafletOwls.pdf) |
| Peregrine scope and long pointed wings | [Cornell Lab: peregrine identification](https://www.allaboutbirds.org/guide/peregrine_falcon/id), [US Fish and Wildlife Service: peregrine](https://www.fws.gov/species/peregrine-falcon-falco-peregrinus) |
| Large colorful toucan bill | [Cornell Lab: keel-billed toucan bill](https://www.allaboutbirds.org/cams/a-closer-look-at-the-keel-billed-toucans-incredible-bill/) |
| Ant pinched waist; cockroach contrast | [University of Minnesota: ants](https://extension.umn.edu/garden-and-home/home-maintenance/household-insects/ants), [University of Maryland: household insect identification](https://www.extension.umd.edu/resource/identifying-common-household-insect-pests) |
| Tortoise/turtle scope and shell | [San Diego Zoo Wildlife Alliance: turtle and tortoise](https://animals.sandiegozoo.org/animals/turtle-and-tortoise) |
| Wall-lizard name and scope | [Natural History Museum: Podarcis muralis](https://www.nhm.ac.uk/our-science/data/uk-species/taxon?tvk=NHMSYS0000080229), [Indiana DNR: wall lizards](https://www.in.gov/dnr/fish-and-wildlife/wildlife-resources/animals/wall-lizards/) |
| Hair/fiber, sheep fleece, male goat beard | [San Diego Zoo Wildlife Alliance: goat and sheep](https://animals.sandiegozoo.org/animals/goat-and-sheep) |
| Squid and octopus color change; eight arms versus additional tentacles | [NOAA Ocean Exploration: squid](https://oceanexplorer.noaa.gov/multimedia/squid/), [Smithsonian Ocean: cephalopods](https://ocean.si.edu/ocean-life/invertebrates/octopuses-squids-and-relatives), [Museum of New Zealand Te Papa: squid and octopus comparison](https://www.tepapa.govt.nz/digital-museum/explore-digital-museum/colossal-squid/difference-between-colossal-squid-giant-squid) |
| Cheetah running distinction | [Smithsonian National Zoo: cheetah](https://nationalzoo.si.edu/animals/cheetah) |
| Beaver broad flat tail | [Smithsonian National Zoo: beaver](https://nationalzoo.si.edu/animals/beaver) |
| Skunk defensive spray | [San Diego Zoo Wildlife Alliance: striped skunk](https://animals.sandiegozoo.org/animals/striped-skunk) |
| Pelican bill pouch | [San Diego Zoo Wildlife Alliance: pelican](https://animals.sandiegozoo.org/animals/pelican) |
| Hamster expandable cheek pouches | [McGill University: hamster biological features](https://www.mcgill.ca/research/files/research/hamsters_handout_2020_revised_april_2020_module_1.pdf) |
| Woodpecker drilling | [US Fish and Wildlife Service: woodpeckers](https://www.fws.gov/story/woodpeckers-and-your-home) |
| Camel padded two-toed feet | [San Diego Zoo Wildlife Alliance: camel physical characteristics](https://ielc.libguides.com/sdzg/factsheets/extantcamels/characteristics) |
| Mosquito blood feeding differs by sex; both sexes use plant sugars | [CDC: about mosquitoes](https://www.cdc.gov/mosquitoes/about/) |
| Beetle wing covers | [Smithsonian: beetle elytra](https://insider.si.edu/2018/01/meet-newest-new-world-canopy-beetle-species-gazillions-await-discovery/) |
| Grasshopper jumping legs | [University of Minnesota: grasshoppers](https://extension.umn.edu/agriculture/crop-production/soybean/grasshoppers-in-minnesota-soybean) |

## Verification boundary

The implementation pass checks unique IDs, complete trait coverage, valid six-valued traits, three display languages, family registration, uniform positive priors, absent removed identifiers, and at least one opposing hard yes/no discriminator for every pair of built-in targets. Pairwise separation means an informative answer exists in this encoded table; it does not mean a player knows the answer, that all facts are correct, or that the engine necessarily asks that question before its turn limit. The engine tests and independent human evaluation must establish those properties separately.
