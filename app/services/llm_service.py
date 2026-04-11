"""
AI Akinator – LLM Service (v2 – Single-Call Architecture)

Design principles:
- ONE LLM call per turn. Extraction, reasoning, and next-question selection
  are fused into a single structured prompt. No separate Extractor or Validator
  agents; all validation logic lives server-side in Python.
- BeliefState is a typed Pydantic model — never a raw dict.
- Hard server-side guards: minimum turns before guessing, confidence threshold,
  duplicate question prevention, trait conflict detection.
- Sessions are stored in-process with enforced TTL eviction.
"""

import json
import re
import time
import logging
from typing import Dict, Any, Optional, Tuple

from openai import AsyncOpenAI
from app.core.config import settings
from app.models.schemas import BeliefState, LLMTurnOutput, ReasoningLog

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
SESSION_TTL: int = 1800          # 30 minutes
MAX_SESSIONS: int = 200
MIN_TURNS_BEFORE_GUESS: int = 5  # Never guess before 5 answered questions
GUESS_CONFIDENCE_THRESHOLD: float = 0.88
MAX_TURNS: int = 20              # Force a best-guess after 20 turns

# ─────────────────────────────────────────────
#  In-process session store  {session_id: {"belief": BeliefState, "created_at": float}}
# ─────────────────────────────────────────────
_sessions: Dict[str, Dict[str, Any]] = {}


# ─────────────────────────────────────────────
#  Master Prompt — fused Extractor + Inquisitor
# ─────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an elite AI playing the Akinator animal-guessing game.
You think in English but ask questions exclusively in Turkish.

═══════════════════════════════════════════════════════
DUAL ROLE — perform BOTH in a single response:
  ROLE A – EXTRACTOR: Parse the user's latest answer and extract new atomic traits.
  ROLE B – INQUISITOR: Use full belief state to pick the single highest-information-gain question next.
═══════════════════════════════════════════════════════

━━━ EXTRACTION RULES ━━━
• Map the user's answer (Evet / Hayır / Bilmiyorum) to `new_traits` dict.
  – "Evet"       → {trait_key: true}
  – "Hayır"      → {trait_key: false}
  – "Bilmiyorum" → {} (no new trait, do not add uncertainty as a trait)
• Use atomic snake_case English keys: is_mammal, can_fly, lives_in_water,
  is_domestic, has_fur, lays_eggs, is_nocturnal, is_carnivore, is_herbivore,
  is_omnivore, has_four_legs, has_wings, is_endangered, lives_in_africa,
  lives_in_asia, lives_in_americas, is_small (< cat), is_large (> human), etc.
• Add clearly eliminated animals to `eliminated_animals` list based on the new traits.
  Example: if is_mammal=false just became confirmed, add all known mammals.
• Do NOT invent trait names not derivable from the question asked.

━━━ INQUISITOR HIERARCHY (follow strictly, in order) ━━━
L1 – Kingdom/Class:   mammal? bird? reptile? fish? amphibian? insect?
L2 – Size/Body:       size, legs, wings, fur/feathers/scales
L3 – Habitat/Range:   water/land/air, continent, wild/domestic
L4 – Diet/Behavior:   carnivore/herbivore/omnivore, nocturnal, social
L5 – Niche/Identity:  specific habitat, endangered, famous feature

Rules:
1. Do NOT ask a question whose answer is already in `confirmed_traits`.
2. CRITICAL — Do NOT repeat ANY question from `asked_questions`. Read every entry before choosing.
3. Pick the question that splits the remaining candidate space ~50/50 (max info gain).
4. Stay at the current hierarchy level until it is exhausted before advancing.
5. NEVER guess before turn {min_turns}. Even if you are 100% sure, ask more questions.
6. Only set action="guess" if confidence >= {threshold} AND turn >= {min_turns}.
7. At turn {max_turns}, you MUST guess (action="guess") with your best candidate.
8. The `question` field must be a natural Turkish yes/no question (ends with "?").
   Never use English in the `question` field.
9. If `candidates_remaining` <= 2 AND turn >= {min_turns}, set action="guess" with your best candidate.
10. If `consecutive_unknowns` >= 3, stop asking behavioral questions; pivot to geographic or
    identifying physical features that uniquely distinguish your top candidate.

━━━ OUTPUT FORMAT ━━━
Return ONLY a valid JSON object with exactly these keys:

{{
  "reasoning": "1-2 sentence English explanation of your choice",
  "new_traits": {{"trait_key": true_or_false}},
  "eliminated_animals": ["animal1", "animal2"],
  "action": "ask_question" | "guess",
  "question": "Turkish yes/no question string or null if guessing",
  "guess": "Animal name in Turkish or null if asking",
  "confidence": 0.0-1.0,
  "candidates_remaining": integer estimate of remaining candidates
}}

Do not include any text outside the JSON object."""


def _build_prompt(belief: BeliefState) -> str:
    return (
        _SYSTEM_PROMPT
        .replace("{min_turns}", str(MIN_TURNS_BEFORE_GUESS))
        .replace("{threshold}", str(GUESS_CONFIDENCE_THRESHOLD))
        .replace("{max_turns}", str(MAX_TURNS))
    )


# ─────────────────────────────────────────────
#  Rate-limit signal & retry-after parser (defined before _llm_call)
# ─────────────────────────────────────────────
class _RateLimitSignal(Exception):
    """Internal signal: rate limit hit, return 503 immediately."""
    def __init__(self, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limited — retry after {retry_after:.0f}s")


def _parse_retry_after(message: str) -> Optional[float]:
    """Extract seconds from strings like 'try again in 5m4.128s' or 'try again in 30s'."""
    m = re.search(r"try again in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s", message, re.I)
    if m:
        minutes = int(m.group(1) or 0)
        seconds = float(m.group(2) or 0)
        return minutes * 60 + seconds
    return None


# ─────────────────────────────────────────────
#  JSON extraction helper
# ─────────────────────────────────────────────
def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────
#  LLM call
# ─────────────────────────────────────────────
async def _llm_call(system: str, user_content: str) -> Optional[Dict[str, Any]]:
    from openai import RateLimitError

    client = AsyncOpenAI(
        api_key=settings.LLM_API_KEY or "dummy_key",
        base_url=settings.LLM_BASE_URL,
    )
    base = settings.LLM_BASE_URL.lower()
    # Providers that don't support / benefit from response_format json_object:
    # Groq doesn't support it; GLM-5.1 is a reasoning model and overthinks with it.
    skip_response_format = "groq" in base or "bigmodel" in base
    kwargs: Dict[str, Any] = dict(
        model=settings.LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0.15,
        max_tokens=2000,
    )
    if not skip_response_format:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = await client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        return _extract_json(raw)
    except RateLimitError as exc:
        msg = str(exc)
        wait = _parse_retry_after(msg)
        if wait is None:
            try:
                ra = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
                wait = float(ra) if ra else 60.0
            except Exception:
                wait = 60.0
        logger.warning("RateLimitError — retry_after=%.0fs", wait)
        raise _RateLimitSignal(retry_after=wait) from exc
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return None


# ─────────────────────────────────────────────
#  Server-side belief state update
# ─────────────────────────────────────────────
def _apply_llm_output(belief: BeliefState, output: LLMTurnOutput, question_asked: str) -> BeliefState:
    """
    Merge LLM output into BeliefState with conflict detection.
    Server owns the state — LLM output is advisory.
    """
    new_confirmed = dict(belief.confirmed)

    for trait, value in output.new_traits.items():
        if not isinstance(trait, str) or not isinstance(value, bool):
            continue
        key = trait.lower().strip().replace(" ", "_")
        # Conflict guard: if we already confirmed the opposite, skip silently
        if key in new_confirmed and new_confirmed[key] != value:
            logger.warning("Trait conflict ignored: %s was %s, LLM proposed %s", key, new_confirmed[key], value)
            continue
        new_confirmed[key] = value

    new_eliminated = list(set(belief.eliminated_animals + [
        a.strip() for a in output.eliminated_animals if isinstance(a, str) and a.strip()
    ]))

    new_scores = dict(belief.candidate_scores)
    for animal in new_eliminated:
        new_scores[animal.lower()] = 0.0

    new_asked = list(belief.asked_questions)
    if question_asked and question_asked not in new_asked:
        new_asked.append(question_asked)

    # Track consecutive unknowns for stagnation detection.
    # Only count when a question was actually answered (belief.last_question exists).
    # First turn has no last_question, so no stagnation is possible yet.
    is_unknown = bool(belief.last_question) and (not output.new_traits)
    new_consecutive = (belief.consecutive_unknowns + 1) if is_unknown else 0

    return BeliefState(
        turn=belief.turn + 1,
        confirmed=new_confirmed,
        candidate_scores=new_scores,
        eliminated_animals=new_eliminated,
        asked_questions=new_asked,
        last_question=question_asked or belief.last_question,
        confidence=output.confidence,
        top_candidate=output.guess or belief.top_candidate,
        consecutive_unknowns=new_consecutive,
    )


# ─────────────────────────────────────────────
#  Server-side action guard
# ─────────────────────────────────────────────
def _validate_action(
    output: LLMTurnOutput, belief: BeliefState, user_answer: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    Returns (final_action, override_reason | None).
    Enforces minimum-turn, confidence, and candidates guards server-side,
    so a hallucinating LLM cannot short-circuit or stall the game.
    """
    if output.action == "guess":
        if belief.turn < MIN_TURNS_BEFORE_GUESS:
            return "ask_question", f"Too early to guess (turn {belief.turn} < {MIN_TURNS_BEFORE_GUESS})"
        if output.confidence < GUESS_CONFIDENCE_THRESHOLD and belief.turn < MAX_TURNS:
            # Allow guess despite low confidence only when LLM explicitly reports 1-2 candidates
            few_left = 1 <= output.candidates_remaining <= 2
            if not few_left:
                return "ask_question", f"Confidence {output.confidence:.2f} < {GUESS_CONFIDENCE_THRESHOLD}"

    if output.action == "ask_question":
        # Force a guess if max turns reached
        if belief.turn >= MAX_TURNS:
            return "guess", f"Max turns ({MAX_TURNS}) reached, forcing guess"
        # Force a guess if LLM explicitly reports 1-2 candidates and we're past min turns
        # (0 means "not reported by LLM" — do NOT treat as near-certain)
        if 1 <= output.candidates_remaining <= 2 and belief.turn >= MIN_TURNS_BEFORE_GUESS:
            return "guess", f"candidates_remaining={output.candidates_remaining} — forcing guess"

    return output.action, None


# ─────────────────────────────────────────────
#  Session management
# ─────────────────────────────────────────────
def _evict_expired() -> None:
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s["created_at"] > SESSION_TTL]
    for sid in expired:
        _sessions.pop(sid, None)
    if len(_sessions) > MAX_SESSIONS:
        oldest = sorted(_sessions.items(), key=lambda x: x[1]["created_at"])
        for sid, _ in oldest[:len(_sessions) - MAX_SESSIONS]:
            _sessions.pop(sid, None)


def _get_or_create(session_id: str) -> BeliefState:
    _evict_expired()
    if session_id not in _sessions:
        _sessions[session_id] = {
            "belief": BeliefState(),
            "created_at": time.time(),
        }
    return _sessions[session_id]["belief"]


def _save(session_id: str, belief: BeliefState) -> None:
    if session_id in _sessions:
        _sessions[session_id]["belief"] = belief


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────
async def process_turn(session_id: str, user_answer: Optional[str] = None) -> Dict[str, Any]:
    """
    Single entry point called by the endpoint for both /start and /ask.
    Returns a dict matching the GameResponse / StartResponse schema.
    """
    belief = _get_or_create(session_id)
    system_prompt = _build_prompt(belief)

    user_content_parts = [f"BELIEF_STATE: {belief.model_dump_json()}"]

    # Inject asked_questions prominently to prevent duplicates
    if belief.asked_questions:
        forbidden = "\n".join(f"  - {q}" for q in belief.asked_questions)
        user_content_parts.append(
            f"FORBIDDEN QUESTIONS (already asked — DO NOT repeat any of these):\n{forbidden}"
        )

    if user_answer and belief.last_question:
        user_content_parts.append(f"LAST_QUESTION: {belief.last_question}")
        user_content_parts.append(f"USER_ANSWER: {user_answer}")
        if belief.consecutive_unknowns >= 2:
            user_content_parts.append(
                f"STAGNATION ALERT: {belief.consecutive_unknowns} consecutive 'Bilmiyorum' answers. "
                "Stop behavioral/social questions. Ask a specific geographic or unique physical "
                "identifying question about your top candidate to break the tie."
            )
    else:
        user_content_parts.append("LAST_QUESTION: null")
        user_content_parts.append("USER_ANSWER: null (this is the first turn, only choose the best first question)")

    try:
        raw = await _llm_call(system_prompt, "\n".join(user_content_parts))
    except _RateLimitSignal as rl:
        return {"type": "rate_limited", "retry_after": rl.retry_after}

    if raw is None:
        return {"type": "error", "content": "LLM call failed — check API key / connectivity."}

    # Parse into typed model with defaults for missing keys
    try:
        llm_output = LLMTurnOutput(
            reasoning=raw.get("reasoning", ""),
            new_traits=raw.get("new_traits") or {},
            eliminated_animals=raw.get("eliminated_animals") or [],
            action=raw.get("action", "ask_question"),
            question=raw.get("question"),
            guess=raw.get("guess"),
            confidence=float(raw.get("confidence", 0.0)),
            candidates_remaining=int(raw.get("candidates_remaining", 0)),
        )
    except Exception as exc:
        logger.error("LLM output parse error: %s | raw: %s", exc, raw)
        return {"type": "error", "content": "LLM returned an unparseable response."}

    # Server-side action guard
    final_action, override_reason = _validate_action(llm_output, belief, user_answer)
    if override_reason:
        logger.info("Action override: %s", override_reason)

    # Update belief state
    question_asked = llm_output.question if final_action == "ask_question" else None
    updated_belief = _apply_llm_output(belief, llm_output, question_asked or "")
    _save(session_id, updated_belief)

    reasoning_log = ReasoningLog(
        analysis=llm_output.reasoning,
        strategy=f"Turn {updated_belief.turn} | Traits confirmed: {len(updated_belief.confirmed)} | Eliminated: {len(updated_belief.eliminated_animals)}",
        confidence=llm_output.confidence,
        candidates_remaining=llm_output.candidates_remaining,
    )

    if final_action == "guess":
        guess_text = llm_output.guess or updated_belief.top_candidate or "Bilinmiyor"
        return {
            "action": "guess",
            "guess": guess_text,
            "reasoning": reasoning_log.model_dump(),
            "turn": updated_belief.turn,
        }

    question_text = llm_output.question
    if not question_text:
        question_text = "Bu bir memeli hayvan mıdır?"

    return {
        "action": "ask_question",
        "question": question_text,
        "reasoning": reasoning_log.model_dump(),
        "turn": updated_belief.turn,
    }
