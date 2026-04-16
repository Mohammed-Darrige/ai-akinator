"""LLM-native Akinator engine with server-side constraint validation."""

import json
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from app.core.config import settings
from app.models.schemas import GameState, ReasoningLog

logger = logging.getLogger(__name__)

SESSION_TTL = 1800
MAX_SESSIONS = 200
MAX_TURNS = 25
MAX_CONVERSATION_MESSAGES = 48
MAX_GENERATION_ATTEMPTS = 2
MIN_FACTS_BEFORE_GUESS = 2
MAX_CANDIDATES_BEFORE_GUESS = 3
MIN_SERVER_GUESS_CONFIDENCE = 0.68

_sessions: Dict[str, Dict[str, Any]] = {}

_LANGUAGE_NAMES = {
    "en": "English",
    "tr": "Turkish",
    "ar": "Arabic",
}

_YES = {"yes", "evet", "نعم", "اجل", "أجل"}
_NO = {"no", "hayır", "hayir", "لا"}
_UNKNOWN = {
    "unknown",
    "i don't know",
    "i dont know",
    "don't know",
    "dont know",
    "bilmiyorum",
    "لا أعرف",
    "لا اعرف",
}

_DISPLAY_ANSWERS: Dict[str, Dict[Optional[bool], str]] = {
    "en": {True: "Yes", False: "No", None: "I don't know"},
    "tr": {True: "Evet", False: "Hayır", None: "Bilmiyorum"},
    "ar": {True: "نعم", False: "لا", None: "لا أعرف"},
}

_START_MESSAGES = {
    "en": "The player is ready. Start the game with your best first yes/no question.",
    "tr": "Oyuncu hazır. Oyunu en iyi ilk evet-hayır sorunuzla başlat.",
    "ar": "اللاعب جاهز. ابدأ اللعبة بأفضل سؤال نعم أو لا أولي لديك.",
}

_REJECTED_GUESS_MESSAGES = {
    "en": "No, that guess is incorrect. Continue the game.",
    "tr": "Hayır, bu tahmin yanlış. Oyuna devam et.",
    "ar": "لا، هذا التخمين غير صحيح. تابع اللعبة.",
}

_GENERIC_QUESTION_PATTERNS = (
    re.compile(r"larger than (an|a) human", re.I),
    re.compile(r"average human", re.I),
    re.compile(r"commonly kept by humans", re.I),
    re.compile(r"humans? (keep|raise|feed)", re.I),
    re.compile(r"what does it eat", re.I),
    re.compile(r"does it eat", re.I),
    re.compile(r"is it carniv", re.I),
)

_TRAIT_KEY_HINTS = {
    "is_bird": ("bird", "avian", "kus", "kuş", "طائر"),
    "is_mammal": ("mammal", "memeli", "ثديي"),
    "has_wings": ("wing", "kanat", "جناح"),
    "can_fly": ("fly", "u\u00e7", "uç", "يطير"),
    "swims": ("swim", "yuz", "yüz", "يسبح"),
    "lives_in_water": ("water", "aquatic", "suda", "su", "ماء", "مائي"),
    "lays_eggs": ("egg", "yumurta", "بيض"),
    "is_nocturnal": ("nocturnal", "night", "gece", "ليلي"),
    "has_fur": ("fur", "hair", "kurk", "kürk", "فرو", "شعر"),
    "has_feathers": ("feather", "tuy", "tüy", "ريش"),
    "camouflages": ("camouflage", "change color", "renk", "تمويه", "لون"),
}


class _RateLimitSignal(Exception):
    def __init__(self, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limited. Retry after {retry_after:.0f}s")


class _ProviderBillingSignal(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _create_client():
    from openai import AsyncOpenAI

    if not settings.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is not configured on the backend.")

    return AsyncOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=_normalize_base_url(settings.LLM_BASE_URL),
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )


def _parse_answer(raw: str) -> Optional[bool]:
    normalized = re.sub(r"\s+", " ", raw.strip().lower())
    if normalized in _YES:
        return True
    if normalized in _NO:
        return False
    if normalized in _UNKNOWN:
        return None
    return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip().replace("```json", "").replace("```", "")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _parse_retry_after(message: str) -> Optional[float]:
    match = re.search(r"try again in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s", message, re.I)
    if not match:
        return None
    return int(match.group(1) or 0) * 60 + float(match.group(2) or 0)


def _extract_provider_error(exc: Exception) -> Tuple[str, str]:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "").strip()
            message = _safe_text(error.get("message"), "")
            if code or message:
                return code, message

    message = _safe_text(str(exc), "")
    code_match = re.search(r'"code"\s*:\s*"?(\d+)"?', message)
    code = code_match.group(1) if code_match else ""
    return code, message


def _is_billing_or_package_issue(code: str, message: str) -> bool:
    lowered = message.lower()
    if code == "1113":
        return True

    markers = (
        "insufficient balance",
        "no resource package",
        "please recharge",
        "insufficient_quota",
        "quota exceeded",
        "余额不足",
        "无可用资源包",
        "请充值",
    )
    return any(marker in lowered for marker in markers)


def _friendly_billing_message(provider_message: str) -> str:
    base = "LLM provider rejected this request due to insufficient API balance or no active resource package for this API key."
    if provider_message:
        return f"{base} Provider message: {provider_message}"
    return base


def _safe_text(value: Any, fallback: str = "") -> str:
    if not isinstance(value, str):
        return fallback
    cleaned = value.strip()
    if cleaned.lower() in {"", "undefined", "null", "none", "n/a"}:
        return fallback
    return cleaned


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_trait(value: str) -> str:
    cleaned = re.sub(r"\s+", "_", value.strip().lower())
    return re.sub(r"[^a-z0-9_]", "", cleaned)


def _display_answer(answer: Optional[bool], language: str) -> str:
    localized = _DISPLAY_ANSWERS.get(language, _DISPLAY_ANSWERS["en"])
    return localized[answer]


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _ledger_lines(state: GameState) -> str:
    if not state.ledger.facts:
        return "(empty)"
    return "\n".join(
        f"- {trait}: {str(value).lower()}"
        for trait, value in state.ledger.facts.items()
    )


def _build_system_prompt(state: GameState) -> str:
    language_name = _LANGUAGE_NAMES.get(state.language, "English")
    asked_traits = ", ".join(state.asked_traits) if state.asked_traits else "(none)"

    return f"""You are the reasoning engine for a live animal deduction game.

The player has exactly one real animal in mind. You decide whether to ask one new yes/no question or make one final guess.

Non-negotiable rules:
- Be fully dynamic. Do not use a canned tree, a fixed catalog, or a memorized question sequence.
- Treat the server constraint ledger as absolute truth.
- Use the full conversation history already provided in this request.
- Ask only one yes/no question per turn.
- Every question must introduce one new `trait_key` written in snake_case English.
- Never reuse a `trait_key` that already appears in the asked trait list or the confirmed ledger.
- Avoid generic or near-universal traits unless they are the single best discriminator at this point.
- Prefer concrete, distinguishing traits over broad survival traits.
- If you are not confident enough to guess, ask a more discriminating question.
- If you guess, the guess must satisfy every confirmed fact.
- All visible text fields must be written entirely in {language_name}.
- Return JSON only. Do not add markdown or prose outside the JSON object.

Current turn index: {state.turn}
Already asked traits: {asked_traits}
Confirmed constraint ledger:
{_ledger_lines(state)}

Return exactly this JSON shape:
{{
  "chain_of_thought": "Reasoning in {language_name}",
  "top_candidates": [
    {{"name": "candidate name in {language_name}", "probability": 0.0}}
  ],
  "action": "ask_question" | "guess",
  "question": "A single yes/no question in {language_name}, or an empty string when guessing",
  "trait_key": "snake_case_english_trait_key, or an empty string when guessing",
  "guess": "Final animal name in {language_name}, or an empty string when asking",
  "confidence": 0.0,
  "candidates_remaining": 1
}}"""


def _runtime_state_packet(
    state: GameState,
    *,
    force_guess: bool,
    corrective_instruction: Optional[str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": "server_state",
        "turn": state.turn,
        "confirmed_constraints": state.ledger.facts,
        "qa_history": [
            {"question": question, "answer": answer}
            for question, answer in state.ledger.qa_history
        ],
        "asked_traits": state.asked_traits,
        "last_question": state.last_question or "",
        "pending_guess": state.pending_guess or "",
    }
    if force_guess:
        payload["instruction"] = (
            "This turn must end with a guess unless the output would break a confirmed constraint."
        )
    if corrective_instruction:
        payload["validation_feedback"] = corrective_instruction
    return payload


def _build_messages(
    state: GameState,
    *,
    force_guess: bool = False,
    corrective_instruction: Optional[str] = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _build_system_prompt(state)}
    ]
    messages.extend(state.conversation[-MAX_CONVERSATION_MESSAGES:])
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                _runtime_state_packet(
                    state,
                    force_guess=force_guess,
                    corrective_instruction=corrective_instruction,
                ),
                ensure_ascii=False,
            ),
        }
    )
    return messages


async def _llm_call_stream(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    from openai import RateLimitError

    client = _create_client()
    base = _normalize_base_url(settings.LLM_BASE_URL).lower()
    supports_response_format = (
        "groq" not in base and "bigmodel" not in base and "z.ai" not in base
    )

    kwargs: Dict[str, Any] = {
        "model": settings.LLM_MODEL_NAME,
        "messages": messages,
        "temperature": 0.15,
        "max_tokens": 2500,
        "stream": True,
    }
    if supports_response_format:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = await client.chat.completions.create(**kwargs)
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except RateLimitError as exc:
        code, message = _extract_provider_error(exc)
        if _is_billing_or_package_issue(code, message):
            raise _ProviderBillingSignal(_friendly_billing_message(message)) from exc

        wait = _parse_retry_after(message) or _parse_retry_after(str(exc)) or 60.0
        raise _RateLimitSignal(wait) from exc


async def _llm_call_json(
    messages: List[Dict[str, str]], *, temperature: float = 0.0
) -> Dict[str, Any]:
    from openai import RateLimitError

    client = _create_client()
    base = _normalize_base_url(settings.LLM_BASE_URL).lower()
    supports_response_format = (
        "groq" not in base and "bigmodel" not in base and "z.ai" not in base
    )

    kwargs: Dict[str, Any] = {
        "model": settings.LLM_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 800,
    }
    if supports_response_format:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = await client.chat.completions.create(**kwargs)
    except RateLimitError as exc:
        code, message = _extract_provider_error(exc)
        if _is_billing_or_package_issue(code, message):
            raise _ProviderBillingSignal(_friendly_billing_message(message)) from exc

        wait = _parse_retry_after(message) or _parse_retry_after(str(exc)) or 60.0
        raise _RateLimitSignal(wait) from exc

    content = response.choices[0].message.content or ""
    parsed = _extract_json(content)
    if parsed is None:
        raise ValueError("Failed to parse JSON response from validator.")
    return parsed


def _parse_llm_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    action = _safe_text(raw.get("action"), "ask_question")
    if action not in {"ask_question", "guess"}:
        action = "ask_question"

    top_candidates: List[Dict[str, Any]] = []
    for item in (raw.get("top_candidates") or [])[:8]:
        if isinstance(item, dict):
            name = _safe_text(item.get("name"))
            probability = max(0.0, min(1.0, _safe_float(item.get("probability"), 0.0)))
            if name:
                top_candidates.append({"name": name, "probability": probability})

    return {
        "chain_of_thought": _safe_text(raw.get("chain_of_thought")),
        "top_candidates": top_candidates,
        "action": action,
        "question": _safe_text(raw.get("question")),
        "trait_key": _normalize_trait(_safe_text(raw.get("trait_key"))),
        "guess": _safe_text(raw.get("guess")),
        "confidence": max(0.0, min(1.0, _safe_float(raw.get("confidence"), 0.0))),
        "candidates_remaining": max(1, _safe_int(raw.get("candidates_remaining"), 1)),
    }


def _derive_server_metrics(state: GameState, parsed: Dict[str, Any]) -> Dict[str, Any]:
    top_candidates = parsed["top_candidates"]
    facts_count = len(state.ledger.facts)

    if top_candidates:
        raw_probabilities = [
            max(0.0, candidate["probability"]) for candidate in top_candidates
        ]
        total = sum(raw_probabilities)
        if total <= 0:
            normalized = [1.0 / len(top_candidates)] * len(top_candidates)
        else:
            normalized = [value / total for value in raw_probabilities]

        for candidate, probability in zip(top_candidates, normalized):
            candidate["probability"] = probability

        top_probability = normalized[0]
        visible_candidates = max(1, sum(1 for value in normalized if value >= 0.08))
        coverage_bonus = min(0.18, facts_count * 0.04)
        derived_confidence = min(0.99, top_probability + coverage_bonus)
        parsed["confidence"] = min(parsed["confidence"] or 1.0, derived_confidence)
        parsed["candidates_remaining"] = visible_candidates
        return parsed

    parsed["confidence"] = min(
        parsed["confidence"], 0.2 + min(0.18, facts_count * 0.04)
    )
    return parsed


def _assistant_history_message(parsed: Dict[str, Any]) -> Dict[str, str]:
    payload = {
        "chain_of_thought": parsed["chain_of_thought"],
        "top_candidates": parsed["top_candidates"],
        "action": parsed["action"],
        "question": parsed["question"],
        "trait_key": parsed["trait_key"],
        "guess": parsed["guess"],
        "confidence": parsed["confidence"],
        "candidates_remaining": parsed["candidates_remaining"],
    }
    return {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}


def _reasoning_log(
    state: GameState,
    parsed: Dict[str, Any],
    *,
    has_contradiction: bool = False,
) -> ReasoningLog:
    return ReasoningLog(
        chain_of_thought=parsed["chain_of_thought"],
        confidence=parsed["confidence"],
        candidates_remaining=parsed["candidates_remaining"],
        constraints=dict(state.ledger.facts),
        qa_history=list(state.ledger.qa_history),
        trait_labels=dict(state.trait_labels),
        top_candidates=parsed["top_candidates"],
        has_contradiction=has_contradiction,
    )


def _validate_turn_output(
    state: GameState, parsed: Dict[str, Any], *, force_guess: bool
) -> Optional[str]:
    action = parsed["action"]
    if force_guess and action != "guess":
        return (
            "You must return action='guess' because the maximum turn limit was reached."
        )

    if action == "ask_question":
        if not parsed["question"]:
            return "Return a non-empty yes/no question when action='ask_question'."
        if not parsed["chain_of_thought"]:
            return (
                "Return a non-empty chain_of_thought explaining the current reasoning."
            )
        if not parsed["trait_key"]:
            return "Return a non-empty snake_case English trait_key when action='ask_question'."
        if (
            parsed["trait_key"] in state.asked_traits
            or parsed["trait_key"] in state.ledger.facts
        ):
            return (
                "The proposed trait_key was already used. Ask about a different trait."
            )
        return None

    if not parsed["chain_of_thought"]:
        return (
            "Return a non-empty chain_of_thought explaining why the guess is justified."
        )
    if not parsed["guess"]:
        return "Return a non-empty guess when action='guess'."
    if not force_guess:
        if len(state.ledger.facts) < MIN_FACTS_BEFORE_GUESS:
            return "There are not enough confirmed facts yet. Ask another question instead of guessing."
        if parsed["candidates_remaining"] > MAX_CANDIDATES_BEFORE_GUESS:
            return "Too many plausible candidates remain. Ask a more discriminating question instead of guessing."
        if parsed["confidence"] < MIN_SERVER_GUESS_CONFIDENCE:
            return (
                "Confidence is too low for a final guess. Ask another question instead."
            )
    return None


async def _validate_guess_against_constraints(
    guess_name: str,
    state: GameState,
) -> Tuple[bool, str]:
    if not state.ledger.facts:
        return True, "No confirmed constraints to validate."

    messages = [
        {
            "role": "system",
            "content": "You are a strict animal consistency validator. Respond with JSON only.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Check whether the proposed guess is consistent with every confirmed fact. If any fact conflicts, return false.",
                    "guess": guess_name,
                    "constraints": state.ledger.facts,
                    "qa_history": [
                        {"question": question, "answer": answer}
                        for question, answer in state.ledger.qa_history
                    ],
                    "required_output": {
                        "is_consistent": True,
                        "failed_checks": ["short description"],
                        "reasoning": "brief explanation",
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]

    parsed = await _llm_call_json(messages)
    is_consistent = parsed.get("is_consistent")
    reasoning = _safe_text(parsed.get("reasoning"), "Validator returned no reasoning.")

    if not isinstance(is_consistent, bool):
        raise ValueError("Validator did not return a boolean is_consistent field.")

    return is_consistent, reasoning


def _validate_question_trait_alignment(
    question: str,
    trait_key: str,
    state: GameState,
) -> Tuple[bool, str]:
    del state

    normalized_question = question.strip().lower()
    if len(normalized_question) < 8:
        return False, "Question is too short to be useful."

    for pattern in _GENERIC_QUESTION_PATTERNS:
        if pattern.search(normalized_question):
            return False, "Question is too generic and not discriminative enough."

    explicit_hints = _TRAIT_KEY_HINTS.get(trait_key, ())
    if explicit_hints and any(hint in normalized_question for hint in explicit_hints):
        return True, "Question matches known trait hints."

    trait_tokens = [token for token in trait_key.split("_") if len(token) >= 3]
    if trait_tokens and any(token in normalized_question for token in trait_tokens):
        return True, "Question contains trait keywords."

    if explicit_hints:
        return False, "Question text does not match the proposed trait_key."

    return True, "No strong contradiction detected for trait alignment."


def _evict_expired() -> None:
    now = time.time()
    expired_ids = [
        session_id
        for session_id, data in _sessions.items()
        if now - data["ts"] > SESSION_TTL
    ]
    for session_id in expired_ids:
        _sessions.pop(session_id, None)

    if len(_sessions) > MAX_SESSIONS:
        oldest_first = sorted(_sessions.items(), key=lambda item: item[1]["ts"])
        for session_id, _ in oldest_first[: len(_sessions) - MAX_SESSIONS]:
            _sessions.pop(session_id, None)


def _get_or_create(session_id: str, language: str) -> GameState:
    _evict_expired()
    if session_id not in _sessions:
        _sessions[session_id] = {
            "state": GameState(language=language),
            "ts": time.time(),
        }

    state = _sessions[session_id]["state"]
    if state.language != language:
        state.language = language
    return state


def _save(session_id: str, state: GameState) -> None:
    _sessions[session_id] = {"state": state, "ts": time.time()}


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def _record_answer(state: GameState, answer: Optional[bool]) -> None:
    answer_text = _display_answer(answer, state.language)
    if state.last_trait_key:
        if answer is not None:
            state.ledger.facts[state.last_trait_key] = answer
        state.ledger.qa_history.append((state.last_question or "", answer_text))
        if state.last_question:
            state.trait_labels[state.last_trait_key] = state.last_question

    state.conversation.append({"role": "user", "content": answer_text})
    state.turn += 1


def _record_guess_rejection(state: GameState) -> None:
    if not state.pending_guess:
        return

    state.conversation.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "action": "guess",
                    "guess": state.pending_guess,
                    "result": "rejected_by_player",
                },
                ensure_ascii=False,
            ),
        }
    )
    state.conversation.append(
        {
            "role": "user",
            "content": _REJECTED_GUESS_MESSAGES.get(
                state.language,
                _REJECTED_GUESS_MESSAGES["en"],
            ),
        }
    )
    state.pending_guess = None
    state.turn += 1


async def process_turn_stream(
    session_id: str,
    user_answer: Optional[str] = None,
    language: str = "en",
) -> AsyncGenerator[str, None]:
    state = _get_or_create(session_id, language)

    if state.game_over:
        yield _sse(
            {
                "type": "error",
                "content": "This session is already complete. Start a new game.",
            }
        )
        return

    if user_answer is None:
        if not state.conversation:
            state.conversation.append(
                {
                    "role": "user",
                    "content": _START_MESSAGES.get(language, _START_MESSAGES["en"]),
                }
            )
    else:
        answer = _parse_answer(user_answer)
        if state.pending_guess:
            if answer is True:
                confirmed_guess = state.pending_guess
                state.pending_guess = None
                state.game_over = True
                reasoning = ReasoningLog(
                    chain_of_thought="",
                    confidence=1.0,
                    candidates_remaining=1,
                    constraints=dict(state.ledger.facts),
                    qa_history=list(state.ledger.qa_history),
                    trait_labels=dict(state.trait_labels),
                    top_candidates=state.top_candidates,
                )
                _save(session_id, state)
                yield _sse(
                    {
                        "type": "result",
                        "action": "guess_correct",
                        "guess": confirmed_guess,
                        "reasoning": reasoning.model_dump(),
                        "turn": state.turn,
                    }
                )
                return

            if answer is None:
                yield _sse(
                    {
                        "type": "error",
                        "content": "Please confirm the guess with yes or no.",
                    }
                )
                return

            _record_guess_rejection(state)
        else:
            _record_answer(state, answer)

    force_guess = state.turn >= MAX_TURNS
    corrective_instruction: Optional[str] = None

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        raw_text = ""
        messages = _build_messages(
            state,
            force_guess=force_guess,
            corrective_instruction=corrective_instruction,
        )

        try:
            async for chunk in _llm_call_stream(messages):
                raw_text += chunk
                yield _sse({"type": "chunk", "text": chunk})
        except _RateLimitSignal as exc:
            yield _sse(
                {
                    "type": "error",
                    "error_type": "rate_limited",
                    "retry_after": int(exc.retry_after),
                    "content": f"Rate limited. Retry after {int(exc.retry_after)}s.",
                }
            )
            return
        except _ProviderBillingSignal as exc:
            yield _sse(
                {
                    "type": "error",
                    "error_type": "billing_required",
                    "content": str(exc),
                }
            )
            return
        except Exception as exc:
            logger.exception("LLM call failed")
            yield _sse({"type": "error", "content": f"AI service error: {exc}"})
            return

        raw_json = _extract_json(raw_text)
        if raw_json is None:
            yield _sse({"type": "error", "content": "Failed to parse AI response."})
            return

        parsed = _parse_llm_response(raw_json)
        parsed = _derive_server_metrics(state, parsed)
        validation_error = _validate_turn_output(state, parsed, force_guess=force_guess)
        if validation_error:
            if attempt < MAX_GENERATION_ATTEMPTS - 1:
                corrective_instruction = validation_error
                yield _sse({"type": "retry", "reason": validation_error})
                continue
            yield _sse({"type": "error", "content": validation_error})
            return

        if parsed["action"] == "guess":
            try:
                (
                    is_consistent,
                    validator_reasoning,
                ) = await _validate_guess_against_constraints(parsed["guess"], state)
            except _RateLimitSignal as exc:
                yield _sse(
                    {
                        "type": "error",
                        "error_type": "rate_limited",
                        "retry_after": int(exc.retry_after),
                        "content": f"Rate limited. Retry after {int(exc.retry_after)}s.",
                    }
                )
                return
            except _ProviderBillingSignal as exc:
                yield _sse(
                    {
                        "type": "error",
                        "error_type": "billing_required",
                        "content": str(exc),
                    }
                )
                return
            except Exception as exc:
                logger.exception("Constraint validation failed")
                yield _sse(
                    {"type": "error", "content": f"Constraint validation failed: {exc}"}
                )
                return

            if not is_consistent:
                if attempt < MAX_GENERATION_ATTEMPTS - 1:
                    corrective_instruction = (
                        f"Your last guess was rejected by the hard constraint validator. "
                        f"Reason: {validator_reasoning}. Produce a new valid response."
                    )
                    yield _sse(
                        {
                            "type": "retry",
                            "reason": "Hard constraint validator rejected the guess.",
                        }
                    )
                    continue
                yield _sse(
                    {
                        "type": "error",
                        "content": "The hard constraint validator rejected the final guess.",
                    }
                )
                return

        if parsed["action"] == "ask_question":
            try:
                is_valid_question, question_reasoning = (
                    _validate_question_trait_alignment(
                        parsed["question"],
                        parsed["trait_key"],
                        state,
                    )
                )
            except Exception as exc:
                logger.exception("Question validation failed")
                yield _sse(
                    {"type": "error", "content": f"Question validation failed: {exc}"}
                )
                return

            if not is_valid_question:
                if attempt < MAX_GENERATION_ATTEMPTS - 1:
                    corrective_instruction = (
                        f"Your last question was rejected by the server validator. "
                        f"Reason: {question_reasoning}. Produce a better question with a correctly aligned trait_key."
                    )
                    yield _sse(
                        {
                            "type": "retry",
                            "reason": "Question validator rejected the proposed trait/question pair.",
                        }
                    )
                    continue
                yield _sse(
                    {
                        "type": "error",
                        "content": "The server validator rejected the proposed question.",
                    }
                )
                return

        state.confidence = parsed["confidence"]
        state.top_candidates = parsed["top_candidates"]
        state.conversation.append(_assistant_history_message(parsed))

        if parsed["action"] == "ask_question":
            state.last_question = parsed["question"]
            state.last_trait_key = parsed["trait_key"]
            state.asked_traits.append(parsed["trait_key"])

            reasoning = _reasoning_log(state, parsed)
            _save(session_id, state)
            yield _sse(
                {
                    "type": "result",
                    "action": "ask_question",
                    "question": parsed["question"],
                    "reasoning": reasoning.model_dump(),
                    "turn": state.turn,
                }
            )
            return

        state.pending_guess = parsed["guess"]
        reasoning = _reasoning_log(state, parsed)
        _save(session_id, state)
        yield _sse(
            {
                "type": "result",
                "action": "guess",
                "guess": parsed["guess"],
                "reasoning": reasoning.model_dump(),
                "turn": state.turn,
            }
        )
        return

    yield _sse(
        {"type": "error", "content": "The model could not produce a valid turn."}
    )
