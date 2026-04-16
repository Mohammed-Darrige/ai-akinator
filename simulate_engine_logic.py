import asyncio
import json
from collections import deque
from contextlib import contextmanager
from typing import Any, AsyncGenerator, Callable, Deque, Dict, Iterable

from app.services import llm_service


def decode_sse(event: str) -> Dict[str, Any]:
    prefix = "data: "
    if not event.startswith(prefix):
        raise ValueError(f"Unexpected SSE payload: {event!r}")
    return json.loads(event[len(prefix) :].strip())


async def collect_events(generator: AsyncGenerator[str, None]) -> list[Dict[str, Any]]:
    events: list[Dict[str, Any]] = []
    async for item in generator:
        events.append(decode_sse(item))
    return events


@contextmanager
def patched_engine(
    scripted_turns: Iterable[Dict[str, Any]],
    *,
    question_validator: Callable[[Dict[str, Any]], Dict[str, Any]],
    guess_validator: Callable[[Dict[str, Any]], Dict[str, Any]],
):
    scripted_queue: Deque[Dict[str, Any]] = deque(scripted_turns)
    original_stream = llm_service._llm_call_stream
    original_json = llm_service._llm_call_json

    async def fake_stream(_: list[Dict[str, str]]) -> AsyncGenerator[str, None]:
        if not scripted_queue:
            raise RuntimeError("No scripted LLM turns remaining.")
        yield json.dumps(scripted_queue.popleft())

    async def fake_json(
        messages: list[Dict[str, str]], *, temperature: float = 0.0
    ) -> Dict[str, Any]:
        del temperature
        payload = json.loads(messages[-1]["content"])
        task = payload.get("task", "")
        if "semantically aligned" in task:
            return question_validator(payload)
        if "proposed guess is consistent" in task:
            return guess_validator(payload)
        raise RuntimeError(f"Unexpected validator task: {task}")

    llm_service._llm_call_stream = fake_stream
    llm_service._llm_call_json = fake_json
    try:
        yield
    finally:
        llm_service._llm_call_stream = original_stream
        llm_service._llm_call_json = original_json
        llm_service._sessions.clear()


def default_question_validator(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = payload["question"].lower()
    trait_key = payload["trait_key"]
    expected_pairs = {
        "is_bird": "bird",
        "can_fly": "fly",
        "swims": "swim",
        "is_mammal": "mammal",
        "has_wings": "wing",
        "is_nocturnal": "nocturnal",
        "lays_eggs": "egg",
        "lives_in_water": "water",
    }
    token = expected_pairs.get(trait_key, trait_key.replace("_", " "))
    return {
        "is_valid": token in question,
        "reasoning": "aligned"
        if token in question
        else "mismatched question and trait_key",
    }


def default_guess_validator(payload: Dict[str, Any]) -> Dict[str, Any]:
    guess = str(payload["guess"]).lower()
    constraints = payload["constraints"]

    if guess == "rabbit" and constraints.get("has_wings") is True:
        return {
            "is_consistent": False,
            "failed_checks": ["rabbit does not have wings"],
            "reasoning": "Rabbit violates has_wings=true.",
        }

    if guess == "eagle" and constraints.get("has_wings") is False:
        return {
            "is_consistent": False,
            "failed_checks": ["eagle has wings"],
            "reasoning": "Eagle violates has_wings=false.",
        }

    if guess == "cat" and constraints.get("lays_eggs") is True:
        return {
            "is_consistent": False,
            "failed_checks": ["cat does not lay eggs"],
            "reasoning": "Cat violates lays_eggs=true.",
        }

    return {
        "is_consistent": True,
        "failed_checks": [],
        "reasoning": "consistent",
    }


def assert_result(events: list[Dict[str, Any]], expected_action: str) -> Dict[str, Any]:
    result = next((event for event in events if event.get("type") == "result"), None)
    if result is None:
        raise AssertionError(f"No result event found in {events!r}")
    if result.get("action") != expected_action:
        raise AssertionError(
            f"Expected action '{expected_action}', got '{result.get('action')}'"
        )
    return result


async def test_penguin_flow() -> None:
    scripted_turns = [
        {
            "chain_of_thought": "Birds and mammals split the candidate space well.",
            "top_candidates": [{"name": "penguin", "probability": 0.52}],
            "action": "ask_question",
            "question": "Is it a bird?",
            "trait_key": "is_bird",
            "guess": "",
            "confidence": 0.52,
            "candidates_remaining": 6,
        },
        {
            "chain_of_thought": "A flight question separates penguins from many other birds.",
            "top_candidates": [{"name": "penguin", "probability": 0.74}],
            "action": "ask_question",
            "question": "Can it fly?",
            "trait_key": "can_fly",
            "guess": "",
            "confidence": 0.74,
            "candidates_remaining": 4,
        },
        {
            "chain_of_thought": "Swimming strongly points toward penguins among flightless birds.",
            "top_candidates": [{"name": "penguin", "probability": 0.91}],
            "action": "ask_question",
            "question": "Does it spend a lot of time swimming?",
            "trait_key": "swims",
            "guess": "",
            "confidence": 0.91,
            "candidates_remaining": 2,
        },
        {
            "chain_of_thought": "Bird, cannot fly, and swims matches penguin best.",
            "top_candidates": [{"name": "penguin", "probability": 0.96}],
            "action": "guess",
            "question": "",
            "trait_key": "",
            "guess": "penguin",
            "confidence": 0.96,
            "candidates_remaining": 1,
        },
    ]

    with patched_engine(
        scripted_turns,
        question_validator=default_question_validator,
        guess_validator=default_guess_validator,
    ):
        session = "penguin-session"
        start_events = await collect_events(
            llm_service.process_turn_stream(session, language="en")
        )
        assert_result(start_events, "ask_question")

        second = await collect_events(
            llm_service.process_turn_stream(session, "yes", language="en")
        )
        assert_result(second, "ask_question")

        third = await collect_events(
            llm_service.process_turn_stream(session, "no", language="en")
        )
        assert_result(third, "ask_question")

        guess_events = await collect_events(
            llm_service.process_turn_stream(session, "yes", language="en")
        )
        guess = assert_result(guess_events, "guess")
        if guess.get("guess") != "penguin":
            raise AssertionError(f"Expected penguin guess, got {guess.get('guess')!r}")

        confirm_events = await collect_events(
            llm_service.process_turn_stream(session, "yes", language="en")
        )
        final = assert_result(confirm_events, "guess_correct")
        if final.get("guess") != "penguin":
            raise AssertionError("Penguin confirmation failed.")


async def test_bat_constraint_gate() -> None:
    scripted_turns = [
        {
            "chain_of_thought": "Mammal is the best first split.",
            "top_candidates": [{"name": "bat", "probability": 0.48}],
            "action": "ask_question",
            "question": "Is it a mammal?",
            "trait_key": "is_mammal",
            "guess": "",
            "confidence": 0.48,
            "candidates_remaining": 7,
        },
        {
            "chain_of_thought": "Wings separate bats from many mammals.",
            "top_candidates": [{"name": "bat", "probability": 0.69}],
            "action": "ask_question",
            "question": "Does it have wings?",
            "trait_key": "has_wings",
            "guess": "",
            "confidence": 0.69,
            "candidates_remaining": 4,
        },
        {
            "chain_of_thought": "Nocturnal behavior narrows the set further.",
            "top_candidates": [{"name": "bat", "probability": 0.86}],
            "action": "ask_question",
            "question": "Is it mostly nocturnal?",
            "trait_key": "is_nocturnal",
            "guess": "",
            "confidence": 0.86,
            "candidates_remaining": 2,
        },
        {
            "chain_of_thought": "I am leaning toward rabbit.",
            "top_candidates": [{"name": "rabbit", "probability": 0.9}],
            "action": "guess",
            "question": "",
            "trait_key": "",
            "guess": "rabbit",
            "confidence": 0.9,
            "candidates_remaining": 1,
        },
        {
            "chain_of_thought": "The constraints actually match bat, not rabbit.",
            "top_candidates": [{"name": "bat", "probability": 0.94}],
            "action": "guess",
            "question": "",
            "trait_key": "",
            "guess": "bat",
            "confidence": 0.94,
            "candidates_remaining": 1,
        },
    ]

    with patched_engine(
        scripted_turns,
        question_validator=default_question_validator,
        guess_validator=default_guess_validator,
    ):
        session = "bat-session"
        await collect_events(llm_service.process_turn_stream(session, language="en"))
        await collect_events(
            llm_service.process_turn_stream(session, "yes", language="en")
        )
        await collect_events(
            llm_service.process_turn_stream(session, "yes", language="en")
        )
        events = await collect_events(
            llm_service.process_turn_stream(session, "yes", language="en")
        )

        if not any(event.get("type") == "retry" for event in events):
            raise AssertionError("Expected retry event after invalid rabbit guess.")

        result = assert_result(events, "guess")
        if result.get("guess") != "bat":
            raise AssertionError(
                f"Expected final guess 'bat', got {result.get('guess')!r}"
            )

        confirm_events = await collect_events(
            llm_service.process_turn_stream(session, "yes", language="en")
        )
        final = assert_result(confirm_events, "guess_correct")
        if final.get("guess") != "bat":
            raise AssertionError("Bat confirmation failed.")


async def test_platypus_flow() -> None:
    scripted_turns = [
        {
            "chain_of_thought": "Mammal is a strong first partition.",
            "top_candidates": [{"name": "platypus", "probability": 0.41}],
            "action": "ask_question",
            "question": "Is it a mammal?",
            "trait_key": "is_mammal",
            "guess": "",
            "confidence": 0.41,
            "candidates_remaining": 8,
        },
        {
            "chain_of_thought": "Egg laying is rare among mammals.",
            "top_candidates": [{"name": "platypus", "probability": 0.7}],
            "action": "ask_question",
            "question": "Does it lay eggs?",
            "trait_key": "lays_eggs",
            "guess": "",
            "confidence": 0.7,
            "candidates_remaining": 3,
        },
        {
            "chain_of_thought": "Water habitat now strongly points to platypus.",
            "top_candidates": [{"name": "platypus", "probability": 0.9}],
            "action": "ask_question",
            "question": "Does it spend much of its time in the water?",
            "trait_key": "lives_in_water",
            "guess": "",
            "confidence": 0.9,
            "candidates_remaining": 2,
        },
        {
            "chain_of_thought": "An egg-laying mammal that lives in water matches platypus.",
            "top_candidates": [{"name": "platypus", "probability": 0.95}],
            "action": "guess",
            "question": "",
            "trait_key": "",
            "guess": "platypus",
            "confidence": 0.95,
            "candidates_remaining": 1,
        },
    ]

    with patched_engine(
        scripted_turns,
        question_validator=default_question_validator,
        guess_validator=default_guess_validator,
    ):
        session = "platypus-session"
        await collect_events(llm_service.process_turn_stream(session, language="en"))
        await collect_events(
            llm_service.process_turn_stream(session, "yes", language="en")
        )
        await collect_events(
            llm_service.process_turn_stream(session, "yes", language="en")
        )
        guess_events = await collect_events(
            llm_service.process_turn_stream(session, "yes", language="en")
        )
        guess = assert_result(guess_events, "guess")
        if guess.get("guess") != "platypus":
            raise AssertionError("Expected platypus guess.")

        confirm_events = await collect_events(
            llm_service.process_turn_stream(session, "yes", language="en")
        )
        final = assert_result(confirm_events, "guess_correct")
        if final.get("guess") != "platypus":
            raise AssertionError("Platypus confirmation failed.")


async def test_mismatched_question_rejected() -> None:
    scripted_turns = [
        {
            "chain_of_thought": "I will ask about wings.",
            "top_candidates": [{"name": "bat", "probability": 0.4}],
            "action": "ask_question",
            "question": "Does it have wings?",
            "trait_key": "is_mammal",
            "guess": "",
            "confidence": 0.4,
            "candidates_remaining": 6,
        },
        {
            "chain_of_thought": "I corrected the trait mapping.",
            "top_candidates": [{"name": "bat", "probability": 0.52}],
            "action": "ask_question",
            "question": "Does it have wings?",
            "trait_key": "has_wings",
            "guess": "",
            "confidence": 0.52,
            "candidates_remaining": 4,
        },
    ]

    with patched_engine(
        scripted_turns,
        question_validator=default_question_validator,
        guess_validator=default_guess_validator,
    ):
        events = await collect_events(
            llm_service.process_turn_stream("mismatch-session", language="en")
        )
        if not any(event.get("type") == "retry" for event in events):
            raise AssertionError("Expected retry for mismatched question/trait pair.")
        result = assert_result(events, "ask_question")
        if result.get("question") != "Does it have wings?":
            raise AssertionError("Expected corrected wings question to survive retry.")


async def main() -> None:
    await test_mismatched_question_rejected()
    await test_penguin_flow()
    await test_bat_constraint_gate()
    await test_platypus_flow()
    print("engine-logic: OK")


if __name__ == "__main__":
    asyncio.run(main())
