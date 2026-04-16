import asyncio
import argparse
import json
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import httpx


class RateLimited(RuntimeError):
    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limited for {retry_after}s")


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern[str]
    answer: str


TARGET_RULES: dict[str, list[Rule]] = {
    "penguin": [
        Rule(
            re.compile(
                r"larger than (an|a) (average )?human|bigger than (an|a) human", re.I
            ),
            "no",
        ),
        Rule(re.compile(r"four (legs|feet)|quadruped", re.I), "no"),
        Rule(re.compile(r"two (legs|feet)|biped", re.I), "yes"),
        Rule(re.compile(r"mammal", re.I), "no"),
        Rule(re.compile(r"bird|avian|feathers?", re.I), "yes"),
        Rule(re.compile(r"wings?", re.I), "yes"),
        Rule(re.compile(r"fly|airborne", re.I), "no"),
        Rule(re.compile(r"swim|aquatic|water|ocean|sea", re.I), "yes"),
        Rule(re.compile(r"lays? eggs?|egg[- ]lay", re.I), "yes"),
        Rule(re.compile(r"fur|hair", re.I), "no"),
        Rule(re.compile(r"carniv|fish|meat", re.I), "yes"),
        Rule(re.compile(r"nocturnal", re.I), "no"),
        Rule(re.compile(r"pet|domesticat", re.I), "no"),
        Rule(re.compile(r"camouflag|change color", re.I), "no"),
    ],
    "bat": [
        Rule(
            re.compile(
                r"larger than (an|a) (average )?human|bigger than (an|a) human", re.I
            ),
            "no",
        ),
        Rule(re.compile(r"mammal", re.I), "yes"),
        Rule(re.compile(r"bird|avian", re.I), "no"),
        Rule(re.compile(r"wings?", re.I), "yes"),
        Rule(re.compile(r"fly|airborne", re.I), "yes"),
        Rule(re.compile(r"lays? eggs?|egg[- ]lay", re.I), "no"),
        Rule(re.compile(r"fur|hair", re.I), "yes"),
        Rule(re.compile(r"nocturnal", re.I), "yes"),
        Rule(re.compile(r"swim|aquatic|water", re.I), "no"),
        Rule(re.compile(r"echolocat|sonar", re.I), "yes"),
        Rule(re.compile(r"pet|domesticat", re.I), "no"),
        Rule(re.compile(r"camouflag|change color", re.I), "no"),
    ],
    "platypus": [
        Rule(
            re.compile(
                r"larger than (an|a) (average )?human|bigger than (an|a) human", re.I
            ),
            "no",
        ),
        Rule(re.compile(r"mammal", re.I), "yes"),
        Rule(re.compile(r"bird|avian", re.I), "no"),
        Rule(re.compile(r"wings?", re.I), "no"),
        Rule(re.compile(r"lays? eggs?|egg[- ]lay", re.I), "yes"),
        Rule(re.compile(r"swim|aquatic|water|river|lake", re.I), "yes"),
        Rule(re.compile(r"fur|hair", re.I), "yes"),
        Rule(re.compile(r"beak|bill|duck[- ]bill", re.I), "yes"),
        Rule(re.compile(r"venom|poison", re.I), "yes"),
        Rule(re.compile(r"nocturnal", re.I), "yes"),
        Rule(re.compile(r"pet|domesticat", re.I), "no"),
        Rule(re.compile(r"camouflag|change color", re.I), "no"),
    ],
}


def answer_question(target: str, question: str) -> str:
    normalized = question.strip()
    for rule in TARGET_RULES[target]:
        if rule.pattern.search(normalized):
            return rule.answer
    return "unknown"


async def parse_sse(response: httpx.Response) -> dict:
    result: dict = {}
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        if payload.get("type") == "session_id":
            result["session_id"] = payload["session_id"]
        elif payload.get("type") == "result":
            result.update(payload)
        elif payload.get("type") == "error":
            if payload.get("error_type") == "rate_limited":
                raise RateLimited(int(payload.get("retry_after", 60)))
            raise RuntimeError(payload.get("content", "Unknown backend error"))
    return result


async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    json_body: dict,
    attempts: int = 3,
) -> dict:
    for attempt in range(attempts):
        response = await client.post(url, json=json_body)
        response.raise_for_status()
        try:
            return await parse_sse(response)
        except RateLimited as exc:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(exc.retry_after + 2)
    raise RuntimeError("Exhausted retries")


def assert_reasoning(result: dict, issues: list[str], turn: int) -> None:
    reasoning = result.get("reasoning") or {}
    chain = reasoning.get("chain_of_thought", "")
    if not isinstance(chain, str) or not chain.strip():
        issues.append(f"T{turn}: empty chain_of_thought")


async def run_target(
    base_url: str, target: str, max_turns: int = 18
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    asked: set[str] = set()

    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        result = await post_with_retry(
            client, "/api/v1/start", json_body={"lang": "en"}
        )
        session_id = result["session_id"]
        turn = int(result.get("turn", 0))
        assert_reasoning(result, issues, turn)

        question = result.get("question", "")
        asked.add(question)

        for _ in range(max_turns):
            answer = answer_question(target, question)
            result = await post_with_retry(
                client,
                f"/api/v1/ask?session_id={session_id}",
                json_body={"user_answer": answer, "lang": "en"},
            )
            turn = int(result.get("turn", turn))
            assert_reasoning(result, issues, turn)

            action = result.get("action")
            if action == "ask_question":
                question = result.get("question", "")
                if question in asked:
                    issues.append(f"T{turn}: repeated question '{question}'")
                asked.add(question)
                continue

            if action == "guess":
                guess = str(result.get("guess", "")).strip().lower()
                if target not in guess:
                    issues.append(
                        f"T{turn}: wrong guess '{guess}' for target '{target}'"
                    )
                    follow_up = await post_with_retry(
                        client,
                        f"/api/v1/ask?session_id={session_id}",
                        json_body={"user_answer": "no", "lang": "en"},
                    )
                    if follow_up.get("action") != "ask_question":
                        issues.append(
                            f"T{turn}: expected a follow-up question after denying guess, got '{follow_up.get('action')}'"
                        )
                        return False, issues
                    question = follow_up.get("question", "")
                    asked.add(question)
                    assert_reasoning(follow_up, issues, turn + 1)
                    continue

                final = await post_with_retry(
                    client,
                    f"/api/v1/ask?session_id={session_id}",
                    json_body={"user_answer": "yes", "lang": "en"},
                )
                if final.get("action") != "guess_correct":
                    issues.append(
                        f"T{turn}: expected guess_correct after confirming guess, got '{final.get('action')}'"
                    )
                    return False, issues
                return len(issues) == 0, issues

            issues.append(f"T{turn}: unexpected action '{action}'")
            return False, issues

    issues.append(f"No correct guess within {max_turns} turns")
    return False, issues


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", default=["penguin", "bat", "platypus"])
    args = parser.parse_args()

    targets = args.targets
    overall_ok = True
    for target in targets:
        ok, issues = await run_target("http://127.0.0.1:8000", target)
        print(f"{target}: {'OK' if ok else 'FAIL'}")
        for issue in issues:
            print(f"  - {issue}")
        overall_ok = overall_ok and ok

    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
