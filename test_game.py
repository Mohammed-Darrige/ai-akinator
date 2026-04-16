"""
Self-Test Simulation – AI Akinator
====================================
Oracle tamamen LLM tabanlıdır. Hardcoded keyword map YOK.
Her AI sorusu, aynı GLM API'sinin hafif bir modeline
"Bu hayvan X mi? Sadece Evet/Hayır/Bilmiyorum ile cevap ver."
diye sorularak yanıtlanır.

Kullanım:
    python test_game.py --animal "Rakun"
    python test_game.py --animal "Zebra" --url http://localhost:8000
"""

import json
import asyncio
import argparse
import os
import sys
import httpx
from dotenv import load_dotenv
from typing import Optional, Any

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
ORACLE_MODEL = "glm-4-flash"  # Hızlı model
ORACLE_TIMEOUT = 20.0


# ─────────────────────────────────────────────────────────────
#  SSE Helper
# ─────────────────────────────────────────────────────────────


async def parse_sse_result(response: httpx.Response) -> dict:
    """Stream'den gelen veriyi oku, son anlamlı SSE objesini döndür."""
    res_data = {}
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                if data.get("type") == "session_id":
                    res_data["session_id"] = data["session_id"]
                if data.get("type") == "result":
                    res_data.update(data)
                if data.get("type") == "retry":
                    res_data["retry_reason"] = data.get("reason")
                if data.get("type") == "error":
                    raise Exception(data.get("content", "Unknown error"))
            except json.JSONDecodeError:
                continue
    return res_data


# ─────────────────────────────────────────────────────────────
#  LLM Oracle — herhangi bir hayvan için çalışır
# ─────────────────────────────────────────────────────────────


async def llm_oracle(animal: str, question: str) -> str:
    """GLM'e soruyu sor, Evet / Hayır / Bilmiyorum döndür."""
    async with httpx.AsyncClient(timeout=ORACLE_TIMEOUT) as cli:
        r = await cli.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": ORACLE_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"Sen bir Akinator oyununda oracle'sın. "
                            f"Kullanıcı '{animal}' hayvanını düşünüyor. "
                            "Sana Türkçe evet/hayır soruları sorulacak. "
                            "YALNIZCA 'Evet', 'Hayır' veya 'Bilmiyorum' yaz. Başka hiçbir şey ekleme."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                "max_tokens": 100,
                "temperature": 0.0,
            },
        )
    try:
        content = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Bilmiyorum"

    low = content.lower()
    if "evet" in low:
        return "Evet"
    if "hayır" in low or "hayir" in low:
        return "Hayır"
    return "Bilmiyorum"


# ─────────────────────────────────────────────────────────────
#  Rate-limit-aware POST helper (game API)
# ─────────────────────────────────────────────────────────────


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    json_body: Optional[dict] = None,
    max_retries: int = 3,
) -> Optional[httpx.Response]:
    for attempt in range(max_retries + 1):
        # httpx post does not like stream=True in post directly for reading later easily with aiter_lines
        # we need to use request(..., stream=True)
        req = client.build_request("POST", url, json=json_body)
        r = await client.send(req, stream=True)

        if r.status_code != 503:
            return r

        await r.aclose()
        wait = int(r.headers.get("retry-after", 60))
        print(f"\n  ⏳ Rate limit — {wait}s bekleniyor (deneme {attempt + 1})...")
        await asyncio.sleep(wait + 2)
    return None


# ─────────────────────────────────────────────────────────────
#  Seans yöneticisi
# ─────────────────────────────────────────────────────────────


async def run_session(
    base_url: str,
    animal: str,
    max_turns: int = 22,
    aliases: Optional[set] = None,
) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"  HEDEF: {animal.upper()}")
    print(f"{'=' * 60}")

    issues: list[str] = []
    questions: list[str] = []
    final_guess: Optional[str] = None
    final_turn: int = 0
    correct: bool = False
    wrong_guesses = 0

    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        # Oyunu başlat
        r = await _post_with_retry(client, "/api/v1/start", json_body={"lang": "tr"})
        if r is None or not r.is_success:
            msg = f"/start başarısız: {r.status_code if r else 'timeout'}"
            print(f"  FAIL {msg}")
            return {
                "animal": animal,
                "correct": False,
                "questions": 0,
                "final_turn": 0,
                "final_guess": None,
                "issues": [msg],
            }

        data = await parse_sse_result(r)
        session_id = data.get("session_id")
        last_q = data.get("question", "")
        turn = data.get("turn", 0)
        rsn = data.get("reasoning", {})

        print(f"\n[T{turn:02d}] AI  : {last_q}")
        print(f"      RLG : {rsn.get('chain_of_thought', '')[:120]}")
        questions.append(last_q)

        for _ in range(max_turns):
            # Oracle'a sor
            ans = await llm_oracle(animal, last_q)
            print(f"      USR : {ans}")

            r = await _post_with_retry(
                client,
                f"/api/v1/ask?session_id={session_id}",
                json_body={"user_answer": ans, "lang": "tr"},
            )
            if r is None or not r.is_success:
                msg = f"T{turn}: HTTP {r.status_code if r else 'timeout'}"
                issues.append(msg)
                print(f"      ERR : {msg}")
                break

            data = await parse_sse_result(r)
            action = data.get("action")
            rsn = data.get("reasoning", {})
            turn = data.get("turn", 0)

            if action == "guess":
                guess = data.get("guess", "?")
                final_guess, final_turn = guess, turn
                print(f"\n[T{turn:02d}] TAHMIN: {guess}")
                print(f"      RLG : {rsn.get('chain_of_thought', '')[:120]}")

                gl = guess.lower()
                targets = {animal.lower()} | {a.lower() for a in (aliases or set())}
                correct = any(t in gl or gl in t for t in targets)
                print(f"\n  SONUC: {'DOGRU' if correct else 'YANLIS'}")

                confirmation = "yes" if correct else "no"
                r = await _post_with_retry(
                    client,
                    f"/api/v1/ask?session_id={session_id}",
                    json_body={"user_answer": confirmation, "lang": "tr"},
                )
                if r is None or not r.is_success:
                    msg = f"T{turn}: guess confirmation HTTP {r.status_code if r else 'timeout'}"
                    issues.append(msg)
                    print(f"      ERR : {msg}")
                    break

                if correct:
                    await parse_sse_result(r)
                    break

                wrong_guesses += 1
                issues.append(f"T{turn}: Beklenen '{animal}', Tahmin '{guess}'")
                if wrong_guesses >= 3:
                    issues.append("Çok fazla yanlış tahmin yapıldı.")
                    break

                data = await parse_sse_result(r)
                action = data.get("action")
                rsn = data.get("reasoning", {})
                turn = data.get("turn", turn)

                if action == "ask_question":
                    q = data.get("question", "")
                    print(f"\n[T{turn:02d}] AI  : {q}")
                    print(f"      RLG : {rsn.get('chain_of_thought', '')[:120]}")
                    if q in questions:
                        issues.append(f"T{turn}: TEKRAR SORU — '{q}'")
                    questions.append(q)
                    last_q = q
                    continue

                issues.append(
                    f"T{turn}: Yanlış tahminden sonra beklenmeyen aksiyon '{action}'"
                )
                break

            elif action == "ask_question":
                q = data.get("question", "")
                print(f"\n[T{turn:02d}] AI  : {q}")
                print(f"      RLG : {rsn.get('chain_of_thought', '')[:120]}")
                if q in questions:
                    issues.append(f"T{turn}: TEKRAR SORU — '{q}'")
                questions.append(q)
                last_q = q
            else:
                issues.append(f"T{turn}: Bilinmeyen aksiyon '{action}'")
                break
        else:
            issues.append(f"max_turns={max_turns} doldu, tahmin yapılamadı.")

    print(f"\n{'-' * 60}")
    print(f"  SORULAN SORU : {len(questions)}")
    print(f"  SORUN SAYISI : {len(issues)}")
    for i in issues:
        print(f"    WARN {i}")
    if not issues:
        print("    OK Sorun yok.")
    print(f"{'-' * 60}")

    return {
        "animal": animal,
        "correct": correct,
        "questions": len(questions),
        "final_turn": final_turn,
        "final_guess": final_guess,
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────


async def main(base_url: str, animal: str, max_turns: int) -> None:
    print("\nAI AKINATOR - SELF-TEST")
    print(f"   Sunucu : {base_url}")
    print(f"   Hayvan : {animal}")
    print(f"   Oracle : {ORACLE_MODEL} (LLM-driven, hardcode yok)\n")

    rep = await run_session(base_url, animal, max_turns)

    print("\n" + "=" * 60)
    print("  RAPOR")
    print("=" * 60)
    status = "DOGRU" if rep["correct"] else "YANLIS"
    print(f"  Hayvan         : {rep['animal']}")
    print(f"  Sonuç          : {status}")
    print(f"  Sorulan soru   : {rep['questions']}")
    print(f"  Tahmin turu    : {rep['final_turn']}")
    print(f"  Tahmin edilen  : {rep['final_guess'] or '-'}")
    print(f"  Tespit edilen sorunlar: {len(rep['issues'])}")
    for i in rep["issues"]:
        print(f"    WARN {i}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--animal", default="Rakun", help="Test edilecek hayvan adı (Türkçe)"
    )
    parser.add_argument("--max-turns", type=int, default=22)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.animal, args.max_turns))
