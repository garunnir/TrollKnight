"""Simulation CLI: feed inbound messages from a file, approve drafts by hand.

    python -m trollknight.cli --persona personas/deadpan.toml --inbound msgs.txt --fake
    (omit --fake and set TROLLKNIGHT_API_KEY / TROLLKNIGHT_MODEL to use a model)
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .convo import Conversation, Limits
from .llm import FakeLLM
from .persona import SafetyPolicy, build_system_prompt, load_persona
from .review import Decision, run_review_session


def _decide(inbound: str, turn) -> Decision:
    print(f"\n[상대] {inbound}")
    print(f"[초안] {turn.text}")
    if turn.repeated:
        print("  ! 이전 응답과 반복됨")
    if turn.violations:
        print("  ! 안전 정책 위반: " + ", ".join(v.category for v in turn.violations))
    while True:
        choice = input("(a)승인 (e)수정 (d)폐기 (s)중단 > ").strip().lower()
        if choice == "a":
            return Decision("approve")
        if choice == "e":
            return Decision("edit", input("수정문> "))
        if choice == "d":
            return Decision("discard")
        if choice == "s":
            return Decision("stop")


async def _main(args: argparse.Namespace) -> None:
    persona = load_persona(args.persona)
    policy = SafetyPolicy()
    if args.fake:
        llm = FakeLLM(["그렇게 생각하시는군요.", "근거가 있으면 링크 주세요."])
    else:
        from .llm.openai_compat import OpenAICompatClient, OpenAICompatConfig

        llm = OpenAICompatClient(OpenAICompatConfig.from_env())

    convo = Conversation(
        llm,
        build_system_prompt(persona, policy),
        Limits(
            max_replies=args.max_replies,
            max_seconds=args.max_seconds,
            max_cost_usd=args.max_cost,
            price_in_per_mtok=args.price_in,
            price_out_per_mtok=args.price_out,
        ),
        guard=policy.check,
    )
    lines = [
        l.strip()
        for l in Path(args.inbound).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    log = Path(args.log) if args.log else None
    result = await run_review_session(convo, lines, _decide, log_path=log)
    print("\n== 결과 ==")
    print(json.dumps(convo.metrics(), ensure_ascii=False, indent=2))
    print(f"승인 {len(result.approved)}건, 폐기 {result.discarded}건")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--persona", required=True)
    p.add_argument("--inbound", required=True, help="상대 메시지 목록(한 줄에 하나)")
    p.add_argument("--fake", action="store_true", help="모델 없이 시뮬레이션")
    p.add_argument("--log", help="JSONL 로그 경로")
    p.add_argument("--max-replies", type=int, default=5)
    p.add_argument("--max-seconds", type=float, default=300)
    p.add_argument("--max-cost", type=float, default=0.5)
    p.add_argument("--price-in", type=float, default=0.0, help="USD / 1M input tokens")
    p.add_argument("--price-out", type=float, default=0.0, help="USD / 1M output tokens")
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
