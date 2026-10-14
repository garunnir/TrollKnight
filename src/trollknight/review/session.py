"""Approval-based session: every draft needs an explicit human decision.

Nothing here posts anywhere. "Approved" only means the text may be copied or
handed to a platform adapter that does not exist yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional

from ..convo.engine import Conversation, Turn

Action = Literal["approve", "edit", "discard", "stop"]


@dataclass(frozen=True)
class Decision:
    action: Action
    text: str = ""  # replacement text for "edit"


@dataclass
class SessionResult:
    approved: list[str] = field(default_factory=list)
    discarded: int = 0
    stopped_by_user: bool = False


def _log(path: Optional[Path], event: dict) -> None:
    if path is None:
        return
    event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


async def run_review_session(
    convo: Conversation,
    inbound: Iterable[str],
    decide: Callable[[str, Turn], Decision],
    *,
    log_path: Optional[Path] = None,
) -> SessionResult:
    """For each inbound message: draft -> human decision -> record.

    - Drafts flagged by the safety guard cannot be approved as-is; the human
      must edit them, and the edited text is re-checked.
    - Discarded drafts are removed from the context the model sees.
    - Edited drafts replace the model text, so context reflects what was sent.
    """
    result = SessionResult()
    for msg in inbound:
        turn = await convo.draft_reply(msg)
        if turn is None:
            _log(log_path, {"event": "stopped", "reason": convo.stop_reason.value})
            break
        _log(
            log_path,
            {
                "event": "draft",
                "inbound": msg,
                "draft": turn.text,
                "repeated": turn.repeated,
                "violations": [v.category for v in turn.violations],
            },
        )

        d = decide(msg, turn)
        if d.action == "stop":
            convo.stop()
            result.stopped_by_user = True
            convo.turns.pop()  # drop unsent draft from context
            _log(log_path, {"event": "user_stop"})
            break
        if d.action == "discard":
            convo.turns.pop()
            result.discarded += 1
            _log(log_path, {"event": "discard"})
            continue
        if d.action == "edit":
            new_text = d.text.strip()
            violations = convo.guard(new_text) if convo.guard else []
            if not new_text or violations:
                convo.turns.pop()
                result.discarded += 1
                _log(
                    log_path,
                    {
                        "event": "edit_rejected",
                        "violations": [v.category for v in violations],
                    },
                )
                continue
            turn.text = new_text
            turn.violations = []
        elif turn.blocked:  # approve of a flagged draft is not allowed
            convo.turns.pop()
            result.discarded += 1
            _log(log_path, {"event": "approve_rejected_blocked"})
            continue

        result.approved.append(turn.text)
        _log(log_path, {"event": "approved", "text": turn.text})
    return result
