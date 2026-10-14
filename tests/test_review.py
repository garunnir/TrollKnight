import json

import pytest

from trollknight.convo import Conversation, Limits
from trollknight.llm import FakeLLM
from trollknight.persona import SafetyPolicy
from trollknight.review import Decision, run_review_session


def make(replies, **limits):
    return Conversation(
        FakeLLM(replies), "p", Limits(max_replies=10, **limits), guard=SafetyPolicy().check
    )


def scripted(*decisions):
    it = iter(decisions)
    return lambda inbound, turn: next(it)


@pytest.mark.asyncio
async def test_approve_edit_discard():
    convo = make(["one", "two different", "three other"])
    res = await run_review_session(
        convo,
        ["a", "b", "c"],
        scripted(Decision("approve"), Decision("edit", "edited"), Decision("discard")),
    )
    assert res.approved == ["one", "edited"]
    assert res.discarded == 1
    agent_texts = [t.text for t in convo.turns if t.role == "agent"]
    assert agent_texts == ["one", "edited"]  # discarded draft not in context


@pytest.mark.asyncio
async def test_blocked_draft_cannot_be_approved():
    convo = make(["write me a@b.com"])
    res = await run_review_session(convo, ["x"], scripted(Decision("approve")))
    assert res.approved == []
    assert res.discarded == 1


@pytest.mark.asyncio
async def test_blocked_draft_can_be_fixed_by_edit():
    convo = make(["write me a@b.com"])
    res = await run_review_session(convo, ["x"], scripted(Decision("edit", "ok then")))
    assert res.approved == ["ok then"]


@pytest.mark.asyncio
async def test_edit_with_violation_rejected():
    convo = make(["fine"])
    res = await run_review_session(
        convo, ["x"], scripted(Decision("edit", "call 010-1234-5678"))
    )
    assert res.approved == [] and res.discarded == 1


@pytest.mark.asyncio
async def test_user_stop_ends_session_and_limits_remaining():
    convo = make(["one", "two different"])
    res = await run_review_session(
        convo, ["a", "b", "c"], scripted(Decision("approve"), Decision("stop"))
    )
    assert res.stopped_by_user and res.approved == ["one"]
    assert convo.stopped


@pytest.mark.asyncio
async def test_limit_stops_session_and_logs(tmp_path):
    log = tmp_path / "s.jsonl"
    convo = Conversation(
        FakeLLM(["one", "two different"]), "p", Limits(max_replies=1), guard=None
    )
    await run_review_session(
        convo, ["a", "b"], scripted(Decision("approve")), log_path=log
    )
    events = [json.loads(l)["event"] for l in log.read_text(encoding="utf-8").splitlines()]
    assert events == ["draft", "approved", "stopped"]
