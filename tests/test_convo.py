import pytest

from trollknight.convo import Conversation, Limits, StopReason, similarity
from trollknight.llm import FakeLLM, Usage


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(replies, **limits):
    clock = Clock()
    llm = FakeLLM(replies)
    convo = Conversation(llm, "persona", Limits(**limits), clock=clock)
    return convo, llm, clock


def test_similarity_ignores_case_and_punctuation():
    assert similarity("Hello, World!", "hello world") == 1.0
    assert similarity("apples", "completely different") < 0.5


@pytest.mark.asyncio
async def test_stops_at_max_replies():
    convo, _, _ = make(["a one", "b two", "c three"], max_replies=2)
    assert await convo.draft_reply("x") is not None
    assert await convo.draft_reply("y") is not None
    assert convo.stop_reason == StopReason.MAX_REPLIES
    assert await convo.draft_reply("z") is None
    assert convo.reply_count == 2


@pytest.mark.asyncio
async def test_stops_at_time_limit():
    convo, _, clock = make(["one", "two"], max_seconds=60)
    await convo.draft_reply("x")
    clock.t = 61
    assert await convo.draft_reply("y") is None
    assert convo.stop_reason == StopReason.MAX_SECONDS


@pytest.mark.asyncio
async def test_stops_at_cost_limit():
    clock = Clock()
    llm = FakeLLM(["one", "two"], usage=Usage(1_000_000, 0))
    convo = Conversation(
        llm,
        "p",
        Limits(max_cost_usd=1.0, price_in_per_mtok=1.0, max_replies=10),
        clock=clock,
    )
    await convo.draft_reply("x")
    assert convo.stop_reason == StopReason.MAX_COST
    assert convo.estimated_cost_usd == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_repetition_detected_and_stops():
    convo, _, _ = make(["same answer"], max_replies=10, max_repeats_in_a_row=2)
    await convo.draft_reply("a")
    t2 = await convo.draft_reply("b")
    t3 = await convo.draft_reply("c")
    assert not convo.turns[1].repeated
    assert t2.repeated and t3.repeated
    assert convo.stop_reason == StopReason.REPETITION
    assert convo.metrics()["repetition_ratio"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_user_stop_blocks_generation():
    convo, llm, _ = make(["one"])
    convo.stop()
    assert await convo.draft_reply("x") is None
    assert llm.calls == []


@pytest.mark.asyncio
async def test_context_is_passed_to_model():
    convo, llm, _ = make(["r1", "r2 different"], max_replies=5)
    await convo.draft_reply("first")
    await convo.draft_reply("second")
    roles = [m["role"] for m in llm.calls[1]]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm.calls[1][-1]["content"] == "second"
