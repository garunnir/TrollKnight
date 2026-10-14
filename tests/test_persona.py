import pytest

from trollknight.convo import Conversation, Limits
from trollknight.llm import FakeLLM
from trollknight.persona import Persona, SafetyPolicy, build_system_prompt, load_persona


def test_load_persona_file():
    p = load_persona("personas/deadpan.toml")
    assert p.id == "deadpan" and p.max_sentences == 2


def test_persona_validation():
    with pytest.raises(ValueError):
        Persona(id="", name="x", style="s")


def test_prompt_contains_style_and_safety_rules():
    prompt = build_system_prompt(Persona("a", "A", "sarcastic"), SafetyPolicy())
    assert "sarcastic" in prompt
    assert "SAFETY RULES" in prompt
    assert prompt.index("sarcastic") < prompt.index("SAFETY RULES")


@pytest.mark.parametrize(
    "text,cat",
    [
        ("연락은 foo@example.com 으로", "personal_info"),
        ("전화번호 010-1234-5678", "personal_info"),
        ("내가 찾아가서 가만 안 둔다", "threat"),
        ("I will find you", "threat"),
    ],
)
def test_policy_flags(text, cat):
    assert cat in {v.category for v in SafetyPolicy().check(text)}


def test_policy_allows_plain_reply():
    assert SafetyPolicy().check("그렇게 생각하시는군요.") == []


def test_bad_regex_fails_fast():
    with pytest.raises(Exception):
        SafetyPolicy(patterns={"x": ("(",)})


@pytest.mark.asyncio
async def test_engine_marks_blocked_turn():
    policy = SafetyPolicy()
    convo = Conversation(
        FakeLLM(["mail me at a@b.com"]), "p", Limits(), guard=policy.check
    )
    turn = await convo.draft_reply("hi")
    assert turn.blocked
    assert convo.metrics()["blocked_count"] == 1
