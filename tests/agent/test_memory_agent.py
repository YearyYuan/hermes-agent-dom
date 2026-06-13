"""Tests for the agentic layer in agent.dual_memory.

All LLM calls are replaced with a mock so tests are fast and offline.
Set log_cli = true in pyproject.toml (or pass -s) to watch INFO decisions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agent.dual_memory import (
    PARA_BUCKETS,
    PersonalWorkspace,
    WorkspaceItem,
    _parse_bucket,
    _parse_file_list,
    _parse_item_blocks,
    agentic_retrieve,
    agentic_route,
    extract_knowledge_items,
    ingest_session,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture()
def workspace(hermes_home):
    ws = PersonalWorkspace()
    ws.initialize()
    return ws


def mock_llm(response: str):
    """Return an LLMCallable that always returns ``response``."""

    def _fn(system: str, messages: list) -> str:  # noqa: ARG001
        return response

    return _fn


# ---------------------------------------------------------------------------
# _parse_bucket
# ---------------------------------------------------------------------------


def test_parse_bucket_all_four():
    for bucket in PARA_BUCKETS:
        assert _parse_bucket(f"BUCKET: {bucket}") == bucket


def test_parse_bucket_case_insensitive():
    assert _parse_bucket("bucket: resources") == "Resources"


def test_parse_bucket_returns_none_on_garbage():
    assert _parse_bucket("I cannot determine the bucket") is None


# ---------------------------------------------------------------------------
# _parse_file_list
# ---------------------------------------------------------------------------


def test_parse_file_list_basic():
    response = (
        "RELEVANCE: policy gradient concepts\n"
        "FILES:\n"
        "- Resources/policy-gradient-basics.md\n"
        "- Areas/rl-notes.md\n"
    )
    assert _parse_file_list(response) == [
        ("Resources", "policy-gradient-basics.md"),
        ("Areas", "rl-notes.md"),
    ]


def test_parse_file_list_none():
    assert _parse_file_list("FILES: none") == []


def test_parse_file_list_case_insensitive_bucket():
    assert _parse_file_list("FILES:\n- resources/pg.md\n") == [("Resources", "pg.md")]


# ---------------------------------------------------------------------------
# agentic_route — no fallback, raises on bad LLM output
# ---------------------------------------------------------------------------


def test_agentic_route_uses_llm_bucket(workspace, caplog):
    item = WorkspaceItem(
        title="Policy Gradient Derivation",
        content="REINFORCE derives the gradient via log-probability times return.",
    )
    with caplog.at_level(logging.INFO, logger="agent.dual_memory"):
        bucket = agentic_route(item, workspace, mock_llm("BUCKET: Resources\nREASON: standalone reference"))
    assert bucket == "Resources"
    assert any("Resources" in r.message for r in caplog.records)
    assert any("standalone reference" in r.message for r in caplog.records)


def test_agentic_route_all_four_buckets(workspace):
    for expected in PARA_BUCKETS:
        item = WorkspaceItem(title="test", content="test content for routing purposes " * 3)
        assert agentic_route(item, workspace, mock_llm(f"BUCKET: {expected}\nREASON: test")) == expected


def test_agentic_route_raises_on_unparseable_response(workspace):
    item = WorkspaceItem(title="X", content="some content that passes the quality gate easily")
    with pytest.raises(ValueError, match="unparseable bucket"):
        agentic_route(item, workspace, mock_llm("I'm not sure which category fits."))


def test_agentic_route_raises_on_llm_exception(workspace):
    item = WorkspaceItem(title="X", content="content content content content content content")

    def _fail(system, messages):
        raise RuntimeError("network timeout")

    with pytest.raises(RuntimeError, match="network timeout"):
        agentic_route(item, workspace, _fail)


# ---------------------------------------------------------------------------
# agentic_retrieve — no fallback, raises when nothing found
# ---------------------------------------------------------------------------


def test_agentic_retrieve_reads_selected_file(workspace, caplog):
    workspace.write_item(
        WorkspaceItem(
            title="Policy Gradient Basics",
            content="∇J(θ) = E[∇logπ(a|s)·G]. The REINFORCE update rule.",
            bucket="Resources",
            tags=["policy-gradient", "reinforce"],
        )
    )
    response = (
        "RELEVANCE: policy gradient reference\n"
        "FILES:\n- Resources/policy-gradient-basics.md\n"
    )
    with caplog.at_level(logging.INFO, logger="agent.dual_memory"):
        results = agentic_retrieve("policy gradient", workspace, mock_llm(response), top_k=3)
    assert len(results) == 1
    assert results[0].record.bucket == "Resources"
    assert "REINFORCE" in results[0].content
    assert any("policy gradient" in r.message for r in caplog.records)


def test_agentic_retrieve_raises_when_llm_selects_none(workspace):
    with pytest.raises(ValueError, match="selected no files"):
        agentic_retrieve("bellman equation", workspace, mock_llm("FILES: none"))


def test_agentic_retrieve_raises_when_llm_names_missing_file(workspace):
    with pytest.raises(ValueError, match="none of the LLM-selected files exist"):
        agentic_retrieve(
            "ppo",
            workspace,
            mock_llm("FILES:\n- Resources/does-not-exist.md\n"),
        )


def test_agentic_retrieve_raises_on_llm_exception(workspace):
    def _fail(system, messages):
        raise ConnectionError("timeout")

    with pytest.raises(ConnectionError):
        agentic_retrieve("q-learning", workspace, _fail)


# ---------------------------------------------------------------------------
# _parse_item_blocks
# ---------------------------------------------------------------------------

_SAMPLE = """\
TITLE: Policy Gradient Theorem
BUCKET: Resources
TAGS: policy-gradient, reinforce, rl
SUMMARY: Derivation of ∇J(θ) via log-probability trick.
CONTENT:
∇J(θ) = E_τ[∑_t ∇logπ_θ(a_t|s_t)·G_t]. Computed via REINFORCE.
---

TITLE: Off-Policy RL Overview
BUCKET: Areas
TAGS: off-policy, importance-sampling
SUMMARY: Survey of off-policy RL methods.
CONTENT:
Off-policy methods learn about a target policy while following a behaviour policy.
Key methods: Q-learning, DQN, SAC.
---
"""


def test_parse_item_blocks_count():
    assert len(_parse_item_blocks(_SAMPLE)) == 2


def test_parse_item_blocks_first():
    item = _parse_item_blocks(_SAMPLE)[0]
    assert item.title == "Policy Gradient Theorem"
    assert item.bucket == "Resources"
    assert "policy-gradient" in item.tags
    assert "REINFORCE" in item.content


def test_parse_item_blocks_second():
    item = _parse_item_blocks(_SAMPLE)[1]
    assert item.title == "Off-Policy RL Overview"
    assert item.bucket == "Areas"


# ---------------------------------------------------------------------------
# extract_knowledge_items
# ---------------------------------------------------------------------------


def test_extract_returns_items(caplog):
    with caplog.at_level(logging.INFO, logger="agent.dual_memory"):
        items = extract_knowledge_items("USER: explain REINFORCE\nASSISTANT: ...", mock_llm(_SAMPLE))
    assert len(items) == 2
    assert any("2 item" in r.message for r in caplog.records)


def test_extract_returns_empty_on_no_items(caplog):
    with caplog.at_level(logging.INFO, logger="agent.dual_memory"):
        items = extract_knowledge_items("ok thanks", mock_llm("NO_ITEMS"))
    assert items == []
    assert any("no durable" in r.message for r in caplog.records)


def test_extract_propagates_llm_exception():
    def _fail(system, messages):
        raise RuntimeError("quota exceeded")

    with pytest.raises(RuntimeError, match="quota exceeded"):
        extract_knowledge_items("some session text here to pass length check", _fail)


# ---------------------------------------------------------------------------
# ingest_session — end-to-end
# ---------------------------------------------------------------------------


def test_ingest_session_writes_to_workspace(workspace, caplog):
    block = (
        "TITLE: REINFORCE Rule\n"
        "BUCKET: Resources\n"
        "TAGS: reinforce, policy-gradient\n"
        "SUMMARY: The REINFORCE policy gradient update.\n"
        "CONTENT:\n"
        "Δθ = α·∇logπ_θ(a|s)·G. Monte Carlo policy gradient, unbiased but high variance.\n"
        "---\n"
    )
    with caplog.at_level(logging.INFO, logger="agent.dual_memory"):
        written = ingest_session("USER: explain REINFORCE\nASSISTANT: ...", workspace, mock_llm(block))
    assert len(written) == 1
    assert written[0].exists()
    assert "Resources" in str(written[0])
    assert any("REINFORCE Rule" in r.message for r in caplog.records)


def test_ingest_session_routes_items_without_bucket(workspace):
    """Items extracted without a bucket trigger a second agentic_route call."""
    block_no_bucket = (
        "TITLE: Actor-Critic Methods\n"
        "TAGS: actor-critic, rl\n"
        "SUMMARY: Combines policy and value learning.\n"
        "CONTENT:\n"
        "Actor-critic uses a policy (actor) and a value function (critic). "
        "The critic reduces variance compared to pure REINFORCE.\n"
        "---\n"
    )
    calls = []

    def _sequential(system, messages):
        calls.append(len(calls))
        if len(calls) == 1:
            return block_no_bucket
        return "BUCKET: Resources\nREASON: standalone reference"

    written = ingest_session("RL session text", workspace, _sequential)
    assert len(written) == 1
    assert "Resources" in str(written[0])
    assert len(calls) == 2  # extract call + route call


def test_ingest_session_empty_returns_no_writes(workspace):
    written = ingest_session("short", workspace, mock_llm("NO_ITEMS"))
    assert written == []


# ---------------------------------------------------------------------------
# run via seed dir (integration of ingest_session loop)
# ---------------------------------------------------------------------------


def test_ingest_seed_dir(workspace, tmp_path, caplog):
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (seed_dir / "session1.md").write_text(
        "USER: explain value functions\nASSISTANT: V(s) = E[∑ γ^t r_t].",
        encoding="utf-8",
    )
    block = (
        "TITLE: Value Function Definition\n"
        "BUCKET: Resources\n"
        "TAGS: value-function, rl\n"
        "SUMMARY: V(s) = expected discounted return from s.\n"
        "CONTENT:\n"
        "V(s) = E[∑_{t=0}^∞ γ^t r_t | s_0 = s]. Baseline in actor-critic methods.\n"
        "---\n"
    )
    # Simulate the CLI seed loop directly
    llm = mock_llm(block)
    written = []
    for f in sorted(seed_dir.glob("*.md")):
        written.extend(ingest_session(f.read_text(encoding="utf-8"), workspace, llm))
    assert len(written) == 1
    assert written[0].exists()
