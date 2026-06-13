"""``hermes memory`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_memory_parser(subparsers, *, cmd_memory: Callable) -> None:
    """Attach the ``memory`` subcommand to ``subparsers``."""
    memory_parser = subparsers.add_parser(
        "memory",
        help="Configure external memory provider",
        description=(
            "Set up and manage external memory provider plugins.\n\n"
            "Available providers: honcho, openviking, mem0, hindsight,\n"
            "holographic, retaindb, byterover.\n\n"
            "Only one external provider can be active at a time.\n"
            "Built-in memory (MEMORY.md/USER.md) is always active."
        ),
    )
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    _setup_parser = memory_sub.add_parser(
        "setup", help="Interactive provider selection and configuration"
    )
    _setup_parser.add_argument(
        "provider",
        nargs="?",
        default=None,
        help="Provider to configure directly (e.g. honcho), skipping the picker",
    )
    memory_sub.add_parser("status", help="Show current memory provider config")
    memory_sub.add_parser("off", help="Disable external provider (built-in only)")
    _reset_parser = memory_sub.add_parser(
        "reset",
        help="Erase all built-in memory (MEMORY.md and USER.md)",
    )
    _reset_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    _reset_parser.add_argument(
        "--target",
        choices=["all", "memory", "user"],
        default="all",
        help="Which store to reset: 'all' (default), 'memory', or 'user'",
    )

    workspace_parser = memory_sub.add_parser(
        "workspace",
        help="Manage the user-facing PARA personal workspace",
        description="Initialize, write, and search the dual-memory personal workspace.",
    )
    workspace_sub = workspace_parser.add_subparsers(dest="workspace_command")
    workspace_sub.add_parser("init", help="Create Projects/Areas/Resources/Archives manifests")
    workspace_add = workspace_sub.add_parser("add", help="Add a markdown item to the PARA workspace")
    workspace_add.add_argument("--title", required=True, help="Workspace item title")
    workspace_add.add_argument(
        "--bucket",
        choices=["Projects", "Areas", "Resources", "Archives"],
        default=None,
        help="Explicit PARA bucket; omitted means auto-route",
    )
    workspace_add.add_argument(
        "--mode",
        choices=["new", "append", "update"],
        default="new",
        help="Write behavior when the slug already exists",
    )
    workspace_add.add_argument("--summary", default="", help="Short manifest summary")
    workspace_add.add_argument("--status-hint", default="", help="Routing hint such as due, ongoing, done")
    workspace_add.add_argument("--tag", action="append", default=[], help="Tag to add; repeatable")
    workspace_add.add_argument("--backlink", action="append", default=[], help="Backlink to add; repeatable")
    workspace_add.add_argument("--file", default="", help="Read item content from a markdown/text file")
    workspace_add.add_argument("content", nargs="*", help="Item content; stdin is accepted when omitted")
    workspace_search = workspace_sub.add_parser("search", help="Search via manifests, then load top files")
    workspace_search.add_argument("query", help="Search query")
    workspace_search.add_argument("--top-k", type=int, default=3, help="Number of files to load")

    workspace_agentic = workspace_sub.add_parser(
        "agentic-search",
        help="LLM-powered search: reads manifests, selects and loads the most relevant files",
    )
    workspace_agentic.add_argument("query", help="Natural-language search query")
    workspace_agentic.add_argument("--top-k", type=int, default=3, help="Number of files to load")
    workspace_agentic.add_argument("--model", default=None, help="Override LLM model for this call")

    workspace_seed = workspace_sub.add_parser(
        "seed",
        help="Ingest knowledge from a directory of session markdown files into the workspace",
    )
    workspace_seed.add_argument(
        "--seed-dir",
        required=True,
        help="Path to a directory of .md session files to extract and ingest",
    )
    workspace_seed.add_argument("--model", default=None, help="Override LLM model for this call")

    procedural_parser = memory_sub.add_parser(
        "procedural",
        help="Manage agent-facing procedural Skill Markdown",
        description="Distill reusable successful workflows into skill drafts.",
    )
    procedural_sub = procedural_parser.add_subparsers(dest="procedural_command")
    procedural_distill = procedural_sub.add_parser("distill", help="Write a procedural skill draft")
    procedural_distill.add_argument("--name", required=True, help="Skill display name")
    procedural_distill.add_argument("--description", required=True, help="Short skill description")
    procedural_distill.add_argument("--trigger", action="append", required=True, help="When to use it; repeatable")
    procedural_distill.add_argument("--step", action="append", required=True, help="Procedure step; repeatable")
    procedural_distill.add_argument("--constraint", action="append", default=[], help="Constraint; repeatable")
    procedural_distill.add_argument("--recovery", action="append", default=[], help="Failure recovery rule; repeatable")
    procedural_distill.add_argument("--file", default="", help="Read provenance/source trace from a file")
    procedural_distill.add_argument("--overwrite", action="store_true", help="Replace an existing skill draft")
    procedural_distill.add_argument("content", nargs="*", help="Optional provenance/source trace")
    agent_parser = memory_sub.add_parser(
        "agent",
        help="Memory agent: nightly knowledge ingestion into the personal workspace",
    )
    agent_sub = agent_parser.add_subparsers(dest="agent_command")

    agent_schedule = agent_sub.add_parser(
        "schedule",
        help="Register a nightly ingestion cron job in hermes cron (default: 2 AM daily)",
    )
    agent_schedule.add_argument(
        "--schedule",
        default="0 2 * * *",
        help="Cron expression or interval (default: '0 2 * * *')",
    )
    agent_schedule.add_argument(
        "--force",
        action="store_true",
        help="Replace existing nightly job if already registered",
    )

    agent_ingest = agent_sub.add_parser(
        "ingest",
        help="Run a one-shot ingestion now (seed dir or sessions from state.db)",
    )
    agent_ingest.add_argument(
        "--seed-dir",
        default=None,
        help="Ingest .md files from this directory instead of state.db",
    )
    agent_ingest.add_argument(
        "--since-hours",
        type=float,
        default=24.0,
        help="How many hours back to look in state.db (default: 24)",
    )
    agent_ingest.add_argument("--model", default=None, help="Override LLM model")

    memory_parser.set_defaults(func=cmd_memory)
