"""CLI helpers for the dual memory framework."""

from __future__ import annotations

import sys
from pathlib import Path

from agent.dual_memory import (
    PARA_BUCKETS,
    PersonalWorkspace,
    ProceduralMemory,
    SkillDraft,
    WorkspaceItem,
    default_procedural_skills_root,
    default_workspace_root,
    filter_workspace_candidate,
    format_retrieval_results,
    make_llm_fn,
)


def _read_content(args) -> str:
    file_value = getattr(args, "file", None)
    if file_value:
        return Path(file_value).expanduser().read_text(encoding="utf-8")
    parts = getattr(args, "content", None) or []
    if parts:
        return " ".join(parts)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def _llm_fn(args=None):
    return make_llm_fn(model=getattr(args, "model", None) or None)


def cmd_workspace(args) -> None:
    """Handle ``hermes memory workspace ...``."""
    action = getattr(args, "workspace_command", None)
    workspace = PersonalWorkspace()

    if action == "init":
        workspace.initialize()
        print(f"\n  Personal workspace initialized: {default_workspace_root()}\n")
        return

    if action == "add":
        content = _read_content(args).strip()
        if not filter_workspace_candidate(content):
            print("\n  Skipped: content is too short or too transient for workspace memory.\n")
            return
        bucket = getattr(args, "bucket", None) or None
        if bucket and bucket not in PARA_BUCKETS:
            print(f"\n  Unknown PARA bucket: {bucket}\n")
            return
        item = WorkspaceItem(
            title=getattr(args, "title", "").strip(),
            content=content,
            bucket=bucket,
            summary=(getattr(args, "summary", "") or "").strip(),
            tags=list(getattr(args, "tag", []) or []),
            backlinks=list(getattr(args, "backlink", []) or []),
            status_hint=(getattr(args, "status_hint", "") or "").strip(),
        )
        path = workspace.write_item(item, mode=getattr(args, "mode", "new"))
        print(f"\n  Wrote workspace item: {path}\n")
        return

    if action == "search":
        query = getattr(args, "query", "").strip()
        results = workspace.retrieve(query, top_k=getattr(args, "top_k", 3))
        if not results:
            print("\n  No workspace matches.\n")
            return
        print()
        print(format_retrieval_results(results))
        print()
        return

    if action == "agentic-search":
        from agent.dual_memory import agentic_retrieve
        query = getattr(args, "query", "").strip()
        print(f"\n  Searching (agentic): {query!r} …\n")
        results = agentic_retrieve(query, workspace, _llm_fn(args), top_k=getattr(args, "top_k", 3))
        if not results:
            print("  No results.\n")
            return
        print(format_retrieval_results(results))
        print()
        return

    if action == "seed":
        from agent.dual_memory import ingest_session
        seed_dir_str = getattr(args, "seed_dir", None)
        if not seed_dir_str:
            print("\n  --seed-dir is required.\n")
            return
        seed_dir = Path(seed_dir_str).expanduser()
        if not seed_dir.is_dir():
            print(f"\n  Directory not found: {seed_dir}\n")
            return
        llm = _llm_fn(args)
        md_files = sorted(seed_dir.glob("*.md"))
        if not md_files:
            print(f"\n  No .md files in {seed_dir}\n")
            return
        print(f"\n  Ingesting {len(md_files)} file(s) from {seed_dir}\n")
        all_written = []
        for md_file in md_files:
            print(f"  → {md_file.name}")
            written = ingest_session(md_file.read_text(encoding="utf-8"), workspace, llm)
            all_written.extend(written)
        print(f"\n  Wrote {len(all_written)} item(s) total:")
        for p in all_written:
            print(f"    {p}")
        print()
        return

    print("\n  Usage: hermes memory workspace {init,add,search,agentic-search,seed}\n")


def cmd_procedural(args) -> None:
    """Handle ``hermes memory procedural ...``."""
    action = getattr(args, "procedural_command", None)
    procedural = ProceduralMemory()

    if action == "distill":
        steps = list(getattr(args, "step", []) or [])
        triggers = list(getattr(args, "trigger", []) or [])
        constraints = list(getattr(args, "constraint", []) or [])
        recovery = list(getattr(args, "recovery", []) or [])
        source = _read_content(args).strip()
        draft = SkillDraft(
            name=getattr(args, "name", "").strip(),
            description=getattr(args, "description", "").strip(),
            triggers=triggers,
            steps=steps,
            constraints=constraints,
            recovery=recovery,
            source=source,
        )
        try:
            path = procedural.write_skill(draft, overwrite=getattr(args, "overwrite", False))
        except FileExistsError as exc:
            print(f"\n  {exc}\n  Re-run with --overwrite to replace it.\n")
            return
        print(f"\n  Wrote procedural skill: {path}")
        print(f"  Skills root: {default_procedural_skills_root()}\n")
        return

    print("\n  Usage: hermes memory procedural distill ...\n")


def cmd_agent(args) -> None:
    """Handle ``hermes memory agent ...``."""
    action = getattr(args, "agent_command", None)

    if action == "schedule":
        _agent_schedule(args)
        return

    if action == "ingest":
        _agent_ingest(args)
        return

    print("\n  Usage: hermes memory agent {schedule,ingest}\n")


def _agent_schedule(args) -> None:
    """Register a nightly memory ingestion cron job via hermes cron."""
    from cron.jobs import create_job, list_jobs

    schedule = getattr(args, "schedule", None) or "0 2 * * *"
    existing = [j for j in list_jobs() if j.get("name") == "memory-agent-nightly"]
    if existing and not getattr(args, "force", False):
        job = existing[0]
        print(f"\n  Nightly memory agent already scheduled (id: {job['id']}).")
        print(f"  Schedule: {job.get('schedule_display', schedule)}")
        print("  Use --force to replace it.\n")
        return

    job = create_job(
        prompt=None,
        schedule=schedule,
        name="memory-agent-nightly",
        script="memory_agent_ingest.py",
        no_agent=True,
    )
    _write_ingest_script()
    print(f"\n  Scheduled nightly memory agent (id: {job['id']})")
    print(f"  Schedule: {schedule}")
    print(f"  Script:   ~/.hermes/scripts/memory_agent_ingest.py\n")


def _agent_ingest(args) -> None:
    """Run one-shot knowledge ingestion (seed dir or last N hours from state.db)."""
    import time
    from agent.dual_memory import ingest_session

    workspace = PersonalWorkspace()
    llm = _llm_fn(args)

    seed_dir_str = getattr(args, "seed_dir", None)
    if seed_dir_str:
        seed_dir = Path(seed_dir_str).expanduser()
        md_files = sorted(seed_dir.glob("*.md"))
        print(f"\n  Ingesting {len(md_files)} seed file(s) …\n")
        all_written = []
        for f in md_files:
            written = ingest_session(f.read_text(encoding="utf-8"), workspace, llm)
            all_written.extend(written)
        print(f"  Done — wrote {len(all_written)} item(s)\n")
        return

    # Production: read from state.db
    since_hours = float(getattr(args, "since_hours", 24))
    cutoff = time.time() - since_hours * 3600.0
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        with db._lock:  # noqa: SLF001
            rows = db._conn.execute(  # noqa: SLF001
                "SELECT id, title FROM sessions WHERE started_at >= ? ORDER BY started_at",
                (cutoff,),
            ).fetchall()
    except Exception as exc:
        print(f"\n  Could not read sessions: {exc}\n")
        return

    if not rows:
        print(f"\n  No sessions in the last {since_hours:.0f}h.\n")
        return

    print(f"\n  Ingesting {len(rows)} session(s) from last {since_hours:.0f}h …\n")
    all_written = []
    for row in rows:
        sid = row[0]
        messages = db.get_messages(sid)
        parts = [
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        ]
        session_text = "\n\n".join(parts)
        if session_text.strip():
            written = ingest_session(session_text, workspace, llm)
            all_written.extend(written)

    print(f"  Done — wrote {len(all_written)} item(s)\n")


def _write_ingest_script() -> None:
    """Write the standalone Python script that the cron job executes."""
    from hermes_constants import get_hermes_home
    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / "memory_agent_ingest.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        '"""Nightly memory agent — run by hermes cron, no agent layer."""\n'
        "import sys, os\n"
        "sys.path.insert(0, os.path.expanduser('~/.hermes'))\n"
        "from agent.dual_memory import PersonalWorkspace, make_llm_fn, ingest_session\n"
        "from hermes_state import SessionDB\n"
        "import time\n\n"
        "workspace = PersonalWorkspace()\n"
        "llm = make_llm_fn()\n"
        "db = SessionDB()\n"
        "cutoff = time.time() - 24 * 3600\n"
        "with db._lock:\n"
        "    rows = db._conn.execute(\n"
        "        'SELECT id FROM sessions WHERE started_at >= ? ORDER BY started_at',\n"
        "        (cutoff,),\n"
        "    ).fetchall()\n"
        "written = []\n"
        "for (sid,) in rows:\n"
        "    msgs = db.get_messages(sid)\n"
        "    text = '\\n\\n'.join(\n"
        "        f\"{m['role'].upper()}: {m['content']}\"\n"
        "        for m in msgs\n"
        "        if m.get('role') in ('user', 'assistant') and isinstance(m.get('content'), str)\n"
        "    )\n"
        "    if text.strip():\n"
        "        written.extend(ingest_session(text, workspace, llm))\n"
        "print(f'memory-agent-nightly: wrote {len(written)} item(s)')\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
