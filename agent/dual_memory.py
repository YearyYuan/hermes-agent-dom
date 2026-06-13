"""Dual memory framework: personal workspace plus procedural skills.

Two memory classes:

* Personal workspace (W): user-visible markdown knowledge assets organized
  with a PARA state machine (Projects / Areas / Resources / Archives).
* Procedural memory (S): agent-facing Skill Markdown distilled from repeated
  successful workflows.

Agentic layer (requires an LLMCallable):

* ``agentic_route``  — LLM reads manifests and classifies a new item.
* ``agentic_retrieve`` — LLM reads manifests, selects files, returns results.
* ``extract_knowledge_items`` — LLM extracts durable knowledge from a session.
* ``ingest_session`` — full pipeline: extract → route → write.
* ``make_llm_fn`` — builds an LLMCallable from the hermes config.

The framework is local-file based and profile-scoped through HERMES_HOME.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Sequence

from hermes_constants import get_hermes_home, get_skills_dir

logger = logging.getLogger(__name__)

ParaBucket = Literal["Projects", "Areas", "Resources", "Archives"]

PARA_BUCKETS: tuple[ParaBucket, ...] = ("Projects", "Areas", "Resources", "Archives")
MANIFEST_NAME = "_manifest.md"


def default_workspace_root() -> Path:
    """Return the default profile-scoped personal workspace root."""
    return get_hermes_home() / "personal_workspace"


def default_procedural_skills_root() -> Path:
    """Return the default profile-scoped procedural skill root."""
    return get_skills_dir() / "procedural-memory"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(text: str, *, fallback: str = "untitled") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or fallback


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}", text.lower())}


def _safe_relative_markdown_path(name: str) -> Path:
    """Return a one-segment markdown filename derived from user/model text."""
    slug = _slugify(name)
    return Path(f"{slug}.md")


def _frontmatter(data: dict[str, object]) -> str:
    lines = ["---"]
    for key, value in data.items():
        if isinstance(value, list):
            rendered = "[" + ", ".join(str(v) for v in value) + "]"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines)


def route_item(title: str, content: str = "", *, status_hint: str = "") -> ParaBucket:
    """Deterministically route a candidate workspace item into PARA.

    A future memory agent can replace this with an LLM classifier that uses the
    same output contract. The heuristic keeps the framework testable and useful
    offline.
    """
    haystack = f"{title}\n{content}\n{status_hint}".lower()
    if re.search(r"\b(done|completed|finished|inactive|archive|archived|retired)\b", haystack):
        return "Archives"
    if re.search(r"\b(deadline|due|milestone|ship|launch|deliver|project|sprint|todo|next step)\b", haystack):
        return "Projects"
    if re.search(r"\b(ongoing|maintain|responsibility|area|habit|routine|standard|policy)\b", haystack):
        return "Areas"
    return "Resources"


@dataclass(frozen=True)
class WorkspaceRecord:
    """Manifest-level metadata for one personal workspace file."""

    bucket: ParaBucket
    path: Path
    title: str
    summary: str = ""
    tags: tuple[str, ...] = ()
    updated_at: str = ""


@dataclass(frozen=True)
class RetrievalResult:
    """One top-k workspace retrieval result."""

    record: WorkspaceRecord
    score: int
    content: str


@dataclass
class WorkspaceItem:
    """A write candidate produced by a memory agent or CLI call."""

    title: str
    content: str
    bucket: ParaBucket | None = None
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    backlinks: list[str] = field(default_factory=list)
    status_hint: str = ""


class PersonalWorkspace:
    """PARA markdown workspace with manifest-based hierarchical retrieval."""

    def __init__(self, root: Path | None = None):
        self.root = root or default_workspace_root()

    def initialize(self) -> None:
        """Create the PARA directory skeleton and manifests."""
        self.root.mkdir(parents=True, exist_ok=True)
        for bucket in PARA_BUCKETS:
            bucket_dir = self.root / bucket
            bucket_dir.mkdir(parents=True, exist_ok=True)
            manifest = bucket_dir / MANIFEST_NAME
            if not manifest.exists():
                manifest.write_text(self._empty_manifest(bucket), encoding="utf-8")

    def write_item(self, item: WorkspaceItem, *, mode: Literal["new", "append", "update"] = "new") -> Path:
        """Write a workspace item and refresh the bucket manifest.

        ``new`` creates a unique file when a slug already exists. ``append``
        appends a dated section to an existing slug. ``update`` replaces the
        body of the existing slug while preserving the same filename.
        """
        if not item.title.strip():
            raise ValueError("Workspace item title cannot be empty")
        if not item.content.strip():
            raise ValueError("Workspace item content cannot be empty")

        self.initialize()
        bucket = item.bucket or route_item(item.title, item.content, status_hint=item.status_hint)
        bucket_dir = self.root / bucket
        rel = _safe_relative_markdown_path(item.title)
        path = bucket_dir / rel
        if mode == "new":
            path = self._unique_path(path)

        now = _utc_now()
        header = _frontmatter(
            {
                "title": item.title.strip(),
                "bucket": bucket,
                "summary": item.summary.strip() or self._summarize(item.content),
                "tags": [t.strip() for t in item.tags if t.strip()],
                "backlinks": [b.strip() for b in item.backlinks if b.strip()],
                "updated_at": now,
                "created_by": "hermes-dual-memory",
            }
        )
        body = item.content.strip() + "\n"

        if mode == "append" and path.exists():
            with path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n\n## Update {now}\n\n{body}")
        else:
            path.write_text(f"{header}\n\n# {item.title.strip()}\n\n{body}", encoding="utf-8")

        self.rebuild_manifest(bucket)
        return path

    def read_manifests(self) -> dict[ParaBucket, str]:
        """Read the four PARA manifests without scanning file bodies."""
        self.initialize()
        return {
            bucket: (self.root / bucket / MANIFEST_NAME).read_text(encoding="utf-8")
            for bucket in PARA_BUCKETS
        }

    def retrieve(self, query: str, *, top_k: int = 3, candidate_limit: int = 8) -> list[RetrievalResult]:
        """Two-stage retrieval using manifests first, then top candidate files."""
        if top_k <= 0:
            return []
        manifests = self.read_manifests()
        query_terms = _tokenize(query)
        bucket_scores = [
            (self._score_text(query_terms, f"{bucket}\n{manifest}"), bucket)
            for bucket, manifest in manifests.items()
        ]
        relevant_buckets = [bucket for score, bucket in sorted(bucket_scores, reverse=True) if score > 0]
        if not relevant_buckets:
            relevant_buckets = list(PARA_BUCKETS)

        records: list[WorkspaceRecord] = []
        for bucket in relevant_buckets:
            records.extend(self.parse_manifest(bucket, manifests[bucket]))

        ranked_records = sorted(
            ((self._score_record(query_terms, record), record) for record in records),
            key=lambda pair: pair[0],
            reverse=True,
        )
        results: list[RetrievalResult] = []
        for score, record in ranked_records[:candidate_limit]:
            path = self.root / record.bucket / record.path
            if not path.exists() or path.name == MANIFEST_NAME:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            full_score = score + self._score_text(query_terms, content)
            if full_score > 0 or not query_terms:
                results.append(RetrievalResult(record=record, score=full_score, content=content))

        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

    def rebuild_manifest(self, bucket: ParaBucket) -> Path:
        """Rebuild a bucket manifest from markdown files in that bucket."""
        if bucket not in PARA_BUCKETS:
            raise ValueError(f"Unknown PARA bucket: {bucket}")
        bucket_dir = self.root / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for path in sorted(bucket_dir.glob("*.md")):
            if path.name == MANIFEST_NAME:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            meta = self._parse_frontmatter(text)
            title = str(meta.get("title") or path.stem.replace("-", " ").title())
            summary = str(meta.get("summary") or self._summarize(text))
            tags = tuple(str(t).strip() for t in meta.get("tags", []) if str(t).strip())
            updated = str(meta.get("updated_at") or "")
            records.append(
                WorkspaceRecord(
                    bucket=bucket,
                    path=Path(path.name),
                    title=title,
                    summary=summary,
                    tags=tags,
                    updated_at=updated,
                )
            )
        manifest = bucket_dir / MANIFEST_NAME
        manifest.write_text(self._render_manifest(bucket, records), encoding="utf-8")
        return manifest

    def parse_manifest(self, bucket: ParaBucket, text: str) -> list[WorkspaceRecord]:
        """Parse records from a generated manifest."""
        records: list[WorkspaceRecord] = []
        for line in text.splitlines():
            if not line.startswith("- ["):
                continue
            match = re.match(
                r"- \[(?P<title>.*?)\]\((?P<path>.*?)\) - (?P<summary>.*?)(?: \| tags: (?P<tags>.*?))?(?: \| updated: (?P<updated>.*))?$",
                line,
            )
            if not match:
                continue
            tags = tuple(
                t.strip()
                for t in (match.group("tags") or "").split(",")
                if t.strip()
            )
            records.append(
                WorkspaceRecord(
                    bucket=bucket,
                    path=Path(match.group("path")),
                    title=match.group("title"),
                    summary=match.group("summary").strip(),
                    tags=tags,
                    updated_at=(match.group("updated") or "").strip(),
                )
            )
        return records

    @staticmethod
    def _empty_manifest(bucket: ParaBucket) -> str:
        return (
            f"# {bucket} Manifest\n\n"
            "This manifest is the retrieval entry point for this PARA bucket.\n\n"
            "- Files: none yet\n"
        )

    @staticmethod
    def _render_manifest(bucket: ParaBucket, records: Sequence[WorkspaceRecord]) -> str:
        lines = [
            f"# {bucket} Manifest",
            "",
            "This manifest is the retrieval entry point for this PARA bucket.",
            "",
        ]
        if not records:
            lines.append("- Files: none yet")
            return "\n".join(lines) + "\n"
        for record in records:
            tags = f" | tags: {', '.join(record.tags)}" if record.tags else ""
            updated = f" | updated: {record.updated_at}" if record.updated_at else ""
            lines.append(
                f"- [{record.title}]({record.path.as_posix()}) - {record.summary}{tags}{updated}"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _summarize(text: str, *, limit: int = 160) -> str:
        squashed = re.sub(r"\s+", " ", text).strip()
        return squashed[: limit - 1] + "..." if len(squashed) > limit else squashed

    @staticmethod
    def _parse_frontmatter(text: str) -> dict[str, object]:
        if not text.startswith("---\n"):
            return {}
        end = text.find("\n---", 4)
        if end == -1:
            return {}
        meta: dict[str, object] = {}
        for line in text[4:end].splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            value = raw.strip()
            if value.startswith("[") and value.endswith("]"):
                items = [v.strip() for v in value[1:-1].split(",") if v.strip()]
                meta[key.strip()] = items
            else:
                meta[key.strip()] = value
        return meta

    @staticmethod
    def _score_text(query_terms: set[str], text: str) -> int:
        if not query_terms:
            return 0
        terms = _tokenize(text)
        return len(query_terms & terms)

    def _score_record(self, query_terms: set[str], record: WorkspaceRecord) -> int:
        weighted = (
            f"{record.title} {record.title} "
            f"{record.summary} "
            f"{' '.join(record.tags)} {' '.join(record.tags)}"
        )
        return self._score_text(query_terms, weighted)

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        for idx in range(2, 1000):
            candidate = path.with_name(f"{stem}-{idx}{suffix}")
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Could not allocate unique path for {path}")


@dataclass
class SkillDraft:
    """A procedural memory candidate distilled from successful work."""

    name: str
    description: str
    triggers: list[str]
    steps: list[str]
    constraints: list[str] = field(default_factory=list)
    recovery: list[str] = field(default_factory=list)
    source: str = ""


class ProceduralMemory:
    """Write procedural memory as normal Hermes Skill Markdown files."""

    def __init__(self, root: Path | None = None):
        self.root = root or default_procedural_skills_root()

    def write_skill(self, draft: SkillDraft, *, overwrite: bool = False) -> Path:
        """Create or update a procedural skill draft."""
        if not draft.name.strip():
            raise ValueError("Skill name cannot be empty")
        if not draft.description.strip():
            raise ValueError("Skill description cannot be empty")
        if not draft.triggers:
            raise ValueError("Skill draft must include at least one trigger")
        if not draft.steps:
            raise ValueError("Skill draft must include at least one step")

        slug = _slugify(draft.name)
        skill_dir = self.root / slug
        skill_path = skill_dir / "SKILL.md"
        if skill_path.exists() and not overwrite:
            raise FileExistsError(f"Procedural skill already exists: {skill_path}")
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(self.render_skill(draft), encoding="utf-8")
        return skill_path

    @staticmethod
    def render_skill(draft: SkillDraft) -> str:
        tags = ["procedural-memory", "agent-distilled"]
        front = _frontmatter(
            {
                "name": _slugify(draft.name),
                "description": draft.description.strip(),
                "version": "0.1.0",
                "author": "Hermes Memory Agent",
                "platforms": "[linux, macos, windows]",
                "metadata.hermes.tags": tags,
                "metadata.hermes.created_by": "agent",
            }
        )
        sections = [
            front,
            "",
            f"# {draft.name.strip()}",
            "",
            "## When To Use",
            "",
            *[f"- {item.strip()}" for item in draft.triggers if item.strip()],
            "",
            "## Procedure",
            "",
            *[f"{idx}. {step.strip()}" for idx, step in enumerate(draft.steps, start=1) if step.strip()],
        ]
        if draft.constraints:
            sections.extend(["", "## Constraints", "", *[f"- {c.strip()}" for c in draft.constraints if c.strip()]])
        if draft.recovery:
            sections.extend(["", "## Recovery", "", *[f"- {r.strip()}" for r in draft.recovery if r.strip()]])
        if draft.source.strip():
            sections.extend(["", "## Provenance", "", draft.source.strip()])
        return "\n".join(sections).rstrip() + "\n"


def filter_workspace_candidate(text: str) -> bool:
    """Return True when content has durable user-facing knowledge value."""
    stripped = text.strip()
    if len(stripped) < 40:
        return False
    low = stripped.lower()
    if re.search(r"\b(thanks|ok|sounds good|temporary|scratch|never mind)\b", low):
        return False
    return True


def format_retrieval_results(results: Iterable[RetrievalResult]) -> str:
    """Render retrieval results for CLI output or future context injection."""
    chunks = []
    for result in results:
        record = result.record
        chunks.append(
            f"## {record.title}\n"
            f"- bucket: {record.bucket}\n"
            f"- path: {record.path.as_posix()}\n"
            f"- score: {result.score}\n\n"
            f"{result.content.strip()}"
        )
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Agentic layer — LLM-powered routing, retrieval, and knowledge extraction
# ---------------------------------------------------------------------------

# ``(system_prompt, messages) -> response_text``
LLMCallable = Callable[[str, list[dict]], str]

_DEFAULT_MEMORY_MODEL = "google/gemini-2.5-flash-lite"


def make_llm_fn(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMCallable:
    """Return an LLMCallable backed by the hermes model config.

    Resolution order per parameter:
    1. Explicit argument
    2. ``~/.hermes/config.yaml`` model section
    3. Environment variables (``OPENROUTER_API_KEY``, ``OPENAI_API_KEY``)
    """
    # Pull ~/.hermes/.env into os.environ so API keys are visible
    try:
        from hermes_cli.config import reload_env
        reload_env()
    except Exception:
        pass

    try:
        from hermes_cli.config import load_config
        cfg = load_config().get("model", {})
    except Exception:
        cfg = {}

    resolved_base_url = (
        base_url
        or cfg.get("base_url")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )
    resolved_api_key = (
        api_key
        or cfg.get("api_key")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "no-key"
    )
    resolved_model = model or cfg.get("model") or _DEFAULT_MEMORY_MODEL

    from openai import OpenAI  # type: ignore[import-untyped]
    client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

    def _call(system: str, messages: list[dict]) -> str:
        full_messages = [{"role": "system", "content": system}] + messages
        response = client.chat.completions.create(
            model=resolved_model,
            max_tokens=4096,
            messages=full_messages,
        )
        return response.choices[0].message.content or ""

    return _call


# ── Prompts ─────────────────────────────────────────────────────────────────

_ROUTE_SYSTEM = """\
You are a personal knowledge management expert applying the PARA method.
Classify a new item into exactly one bucket.

PARA definitions:

Projects  — Active work toward a specific, bounded outcome with a deadline or
            clear end state requiring action now.
            Examples: "Implement PPO for CartPole (due Friday)", "Write Chapter 3".

Areas     — Ongoing domains of mastery or responsibility with no fixed end date.
            Knowledge you continuously maintain and develop.
            Examples: "Reinforcement Learning study notes", "Python practices".

Resources — Reference material that stands on its own; useful to retrieve later
            regardless of any active project. Concepts, algorithms, comparisons.
            Examples: "Policy Gradient derivation", "Off-policy RL comparison".

Archives  — Inactive, completed, or paused items.
            Examples: "Completed: RL Homework 4", "Retired fitness tracker".

Priority (apply in order):
1. Completion/inactivity signals (done, finished, archived) → Archives
2. Specific goal + deadline or next-action required → Projects
3. Ongoing learning domain / living knowledge base → Areas
4. Standalone concept, technique, or reference → Resources
"""

_ROUTE_USER_TMPL = """\
## Current Workspace Manifests

### Projects
{projects}

### Areas
{areas}

### Resources
{resources}

### Archives
{archives}

---
## Item to Classify

**Title**: {title}

**Content preview** (first 600 chars):
{content_excerpt}

---
Respond in EXACTLY this format (two lines only):

BUCKET: <Projects | Areas | Resources | Archives>
REASON: <one sentence>

Note: The user is taking a RL class now. So, RL related notes are likely to be in the RL folder under Projects.
"""

_RETRIEVE_SYSTEM = """\
You are a personal knowledge retrieval assistant.
Given a query and four PARA workspace manifests, select the files most likely
to answer the query. Be selective — choose only genuinely relevant files.
"""

_RETRIEVE_USER_TMPL = """\
## Query
{query}

## Manifests

### Projects
{projects}

### Areas
{areas}

### Resources
{resources}

### Archives
{archives}

---
List up to {top_k} files, most relevant first.
Only list files that appear in the manifests above.

Respond in EXACTLY this format:

RELEVANCE: <one sentence on what you expect to find>
FILES:
- <Bucket>/<filename.md>
- <Bucket>/<filename.md>

If nothing is relevant write: FILES: none
"""

_EXTRACT_SYSTEM = """\
You are a personal knowledge distillation expert.
Read a conversation and extract items with DURABLE EPISTEMIC VALUE worth
keeping in a PARA workspace.

Extract when the content is:
- A conceptual explanation of an algorithm, method, or theorem
- A technical derivation or proof worth referencing later
- A comparison or trade-off analysis between approaches
- A decision with rationale that shapes future study or work

Do NOT extract:
- Task completion records ("I finished X")
- Transient requests or clarifying questions
- Conversational filler
- Raw tool outputs (unless they contain standalone durable knowledge)

For each item output this block, then a line with only ---:

TITLE: <concise, search-friendly title, ≤10 words>
BUCKET: <Projects | Areas | Resources | Archives>
TAGS: <3–5 comma-separated lowercase tags>
SUMMARY: <1–2 sentences>
CONTENT:
<cleaned-up, self-contained markdown; rewrite for standalone readability>
---
"""

_EXTRACT_USER_TMPL = """\
Extract all durable knowledge items from the session below.
If there are none, respond with exactly: NO_ITEMS

Session:
---
{session_text}
---
"""


# ── Routing ──────────────────────────────────────────────────────────────────

def agentic_route(
    item: WorkspaceItem,
    workspace: PersonalWorkspace,
    llm_fn: LLMCallable,
) -> ParaBucket:
    """Ask the LLM to classify ``item`` into a PARA bucket.

    The LLM receives all four manifests so it classifies consistently with
    the existing taxonomy.  Raises ``ValueError`` if the response cannot be
    parsed into a valid bucket.
    """
    manifests = workspace.read_manifests()
    prompt = _ROUTE_USER_TMPL.format(
        projects=manifests["Projects"].strip(),
        areas=manifests["Areas"].strip(),
        resources=manifests["Resources"].strip(),
        archives=manifests["Archives"].strip(),
        title=item.title.strip(),
        content_excerpt=item.content.strip()[:600],
    )
    response = llm_fn(_ROUTE_SYSTEM, [{"role": "user", "content": prompt}])
    bucket = _parse_bucket(response)
    if bucket is None:
        raise ValueError(
            f"agentic_route: LLM returned unparseable bucket for {item.title!r}.\n"
            f"Raw response:\n{response}"
        )
    reason = _parse_reason(response)
    logger.info("agentic_route: %r → %s  (%s)", item.title, bucket, reason)
    return bucket


# ── Retrieval ────────────────────────────────────────────────────────────────

def agentic_retrieve(
    query: str,
    workspace: PersonalWorkspace,
    llm_fn: LLMCallable,
    *,
    top_k: int = 3,
) -> list[RetrievalResult]:
    """LLM-driven two-step retrieval: manifest scan → file selection → read.

    Step 1 — LLM reads all four manifests and names which files to read.
    Step 2 — Those files are loaded and returned as ``RetrievalResult`` objects
             in the LLM's ranked order (most relevant first).

    Raises ``ValueError`` if the LLM selects no files or names files that do
    not exist in the workspace.
    """
    workspace.initialize()
    manifests = workspace.read_manifests()
    prompt = _RETRIEVE_USER_TMPL.format(
        query=query.strip(),
        projects=manifests["Projects"].strip(),
        areas=manifests["Areas"].strip(),
        resources=manifests["Resources"].strip(),
        archives=manifests["Archives"].strip(),
        top_k=top_k * 2,
    )
    response = llm_fn(_RETRIEVE_SYSTEM, [{"role": "user", "content": prompt}])
    selected = _parse_file_list(response)

    relevance = _parse_relevance(response)
    logger.info("agentic_retrieve: query=%r  relevance=%s  files=%s", query, relevance, selected)

    if not selected:
        raise ValueError(
            f"agentic_retrieve: LLM selected no files for query {query!r}.\n"
            f"Raw response:\n{response}"
        )

    results: list[RetrievalResult] = []
    missing: list[str] = []
    for bucket, filename in selected[:top_k]:
        file_path = workspace.root / bucket / filename
        if not file_path.exists() or file_path.name == MANIFEST_NAME:
            missing.append(f"{bucket}/{filename}")
            continue
        content = file_path.read_text(encoding="utf-8")
        record = WorkspaceRecord(
            bucket=bucket,  # type: ignore[arg-type]
            path=Path(filename),
            title=_extract_title(content, filename),
            summary=_extract_summary(content),
        )
        results.append(RetrievalResult(record=record, score=len(results), content=content))

    if missing:
        logger.warning("agentic_retrieve: LLM named non-existent files: %s", missing)
    if not results:
        raise ValueError(
            f"agentic_retrieve: none of the LLM-selected files exist in the workspace.\n"
            f"Selected: {selected}\nMissing: {missing}"
        )
    return results


# ── Knowledge extraction ─────────────────────────────────────────────────────

def extract_knowledge_items(
    session_text: str,
    llm_fn: LLMCallable,
) -> list[WorkspaceItem]:
    """Extract durable knowledge items from raw conversation text.

    Returns a list of ``WorkspaceItem`` objects with ``bucket`` pre-assigned.
    Items that fail the ``filter_workspace_candidate`` quality gate are dropped.
    Returns an empty list (never raises) when the session has no durable content.
    """
    if not session_text.strip():
        return []
    prompt = _EXTRACT_USER_TMPL.format(session_text=session_text.strip())
    response = llm_fn(_EXTRACT_SYSTEM, [{"role": "user", "content": prompt}])
    if "NO_ITEMS" in response.upper():
        logger.info("extract_knowledge_items: no durable items in session")
        return []
    items = _parse_item_blocks(response)
    logger.info("extract_knowledge_items: extracted %d item(s)", len(items))
    for item in items:
        logger.info("  · [%s] %s", item.bucket or "?", item.title)
    return items


def ingest_session(
    session_text: str,
    workspace: PersonalWorkspace,
    llm_fn: LLMCallable,
    *,
    write_mode: Literal["new", "append", "update"] = "new",
) -> list[Path]:
    """Extract knowledge from ``session_text``, route each item, write to workspace.

    Items with a bucket pre-assigned by the extractor skip a second routing
    call.  Items without a bucket go through ``agentic_route``.

    Returns the list of paths written.
    """
    items = extract_knowledge_items(session_text, llm_fn)
    if not items:
        return []
    workspace.initialize()
    written: list[Path] = []
    for item in items:
        if item.bucket is None:
            item.bucket = agentic_route(item, workspace, llm_fn)
        path = workspace.write_item(item, mode=write_mode)
        logger.info("ingest_session: wrote %s  →  %s", item.title, path)
        written.append(path)
    return written


# ── Parse helpers ────────────────────────────────────────────────────────────

def _parse_bucket(text: str) -> Optional[ParaBucket]:
    m = re.search(r"BUCKET\s*:\s*(Projects|Areas|Resources|Archives)", text, re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).strip()
    for b in PARA_BUCKETS:
        if b.lower() == raw.lower():
            return b  # type: ignore[return-value]
    return None


def _parse_reason(text: str) -> str:
    m = re.search(r"REASON\s*:\s*(.+)", text)
    return m.group(1).strip() if m else ""


def _parse_relevance(text: str) -> str:
    m = re.search(r"RELEVANCE\s*:\s*(.+)", text)
    return m.group(1).strip() if m else ""


def _parse_file_list(text: str) -> list[tuple[str, str]]:
    """Parse ``FILES:\\n- Bucket/file.md`` lines from an LLM response."""
    results: list[tuple[str, str]] = []
    in_files = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FILES:"):
            rest = stripped.split(":", 1)[1].strip()
            if rest.lower() == "none":
                return []
            in_files = True
            continue
        if in_files and stripped.startswith("-"):
            entry = stripped.lstrip("- ").strip()
            if "/" in entry:
                bucket_raw, filename = entry.split("/", 1)
                for b in PARA_BUCKETS:
                    if b.lower() == bucket_raw.strip().lower():
                        results.append((b, filename.strip()))
                        break
        elif in_files and stripped and not stripped.startswith("-"):
            break
    return results


def _parse_item_blocks(text: str) -> list[WorkspaceItem]:
    items: list[WorkspaceItem] = []
    for block in re.split(r"\n---\n", text):
        block = block.strip()
        if not block:
            continue
        item = _parse_single_block(block)
        if item is not None and filter_workspace_candidate(item.content):
            items.append(item)
    return items


def _parse_single_block(block: str) -> Optional[WorkspaceItem]:
    title_m = re.search(r"^TITLE:\s*(.+)$", block, re.MULTILINE)
    bucket_m = re.search(r"^BUCKET:\s*(Projects|Areas|Resources|Archives)", block, re.MULTILINE | re.IGNORECASE)
    tags_m = re.search(r"^TAGS:\s*(.+)$", block, re.MULTILINE)
    summary_m = re.search(r"^SUMMARY:\s*(.+)$", block, re.MULTILINE)
    content_m = re.search(r"^CONTENT:\s*\n(.*)", block, re.MULTILINE | re.DOTALL)
    if not title_m or not content_m:
        return None
    title = title_m.group(1).strip()
    content = content_m.group(1).strip()
    if not title or not content:
        return None
    bucket: Optional[ParaBucket] = None
    if bucket_m:
        raw = bucket_m.group(1).strip()
        for b in PARA_BUCKETS:
            if b.lower() == raw.lower():
                bucket = b  # type: ignore[assignment]
                break
    tags = [t.strip() for t in tags_m.group(1).split(",") if t.strip()] if tags_m else []
    summary = summary_m.group(1).strip() if summary_m else ""
    return WorkspaceItem(title=title, content=content, bucket=bucket, summary=summary, tags=tags)


def _extract_title(content: str, fallback: str) -> str:
    for pat in (r'^title:\s*["\']?(.+?)["\']?\s*$', r'^#\s+(.+)$'):
        m = re.search(pat, content, re.MULTILINE)
        if m:
            return m.group(1).strip()
    return Path(fallback).stem.replace("-", " ").title()


def _extract_summary(content: str) -> str:
    m = re.search(r'^summary:\s*["\']?(.+?)["\']?\s*$', content, re.MULTILINE)
    return m.group(1).strip() if m else ""
