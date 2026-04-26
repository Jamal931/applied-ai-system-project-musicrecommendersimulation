"""
AI-powered music recommender — Retrieval-Augmented Generation + Agentic Workflow.

Stretch features implemented
-----------------------------
RAG Enhancement   — second data source: genre_profiles.json, exposed via the
                    get_genre_info tool so Claude can look up genre context
                    before or while searching for songs.

Agentic Enhancement — multi-step observable reasoning: when verbose=True the
                    recommender runs an explicit planning phase (Claude states
                    its search strategy without making tool calls), then an
                    execution phase (Claude searches and generates), logging
                    each step as [PLAN] / [TOOL CALL] / [OBSERVATION] / [GENERATE].

Specialization    — few-shot system prompt (SPECIALIZED_SYSTEM) with two
                    worked examples that train Claude to cite exact feature
                    values and explain listening-context fit. Enabled via
                    specialized=True on recommend().

Baseline          — original _SYSTEM prompt with no few-shot examples,
                    used when specialized=False for comparison.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_CATALOG: list[dict] = []
_GENRE_PROFILES: dict = {}

_GENRE_PROFILES_PATH = Path(__file__).parent.parent / "data" / "genre_profiles.json"


def _load_genre_profiles() -> dict:
    try:
        with open(_GENRE_PROFILES_PATH, encoding="utf-8") as f:
            profiles = json.load(f)
        logger.info("Loaded %d genre profiles from %s", len(profiles), _GENRE_PROFILES_PATH)
        return profiles
    except FileNotFoundError:
        logger.warning("genre_profiles.json not found — get_genre_info tool will return empty results")
        return {}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _search_songs(
    genre: str | None = None,
    mood: str | None = None,
    min_energy: float | None = None,
    max_energy: float | None = None,
    limit: int = 5,
) -> list[dict]:
    """Filter the in-memory catalog and return matching songs (tool backend)."""
    results = _CATALOG
    if genre:
        results = [s for s in results if s["genre"].lower() == genre.lower()]
    if mood:
        results = [s for s in results if s["mood"].lower() == mood.lower()]
    if min_energy is not None:
        results = [s for s in results if s["energy"] >= min_energy]
    if max_energy is not None:
        results = [s for s in results if s["energy"] <= max_energy]
    return results[:limit]


def _get_genre_info(genre: str) -> dict | str:
    """
    Return the genre profile for *genre* from genre_profiles.json.
    RAG Enhancement: second data source that gives Claude richer context
    about typical energy ranges, tempo, best use cases, and mood pairings
    before it starts searching for songs.
    """
    profile = _GENRE_PROFILES.get(genre.lower())
    if not profile:
        available = ", ".join(_GENRE_PROFILES.keys()) or "none loaded"
        return f"No profile found for '{genre}'. Available genres: {available}."
    return profile


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "search_songs",
        "description": (
            "Search the music catalog. All parameters are optional — omit any "
            "you don't need. Returns a list of matching song objects with fields: "
            "title, artist, genre, mood, energy (0–1), tempo_bpm, valence (0–1), "
            "danceability (0–1), acousticness (0–1)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genre": {
                    "type": "string",
                    "description": "Genre to filter by (pop, lofi, rock, jazz, ambient, synthwave, indie pop).",
                },
                "mood": {
                    "type": "string",
                    "description": "Mood to filter by (happy, chill, intense, moody, focused, relaxed).",
                },
                "min_energy": {"type": "number", "description": "Minimum energy level, 0.0–1.0."},
                "max_energy": {"type": "number", "description": "Maximum energy level, 0.0–1.0."},
                "limit": {"type": "integer", "description": "Maximum results to return (default 5)."},
            },
        },
    },
    {
        "name": "get_genre_info",
        "description": (
            "Look up detailed context for a music genre: typical energy range, tempo, "
            "acousticness, best listening situations, and mood pairings. "
            "Use this BEFORE searching for songs when you need to understand what a "
            "genre sounds like or whether it fits the user's request."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genre": {
                    "type": "string",
                    "description": "Genre name to look up (pop, lofi, rock, jazz, ambient, synthwave, indie pop).",
                }
            },
            "required": ["genre"],
        },
    },
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a music recommendation assistant with access to a small song catalog.

When the user describes their taste or listening situation, use the search_songs
tool to retrieve candidates. You may also call get_genre_info to understand a
genre before searching. Call tools multiple times with different filters to
gather a diverse set of candidates.

After retrieving candidates, recommend the best 3–5 songs and explain why each
fits — reference concrete features like energy level, tempo, mood, and genre.
Keep responses friendly, specific, and concise."""

# Specialization: two few-shot examples that model expert-level output.
# This prompt measurably increases how often Claude cites exact feature values
# and explains listening-context fit, compared to the baseline _SYSTEM above.
_SPECIALIZED_SYSTEM = """\
You are an expert music curator with deep knowledge of acoustic properties and
listening contexts.

When recommending songs, always:
- Reference specific numeric features (energy level, tempo BPM, acousticness).
- Explain the listening-context fit (why this song works for the situation).
- Note any catalog limitations (e.g. only one jazz track available).
- Use get_genre_info to look up a genre before searching if you are uncertain.

--- EXAMPLE 1 ---

User: I need music for deep focus while coding at night.

Assistant (search plan): I'll look for tracks tagged "focused" or "chill" with
energy below 0.5, and check the lofi genre profile to confirm it fits.

[Tool: get_genre_info(genre="lofi")]
[Tool: search_songs(mood="focused")]
[Tool: search_songs(mood="chill", max_energy=0.5)]

Here are my recommendations:

1. **Focus Flow** (LoRoom) — lofi | energy 0.40 | 80 BPM | mood: focused
   The only track tagged "focused" in the catalog. 80 BPM sits in the ideal
   range for sustained cognitive work — alert but not distracting. Very high
   acousticness (0.78) provides warmth without electronic harshness.

2. **Library Rain** (Paper Lanterns) — lofi | energy 0.35 | 72 BPM | mood: chill
   Even lower energy than Focus Flow, making it better for reading-intensive
   sessions. Acousticness of 0.86 is the highest in the catalog — closest to
   a silent acoustic environment.

--- EXAMPLE 2 ---

User: Something energetic for a Saturday morning run.

Assistant (search plan): I'll check the pop and rock profiles for high-energy
characteristics, then search by energy ≥ 0.85 and intense/happy mood.

[Tool: get_genre_info(genre="rock")]
[Tool: search_songs(min_energy=0.85)]
[Tool: search_songs(mood="intense")]

Here are my recommendations:

1. **Gym Hero** (Max Pulse) — pop | energy 0.93 | 132 BPM | mood: intense
   The single highest-energy track in the catalog. 132 BPM matches a brisk
   running cadence. Near-zero acousticness (0.05) means a clean, punchy
   electronic mix with no softening warmth.

2. **Storm Runner** (Voltline) — rock | energy 0.91 | 152 BPM | mood: intense
   152 BPM is ideal for interval sprints. The raw rock instrumentation adds
   a physical, push-through-it quality that pure electronic tracks lack.

--- END EXAMPLES ---

Now respond to the user's actual request with the same level of specificity."""

# Prompt used in Phase 1 of the verbose agentic mode (planning only, no tools).
_PLANNING_PROMPT = """\
You are planning music catalog searches. The user has made a request.
Output ONLY a 2–3 sentence search plan: which genres, moods, and energy
ranges you will query, and why. Do not make any tool calls and do not
write recommendations yet. Just describe your plan."""


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _dispatch_tool(name: str, args: dict) -> str:
    if name == "search_songs":
        result = _search_songs(**args)
    elif name == "get_genre_info":
        result = _get_genre_info(**args)
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Core agentic loop (shared by both modes)
# ---------------------------------------------------------------------------

def _run_loop(
    client: anthropic.Anthropic,
    system: str,
    messages: list[dict[str, Any]],
    tool_log: list[dict],
    max_iterations: int,
) -> str:
    for iteration in range(max_iterations):
        logger.debug("Loop iteration %d", iteration)
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1200,
                system=system,
                tools=_TOOLS,
                messages=messages,
            )
        except anthropic.APIError as exc:
            logger.error("Claude API error: %s", exc)
            raise

        if response.stop_reason == "end_turn":
            text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            logger.info("[GENERATE] %d char(s) after %d iteration(s)", len(text), iteration + 1)
            return text

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                args = block.input
                log_entry = {"tool": block.name, "args": args}
                logger.info(
                    "[TOOL CALL] %s(%s)",
                    block.name,
                    ", ".join(f"{k}={v!r}" for k, v in args.items()),
                )
                result_str = _dispatch_tool(block.name, args)
                result_obj = json.loads(result_str)
                hit_count = len(result_obj) if isinstance(result_obj, list) else 1
                logger.info("[OBSERVATION] %s returned %s result(s)", block.name, hit_count)
                log_entry["result_count"] = hit_count
                tool_log.append(log_entry)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result_str}
                )
            messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

    raise RuntimeError(f"Agentic loop did not converge after {max_iterations} iterations")


# ---------------------------------------------------------------------------
# AIRecommender
# ---------------------------------------------------------------------------

class AIRecommender:
    """
    Claude-powered recommender with RAG, agentic tool-use, and specialization.

    Parameters
    ----------
    songs : list[dict]
        Song catalog (output of ``load_songs``).
    genre_profiles_path : str | None
        Override path to genre_profiles.json.  ``None`` uses the default location.
    """

    def __init__(
        self,
        songs: list[dict],
        genre_profiles_path: str | None = None,
    ) -> None:
        global _CATALOG, _GENRE_PROFILES
        _CATALOG = songs
        path = genre_profiles_path or str(_GENRE_PROFILES_PATH)
        _GENRE_PROFILES = _load_genre_profiles() if Path(path).exists() else {}
        self.client = anthropic.Anthropic()
        logger.info(
            "AIRecommender ready — %d songs, %d genre profiles",
            len(songs), len(_GENRE_PROFILES),
        )

    def recommend(
        self,
        user_description: str,
        *,
        specialized: bool = False,
        verbose: bool = False,
        max_iterations: int = 6,
    ) -> "str | dict":
        """
        Generate music recommendations.

        Parameters
        ----------
        user_description : str
            Natural-language preference description from the user.
        specialized : bool
            Use the few-shot specialized system prompt instead of the baseline.
            Produces more feature-specific, expert-style output.
        verbose : bool
            Enable observable multi-step agentic mode.  Runs a planning phase
            (Phase 1) before the main RAG loop (Phase 2), and returns a dict
            with ``plan``, ``tool_calls``, and ``recommendation`` keys instead
            of a plain string.
        max_iterations : int
            Maximum tool-use iterations in the agentic loop.

        Returns
        -------
        str
            Recommendation text (when ``verbose=False``).
        dict
            ``{"plan": str, "tool_calls": list, "recommendation": str}``
            (when ``verbose=True``).
        """
        logger.info(
            "recommend() query=%r specialized=%s verbose=%s",
            user_description, specialized, verbose,
        )
        system = _SPECIALIZED_SYSTEM if specialized else _SYSTEM
        tool_log: list[dict] = []

        if verbose:
            # ----------------------------------------------------------------
            # Phase 1 — Planning (no tools, observable reasoning step)
            # ----------------------------------------------------------------
            plan_response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=_PLANNING_PROMPT,
                messages=[{"role": "user", "content": user_description}],
            )
            plan = next(
                (b.text for b in plan_response.content if hasattr(b, "text")), ""
            ).strip()
            logger.info("[PLAN] %s", plan)

            # ----------------------------------------------------------------
            # Phase 2 — Execution: inject plan into the conversation
            # ----------------------------------------------------------------
            augmented_system = (
                system
                + f"\n\nSearch plan for this request:\n{plan}\n\nExecute the plan now."
            )
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": user_description}
            ]
            recommendation = _run_loop(
                self.client, augmented_system, messages, tool_log, max_iterations
            )
            return {
                "plan": plan,
                "tool_calls": tool_log,
                "recommendation": recommendation,
            }

        # --------------------------------------------------------------------
        # Standard mode (same as before, single loop)
        # --------------------------------------------------------------------
        messages = [{"role": "user", "content": user_description}]
        return _run_loop(self.client, system, messages, tool_log, max_iterations)
