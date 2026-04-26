"""
Music Recommender Simulation — entry point.

Rule-based demo runs unconditionally.
AI demo (all four stretch features) runs only when ANTHROPIC_API_KEY is set.

Stretch features demonstrated
------------------------------
  RAG Enhancement    — AIRecommender now has a second tool (get_genre_info)
                       backed by data/genre_profiles.json; Claude calls it
                       to look up genre context before searching for songs.

  Agentic Enhancement — verbose=True triggers a two-phase loop:
                       Phase 1 generates an explicit search plan (observable),
                       Phase 2 executes the plan with tool calls (observable).
                       Every step is logged as [PLAN] / [TOOL CALL] / [OBSERVATION].

  Specialization      — specialized=True activates the few-shot system prompt
                       with two worked examples that train Claude to cite exact
                       feature values and explain listening-context fit.
                       The same query is run in both modes so the difference
                       is visible side-by-side.

  Test Harness        — run tests/eval_harness.py for the automated pass/fail
                       evaluation script with confidence scoring.
"""

import logging
import os
import sys

from recommender import load_songs, recommend_songs

# ---------------------------------------------------------------------------
# Logging — stdout + recommender.log
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("recommender.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based demo profiles
# ---------------------------------------------------------------------------

PROFILES = [
    {
        "label": "High-Energy Pop",
        "genre": "pop", "mood": "happy",
        "energy": 0.9, "tempo_bpm": 128,
        "valence": 0.85, "danceability": 0.88, "acousticness": 0.10,
    },
    {
        "label": "Chill Lofi",
        "genre": "lofi", "mood": "chill",
        "energy": 0.35, "tempo_bpm": 75,
        "valence": 0.58, "danceability": 0.55, "acousticness": 0.80,
    },
    {
        "label": "Deep Intense Rock",
        "genre": "rock", "mood": "intense",
        "energy": 0.92, "tempo_bpm": 150,
        "valence": 0.45, "danceability": 0.65, "acousticness": 0.08,
    },
    {
        "label": "EDGE: High Energy + Sad Mood (conflicting)",
        "genre": "ambient", "mood": "sad",
        "energy": 0.95, "tempo_bpm": 60,
        "valence": 0.20, "danceability": 0.30, "acousticness": 0.90,
    },
    {
        "label": "EDGE: Genre Miss — every feature matches pop but genre=jazz",
        "genre": "jazz", "mood": "happy",
        "energy": 0.82, "tempo_bpm": 118,
        "valence": 0.84, "danceability": 0.79, "acousticness": 0.18,
    },
    {
        "label": "EDGE: All-zeros numeric profile",
        "genre": "lofi", "mood": "chill",
        "energy": 0.0, "tempo_bpm": 0,
        "valence": 0.0, "danceability": 0.0, "acousticness": 0.0,
    },
]


def print_recommendations(label: str, user_prefs: dict, recommendations: list) -> None:
    width = 64
    print()
    print("=" * width)
    print(f" {label}".center(width))
    print("=" * width)
    print(
        f"  genre={user_prefs['genre']}  mood={user_prefs['mood']}  "
        f"energy={user_prefs['energy']}  tempo={user_prefs['tempo_bpm']}"
    )
    print("-" * width)
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"  #{rank}  {song['title']}  ({song['artist']})")
        print(f"       Score : {score:.2f}")
        for reason in explanation.split(" | "):
            print(f"         • {reason}")
        print()
    print("=" * width)


# ---------------------------------------------------------------------------
# AI demo helpers
# ---------------------------------------------------------------------------

def _divider(title: str = "", width: int = 64) -> None:
    if title:
        print(f"\n{'─' * width}")
        print(f"  {title}")
        print(f"{'─' * width}")
    else:
        print("─" * width)


def run_ai_demo(songs: list) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set — skipping AI demo. "
            "Export the variable and re-run to see Claude-powered recommendations."
        )
        return

    try:
        from ai_recommender import AIRecommender
    except ImportError:
        logger.error("anthropic not installed. Run: pip install anthropic")
        return

    print()
    print("=" * 64)
    print("  AI-POWERED RECOMMENDATIONS".center(64))
    print("=" * 64)

    ai_rec = AIRecommender(songs)

    # -----------------------------------------------------------------------
    # Demo 1: RAG Enhancement — get_genre_info tool + search_songs
    # -----------------------------------------------------------------------
    _divider("DEMO 1 — RAG Enhancement (two-tool retrieval)")
    query = "I want upbeat, feel-good music for a sunny road trip."
    print(f"\n> {query}\n")
    print("  Claude will call get_genre_info AND search_songs — watch the logs.\n")
    try:
        result = ai_rec.recommend(query)
        print(result)
    except Exception as exc:
        logger.error("AI recommendation failed: %s", exc)

    # -----------------------------------------------------------------------
    # Demo 2: Agentic Enhancement — verbose two-phase mode
    # -----------------------------------------------------------------------
    _divider("DEMO 2 — Agentic Enhancement (verbose multi-step reasoning)")
    query = "I need something calm and focused for studying late at night."
    print(f"\n> {query}\n")
    try:
        result = ai_rec.recommend(query, verbose=True)
        print(f"[PLAN]\n{result['plan']}\n")
        print(f"[TOOL CALLS] {len(result['tool_calls'])} call(s) made")
        for tc in result["tool_calls"]:
            print(f"  • {tc['tool']}({tc['args']}) → {tc['result_count']} result(s)")
        print()
        print("[RECOMMENDATION]")
        print(result["recommendation"])
    except Exception as exc:
        logger.error("Verbose recommendation failed: %s", exc)

    # -----------------------------------------------------------------------
    # Demo 3: Specialization — baseline vs few-shot side-by-side
    # -----------------------------------------------------------------------
    _divider("DEMO 3 — Specialization (baseline vs few-shot comparison)")
    query = "High-energy workout music — fast tempo, no acoustic softness."
    print(f"\n> {query}\n")
    try:
        baseline = ai_rec.recommend(query, specialized=False)
        specialized = ai_rec.recommend(query, specialized=True)

        print("── BASELINE (no few-shot) ──")
        print(baseline)
        print()
        print("── SPECIALIZED (few-shot examples in system prompt) ──")
        print(specialized)
    except Exception as exc:
        logger.error("Specialization comparison failed: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Music Recommender starting")
    songs = load_songs("data/songs.csv")

    # Rule-based demo
    for profile in PROFILES:
        label = profile.pop("label")
        recommendations = recommend_songs(profile, songs, k=5)
        print_recommendations(label, profile, recommendations)
        profile["label"] = label

    # AI demo (all four stretch features)
    run_ai_demo(songs)
    logger.info("Done. See recommender.log for full trace.")


if __name__ == "__main__":
    main()
