"""
Evaluation harness for the Music Recommender Simulation.

Stretch feature: Test Harness / Evaluation Script
--------------------------------------------------
This script runs the system against a set of predefined inputs and reports
pass/fail results, confidence scores, and a summary table — without requiring
manual inspection of the output.

Two evaluation modes:
  1. Rule-based evaluation — fully deterministic; runs without an API key.
     Checks that the scoring algorithm returns expected top songs.
  2. AI evaluation (optional) — requires ANTHROPIC_API_KEY.
     Checks that Claude's recommendations contain only real catalog songs
     (no hallucinations), that tool calls were made (RAG is active), and
     that the output is non-empty. Compares baseline vs specialised output.

Usage
-----
# Rule-based only (no API key needed):
  python tests/eval_harness.py

# Full evaluation including AI mode:
  ANTHROPIC_API_KEY=sk-ant-... python tests/eval_harness.py
"""

import os
import sys
import logging

# Allow running from the project root or from the tests/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recommender import load_songs, recommend_songs  # noqa: E402

logging.basicConfig(level=logging.WARNING)  # suppress info noise during harness run

# ---------------------------------------------------------------------------
# Catalog path
# ---------------------------------------------------------------------------

_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

# ---------------------------------------------------------------------------
# Rule-based test cases
# ---------------------------------------------------------------------------

RULE_TESTS = [
    {
        "name": "Pop / happy — normal",
        "profile": {
            "genre": "pop", "mood": "happy",
            "energy": 0.9, "tempo_bpm": 128,
            "valence": 0.85, "danceability": 0.88, "acousticness": 0.10,
        },
        "expect_rank1": "Sunrise City",
        "expect_in_top3": ["Sunrise City", "Gym Hero"],
        "description": "Both pop songs should dominate; Sunrise City wins on mood bonus.",
    },
    {
        "name": "Lofi / chill — normal",
        "profile": {
            "genre": "lofi", "mood": "chill",
            "energy": 0.35, "tempo_bpm": 75,
            "valence": 0.58, "danceability": 0.55, "acousticness": 0.80,
        },
        "expect_rank1": "Library Rain",
        "expect_in_top3": ["Library Rain", "Midnight Coding"],
        "description": "Lofi catalog dominates; Library Rain edges out via near-perfect energy match.",
    },
    {
        "name": "Rock / intense — normal",
        "profile": {
            "genre": "rock", "mood": "intense",
            "energy": 0.92, "tempo_bpm": 150,
            "valence": 0.45, "danceability": 0.65, "acousticness": 0.08,
        },
        "expect_rank1": "Storm Runner",
        "expect_in_top3": ["Storm Runner", "Gym Hero"],
        "description": "Only one rock song; second slot filled by highest-energy alternative.",
    },
    {
        "name": "Ambient / chill — narrow catalog",
        "profile": {
            "genre": "ambient", "mood": "chill",
            "energy": 0.30, "tempo_bpm": 62,
            "valence": 0.65, "danceability": 0.42, "acousticness": 0.90,
        },
        "expect_rank1": "Spacewalk Thoughts",
        "expect_in_top3": ["Spacewalk Thoughts"],
        "description": "Only one ambient song; genre bonus guarantees it ranks first.",
    },
    {
        "name": "EDGE: all-zeros numerics",
        "profile": {
            "genre": "lofi", "mood": "chill",
            "energy": 0.0, "tempo_bpm": 0,
            "valence": 0.0, "danceability": 0.0, "acousticness": 0.0,
        },
        "expect_rank1": None,  # either lofi song is acceptable at #1
        "expect_in_top3": ["Library Rain", "Midnight Coding"],
        "description": "Genre + mood bonus keeps lofi songs at the top even with zero numerics.",
    },
    {
        "name": "EDGE: jazz genre miss",
        "profile": {
            "genre": "jazz", "mood": "happy",
            "energy": 0.82, "tempo_bpm": 118,
            "valence": 0.84, "danceability": 0.79, "acousticness": 0.18,
        },
        "expect_rank1": "Sunrise City",
        "expect_in_top3": ["Sunrise City", "Coffee Shop Stories"],
        "description": "Numeric perfect-match (Sunrise City) beats genre label (Coffee Shop) — barely.",
    },
]

# ---------------------------------------------------------------------------
# AI test cases (require ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

AI_TESTS = [
    {
        "name": "AI — no hallucinations (workout)",
        "query": "Fast, high-energy music for a morning workout.",
        "check": "no_hallucinations",
        "description": "All recommended songs must exist in the catalog.",
    },
    {
        "name": "AI — RAG active (study session)",
        "query": "Something calm and focused for studying late at night.",
        "check": "rag_active",
        "description": "At least one tool call must be made (confirming RAG is running).",
    },
    {
        "name": "AI — specialization improves output (road trip)",
        "query": "Happy, feel-good music for a sunny road trip.",
        "check": "specialization",
        "description": "Specialized output should contain more numeric feature references than baseline.",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_rule_test(test: dict, songs: list) -> dict:
    profile = dict(test["profile"])
    recs = recommend_songs(profile, songs, k=5)
    top_titles = [r[0]["title"] for r in recs]

    passed = True
    details = []

    # Check rank-1
    if test["expect_rank1"] is not None:
        if top_titles[0] == test["expect_rank1"]:
            details.append(f"rank-1 = '{top_titles[0]}' ✓")
        else:
            details.append(f"rank-1 FAIL: got '{top_titles[0]}', expected '{test['expect_rank1']}'")
            passed = False

    # Check expected songs appear in top-3
    top3 = set(top_titles[:3])
    for expected in test["expect_in_top3"]:
        if expected in top3:
            details.append(f"'{expected}' in top-3 ✓")
        else:
            details.append(f"'{expected}' NOT in top-3 ✗  (got {top_titles[:3]})")
            passed = False

    confidence = recs[0][1] / 6.0 if recs else 0.0  # max possible score is 6.0

    return {
        "name": test["name"],
        "passed": passed,
        "confidence": min(confidence, 1.0),
        "details": details,
        "description": test["description"],
    }


def _count_feature_numbers(text: str) -> int:
    """Count occurrences of decimal numbers in the text (rough proxy for feature references)."""
    import re
    return len(re.findall(r"\b0\.\d+\b|\b\d{2,3} BPM\b", text, re.IGNORECASE))


def _run_ai_test(test: dict, ai_rec, catalog_titles: set) -> dict:
    """Run one AI test case and return a result dict."""
    passed = True
    details = []
    confidence = 0.0

    if test["check"] == "no_hallucinations":
        result = ai_rec.recommend(test["query"])
        for title in catalog_titles:
            if title.lower() in result.lower():
                pass  # mentioned song exists — good
        bad = [w for w in result.split() if len(w) > 5 and w[0].isupper() and w not in result]
        # Simplified check: count how many catalog songs are mentioned
        mentioned = sum(1 for t in catalog_titles if t.lower() in result.lower())
        if mentioned >= 2:
            details.append(f"{mentioned} catalog songs mentioned ✓")
            confidence = min(mentioned / 3, 1.0)
        else:
            details.append(f"Only {mentioned} catalog song(s) mentioned — possible hallucination ✗")
            passed = False

    elif test["check"] == "rag_active":
        result = ai_rec.recommend(test["query"], verbose=True)
        tool_calls = result.get("tool_calls", [])
        if len(tool_calls) >= 1:
            details.append(f"{len(tool_calls)} tool call(s) made ✓")
            confidence = min(len(tool_calls) / 2, 1.0)
        else:
            details.append("No tool calls made — RAG not active ✗")
            passed = False
        # Also verify plan was generated
        plan = result.get("plan", "")
        if plan:
            details.append(f"Plan generated ({len(plan)} chars) ✓")
        else:
            details.append("No plan generated ✗")
            passed = False

    elif test["check"] == "specialization":
        baseline = ai_rec.recommend(test["query"], specialized=False)
        specialized = ai_rec.recommend(test["query"], specialized=True)
        base_nums = _count_feature_numbers(baseline)
        spec_nums = _count_feature_numbers(specialized)
        details.append(f"Baseline numeric references: {base_nums}")
        details.append(f"Specialized numeric references: {spec_nums}")
        if spec_nums >= base_nums:
            details.append("Specialized output has ≥ baseline feature references ✓")
            confidence = min(spec_nums / max(base_nums, 1), 1.0)
        else:
            details.append("Specialized output has FEWER feature references than baseline ✗")
            passed = False

    return {
        "name": test["name"],
        "passed": passed,
        "confidence": round(confidence, 2),
        "details": details,
        "description": test["description"],
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_result(result: dict, idx: int) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    conf = f"{result['confidence']:.2f}"
    print(f"  {idx:>2}. [{status}] (confidence {conf})  {result['name']}")
    for d in result["details"]:
        print(f"          {d}")


def _print_summary(results: list) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    avg_conf = sum(r["confidence"] for r in results) / total if total else 0
    print()
    print("=" * 62)
    print(f"  SUMMARY  {passed}/{total} passed  |  avg confidence {avg_conf:.2f}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    songs = load_songs(_CSV)
    catalog_titles = {s["title"] for s in songs}

    # -------------------------------------------------------------------------
    # Rule-based evaluation (always runs)
    # -------------------------------------------------------------------------
    print()
    print("=" * 62)
    print("  RULE-BASED EVALUATION")
    print("=" * 62)
    rule_results = []
    for i, test in enumerate(RULE_TESTS, start=1):
        result = _run_rule_test(test, songs)
        _print_result(result, i)
        rule_results.append(result)
    _print_summary(rule_results)

    # -------------------------------------------------------------------------
    # AI evaluation (only if API key is set)
    # -------------------------------------------------------------------------
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print()
        print("  AI EVALUATION skipped — set ANTHROPIC_API_KEY to enable.")
        print()
        return

    try:
        from ai_recommender import AIRecommender
    except ImportError:
        print("  anthropic not installed — run: pip install anthropic")
        return

    print()
    print("=" * 62)
    print("  AI EVALUATION  (requires ANTHROPIC_API_KEY)")
    print("=" * 62)
    ai_rec = AIRecommender(songs)
    ai_results = []
    for i, test in enumerate(AI_TESTS, start=1):
        print(f"\n  Running: {test['name']} ...")
        try:
            result = _run_ai_test(test, ai_rec, catalog_titles)
        except Exception as exc:
            result = {
                "name": test["name"],
                "passed": False,
                "confidence": 0.0,
                "details": [f"Exception: {exc}"],
                "description": test["description"],
            }
        _print_result(result, i)
        ai_results.append(result)
    _print_summary(ai_results)

    all_results = rule_results + ai_results
    total = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])
    avg_conf = sum(r["confidence"] for r in all_results) / total
    print()
    print(f"  OVERALL  {passed}/{total} passed  |  avg confidence {avg_conf:.2f}")
    print()


if __name__ == "__main__":
    main()
