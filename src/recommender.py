import csv
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


class Recommender:
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "acousticness": 0.8 if user.likes_acoustic else 0.2,
        }
        scored = [
            (song, score_song(user_prefs, asdict(song))[0])
            for song in self.songs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            "recommend(): scored %d songs for genre=%s mood=%s",
            len(self.songs), user.favorite_genre, user.favorite_mood,
        )
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "acousticness": 0.8 if user.likes_acoustic else 0.2,
        }
        _, reasons = score_song(user_prefs, asdict(song))
        explanation = " | ".join(reasons) if reasons else "No matching features."
        logger.debug("explain_recommendation(): %s → %s", song.title, explanation)
        return explanation


def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv and return a list of dicts with numeric fields cast correctly."""
    songs: List[Dict] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["id"] = int(row["id"])
                row["energy"] = float(row["energy"])
                row["tempo_bpm"] = float(row["tempo_bpm"])
                row["valence"] = float(row["valence"])
                row["danceability"] = float(row["danceability"])
                row["acousticness"] = float(row["acousticness"])
                songs.append(row)
        logger.info("Loaded %d songs from %s", len(songs), csv_path)
    except FileNotFoundError:
        logger.error("Catalog not found: %s", csv_path)
        raise
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Return (total_score, reasons) for one song judged against user_prefs."""
    score = 0.0
    reasons: List[str] = []

    if song["genre"].lower() == user_prefs.get("genre", "").lower():
        score += 2.0
        reasons.append("genre match (+2.0)")

    if song["mood"].lower() == user_prefs.get("mood", "").lower():
        score += 1.0
        reasons.append("mood match (+1.0)")

    if "energy" in user_prefs:
        e = 1.0 - abs(song["energy"] - float(user_prefs["energy"]))
        score += e
        reasons.append(f"energy similarity ({e:+.2f})")

    if "tempo_bpm" in user_prefs:
        t = max(0.0, 1.0 - abs(song["tempo_bpm"] - float(user_prefs["tempo_bpm"])) / 80)
        score += 0.5 * t
        reasons.append(f"tempo similarity ({0.5 * t:+.2f})")

    if "valence" in user_prefs:
        v = 1.0 - abs(song["valence"] - float(user_prefs["valence"]))
        score += 0.5 * v
        reasons.append(f"valence similarity ({0.5 * v:+.2f})")

    if "danceability" in user_prefs:
        d = 1.0 - abs(song["danceability"] - float(user_prefs["danceability"]))
        score += 0.5 * d
        reasons.append(f"danceability similarity ({0.5 * d:+.2f})")

    if "acousticness" in user_prefs:
        a = 1.0 - abs(song["acousticness"] - float(user_prefs["acousticness"]))
        score += 0.5 * a
        reasons.append(f"acousticness similarity ({0.5 * a:+.2f})")

    return score, reasons


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """Score every song, sort descending, return top k as (song, score, explanation)."""
    scored = [
        (song, *score_song(user_prefs, song))
        for song in songs
    ]
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    logger.debug("recommend_songs(): ranked %d songs, returning top %d", len(songs), k)
    return [(song, score, " | ".join(reasons)) for song, score, reasons in ranked[:k]]
