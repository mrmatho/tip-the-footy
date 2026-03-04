"""
build_features.py – Feature engineering pipeline for the AFL Tipping Model.

Takes raw historical game data (as returned by fetch_data) and produces a
feature matrix suitable for training the winner classifier and margin
regressor.
"""

import os

import numpy as np
import pandas as pd

# ── Feature / target column names ────────────────────────────────────────────

FEATURE_COLS = [
    "home_win_rate_last_5",
    "away_win_rate_last_5",
    "home_avg_score_last_5",
    "away_avg_score_last_5",
    "home_avg_margin_last_5",
    "away_avg_margin_last_5",
    "head_to_head_win_rate",
    "venue_home_win_rate",
    "days_since_last_game_home",
    "days_since_last_game_away",
    "home_ladder_position",
    "away_ladder_position",
]

TARGET_CLF = "home_win"  # 1 if home team wins, 0 otherwise
TARGET_REG = "margin"    # home_score − away_score (negative = away win)

_WINDOW = 5  # rolling-window size for recent-form features


# ── Private helpers ───────────────────────────────────────────────────────────


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with an extra ``date_dt`` column parsed from ``date``."""
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _build_team_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a team-centric view where each game appears twice.

    One row per (team, game) combination so that per-team rolling stats can
    be computed without distinguishing home vs. away.
    """
    shared = ["id", "date_dt"]

    home = (
        df[shared + ["hteam", "ateam", "hscore", "ascore"]]
        .rename(columns={"hteam": "team", "ateam": "opponent",
                         "hscore": "score", "ascore": "opp_score"})
        .assign(is_home=True)
    )

    away = (
        df[shared + ["hteam", "ateam", "hscore", "ascore"]]
        .rename(columns={"ateam": "team", "hteam": "opponent",
                         "ascore": "score", "hscore": "opp_score"})
        .assign(is_home=False)
    )

    view = pd.concat([home, away], ignore_index=True)
    view["margin"] = view["score"] - view["opp_score"]
    view["won"] = (view["margin"] > 0).astype(int)
    return view.sort_values(["team", "date_dt"]).reset_index(drop=True)


def _rolling_stats(
    team_view: pd.DataFrame,
    team: str,
    before_dt,
    window: int = _WINDOW,
) -> dict:
    """Return win_rate, avg_score, avg_margin for a team's last *window* games."""
    past = team_view[
        (team_view["team"] == team) & (team_view["date_dt"] < before_dt)
    ]
    recent = past.tail(window)
    if recent.empty:
        return {"win_rate": np.nan, "avg_score": np.nan, "avg_margin": np.nan}
    return {
        "win_rate": float(recent["won"].mean()),
        "avg_score": float(recent["score"].mean()),
        "avg_margin": float(recent["margin"].mean()),
    }


def _h2h_win_rate(
    team_view: pd.DataFrame,
    home_team: str,
    away_team: str,
    before_dt,
) -> float:
    """Return the home team's historical win rate against the away team."""
    mask = (
        (team_view["team"] == home_team)
        & (team_view["opponent"] == away_team)
        & (team_view["date_dt"] < before_dt)
    )
    past = team_view[mask]
    if past.empty:
        return np.nan
    return float(past["won"].mean())


def _venue_win_rate(
    df: pd.DataFrame,
    home_team: str,
    venue: str,
    before_dt,
) -> float:
    """Return the home team's win rate at this venue."""
    mask = (
        (df["hteam"] == home_team)
        & (df["venue"] == venue)
        & (df["date_dt"] < before_dt)
    )
    past = df[mask]
    if past.empty:
        return np.nan
    won = int((past["hscore"] > past["ascore"]).sum())
    return float(won / len(past))


def _days_rest(team_view: pd.DataFrame, team: str, before_dt) -> float:
    """Return the number of days since the team's last game before *before_dt*."""
    past = team_view[
        (team_view["team"] == team) & (team_view["date_dt"] < before_dt)
    ]
    if past.empty:
        return np.nan
    last_dt = past["date_dt"].max()
    return float((before_dt - last_dt).days)


def _ladder_positions(
    df: pd.DataFrame,
    season: int,
    before_dt,
) -> dict:
    """Return a ``{team: ladder_position}`` dict for *season* as of *before_dt*.

    Position 1 = most wins. Teams not yet on the ladder are omitted.
    """
    played = df[
        (df["season"] == season)
        & (df["date_dt"] < before_dt)
        & df["hscore"].notna()
        & df["ascore"].notna()
    ]
    if played.empty:
        return {}
    home_wins = played[played["hscore"] > played["ascore"]].groupby("hteam").size()
    away_wins = played[played["ascore"] > played["hscore"]].groupby("ateam").size()
    wins = home_wins.add(away_wins, fill_value=0).sort_values(ascending=False)
    return {team: rank for rank, team in enumerate(wins.index, start=1)}


# ── Public API ────────────────────────────────────────────────────────────────


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix from raw historical game data.

    Args:
        df: Raw DataFrame from ``fetch_data``.  Must contain columns:
            ``id``, ``round``, ``season``, ``venue``, ``hteam``, ``ateam``,
            ``hscore``, ``ascore``, ``date``.

    Returns:
        DataFrame with :data:`FEATURE_COLS` plus ``match_id``, ``season``,
        ``round``, ``home_team``, ``away_team``, :data:`TARGET_CLF`, and
        :data:`TARGET_REG` columns.
    """
    df = _parse_dates(df)
    df = df.dropna(subset=["date_dt"]).reset_index(drop=True)

    # Only completed games have valid scores (and therefore valid targets).
    if "complete" in df.columns:
        df = df[df["complete"] == 100].reset_index(drop=True)

    team_view = _build_team_view(df)

    all_teams = set(df["hteam"].tolist() + df["ateam"].tolist())
    mid_position = len(all_teams) // 2 + 1

    records = []
    for _, row in df.iterrows():
        dt = row["date_dt"]
        home = str(row["hteam"])
        away = str(row["ateam"])

        h_stats = _rolling_stats(team_view, home, dt)
        a_stats = _rolling_stats(team_view, away, dt)
        ladder = _ladder_positions(df, row["season"], dt)

        records.append({
            "match_id": row["id"],
            "season": row["season"],
            "round": row["round"],
            "home_team": home,
            "away_team": away,
            "home_win_rate_last_5": h_stats["win_rate"],
            "away_win_rate_last_5": a_stats["win_rate"],
            "home_avg_score_last_5": h_stats["avg_score"],
            "away_avg_score_last_5": a_stats["avg_score"],
            "home_avg_margin_last_5": h_stats["avg_margin"],
            "away_avg_margin_last_5": a_stats["avg_margin"],
            "head_to_head_win_rate": _h2h_win_rate(team_view, home, away, dt),
            "venue_home_win_rate": _venue_win_rate(df, home, str(row["venue"]), dt),
            "days_since_last_game_home": _days_rest(team_view, home, dt),
            "days_since_last_game_away": _days_rest(team_view, away, dt),
            "home_ladder_position": float(ladder.get(home, mid_position)),
            "away_ladder_position": float(ladder.get(away, mid_position)),
            TARGET_REG: float(row["hscore"]) - float(row["ascore"]),
            TARGET_CLF: int(row["hscore"] > row["ascore"]),
        })

    return pd.DataFrame(records)


def build_game_features(game: dict, historical_df: pd.DataFrame) -> pd.DataFrame:
    """Build features for a single *upcoming* game using historical data.

    Args:
        game: Dict with at minimum ``hteam``, ``ateam``, ``venue``, ``year``
              keys (Squiggle API format).  An optional ``date`` key is used to
              anchor the lookback window; if absent the latest known date + 7
              days is used.
        historical_df: Completed historical games DataFrame from
                       ``fetch_data.fetch_historical``.

    Returns:
        Single-row DataFrame with :data:`FEATURE_COLS` columns (no target
        columns, since the match has not been played yet).
    """
    df = _parse_dates(historical_df.copy())
    df = df.dropna(subset=["date_dt"]).reset_index(drop=True)
    if "complete" in df.columns:
        df = df[df["complete"] == 100].reset_index(drop=True)

    team_view = _build_team_view(df)

    if "date" in game and game["date"]:
        dt = pd.to_datetime(game["date"], errors="coerce")
    else:
        dt = None
    if dt is None or pd.isna(dt):
        dt = df["date_dt"].max() + pd.Timedelta(days=7)

    home = str(game["hteam"])
    away = str(game["ateam"])
    venue = str(game.get("venue", ""))
    season = int(game.get("year", df["season"].max()))

    all_teams = set(df["hteam"].tolist() + df["ateam"].tolist())
    mid_position = len(all_teams) // 2 + 1

    h_stats = _rolling_stats(team_view, home, dt)
    a_stats = _rolling_stats(team_view, away, dt)
    ladder = _ladder_positions(df, season, dt)

    return pd.DataFrame([{
        "home_win_rate_last_5": h_stats["win_rate"],
        "away_win_rate_last_5": a_stats["win_rate"],
        "home_avg_score_last_5": h_stats["avg_score"],
        "away_avg_score_last_5": a_stats["avg_score"],
        "home_avg_margin_last_5": h_stats["avg_margin"],
        "away_avg_margin_last_5": a_stats["avg_margin"],
        "head_to_head_win_rate": _h2h_win_rate(team_view, home, away, dt),
        "venue_home_win_rate": _venue_win_rate(df, home, venue, dt),
        "days_since_last_game_home": _days_rest(team_view, home, dt),
        "days_since_last_game_away": _days_rest(team_view, away, dt),
        "home_ladder_position": float(ladder.get(home, mid_position)),
        "away_ladder_position": float(ladder.get(away, mid_position)),
    }])


def save_features(df: pd.DataFrame, path: str = "data/features.csv") -> None:
    """Save the feature DataFrame to a CSV file, creating directories as needed.

    Args:
        df: Feature DataFrame to save.
        path: Destination file path.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    raw = pd.read_csv("data/historical_games.csv")
    print(f"Building features for {len(raw)} games…")
    features = build_features(raw)
    save_features(features)
    print(f"Saved {len(features)} feature rows to data/features.csv")
