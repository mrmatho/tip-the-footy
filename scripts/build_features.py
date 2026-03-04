"""
build_features.py – Feature engineering pipeline for the AFL Tipping Model.

Takes raw historical game data (as returned by fetch_data) and produces a
feature matrix suitable for training the winner classifier and margin
regressor.
"""

import os

import numpy as np
import pandas as pd

from scripts.app_config import load_model_config

# ── Feature / target column names ────────────────────────────────────────────

FEATURE_COLS = [
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff_pre",
    "elo_expected_home_win",
    "home_win_rate_last_5",
    "away_win_rate_last_5",
    "home_avg_score_last_5",
    "away_avg_score_last_5",
    "home_avg_margin_last_5",
    "away_avg_margin_last_5",
    "home_win_rate_last_3",
    "away_win_rate_last_3",
    "home_avg_score_last_3",
    "away_avg_score_last_3",
    "home_avg_margin_last_3",
    "away_avg_margin_last_3",
    "win_rate_diff",
    "avg_margin_diff",
    "head_to_head_win_rate",
    "venue_home_win_rate",
    "days_since_last_game_home",
    "days_since_last_game_away",
    "home_ladder_position",
    "away_ladder_position",
]

TARGET_CLF = "home_win"  # 1 if home team wins, 0 otherwise
TARGET_REG = "margin"    # home_score − away_score (negative = away win)

_WINDOW = 5   # rolling-window size for recent-form features
_WINDOW_SHORT = 3  # shorter rolling window to capture very recent form
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 50.0
ELO_SEASON_REVERSION = 0.10

_CFG = load_model_config()
_WINDOW = int(_CFG["features"]["rolling_window"])
_WINDOW_SHORT = int(_CFG["features"]["rolling_window_short"])
ELO_START = float(_CFG["elo"]["start_rating"])
ELO_K = float(_CFG["elo"]["k_factor"])
ELO_HOME_ADV = float(_CFG["elo"]["home_advantage"])
ELO_SEASON_REVERSION = float(_CFG["elo"]["season_reversion"])


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


def _elo_expected_home(home_elo: float, away_elo: float) -> float:
    """Return expected home-team win probability from Elo ratings."""
    return float(1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + ELO_HOME_ADV)) / 400.0)))


def _apply_season_reversion(ratings: dict[str, float]) -> dict[str, float]:
    """Move every rating a fraction back toward the Elo mean at season boundaries."""
    if ELO_SEASON_REVERSION <= 0:
        return ratings
    return {
        team: float(ELO_START + (rating - ELO_START) * (1.0 - ELO_SEASON_REVERSION))
        for team, rating in ratings.items()
    }


def _compute_pre_match_elos(df: pd.DataFrame) -> dict:
    """Return ``{match_id: (home_elo_pre, away_elo_pre, expected_home_win)}``."""
    ratings: dict[str, float] = {}
    by_id: dict = {}
    ordered = df.sort_values(["season", "date_dt", "id"]).reset_index(drop=True)
    active_season = None

    for _, row in ordered.iterrows():
        season = int(row["season"])
        if active_season is None:
            active_season = season
        elif season != active_season:
            ratings = _apply_season_reversion(ratings)
            active_season = season

        home = str(row["hteam"])
        away = str(row["ateam"])

        home_pre = float(ratings.get(home, ELO_START))
        away_pre = float(ratings.get(away, ELO_START))
        expected_home = _elo_expected_home(home_pre, away_pre)

        by_id[row["id"]] = (home_pre, away_pre, expected_home)

        hscore = float(row["hscore"])
        ascore = float(row["ascore"])
        if hscore > ascore:
            actual_home = 1.0
        elif hscore < ascore:
            actual_home = 0.0
        else:
            actual_home = 0.5

        delta = ELO_K * (actual_home - expected_home)
        ratings[home] = home_pre + delta
        ratings[away] = away_pre - delta

    return by_id


def _elo_for_upcoming(df: pd.DataFrame, home_team: str, away_team: str, before_dt) -> tuple:
    """Return ``(home_elo_pre, away_elo_pre, expected_home_win)`` before *before_dt*."""
    played = df[
        (df["date_dt"] < before_dt)
        & df["hscore"].notna()
        & df["ascore"].notna()
    ]
    if played.empty:
        home_pre = float(ELO_START)
        away_pre = float(ELO_START)
        return home_pre, away_pre, _elo_expected_home(home_pre, away_pre)

    ratings: dict[str, float] = {}
    ordered = played.sort_values(["season", "date_dt", "id"]).reset_index(drop=True)
    active_season = None
    for _, row in ordered.iterrows():
        season = int(row["season"])
        if active_season is None:
            active_season = season
        elif season != active_season:
            ratings = _apply_season_reversion(ratings)
            active_season = season

        home = str(row["hteam"])
        away = str(row["ateam"])
        home_pre = float(ratings.get(home, ELO_START))
        away_pre = float(ratings.get(away, ELO_START))
        expected_home = _elo_expected_home(home_pre, away_pre)

        hscore = float(row["hscore"])
        ascore = float(row["ascore"])
        if hscore > ascore:
            actual_home = 1.0
        elif hscore < ascore:
            actual_home = 0.0
        else:
            actual_home = 0.5

        delta = ELO_K * (actual_home - expected_home)
        ratings[home] = home_pre + delta
        ratings[away] = away_pre - delta

    home_pre = float(ratings.get(home_team, ELO_START))
    away_pre = float(ratings.get(away_team, ELO_START))
    return home_pre, away_pre, _elo_expected_home(home_pre, away_pre)


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

    # ── Vectorised rolling form and rest-days (O(n log n)) ───────────────────
    tv = team_view.sort_values(["team", "date_dt"]).reset_index(drop=True)
    _grp = tv.groupby("team", sort=False)
    tv["_win_rate"]   = _grp["won"].transform(
        lambda x: x.shift(1).rolling(_WINDOW, min_periods=1).mean()
    )
    tv["_avg_score"]  = _grp["score"].transform(
        lambda x: x.shift(1).rolling(_WINDOW, min_periods=1).mean()
    )
    tv["_avg_margin"] = _grp["margin"].transform(
        lambda x: x.shift(1).rolling(_WINDOW, min_periods=1).mean()
    )
    tv["_win_rate_3"]   = _grp["won"].transform(
        lambda x: x.shift(1).rolling(_WINDOW_SHORT, min_periods=1).mean()
    )
    tv["_avg_score_3"]  = _grp["score"].transform(
        lambda x: x.shift(1).rolling(_WINDOW_SHORT, min_periods=1).mean()
    )
    tv["_avg_margin_3"] = _grp["margin"].transform(
        lambda x: x.shift(1).rolling(_WINDOW_SHORT, min_periods=1).mean()
    )
    _prev_dt = _grp["date_dt"].transform(lambda x: x.shift(1))
    tv["_days_rest"]  = (tv["date_dt"] - _prev_dt).dt.days

    # Index by game id for O(1) lookups per game.
    home_tv = tv[tv["is_home"]].set_index("id")
    away_tv = tv[~tv["is_home"]].set_index("id")

    # ── Vectorised h2h win rate ──────────────────────────────────────────────
    h2h = team_view.sort_values(["team", "opponent", "date_dt"]).reset_index(drop=True)
    h2h["_h2h"] = (
        h2h.groupby(["team", "opponent"], sort=False)["won"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    h2h_by_id = h2h[h2h["is_home"]].set_index("id")["_h2h"]

    # ── Vectorised venue home win rate ───────────────────────────────────────
    df_v = df.sort_values("date_dt").copy()
    df_v["_hw"] = (df_v["hscore"] > df_v["ascore"]).astype(float)
    df_v["_venue_wr"] = (
        df_v.groupby(["hteam", "venue"], sort=False)["_hw"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    venue_wr_by_id = df_v.set_index("id")["_venue_wr"]

    # ── Ladder positions (cached once per round, not once per game) ──────────
    ladder_cache: dict = {}
    for _, rrow in (
        df[["season", "round", "date_dt"]]
        .drop_duplicates(subset=["season", "round"])
        .iterrows()
    ):
        ladder_cache[(rrow["season"], rrow["round"])] = _ladder_positions(
            df, rrow["season"], rrow["date_dt"]
        )

    elo_by_id = _compute_pre_match_elos(df)

    # ── Assemble feature rows ─────────────────────────────────────────────────
    records = []
    for _, row in df.iterrows():
        gid = row["id"]
        ladder = ladder_cache.get((row["season"], row["round"]), {})
        home_elo_pre, away_elo_pre, elo_expected_home = elo_by_id.get(
            gid, (ELO_START, ELO_START, _elo_expected_home(ELO_START, ELO_START))
        )
        h_wr5 = home_tv["_win_rate"].get(gid, np.nan)
        a_wr5 = away_tv["_win_rate"].get(gid, np.nan)
        h_mg5 = home_tv["_avg_margin"].get(gid, np.nan)
        a_mg5 = away_tv["_avg_margin"].get(gid, np.nan)
        records.append({
            "match_id": gid,
            "season": row["season"],
            "round": row["round"],
            "home_team": str(row["hteam"]),
            "away_team": str(row["ateam"]),
            "home_elo_pre":              home_elo_pre,
            "away_elo_pre":              away_elo_pre,
            "elo_diff_pre":              home_elo_pre - away_elo_pre,
            "elo_expected_home_win":     elo_expected_home,
            "home_win_rate_last_5":      h_wr5,
            "away_win_rate_last_5":      a_wr5,
            "home_avg_score_last_5":     home_tv["_avg_score"].get(gid, np.nan),
            "away_avg_score_last_5":     away_tv["_avg_score"].get(gid, np.nan),
            "home_avg_margin_last_5":    h_mg5,
            "away_avg_margin_last_5":    a_mg5,
            "home_win_rate_last_3":      home_tv["_win_rate_3"].get(gid, np.nan),
            "away_win_rate_last_3":      away_tv["_win_rate_3"].get(gid, np.nan),
            "home_avg_score_last_3":     home_tv["_avg_score_3"].get(gid, np.nan),
            "away_avg_score_last_3":     away_tv["_avg_score_3"].get(gid, np.nan),
            "home_avg_margin_last_3":    home_tv["_avg_margin_3"].get(gid, np.nan),
            "away_avg_margin_last_3":    away_tv["_avg_margin_3"].get(gid, np.nan),
            "win_rate_diff":             h_wr5 - a_wr5,
            "avg_margin_diff":           h_mg5 - a_mg5,
            "head_to_head_win_rate":     h2h_by_id.get(gid, np.nan),
            "venue_home_win_rate":       venue_wr_by_id.get(gid, np.nan),
            "days_since_last_game_home": home_tv["_days_rest"].get(gid, np.nan),
            "days_since_last_game_away": away_tv["_days_rest"].get(gid, np.nan),
            "home_ladder_position":      float(ladder.get(str(row["hteam"]), mid_position)),
            "away_ladder_position":      float(ladder.get(str(row["ateam"]), mid_position)),
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

    Raises:
        ValueError: If ``historical_df`` is empty or missing required columns.
    """
    if historical_df is None or historical_df.empty:
        raise ValueError(
            "build_game_features requires non-empty historical data. "
            "Call fetch_historical (or equivalent) and ensure it returns at "
            "least one completed game before building game-level features."
        )
    required_cols = {"date", "hteam", "ateam", "season"}
    missing_cols = required_cols.difference(historical_df.columns)
    if missing_cols:
        raise ValueError(
            "build_game_features expected historical_df to contain columns "
            f"{sorted(required_cols)}, but these columns are missing: "
            f"{sorted(missing_cols)}"
        )
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
    h_stats_3 = _rolling_stats(team_view, home, dt, window=_WINDOW_SHORT)
    a_stats_3 = _rolling_stats(team_view, away, dt, window=_WINDOW_SHORT)
    ladder = _ladder_positions(df, season, dt)
    home_elo_pre, away_elo_pre, elo_expected_home = _elo_for_upcoming(
        df, home, away, dt
    )

    return pd.DataFrame([{
        "home_elo_pre": home_elo_pre,
        "away_elo_pre": away_elo_pre,
        "elo_diff_pre": home_elo_pre - away_elo_pre,
        "elo_expected_home_win": elo_expected_home,
        "home_win_rate_last_5": h_stats["win_rate"],
        "away_win_rate_last_5": a_stats["win_rate"],
        "home_avg_score_last_5": h_stats["avg_score"],
        "away_avg_score_last_5": a_stats["avg_score"],
        "home_avg_margin_last_5": h_stats["avg_margin"],
        "away_avg_margin_last_5": a_stats["avg_margin"],
        "home_win_rate_last_3": h_stats_3["win_rate"],
        "away_win_rate_last_3": a_stats_3["win_rate"],
        "home_avg_score_last_3": h_stats_3["avg_score"],
        "away_avg_score_last_3": a_stats_3["avg_score"],
        "home_avg_margin_last_3": h_stats_3["avg_margin"],
        "away_avg_margin_last_3": a_stats_3["avg_margin"],
        "win_rate_diff": h_stats["win_rate"] - a_stats["win_rate"],
        "avg_margin_diff": h_stats["avg_margin"] - a_stats["avg_margin"],
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
