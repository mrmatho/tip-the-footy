"""
fetch_data.py – Data sourcing module for the AFL Tipping Model.

Fetches historical and current-season match data from the Squiggle API
(https://api.squiggle.com.au/) and saves it locally for model training.
"""

import datetime
import os

import pandas as pd
import requests

from scripts.app_config import load_model_config

_CFG = load_model_config()
BASE_URL = _CFG["data"]["base_url"]
REQUEST_TIMEOUT = int(_CFG["data"]["request_timeout"])
USER_AGENT = _CFG["data"]["user_agent"]
DEFAULT_START_YEAR = int(_CFG["pipeline"]["historical_start_year"])
DEFAULT_END_YEAR = int(_CFG["pipeline"]["historical_end_year"])
HISTORICAL_GAMES_PATH = "data/historical_games.csv"


def _load_cached_historical(path: str = HISTORICAL_GAMES_PATH) -> pd.DataFrame:
    """Load cached historical games if the CSV exists and contains rows."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()

    cached = pd.read_csv(path)
    if cached.empty:
        return pd.DataFrame()
    return cached


def fetch_games(season: int, force: bool = False) -> pd.DataFrame:
    """Fetch all completed games for a given AFL season from the Squiggle API.

    Args:
        season: The AFL season year (e.g. 2023).
        force: When ``True``, always fetch from Squiggle even if
            ``data/historical_games.csv`` already has data.

    Returns:
        A DataFrame with one row per game and columns provided by the API.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status code.
    """
    if not force:
        cached = _load_cached_historical()
        if not cached.empty:
            if "season" in cached.columns:
                season_games = cached[cached["season"] == season].copy()
                if not season_games.empty:
                    return season_games.reset_index(drop=True)
            else:
                return cached.reset_index(drop=True)

    params = {"q": "games", "year": season}
    response = requests.get(
        BASE_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    games = response.json().get("games", [])
    df = pd.DataFrame(games)
    if not df.empty:
        df["season"] = season
    return df


def fetch_historical(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch and combine game data for a range of AFL seasons.

    Args:
        start_year: First season to include (inclusive).
        end_year: Last season to include (inclusive).
        force: When ``True``, always fetch from Squiggle.

    Returns:
        A single DataFrame containing all games across the requested seasons.
    """
    if not force:
        cached = _load_cached_historical()
        if not cached.empty:
            return cached.reset_index(drop=True)

    frames = [fetch_games(year, force=force) for year in range(start_year, end_year + 1)]
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def fetch_current_season() -> pd.DataFrame:
    """Fetch all games for the current AFL season.

    Returns:
        A DataFrame containing games for the current calendar year.
    """
    current_year = datetime.date.today().year
    return fetch_games(current_year, force=True)


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to a CSV file, creating parent directories if needed.

    Args:
        df: The DataFrame to save.
        path: Destination file path (e.g. ``data/historical_games.csv``).
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    print(f"Fetching historical AFL data ({DEFAULT_START_YEAR}–{DEFAULT_END_YEAR})…")
    historical = fetch_historical()
    save_data(historical, "data/historical_games.csv")
    print(f"Saved {len(historical)} games to data/historical_games.csv")
