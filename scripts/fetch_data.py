"""
fetch_data.py – Data sourcing module for the AFL Tipping Model.

Fetches historical and current-season match data from the Squiggle API
(https://api.squiggle.com.au/) and saves it locally for model training.
"""

import datetime
import os

import pandas as pd
import requests

BASE_URL = "https://api.squiggle.com.au/"


def fetch_games(season: int) -> pd.DataFrame:
    """Fetch all completed games for a given AFL season from the Squiggle API.

    Args:
        season: The AFL season year (e.g. 2023).

    Returns:
        A DataFrame with one row per game and columns provided by the API.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status code.
    """
    params = {"q": "games", "year": season}
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    games = response.json().get("games", [])
    df = pd.DataFrame(games)
    if not df.empty:
        df["season"] = season
    return df


def fetch_historical(start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
    """Fetch and combine game data for a range of AFL seasons.

    Args:
        start_year: First season to include (inclusive).
        end_year: Last season to include (inclusive).

    Returns:
        A single DataFrame containing all games across the requested seasons.
    """
    frames = [fetch_games(year) for year in range(start_year, end_year + 1)]
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
    return fetch_games(current_year)


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
    print("Fetching historical AFL data (2010–2024)…")
    historical = fetch_historical()
    save_data(historical, "data/historical_games.csv")
    print(f"Saved {len(historical)} games to data/historical_games.csv")
