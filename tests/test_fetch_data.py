"""Tests for scripts/fetch_data.py – data sourcing module."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Allow importing from the scripts package without installing the project.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.fetch_data import (
    BASE_URL,
    fetch_current_season,
    fetch_games,
    fetch_historical,
    save_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_GAMES = [
    {
        "id": 1,
        "round": 1,
        "year": 2023,
        "venue": "MCG",
        "hteam": "Collingwood",
        "ateam": "Carlton",
        "hscore": 110,
        "ascore": 90,
        "date": "2023-03-16 19:30:00",
        "complete": 100,
    },
    {
        "id": 2,
        "round": 1,
        "year": 2023,
        "venue": "Docklands",
        "hteam": "Essendon",
        "ateam": "Hawthorn",
        "hscore": 85,
        "ascore": 92,
        "date": "2023-03-17 19:30:00",
        "complete": 100,
    },
]


def _make_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Return a mock requests.Response with the given JSON payload."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = Exception(
            f"HTTP Error {status_code}"
        )
    return mock_resp


# ---------------------------------------------------------------------------
# fetch_games
# ---------------------------------------------------------------------------


class TestFetchGames:
    def test_returns_dataframe_with_correct_rows(self):
        """fetch_games should return a DataFrame with one row per game."""
        with patch("scripts.fetch_data.requests.get") as mock_get:
            mock_get.return_value = _make_response({"games": SAMPLE_GAMES})
            df = fetch_games(2023)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_adds_season_column(self):
        """fetch_games should add a 'season' column equal to the requested year."""
        with patch("scripts.fetch_data.requests.get") as mock_get:
            mock_get.return_value = _make_response({"games": SAMPLE_GAMES})
            df = fetch_games(2023)

        assert "season" in df.columns
        assert (df["season"] == 2023).all()

    def test_calls_correct_url_and_params(self):
        """fetch_games should call the Squiggle API with the right parameters."""
        with patch("scripts.fetch_data.requests.get") as mock_get:
            mock_get.return_value = _make_response({"games": SAMPLE_GAMES})
            fetch_games(2022)

        mock_get.assert_called_once_with(
            BASE_URL, params={"q": "games", "year": 2022}, timeout=30
        )

    def test_empty_response_returns_empty_dataframe(self):
        """fetch_games should return an empty DataFrame when the API returns no games."""
        with patch("scripts.fetch_data.requests.get") as mock_get:
            mock_get.return_value = _make_response({"games": []})
            df = fetch_games(2023)

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_raises_on_http_error(self):
        """fetch_games should propagate HTTP errors from the API."""
        with patch("scripts.fetch_data.requests.get") as mock_get:
            mock_get.return_value = _make_response({}, status_code=500)
            with pytest.raises(Exception):
                fetch_games(2023)


# ---------------------------------------------------------------------------
# fetch_historical
# ---------------------------------------------------------------------------


class TestFetchHistorical:
    def test_combines_multiple_seasons(self):
        """fetch_historical should concatenate data from each requested season."""
        with patch("scripts.fetch_data.fetch_games") as mock_fetch:
            mock_fetch.side_effect = lambda year: pd.DataFrame(
                [{"id": year, "season": year}]
            )
            df = fetch_historical(start_year=2021, end_year=2023)

        assert len(df) == 3
        assert set(df["season"]) == {2021, 2022, 2023}

    def test_calls_fetch_games_for_each_year(self):
        """fetch_historical should invoke fetch_games once per season."""
        with patch("scripts.fetch_data.fetch_games") as mock_fetch:
            mock_fetch.side_effect = lambda year: pd.DataFrame(
                [{"id": year, "season": year}]
            )
            fetch_historical(start_year=2020, end_year=2022)

        assert mock_fetch.call_count == 3
        mock_fetch.assert_any_call(2020)
        mock_fetch.assert_any_call(2021)
        mock_fetch.assert_any_call(2022)

    def test_returns_empty_dataframe_when_all_seasons_empty(self):
        """fetch_historical should return an empty DataFrame when all seasons have no data."""
        with patch("scripts.fetch_data.fetch_games") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()
            df = fetch_historical(start_year=2021, end_year=2021)

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_ignores_empty_season_frames(self):
        """fetch_historical should skip empty season DataFrames when concatenating."""
        with patch("scripts.fetch_data.fetch_games") as mock_fetch:
            mock_fetch.side_effect = [
                pd.DataFrame([{"id": 1, "season": 2021}]),
                pd.DataFrame(),  # 2022 has no data
                pd.DataFrame([{"id": 2, "season": 2023}]),
            ]
            df = fetch_historical(start_year=2021, end_year=2023)

        assert len(df) == 2


# ---------------------------------------------------------------------------
# fetch_current_season
# ---------------------------------------------------------------------------


class TestFetchCurrentSeason:
    def test_fetches_current_year(self):
        """fetch_current_season should call fetch_games with the current calendar year."""
        with patch("scripts.fetch_data.fetch_games") as mock_fetch:
            with patch("scripts.fetch_data.datetime") as mock_dt:
                mock_dt.date.today.return_value = MagicMock(year=2025)
                mock_fetch.return_value = pd.DataFrame(SAMPLE_GAMES)
                df = fetch_current_season()

        mock_fetch.assert_called_once_with(2025)
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# save_data
# ---------------------------------------------------------------------------


class TestSaveData:
    def test_saves_csv_to_given_path(self):
        """save_data should write the DataFrame to the specified CSV file."""
        df = pd.DataFrame(SAMPLE_GAMES)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "games.csv")
            save_data(df, path)
            assert os.path.exists(path)
            loaded = pd.read_csv(path)
            assert len(loaded) == len(df)

    def test_creates_parent_directories(self):
        """save_data should create missing parent directories before saving."""
        df = pd.DataFrame(SAMPLE_GAMES)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "games.csv")
            save_data(df, path)
            assert os.path.exists(path)

    def test_csv_contains_expected_columns(self):
        """save_data should preserve all DataFrame columns in the CSV."""
        df = pd.DataFrame(SAMPLE_GAMES)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "games.csv")
            save_data(df, path)
            loaded = pd.read_csv(path)
            assert set(df.columns) == set(loaded.columns)
