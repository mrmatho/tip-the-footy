"""Tests for scripts/build_features.py – feature engineering module."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.build_features import (
    FEATURE_COLS,
    TARGET_CLF,
    TARGET_REG,
    _build_team_view,
    _days_rest,
    _h2h_win_rate,
    _ladder_positions,
    _parse_dates,
    _rolling_stats,
    _venue_win_rate,
    build_features,
    build_game_features,
    save_features,
)

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

SAMPLE_GAMES = pd.DataFrame([
    {
        "id": 1, "round": 1, "season": 2023, "year": 2023,
        "venue": "MCG", "hteam": "Collingwood", "ateam": "Carlton",
        "hscore": 110, "ascore": 90,
        "date": "2023-03-16 19:30:00", "complete": 100,
    },
    {
        "id": 2, "round": 1, "season": 2023, "year": 2023,
        "venue": "Docklands", "hteam": "Essendon", "ateam": "Hawthorn",
        "hscore": 85, "ascore": 92,
        "date": "2023-03-17 19:30:00", "complete": 100,
    },
    {
        "id": 3, "round": 2, "season": 2023, "year": 2023,
        "venue": "MCG", "hteam": "Carlton", "ateam": "Essendon",
        "hscore": 95, "ascore": 80,
        "date": "2023-03-24 19:30:00", "complete": 100,
    },
])


# ---------------------------------------------------------------------------
# _parse_dates
# ---------------------------------------------------------------------------


class TestParseDates:
    def test_adds_date_dt_column(self):
        df = _parse_dates(SAMPLE_GAMES)
        assert "date_dt" in df.columns

    def test_parses_dates_correctly(self):
        df = _parse_dates(SAMPLE_GAMES)
        assert pd.api.types.is_datetime64_any_dtype(df["date_dt"])
        assert not df["date_dt"].isna().any()

    def test_does_not_mutate_input(self):
        original = SAMPLE_GAMES.copy()
        _parse_dates(SAMPLE_GAMES)
        pd.testing.assert_frame_equal(SAMPLE_GAMES, original)


# ---------------------------------------------------------------------------
# _build_team_view
# ---------------------------------------------------------------------------


class TestBuildTeamView:
    def test_doubles_row_count(self):
        df = _parse_dates(SAMPLE_GAMES)
        view = _build_team_view(df)
        assert len(view) == len(df) * 2

    def test_has_required_columns(self):
        df = _parse_dates(SAMPLE_GAMES)
        view = _build_team_view(df)
        for col in ["team", "opponent", "score", "opp_score", "margin", "won"]:
            assert col in view.columns

    def test_won_column_is_binary(self):
        df = _parse_dates(SAMPLE_GAMES)
        view = _build_team_view(df)
        assert set(view["won"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# _rolling_stats
# ---------------------------------------------------------------------------


class TestRollingStats:
    def test_empty_history_returns_nan(self):
        df = _parse_dates(SAMPLE_GAMES)
        view = _build_team_view(df)
        stats = _rolling_stats(view, "Collingwood", pd.Timestamp("2022-01-01"))
        assert np.isnan(stats["win_rate"])
        assert np.isnan(stats["avg_score"])
        assert np.isnan(stats["avg_margin"])

    def test_win_rate_after_loss(self):
        """Carlton lost game 1; their win rate before game 3 should be 0."""
        df = _parse_dates(SAMPLE_GAMES)
        view = _build_team_view(df)
        stats = _rolling_stats(view, "Carlton", pd.Timestamp("2023-03-24"))
        assert stats["win_rate"] == 0.0

    def test_returns_dict_with_three_keys(self):
        df = _parse_dates(SAMPLE_GAMES)
        view = _build_team_view(df)
        stats = _rolling_stats(view, "Carlton", pd.Timestamp("2023-12-31"))
        assert set(stats.keys()) == {"win_rate", "avg_score", "avg_margin"}


# ---------------------------------------------------------------------------
# _h2h_win_rate
# ---------------------------------------------------------------------------


class TestH2hWinRate:
    def test_no_history_returns_nan(self):
        df = _parse_dates(SAMPLE_GAMES)
        view = _build_team_view(df)
        rate = _h2h_win_rate(view, "Collingwood", "Essendon", pd.Timestamp("2023-01-01"))
        assert np.isnan(rate)

    def test_correct_win_rate_after_game(self):
        """Collingwood beat Carlton in game 1; h2h rate should be 1.0."""
        df = _parse_dates(SAMPLE_GAMES)
        view = _build_team_view(df)
        rate = _h2h_win_rate(view, "Collingwood", "Carlton", pd.Timestamp("2023-12-31"))
        assert rate == 1.0


# ---------------------------------------------------------------------------
# _venue_win_rate
# ---------------------------------------------------------------------------


class TestVenueWinRate:
    def test_no_history_returns_nan(self):
        df = _parse_dates(SAMPLE_GAMES)
        rate = _venue_win_rate(df, "Collingwood", "MCG", pd.Timestamp("2022-01-01"))
        assert np.isnan(rate)

    def test_correct_rate_after_win(self):
        """Collingwood won at MCG in game 1; venue rate should be 1.0."""
        df = _parse_dates(SAMPLE_GAMES)
        rate = _venue_win_rate(df, "Collingwood", "MCG", pd.Timestamp("2023-12-31"))
        assert rate == 1.0


# ---------------------------------------------------------------------------
# _days_rest
# ---------------------------------------------------------------------------


class TestDaysRest:
    def test_no_history_returns_nan(self):
        df = _parse_dates(SAMPLE_GAMES)
        view = _build_team_view(df)
        days = _days_rest(view, "Collingwood", pd.Timestamp("2022-01-01"))
        assert np.isnan(days)

    def test_correct_days_between_games(self):
        """Carlton played 2023-03-16; game 3 is 2023-03-24 → 8 days rest."""
        df = _parse_dates(SAMPLE_GAMES)
        view = _build_team_view(df)
        days = _days_rest(view, "Carlton", pd.Timestamp("2023-03-24 19:30:00"))
        assert days == 8.0


# ---------------------------------------------------------------------------
# _ladder_positions
# ---------------------------------------------------------------------------


class TestLadderPositions:
    def test_empty_before_season_starts(self):
        df = _parse_dates(SAMPLE_GAMES)
        ladder = _ladder_positions(df, 2023, pd.Timestamp("2022-01-01"))
        assert ladder == {}

    def test_winners_appear_in_ladder(self):
        """After round 1 Collingwood and Hawthorn both have 1 win."""
        df = _parse_dates(SAMPLE_GAMES)
        ladder = _ladder_positions(df, 2023, pd.Timestamp("2023-03-24"))
        assert "Collingwood" in ladder
        assert "Hawthorn" in ladder

    def test_positions_are_positive_integers(self):
        df = _parse_dates(SAMPLE_GAMES)
        ladder = _ladder_positions(df, 2023, pd.Timestamp("2023-12-31"))
        for pos in ladder.values():
            assert pos >= 1


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    def test_returns_dataframe(self):
        df = build_features(SAMPLE_GAMES)
        assert isinstance(df, pd.DataFrame)

    def test_has_all_feature_columns(self):
        df = build_features(SAMPLE_GAMES)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_has_target_columns(self):
        df = build_features(SAMPLE_GAMES)
        assert TARGET_CLF in df.columns
        assert TARGET_REG in df.columns

    def test_row_count_matches_completed_games(self):
        df = build_features(SAMPLE_GAMES)
        assert len(df) == len(SAMPLE_GAMES)

    def test_home_win_target_correct(self):
        """Game 1: Collingwood 110 > Carlton 90 → home_win = 1."""
        df = build_features(SAMPLE_GAMES)
        first = df[df["match_id"] == 1].iloc[0]
        assert first[TARGET_CLF] == 1

    def test_away_win_target_correct(self):
        """Game 2: Essendon 85 < Hawthorn 92 → home_win = 0."""
        df = build_features(SAMPLE_GAMES)
        second = df[df["match_id"] == 2].iloc[0]
        assert second[TARGET_CLF] == 0

    def test_margin_target_correct(self):
        """Game 1: 110 − 90 = 20."""
        df = build_features(SAMPLE_GAMES)
        first = df[df["match_id"] == 1].iloc[0]
        assert first[TARGET_REG] == pytest.approx(20.0)

    def test_incomplete_games_excluded(self):
        """Games with complete != 100 should be dropped."""
        df_with_incomplete = SAMPLE_GAMES.copy()
        df_with_incomplete.loc[0, "complete"] = 0
        result = build_features(df_with_incomplete)
        assert len(result) == len(SAMPLE_GAMES) - 1

    def test_first_games_have_nan_rolling_features(self):
        """First game for each team has no prior history → NaN rolling stats."""
        df = build_features(SAMPLE_GAMES)
        first = df[df["match_id"] == 1].iloc[0]
        assert np.isnan(first["home_win_rate_last_5"])
        assert np.isnan(first["away_win_rate_last_5"])


# ---------------------------------------------------------------------------
# build_game_features
# ---------------------------------------------------------------------------


class TestBuildGameFeatures:
    def test_returns_single_row_dataframe(self):
        game = {
            "hteam": "Collingwood", "ateam": "Carlton",
            "venue": "MCG", "date": "2023-03-31 19:30:00", "year": 2023,
        }
        df = build_game_features(game, SAMPLE_GAMES)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_has_all_feature_columns(self):
        game = {
            "hteam": "Collingwood", "ateam": "Carlton",
            "venue": "MCG", "date": "2023-03-31 19:30:00", "year": 2023,
        }
        df = build_game_features(game, SAMPLE_GAMES)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_no_target_columns(self):
        """Upcoming game features should not include target columns."""
        game = {
            "hteam": "Collingwood", "ateam": "Carlton",
            "venue": "MCG", "date": "2023-03-31 19:30:00", "year": 2023,
        }
        df = build_game_features(game, SAMPLE_GAMES)
        assert TARGET_CLF not in df.columns
        assert TARGET_REG not in df.columns

    def test_works_without_date(self):
        """If the game dict has no date, the function should still return a row."""
        game = {
            "hteam": "Collingwood", "ateam": "Carlton",
            "venue": "MCG", "year": 2023,
        }
        df = build_game_features(game, SAMPLE_GAMES)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# save_features
# ---------------------------------------------------------------------------


class TestSaveFeatures:
    def test_saves_csv_file(self):
        features = build_features(SAMPLE_GAMES)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "features.csv")
            save_features(features, path)
            assert os.path.exists(path)

    def test_creates_parent_directories(self):
        features = build_features(SAMPLE_GAMES)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data", "features.csv")
            save_features(features, path)
            assert os.path.exists(path)

    def test_saved_csv_has_correct_columns(self):
        features = build_features(SAMPLE_GAMES)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "features.csv")
            save_features(features, path)
            loaded = pd.read_csv(path)
            assert set(features.columns) == set(loaded.columns)
