"""
Phase 3 integration tests: ADP fetch + LLM endpoint.

Tests cover:
- normalise_name (suffix stripping, case, whitespace)
- generate_synthetic_adp (shape, columns, determinism, clamping)
- fetch_adp routing (synthetic direct, fallback on failure)
- save_adp_csv (writes correct file, returns Path)
- Draft API: no-key LLM endpoint returns expected message
- Draft API: LLM endpoint with mocked anthropic client
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_board_csv(tmp_path: Path, n: int = 20) -> Path:
    """Write a minimal draft board CSV to tmp_path and return its path."""
    positions = ["QB", "RB", "WR", "TE", "K"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "player_name": f"Player {i}",
                "position": positions[i % len(positions)],
                "team": "NE",
                "overall_rank": i + 1,
                "pos_rank": (i // len(positions)) + 1,
                "proj_p50": 200.0 - i * 5,
                "proj_p10": 150.0 - i * 5,
                "proj_p90": 250.0 - i * 5,
                "vor": 50.0 - i * 2,
                "tier": 1 + i // 5,
                "pos_tier": 1,
            }
        )
    p = tmp_path / "draft_board_2026.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# normalise_name
# ---------------------------------------------------------------------------


class TestNormaliseName:
    def _norm(self, name: str) -> str:
        from nfl_predict.adp_fetch import normalise_name

        return normalise_name(name)

    def test_lowercase(self):
        assert self._norm("Patrick Mahomes") == "patrick mahomes"

    def test_strip_jr(self):
        assert self._norm("Calvin Ridley Jr.") == "calvin ridley"

    def test_strip_jr_no_dot(self):
        assert self._norm("Odell Beckham Jr") == "odell beckham"

    def test_strip_sr(self):
        assert self._norm("Lorenzo Neal Sr") == "lorenzo neal"

    def test_strip_ii(self):
        assert self._norm("Michael Pittman II") == "michael pittman"

    def test_strip_iii(self):
        assert self._norm("Will Fuller III") == "will fuller"

    def test_strip_iv(self):
        assert self._norm("DeShawn Watson IV") == "deshawn watson"

    def test_collapse_whitespace(self):
        assert self._norm("  Josh   Allen  ") == "josh allen"

    def test_already_normalised(self):
        assert self._norm("josh allen") == "josh allen"


# ---------------------------------------------------------------------------
# generate_synthetic_adp
# ---------------------------------------------------------------------------


class TestGenerateSyntheticAdp:
    def test_columns(self, tmp_path: Path):
        from nfl_predict.adp_fetch import generate_synthetic_adp

        board = _make_board_csv(tmp_path)
        df = generate_synthetic_adp(board)
        assert list(df.columns) == ["player_name", "position", "team", "adp"]

    def test_row_count(self, tmp_path: Path):
        from nfl_predict.adp_fetch import generate_synthetic_adp

        board = _make_board_csv(tmp_path, n=30)
        df = generate_synthetic_adp(board)
        assert len(df) == 30

    def test_adp_all_positive(self, tmp_path: Path):
        from nfl_predict.adp_fetch import generate_synthetic_adp

        board = _make_board_csv(tmp_path)
        df = generate_synthetic_adp(board)
        assert (df["adp"] >= 1.0).all()

    def test_sorted_by_adp(self, tmp_path: Path):
        from nfl_predict.adp_fetch import generate_synthetic_adp

        board = _make_board_csv(tmp_path)
        df = generate_synthetic_adp(board)
        assert df["adp"].is_monotonic_increasing

    def test_deterministic_with_seed(self, tmp_path: Path):
        """Same board → same output (seed=42 fixed in implementation)."""
        from nfl_predict.adp_fetch import generate_synthetic_adp

        board = _make_board_csv(tmp_path)
        df1 = generate_synthetic_adp(board)
        df2 = generate_synthetic_adp(board)
        pd.testing.assert_frame_equal(df1, df2)

    def test_no_board_raises(self, tmp_path: Path, monkeypatch):
        from nfl_predict.adp_fetch import generate_synthetic_adp

        # Point glob to an empty directory
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            generate_synthetic_adp()


# ---------------------------------------------------------------------------
# fetch_adp routing
# ---------------------------------------------------------------------------


class TestFetchAdp:
    def test_synthetic_source_direct(self, tmp_path: Path):
        """source='synthetic' skips live fetch entirely."""
        from nfl_predict.adp_fetch import fetch_adp

        _make_board_csv(tmp_path)
        # Patch generate_synthetic_adp so the test is self-contained
        with patch("nfl_predict.adp_fetch.generate_synthetic_adp") as mock_gen:
            mock_gen.return_value = pd.DataFrame(
                {"player_name": ["A"], "position": ["QB"], "team": ["NE"], "adp": [1.0]}
            )
            result = fetch_adp(source="synthetic")
            mock_gen.assert_called_once()
            assert len(result) == 1

    def test_fallback_to_synthetic_on_sleeper_failure(self, tmp_path: Path):
        """When Sleeper raises, synthetic fallback is used."""
        from nfl_predict.adp_fetch import fetch_adp

        synthetic_df = pd.DataFrame(
            {"player_name": ["B"], "position": ["WR"], "team": ["KC"], "adp": [5.0]}
        )

        with (
            patch(
                "nfl_predict.adp_fetch.fetch_from_sleeper",
                side_effect=ConnectionError("network down"),
            ),
            patch(
                "nfl_predict.adp_fetch.generate_synthetic_adp",
                return_value=synthetic_df,
            ),
        ):
            result = fetch_adp(source="sleeper", fallback_to_synthetic=True)
            assert len(result) == 1
            assert result.iloc[0]["player_name"] == "B"

    def test_no_fallback_returns_empty_on_failure(self):
        """fallback_to_synthetic=False → empty DataFrame on live failure."""
        from nfl_predict.adp_fetch import fetch_adp

        with patch(
            "nfl_predict.adp_fetch.fetch_from_sleeper",
            side_effect=ConnectionError("network down"),
        ):
            result = fetch_adp(source="sleeper", fallback_to_synthetic=False)
            assert result.empty

    def test_unknown_source_returns_empty(self):
        """Unknown source is caught and returns empty (ValueError swallowed by try/except)."""
        from nfl_predict.adp_fetch import fetch_adp

        result = fetch_adp(source="unknown_xyz", fallback_to_synthetic=False)
        assert result.empty

    def test_successful_sleeper_fetch(self):
        """When Sleeper returns data, it is passed through."""
        from nfl_predict.adp_fetch import fetch_adp

        sleeper_df = pd.DataFrame(
            {
                "player_name": ["Drake Maye", "Bijan Robinson"],
                "position": ["QB", "RB"],
                "team": ["NE", "ATL"],
                "adp": [1.0, 2.0],
            }
        )

        with patch("nfl_predict.adp_fetch.fetch_from_sleeper", return_value=sleeper_df):
            result = fetch_adp(source="sleeper", fallback_to_synthetic=False)
            assert len(result) == 2
            assert "Drake Maye" in result["player_name"].values


# ---------------------------------------------------------------------------
# save_adp_csv
# ---------------------------------------------------------------------------


class TestSaveAdpCsv:
    def test_creates_file(self, tmp_path: Path):
        from nfl_predict.adp_fetch import save_adp_csv

        df = pd.DataFrame(
            {"player_name": ["A"], "position": ["QB"], "team": ["NE"], "adp": [1.0]}
        )
        out = tmp_path / "adp.csv"
        result = save_adp_csv(df, path=out)
        assert result == out
        assert out.exists()

    def test_round_trip(self, tmp_path: Path):
        from nfl_predict.adp_fetch import save_adp_csv

        df = pd.DataFrame(
            {
                "player_name": ["A", "B"],
                "position": ["QB", "RB"],
                "team": ["NE", "KC"],
                "adp": [1.0, 2.5],
            }
        )
        out = tmp_path / "adp.csv"
        save_adp_csv(df, path=out)
        loaded = pd.read_csv(out)
        assert list(loaded["player_name"]) == ["A", "B"]
        assert loaded["adp"].tolist() == [1.0, 2.5]

    def test_creates_parent_dir(self, tmp_path: Path):
        from nfl_predict.adp_fetch import save_adp_csv

        df = pd.DataFrame(
            {"player_name": ["A"], "position": ["QB"], "team": ["NE"], "adp": [1.0]}
        )
        nested = tmp_path / "sub" / "dir" / "adp.csv"
        save_adp_csv(df, path=nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Draft API — LLM endpoint
# ---------------------------------------------------------------------------


def _build_state_json(tmp_path: Path) -> Path:
    """Write a minimal draft state JSON for API tests (matches load_state schema)."""
    import json

    board = _make_board_csv(tmp_path, n=20)
    board_df = pd.read_csv(board)

    state = {
        "board_csv": board_df.to_csv(index=False),
        "available_csv": board_df.to_csv(index=False),
        "picks": [],
        "my_picks": [],
        "my_roster": {"QB": [], "RB": [], "WR": [], "TE": [], "K": [], "FLEX": []},
        "league_size": 10,
        "draft_position": 5,
        "current_pick": 1,
    }
    p = tmp_path / "draft_state.json"
    p.write_text(json.dumps(state))
    return p


class TestDraftApiLlm:
    """Test the /draft/llm-suggest endpoint via FastAPI TestClient."""

    @pytest.fixture()
    def client_and_state(self, tmp_path: Path):
        """Return a TestClient with STATE_PATH patched to tmp_path."""
        from fastapi.testclient import TestClient

        import nfl_predict.draft_api as draft_api_mod
        from nfl_predict.api import app

        state_path = _build_state_json(tmp_path)
        original = draft_api_mod.STATE_PATH
        draft_api_mod.STATE_PATH = state_path
        try:
            with TestClient(app) as client:
                yield client, state_path
        finally:
            draft_api_mod.STATE_PATH = original

    def test_no_key_returns_message(self, client_and_state):
        """Without ANTHROPIC_API_KEY, endpoint returns the no-key message."""
        client, _ = client_and_state
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            resp = client.post("/draft/llm-suggest")
        assert resp.status_code == 200
        assert "ANTHROPIC_API_KEY" in resp.text

    def test_with_mocked_anthropic(self, client_and_state):
        """With key set, mock get_llm_suggestion to verify the endpoint renders advice."""
        client, _ = client_and_state

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-fake"}),
            patch(
                "nfl_predict.draft_assistant.get_llm_suggestion",
                return_value="Take Bijan Robinson at pick 3.",
            ),
            patch("nfl_predict.draft_api._llm_available", return_value=True),
        ):
            resp = client.post("/draft/llm-suggest")

        assert resp.status_code == 200
        assert "Bijan Robinson" in resp.text


# ---------------------------------------------------------------------------
# CLI: fetch-adp command
# ---------------------------------------------------------------------------


class TestFetchAdpCli:
    def test_synthetic_source_writes_csv(self, tmp_path: Path):
        from typer.testing import CliRunner

        from nfl_predict.cli import app

        _make_board_csv(tmp_path)
        out = tmp_path / "adp_out.csv"

        with patch(
            "nfl_predict.adp_fetch.generate_synthetic_adp",
            return_value=pd.DataFrame(
                {
                    "player_name": ["A", "B"],
                    "position": ["QB", "RB"],
                    "team": ["NE", "KC"],
                    "adp": [1.0, 2.0],
                }
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(
                app,
                ["fetch-adp", "--source", "synthetic", "--out", str(out)],
            )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_no_fallback_exits_1_on_failure(self):
        from typer.testing import CliRunner

        from nfl_predict.cli import app

        with patch(
            "nfl_predict.adp_fetch.fetch_from_sleeper",
            side_effect=ConnectionError("net fail"),
        ):
            runner = CliRunner()
            result = runner.invoke(
                app,
                ["fetch-adp", "--source", "sleeper", "--no-fallback"],
            )
        assert result.exit_code == 1
