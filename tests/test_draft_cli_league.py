"""
The draft CLI must act on the league's own session file.

`draft-start --league hoh` writes outputs/draft_state_hoh.json, but
`draft-pick` defaulted to the old un-namespaced outputs/draft_state.json, so
it could not see the session that had just been created. Found mid-practice
draft, where it looked like the pick simply failed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from nfl_predict.cli import app
from nfl_predict.leagues import get_profile

runner = CliRunner()


def _board() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["00-1", "00-2", "00-3"],
            "overall_rank": [1, 2, 3],
            "tier": [1, 1, 2],
            "pos_rank": [1, 1, 1],
            "pos_tier": [1, 1, 1],
            "player_name": ["Bijan Robinson", "Josh Allen", "Justin Jefferson"],
            "position": ["RB", "QB", "WR"],
            "team": ["ATL", "BUF", "MIN"],
            "proj_p10": [200.0, 320.0, 190.0],
            "proj_p50": [280.0, 400.0, 260.0],
            "proj_p90": [350.0, 470.0, 330.0],
            "vor": [90.0, 70.0, 85.0],
            "replacement_baseline": [120.0, 240.0, 130.0],
            "proj_ppg_p50": [17.5, 23.5, 16.3],
            "proj_games_p50": [16.0, 17.0, 16.0],
        }
    )


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    monkeypatch.delenv("NFL_PREDICT_LEAGUE", raising=False)
    return tmp_path


def _start(league: str, workdir: Path):
    board = f"outputs/draft_board_2026_{league}.csv"
    _board().to_csv(workdir / board, index=False)
    return runner.invoke(
        app,
        ["draft-start", "--league", league, "--board-path", board, "--draft-position", "1"],
    )


class TestStatePath:
    def test_start_writes_the_leagues_file(self, workdir: Path) -> None:
        assert _start("hoh", workdir).exit_code == 0
        assert (workdir / "outputs/draft_state_hoh.json").exists()
        assert not (workdir / "outputs/draft_state.json").exists()

    def test_start_matches_the_profile(self, workdir: Path) -> None:
        _start("hoh", workdir)
        assert (workdir / get_profile("hoh").state_path).exists()

    def test_pick_finds_the_session_start_created(self, workdir: Path) -> None:
        """The bug: draft-pick looked in outputs/draft_state.json and failed."""
        _start("hoh", workdir)
        result = runner.invoke(app, ["draft-pick", "Bijan Robinson", "--league", "hoh"])
        assert result.exit_code == 0, result.output
        assert "Bijan Robinson" in result.output

    def test_pick_follows_the_environment_league(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _start("hoh", workdir)
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        assert runner.invoke(app, ["draft-pick", "Josh Allen"]).exit_code == 0

    def test_two_leagues_keep_separate_sessions(self, workdir: Path) -> None:
        """Two drafts on one evening must not write over each other."""
        _start("hoh", workdir)
        _start("rumble", workdir)

        runner.invoke(app, ["draft-pick", "Bijan Robinson", "--league", "hoh"])

        import json

        hoh = json.loads((workdir / "outputs/draft_state_hoh.json").read_text())
        rumble = json.loads((workdir / "outputs/draft_state_rumble.json").read_text())
        assert len(hoh["picks"]) == 1
        assert len(rumble["picks"]) == 0

    def test_an_explicit_state_path_still_wins(self, workdir: Path) -> None:
        _start("hoh", workdir)
        result = runner.invoke(
            app,
            [
                "draft-pick",
                "Bijan Robinson",
                "--state-path",
                "outputs/draft_state_hoh.json",
            ],
        )
        assert result.exit_code == 0


class TestMineFlag:
    def test_mine_records_the_pick_as_yours(self, workdir: Path) -> None:
        """CLAUDE.md documented --mine; only --drafter me existed."""
        _start("hoh", workdir)
        result = runner.invoke(
            app, ["draft-pick", "Bijan Robinson", "--mine", "--league", "hoh"]
        )
        assert result.exit_code == 0, result.output
        assert "YOUR PICK" in result.output

    def test_without_mine_the_pick_is_an_opponents(self, workdir: Path) -> None:
        _start("hoh", workdir)
        result = runner.invoke(app, ["draft-pick", "Bijan Robinson", "--league", "hoh"])
        assert "YOUR PICK" not in result.output

    def test_mine_and_drafter_agree(self, workdir: Path) -> None:
        _start("hoh", workdir)
        a = runner.invoke(app, ["draft-pick", "Bijan Robinson", "--mine", "--league", "hoh"])
        _start("rumble", workdir)
        b = runner.invoke(
            app, ["draft-pick", "Bijan Robinson", "--drafter", "me", "--league", "rumble"]
        )
        assert ("YOUR PICK" in a.output) == ("YOUR PICK" in b.output) is True
