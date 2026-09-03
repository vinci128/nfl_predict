"""
`nfl-predict keepers` writes the keeper exclusion list from ESPN.

ESPN locks keepers an hour before the draft; from then each team's roster is
its keepers. The hour before a draft is the wrong time to type sixty names, and
the wrong time to discover you typed one wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from nfl_predict.cli import app

runner = CliRunner()


def _rosters(per_team: int, teams: int = 10) -> list[dict]:
    return [
        {
            "team_id": str(t),
            "team_name": f"Team {t}",
            "player_name": f"Player {t}-{i}",
            "player_id": f"00-{t}{i}",
            "position": "RB",
            "is_mine": t == 1,
        }
        for t in range(1, teams + 1)
        for i in range(per_team)
    ]


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    return tmp_path


def _patch(monkeypatch: pytest.MonkeyPatch, rosters: list[dict]) -> None:
    from nfl_predict import espn_fantasy

    class _Fake:
        @staticmethod
        def from_env(_league=None):
            class _C:
                @staticmethod
                def fetch_rosters():
                    return rosters

            return _C()

    monkeypatch.setattr(espn_fantasy, "EspnFantasyClient", _Fake)


class TestBeforeTheLock:
    def test_full_rosters_are_refused(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Writing 220 names as 'keepers' would empty the board."""
        _patch(monkeypatch, _rosters(per_team=22))
        result = runner.invoke(app, ["keepers", "--league", "ludopathy"])

        assert result.exit_code == 1
        assert "not locked yet" in result.output
        assert not (workdir / "data/keepers_ludopathy_2026.txt").exists()

    def test_it_says_which_teams_are_wrong(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rosters = _rosters(per_team=6)
        rosters.append({**rosters[0], "player_name": "One Extra"})
        _patch(monkeypatch, rosters)

        result = runner.invoke(app, ["keepers", "--league", "ludopathy"])

        assert result.exit_code == 1
        assert "expected 6" in result.output


class TestAfterTheLock:
    def test_it_writes_every_keeper(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch(monkeypatch, _rosters(per_team=6))
        result = runner.invoke(app, ["keepers", "--league", "ludopathy"])

        assert result.exit_code == 0
        body = (workdir / "data/keepers_ludopathy_2026.txt").read_text()
        names = [ln for ln in body.splitlines() if ln and not ln.startswith("#")]
        assert len(names) == 60

    def test_the_file_is_readable_by_the_board_builder(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_exclusions skips comments and blanks; this must match."""
        from nfl_predict.draft_board import load_exclusions

        _patch(monkeypatch, _rosters(per_team=6))
        runner.invoke(app, ["keepers", "--league", "ludopathy"])

        assert len(load_exclusions(workdir / "data/keepers_ludopathy_2026.txt")) == 60

    def test_dry_run_writes_nothing(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch(monkeypatch, _rosters(per_team=6))
        runner.invoke(app, ["keepers", "--league", "ludopathy", "--no-write"])
        assert not (workdir / "data/keepers_ludopathy_2026.txt").exists()


class TestLeaguesWithoutKeepers:
    @pytest.mark.parametrize("league", ["hoh", "rumble"])
    def test_it_declines_politely(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch, league: str
    ) -> None:
        result = runner.invoke(app, ["keepers", "--league", league])
        assert result.exit_code == 0
        assert "no keepers" in result.output.lower()
