"""
The draft web UI must follow the active league profile.

Two defects this covers, both introduced when the pipeline was namespaced by
league and the CLI was updated but the web layer was not:

  - the UI wrote a hardcoded ``outputs/draft_state.json`` while ``nfl-sync``
    read ``outputs/draft_state_{league}.json``, so ESPN picks pulled in a
    second terminal never reached the browser
  - the setup form defaulted to 12 teams, which neither league is
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_predict import draft_api
from nfl_predict.leagues import get_profile


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run against a throwaway outputs/ so no real session is touched."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    return tmp_path


def _board() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["00-0011111", "00-0022222", "00-0033333"],
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


def _write_board(workdir: Path, league: str) -> str:
    path = f"outputs/draft_board_2026_{league}.csv"
    _board().to_csv(workdir / path, index=False)
    return path


class TestStatePath:
    @pytest.mark.parametrize("league", ["ludopathy", "hoh"])
    def test_state_path_matches_the_cli(self, league: str) -> None:
        """nfl-sync uses profile.state_path; the UI must use the same file."""
        assert draft_api._state_path(league) == get_profile(league).state_path

    def test_state_path_is_namespaced_per_league(self) -> None:
        assert draft_api._state_path("hoh") != draft_api._state_path("ludopathy")

    def test_start_writes_the_board_leagues_state_file(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        board_path = _write_board(workdir, "hoh")

        client = _client()
        resp = client.post(
            "/draft/start",
            data={"board_path": board_path, "draft_position": "1"},
            follow_redirects=False,
        )

        assert resp.status_code == 303
        assert (workdir / "outputs/draft_state_hoh.json").exists()
        assert not (workdir / "outputs/draft_state.json").exists()

    def test_a_session_started_from_another_leagues_board_is_still_found(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The board picker offers every board, not just the active league's."""
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        board_path = _write_board(workdir, "ludopathy")

        client = _client()
        client.post(
            "/draft/start",
            data={"board_path": board_path, "draft_position": "1"},
            follow_redirects=False,
        )

        assert (workdir / "outputs/draft_state_ludopathy.json").exists()
        assert draft_api._state_exists()
        assert client.get("/draft/board").status_code == 200

    def test_no_session_is_not_a_session(self, workdir: Path) -> None:
        assert not draft_api._state_exists()


class TestSetupDefaults:
    @pytest.mark.parametrize(
        ("league", "size"),
        [("ludopathy", 10), ("hoh", 14)],
    )
    def test_league_size_comes_from_the_profile(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch, league: str, size: int
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", league)
        _write_board(workdir, league)

        body = _client().get("/draft").text

        assert f'<option value="{size}" selected>{size} teams</option>' in body
        assert '<option value="12" selected>' not in body

    def test_the_leagues_own_board_is_preselected(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        _write_board(workdir, "ludopathy")
        _write_board(workdir, "hoh")

        body = _client().get("/draft").text

        assert 'value="outputs/draft_board_2026_hoh.csv" selected' in body
        assert 'value="outputs/draft_board_2026_ludopathy.csv" selected' not in body

    def test_draft_position_cannot_exceed_the_league_size(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "ludopathy")
        _write_board(workdir, "ludopathy")

        assert 'max="10"' in _client().get("/draft").text

    def test_omitted_league_size_falls_back_to_the_profile(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A post without the field must not silently become a 12-team draft."""
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        board_path = _write_board(workdir, "hoh")

        client = _client()
        client.post(
            "/draft/start",
            data={"board_path": board_path, "draft_position": "1"},
            follow_redirects=False,
        )

        from nfl_predict.draft_assistant import load_state

        state = load_state(workdir / "outputs/draft_state_hoh.json")
        assert state.league_size == 14

    def test_an_explicit_league_size_is_still_honoured(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        board_path = _write_board(workdir, "hoh")

        client = _client()
        client.post(
            "/draft/start",
            data={"board_path": board_path, "league_size": "8", "draft_position": "1"},
            follow_redirects=False,
        )

        from nfl_predict.draft_assistant import load_state

        state = load_state(workdir / "outputs/draft_state_hoh.json")
        assert state.league_size == 8


def _client():
    from fastapi.testclient import TestClient

    from nfl_predict.api import app

    return TestClient(app)


class TestNoLeakedScripts:
    """
    htmx cannot execute a <script> delivered by an out-of-band swap: it appends
    the element's text to the body instead. pick_response.html carried one, so
    after every pick the JavaScript source rendered as visible text under the
    board. draft_board.html already does the same work on htmx:afterSettle.
    """

    @staticmethod
    def _partials() -> list[Path]:
        root = Path(__file__).parent.parent / "src/nfl_predict/templates/partials"
        return sorted(root.glob("*.html"))

    def test_no_partial_ships_a_script_tag(self) -> None:
        offenders = [p.name for p in self._partials() if "<script" in p.read_text()]
        assert offenders == [], (
            f"htmx swaps these in as text, not code: {offenders}. "
            "Put the behaviour in draft_board.html's htmx:afterSettle handler."
        )

    def test_the_board_still_refocuses_and_rechecks_sync(self) -> None:
        """Deleting the partial's script is only safe because this exists."""
        board = (
            Path(__file__).parent.parent / "src/nfl_predict/templates/draft_board.html"
        ).read_text()
        assert "htmx:afterSettle" in board
        assert "nfl-sync-status" in board
        assert "player-input" in board


class TestEspnSetupPartial:
    """
    Reading league size and draft slot from ESPN instead of typing them.

    A wrong draft position throws the snake order off from the first pick, and
    two of the three leagues randomise their order at draft start — so the
    partial has to say whether the slot it found is settled.
    """

    @staticmethod
    def _patch(monkeypatch: pytest.MonkeyPatch, **setup):
        from nfl_predict import espn_fantasy

        base = {
            "league_size": 14,
            "draft_position": 8,
            "order_is_final": True,
            "draft_type": "SNAKE",
            "team_id": "9",
        }
        base.update(setup)

        class _Fake:
            @staticmethod
            def from_env(_league=None):
                class _C:
                    @staticmethod
                    def fetch_draft_setup():
                        return base

                return _C()

        monkeypatch.setattr(espn_fantasy, "EspnFantasyClient", _Fake)

    def test_it_preselects_the_size_espn_reports(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        self._patch(monkeypatch, league_size=12)
        body = _client().get("/draft/espn-setup").text
        assert '<option value="12" selected>' in body

    def test_it_fills_the_draft_position(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        self._patch(monkeypatch, draft_position=8)
        assert (
            'name="draft_position" value="8"' in _client().get("/draft/espn-setup").text
        )

    def test_a_settled_order_is_reported_as_confirmed(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "ludopathy")
        self._patch(monkeypatch, order_is_final=True, draft_position=3, league_size=10)
        assert "confirmed by ESPN" in _client().get("/draft/espn-setup").text

    def test_a_randomised_order_warns_that_the_slot_may_change(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        self._patch(monkeypatch, order_is_final=False)
        # The template wraps the sentence, so compare on collapsed whitespace.
        body = " ".join(_client().get("/draft/espn-setup").text.split())
        assert "randomises the order at draft start" in body
        assert "re-sync once it is locked" in body

    def test_no_order_yet_says_so(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        self._patch(monkeypatch, draft_position=None)
        body = _client().get("/draft/espn-setup").text
        assert "no draft order yet" in body
        assert 'name="draft_position" value="1"' in body

    def test_an_espn_failure_degrades_to_profile_defaults(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The setup page must still work when ESPN is down or a cookie expired."""
        from nfl_predict import espn_fantasy
        from nfl_predict.espn_fantasy import EspnFantasyError

        class _Boom:
            @staticmethod
            def from_env(_league=None):
                raise EspnFantasyError("HTTP 401 for league 1773102615")

        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        monkeypatch.setattr(espn_fantasy, "EspnFantasyClient", _Boom)

        resp = _client().get("/draft/espn-setup")
        assert resp.status_code == 200
        assert "401" in resp.text
        assert '<option value="14" selected>' in resp.text

    def test_the_button_only_shows_for_a_league_with_an_espn_id(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        _write_board(workdir, "hoh")
        assert "/draft/espn-setup" in _client().get("/draft").text
