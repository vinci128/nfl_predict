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

    def test_another_leagues_session_is_not_shown_as_this_ones(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Switching league must not surface the other draft.

        An earlier fallback used the sole session on disk when the active
        league had none, so with a Hell or Highwater draft running, switching
        to Royal Rumble showed that 14-team slate under the Royal Rumble
        heading. Two drafts in one evening makes that a real hazard.
        """
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        board_path = _write_board(workdir, "hoh")

        client = _client()
        client.post(
            "/draft/start",
            data={"board_path": board_path, "draft_position": "1"},
            follow_redirects=False,
        )
        assert (workdir / "outputs/draft_state_hoh.json").exists()

        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "rumble")
        assert not draft_api._state_exists()
        assert (
            _client().get("/draft/board", follow_redirects=False).headers["location"]
            == "/draft"
        )

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

    def test_the_setup_page_reads_from_espn_without_being_asked(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """League size and slot should be automatic, not a button press."""
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        _write_board(workdir, "hoh")
        body = " ".join(_client().get("/draft").text.split())
        assert 'hx-get="/draft/espn-setup" hx-trigger="load"' in body

    def test_the_fetched_response_does_not_trigger_itself_again(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without this the swap would re-fire on every load, forever."""
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        self._patch(monkeypatch)
        assert 'hx-trigger="load"' not in _client().get("/draft/espn-setup").text

    def test_the_form_is_usable_before_espn_answers(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first paint carries the profile's own values, not empty fields."""
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        _write_board(workdir, "hoh")
        body = _client().get("/draft").text
        assert '<option value="14" selected>' in body
        assert 'name="draft_position" value="1"' in body

    def test_a_league_without_an_espn_id_does_not_auto_fetch(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dataclasses import replace

        from nfl_predict.leagues import PROFILES

        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        _write_board(workdir, "hoh")
        monkeypatch.setitem(
            PROFILES, "hoh", replace(PROFILES["hoh"], espn_league_id=None)
        )

        body = _client().get("/draft").text
        assert 'hx-trigger="load"' not in body


class TestAutomaticRouting:
    """
    Opening the app should land on the right board with no clicking.

    Two drafts two hours apart means the league switcher gets used mid-evening,
    and it has to put you on that league's draft rather than a form.
    """

    def _start(self, workdir: Path, league: str) -> None:
        _write_board(workdir, league)
        _client().post(
            "/draft/start",
            data={
                "board_path": f"outputs/draft_board_2026_{league}.csv",
                "draft_position": "1",
            },
            follow_redirects=False,
        )

    def test_setup_goes_straight_to_a_live_board(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        self._start(workdir, "hoh")

        resp = _client().get("/draft", follow_redirects=False)

        assert resp.status_code == 303
        assert resp.headers["location"] == "/draft/board"

    def test_setup_is_shown_when_there_is_no_session(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        _write_board(workdir, "hoh")

        resp = _client().get("/draft", follow_redirects=False)

        assert resp.status_code == 200
        assert "New Draft Session" in resp.text

    def test_the_board_sends_you_to_setup_rather_than_404ing(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")

        resp = _client().get("/draft/board", follow_redirects=False)

        assert resp.status_code == 303
        assert resp.headers["location"] == "/draft"

    def test_each_league_lands_on_its_own_draft(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The evening's actual shape: two sessions, switching between them."""
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        self._start(workdir, "hoh")
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "rumble")
        self._start(workdir, "rumble")

        for league, teams in (("hoh", 14), ("rumble", 8)):
            monkeypatch.setenv("NFL_PREDICT_LEAGUE", league)
            body = _client().get("/draft/board").text
            assert f"{teams}-team league" in body
