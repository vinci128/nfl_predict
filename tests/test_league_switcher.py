"""
Switching leagues in the web UI.

Both leagues are served from one process, so the active league cannot be a
process-wide setting: it comes from the viewer's cookie and every
`get_profile()` call under that request picks it up.
"""

from __future__ import annotations

import pytest

from nfl_predict.leagues import (
    get_profile,
    league_nav,
    reset_active_league,
    set_active_league,
)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from nfl_predict.api import app

    return TestClient(app)


class TestActiveLeagueContext:
    def test_override_wins_over_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "ludopathy")
        token = set_active_league("hoh")
        try:
            assert get_profile().key == "hoh"
        finally:
            reset_active_league(token)
        assert get_profile().key == "ludopathy"

    def test_no_override_falls_back_to_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        token = set_active_league(None)
        try:
            assert get_profile().key == "hoh"
        finally:
            reset_active_league(token)

    def test_a_stale_cookie_does_not_raise(self) -> None:
        """An unknown league must degrade to the default, not 500 the page."""
        token = set_active_league("a-league-that-was-deleted")
        try:
            assert get_profile().key in {"ludopathy", "hoh"}
        finally:
            reset_active_league(token)

    def test_an_explicit_key_still_beats_the_override(self) -> None:
        token = set_active_league("hoh")
        try:
            assert get_profile("ludopathy").key == "ludopathy"
        finally:
            reset_active_league(token)

    def test_nav_reports_the_active_league_and_every_option(self) -> None:
        token = set_active_league("hoh")
        try:
            nav = league_nav()
        finally:
            reset_active_league(token)

        assert nav["active"] == "hoh"
        assert nav["name"] == "Hell or Highwater"
        assert dict(nav["options"]) == {
            "ludopathy": "Ludopathy Bowl",
            "hoh": "Hell or Highwater",
            "rumble": "Royal Rumble",
        }


class TestSwitchEndpoint:
    def test_sets_the_cookie(self, client) -> None:
        r = client.get("/league/hoh", follow_redirects=False)
        assert r.status_code == 303
        assert r.cookies["league"] == "hoh"

    def test_accepts_a_league_alias(self, client) -> None:
        r = client.get("/league/hell_or_highwater", follow_redirects=False)
        assert r.cookies["league"] == "hoh"

    def test_unknown_league_is_404_not_500(self, client) -> None:
        assert client.get("/league/nope", follow_redirects=False).status_code == 404

    def test_returns_to_the_page_you_came_from(self, client) -> None:
        r = client.get(
            "/league/hoh",
            headers={"referer": "http://testserver/weekly"},
            follow_redirects=False,
        )
        assert r.headers["location"] == "/weekly"

    def test_a_foreign_referer_cannot_redirect_off_site(self, client) -> None:
        """Only the path is reused, so this cannot become an open redirect."""
        r = client.get(
            "/league/hoh",
            headers={"referer": "https://evil.example/phish"},
            follow_redirects=False,
        )
        assert r.headers["location"] == "/phish"
        assert "evil.example" not in r.headers["location"]

    def test_no_referer_returns_home(self, client) -> None:
        r = client.get("/league/hoh", follow_redirects=False)
        assert r.headers["location"] == "/"


class TestCookieDrivesThePage:
    def test_the_switcher_marks_the_cookie_league_selected(self, client) -> None:
        client.cookies.set("league", "hoh")
        body = client.get("/draft").text

        assert '<option value="hoh" selected>Hell or Highwater</option>' in body
        assert '<option value="ludopathy" selected>' not in body

    def test_switching_changes_which_league_the_page_describes(self, client) -> None:
        client.cookies.set("league", "ludopathy")
        assert "Ludopathy Bowl" in client.get("/draft").text

        client.cookies.set("league", "hoh")
        assert "Hell or Highwater" in client.get("/draft").text

    def test_league_size_follows_the_cookie(self, client) -> None:
        client.cookies.set("league", "hoh")
        assert (
            '<option value="14" selected>14 teams</option>' in client.get("/draft").text
        )

        client.cookies.set("league", "ludopathy")
        assert (
            '<option value="10" selected>10 teams</option>' in client.get("/draft").text
        )

    def test_the_cookie_does_not_leak_between_requests(self, client) -> None:
        """One viewer on hoh must not change what another viewer sees."""
        client.cookies.set("league", "hoh")
        client.get("/draft")

        client.cookies.clear()
        body = client.get("/draft").text
        assert '<option value="ludopathy" selected>' in body
