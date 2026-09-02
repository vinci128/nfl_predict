"""
Tests for the ESPN draft-sync provider and the provider abstraction.

No live ESPN calls: `_get` is patched with a recorded-shape payload, taken
from the documented `?view=mDraftDetail` response
(draftDetail.picks[] -> teamId, playerId, roundId, roundPickNumber).
"""

from __future__ import annotations

import email.message
import json
import os
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from nfl_predict.draft_sync import DraftSyncError, available_providers, make_client
from nfl_predict.espn_fantasy import (
    EspnFantasyClient,
    EspnFantasyError,
    _explain_http_error,
    _normalise_position,
    _normalise_swid,
    _normalise_team,
    _position_from_slots,
    _team_names,
    _week_points,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _draft_payload() -> dict:
    """A minimal mDraftDetail response, picks deliberately out of order."""
    return {
        "draftDetail": {
            "drafted": True,
            "picks": [
                {
                    "teamId": 7,
                    "playerId": 222,
                    "roundId": 1,
                    "roundPickNumber": 2,
                },
                {
                    "teamId": 3,
                    "playerId": 111,
                    "roundId": 1,
                    "roundPickNumber": 1,
                },
                {
                    "teamId": 3,
                    "playerId": 999,  # not in the crosswalk
                    "roundId": 2,
                    "roundPickNumber": 1,
                },
            ],
        }
    }


_PLAYER_MAP = {
    111: {
        "player_id": "00-0011111",
        "name": "Bijan Robinson",
        "position": "RB",
        "team": "ATL",
    },
    222: {
        "player_id": "00-0022222",
        "name": "Justin Jefferson",
        "position": "WR",
        "team": "MIN",
    },
}


@pytest.fixture()
def client() -> EspnFantasyClient:
    c = EspnFantasyClient(league_id="123", season=2026, team_id="3", league_size=12)
    c._player_map = dict(_PLAYER_MAP)
    return c


@pytest.fixture()
def private_client() -> EspnFantasyClient:
    """Client with ESPN cookies (private league)."""
    return EspnFantasyClient(
        league_id="456",
        season=2026,
        espn_s2="test_s2_value",
        swid="{ABC-123}",
        team_id="5",
        league_size=10,
    )


# ---------------------------------------------------------------------------
# Pick parsing
# ---------------------------------------------------------------------------


class TestFetchAllPicks:
    def test_picks_sorted_into_draft_order(self, client: EspnFantasyClient) -> None:
        """ESPN order is not guaranteed, and the caller slices by count -- a
        mis-ordered list would record picks against the wrong players."""
        with patch.object(client, "_get", return_value=_draft_payload()):
            picks = client.fetch_all_picks()

        assert [p["overall_pick"] for p in picks] == [1, 2, 3]
        assert picks[0]["player_name"] == "Bijan Robinson"
        assert picks[1]["player_name"] == "Justin Jefferson"

    def test_round_and_pick_come_from_espn(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value=_draft_payload()):
            picks = client.fetch_all_picks()

        assert (picks[0]["round"], picks[0]["pick_in_round"]) == (1, 1)
        assert (picks[2]["round"], picks[2]["pick_in_round"]) == (2, 1)

    def test_resolves_gsis_id_for_exact_matching(
        self, client: EspnFantasyClient
    ) -> None:
        """player_id lets mark_drafted match exactly -- abbreviated names
        collide (Bijan and Brian Robinson are both B.Robinson)."""
        with patch.object(client, "_get", return_value=_draft_payload()):
            picks = client.fetch_all_picks()

        assert picks[0]["player_id"] == "00-0011111"
        assert picks[0]["position"] == "RB"
        assert picks[0]["nfl_team"] == "ATL"

    def test_unknown_player_still_recorded(self, client: EspnFantasyClient) -> None:
        """A crosswalk miss must not drop the pick -- ordering would shift and
        every later pick would be attributed to the wrong player."""
        with patch.object(client, "_get", return_value=_draft_payload()):
            picks = client.fetch_all_picks()

        assert len(picks) == 3
        assert picks[2]["player_name"] == "ESPN player 999"
        assert picks[2]["player_id"] == ""

    def test_is_mine_matches_team_id(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value=_draft_payload()):
            picks = client.fetch_all_picks()

        assert picks[0]["is_mine"] is True  # teamId 3
        assert picks[1]["is_mine"] is False  # teamId 7

    def test_empty_before_draft_starts(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value={"draftDetail": {}}):
            assert client.fetch_all_picks() == []

    def test_fetch_new_picks_skips_recorded(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value=_draft_payload()):
            new = client.fetch_new_picks(already_recorded=2)

        assert len(new) == 1
        assert new[0]["overall_pick"] == 3

    def test_fetch_new_picks_all_recorded_returns_empty(
        self, client: EspnFantasyClient
    ) -> None:
        with patch.object(client, "_get", return_value=_draft_payload()):
            new = client.fetch_new_picks(already_recorded=3)
        assert new == []

    def test_fetch_new_picks_more_than_available_returns_empty(
        self, client: EspnFantasyClient
    ) -> None:
        with patch.object(client, "_get", return_value=_draft_payload()):
            new = client.fetch_new_picks(already_recorded=99)
        assert new == []

    def test_missing_draft_detail_key(self, client: EspnFantasyClient) -> None:
        """Response with no draftDetail key at all."""
        with patch.object(client, "_get", return_value={"something": "else"}):
            assert client.fetch_all_picks() == []

    def test_missing_picks_key(self, client: EspnFantasyClient) -> None:
        """draftDetail exists but has no picks key."""
        with patch.object(
            client, "_get", return_value={"draftDetail": {"drafted": True}}
        ):
            assert client.fetch_all_picks() == []

    def test_fallback_round_when_espn_omits_roundid(
        self, client: EspnFantasyClient
    ) -> None:
        """If ESPN omits roundId, the client falls back to league_size math."""
        payload = {
            "draftDetail": {
                "picks": [
                    {"teamId": 1, "playerId": 111, "roundPickNumber": 1},
                    {"teamId": 2, "playerId": 222, "roundPickNumber": 2},
                ],
            }
        }
        with patch.object(client, "_get", return_value=payload):
            picks = client.fetch_all_picks()
        # With league_size=12 and no roundId: pick 1 is R1.1, pick 2 is R1.2
        assert picks[0]["round"] == 1
        assert picks[1]["round"] == 1

    def test_fallback_pick_when_espn_omits_roundpicknumber(
        self, client: EspnFantasyClient
    ) -> None:
        """If ESPN omits roundPickNumber, the client falls back to league_size math."""
        payload = {
            "draftDetail": {
                "picks": [
                    {"teamId": 1, "playerId": 111, "roundId": 1},
                    {"teamId": 2, "playerId": 222, "roundId": 1},
                ],
            }
        }
        with patch.object(client, "_get", return_value=payload):
            picks = client.fetch_all_picks()
        assert picks[0]["pick_in_round"] == 1
        assert picks[1]["pick_in_round"] == 2

    def test_no_player_id_field_in_pick(self, client: EspnFantasyClient) -> None:
        """A pick with no playerId key at all is an unmade slot -- skip it."""
        payload = {
            "draftDetail": {
                "picks": [
                    {"teamId": 1, "roundId": 1, "roundPickNumber": 1},
                ],
            }
        }
        with patch.object(client, "_get", return_value=payload):
            picks = client.fetch_all_picks()
        assert picks == []

    def test_unmade_slots_are_skipped(self, client: EspnFantasyClient) -> None:
        """
        ESPN pre-seeds the whole slate once the draft is scheduled: unmade
        slots carry playerId -1. Recording those would fill the local draft
        state with phantom picks before anyone has drafted.
        """
        payload = {
            "draftDetail": {
                "picks": [
                    {"teamId": 1, "playerId": 111, "roundId": 1, "roundPickNumber": 1},
                    {"teamId": 2, "playerId": -1, "roundId": 1, "roundPickNumber": 2},
                    {"teamId": 3, "playerId": -1, "roundId": 1, "roundPickNumber": 3},
                ],
            }
        }
        with patch.object(client, "_get", return_value=payload):
            picks = client.fetch_all_picks()
        assert len(picks) == 1
        assert picks[0]["player_name"] == "Bijan Robinson"

    def test_full_unstarted_slate_reads_as_no_picks(
        self, client: EspnFantasyClient
    ) -> None:
        """A scheduled but unstarted draft returns picks for every slot."""
        payload = {
            "draftDetail": {
                "drafted": False,
                "inProgress": False,
                "picks": [
                    {
                        "teamId": (i % 12) + 1,
                        "playerId": -1,
                        "roundId": (i // 12) + 1,
                        "roundPickNumber": (i % 12) + 1,
                    }
                    for i in range(180)
                ],
            }
        }
        with patch.object(client, "_get", return_value=payload):
            assert client.fetch_all_picks() == []

    def test_empty_team_id(self, client: EspnFantasyClient) -> None:
        """A pick with no teamId -- should still record."""
        payload = {
            "draftDetail": {
                "picks": [
                    {"playerId": 111, "roundId": 1, "roundPickNumber": 1},
                ],
            }
        }
        with patch.object(client, "_get", return_value=payload):
            picks = client.fetch_all_picks()
        assert picks[0]["team_id"] == ""
        assert picks[0]["is_mine"] is False


# ---------------------------------------------------------------------------
# Position mapping
# ---------------------------------------------------------------------------


class TestPositionMapping:
    def test_kicker_pk_maps_to_k(self, client: EspnFantasyClient) -> None:
        """ESPN uses 'PK' for kickers; our board uses 'K'."""
        payload = {
            "draftDetail": {
                "picks": [
                    {"teamId": 1, "playerId": 888, "roundId": 1, "roundPickNumber": 1},
                ],
            }
        }
        client._player_map = {
            888: {
                "player_id": "",
                "name": "Justin Tucker",
                "position": "PK",
                "team": "BAL",
            },
        }
        with patch.object(client, "_get", return_value=payload):
            picks = client.fetch_all_picks()
        assert picks[0]["position"] == "K"

    def test_defense_def_maps_to_dst(self, client: EspnFantasyClient) -> None:
        """ESPN uses 'DEF' for defenses; our board uses 'DST'."""
        payload = {
            "draftDetail": {
                "picks": [
                    {"teamId": 1, "playerId": 777, "roundId": 1, "roundPickNumber": 1},
                ],
            }
        }
        client._player_map = {
            777: {
                "player_id": "",
                "name": "49ers D/ST",
                "position": "DEF",
                "team": "SF",
            },
        }
        with patch.object(client, "_get", return_value=payload):
            picks = client.fetch_all_picks()
        assert picks[0]["position"] == "DST"

    def test_defense_dst_maps_to_dst(self, client: EspnFantasyClient) -> None:
        """ESPN sometimes uses 'DST' directly."""
        payload = {
            "draftDetail": {
                "picks": [
                    {"teamId": 1, "playerId": 777, "roundId": 1, "roundPickNumber": 1},
                ],
            }
        }
        client._player_map = {
            777: {
                "player_id": "",
                "name": "Cowboys D/ST",
                "position": "DST",
                "team": "DAL",
            },
        }
        with patch.object(client, "_get", return_value=payload):
            picks = client.fetch_all_picks()
        assert picks[0]["position"] == "DST"

    def test_kicker_k_passthrough(self, client: EspnFantasyClient) -> None:
        """ESPN sometimes uses 'K' directly."""
        payload = {
            "draftDetail": {
                "picks": [
                    {"teamId": 1, "playerId": 888, "roundId": 1, "roundPickNumber": 1},
                ],
            }
        }
        client._player_map = {
            888: {
                "player_id": "",
                "name": "Harrison Butker",
                "position": "K",
                "team": "KC",
            },
        }
        with patch.object(client, "_get", return_value=payload):
            picks = client.fetch_all_picks()
        assert picks[0]["position"] == "K"

    def test_unknown_position_uppercased(self, client: EspnFantasyClient) -> None:
        """Unknown positions should be uppercased as-is."""
        payload = {
            "draftDetail": {
                "picks": [
                    {"teamId": 1, "playerId": 666, "roundId": 1, "roundPickNumber": 1},
                ],
            }
        }
        client._player_map = {
            666: {
                "player_id": "",
                "name": "Flex Star",
                "position": "FLEX",
                "team": "NYG",
            },
        }
        with patch.object(client, "_get", return_value=payload):
            picks = client.fetch_all_picks()
        assert picks[0]["position"] == "FLEX"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_public_league_needs_no_cookies(self) -> None:
        """Requiring cookies would hide a working public-league setup."""
        with patch.dict(os.environ, {"ESPN_LEAGUE_ID": "123"}, clear=True):
            assert EspnFantasyClient.credentials_available() is True
            client = EspnFantasyClient.from_env()
        assert client.espn_s2 is None

    def test_missing_league_id_raises_when_the_profile_has_none(self) -> None:
        """
        With no id in the environment and none on the league profile there is
        nothing to talk to, so the client must refuse rather than guess.
        """
        from dataclasses import replace

        from nfl_predict import leagues

        anonymous = replace(leagues.LUDOPATHY, espn_league_id=None)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(leagues.PROFILES, {"ludopathy": anonymous}),
        ):
            assert EspnFantasyClient.credentials_available() is False
            with pytest.raises(EspnFantasyError, match="ESPN_LEAGUE_ID"):
                EspnFantasyClient.from_env()

    def test_league_profile_supplies_the_id_without_any_environment(self) -> None:
        """A configured league needs no env vars at all for a public league."""
        with patch.dict(os.environ, {}, clear=True):
            c = EspnFantasyClient.from_env("hoh")
        assert c.league_id == "581348581"
        assert c.league_size == 14
        assert c.team_id == "9"

    def test_environment_overrides_the_profile(self) -> None:
        with patch.dict(os.environ, {"ESPN_LEAGUE_ID": "999"}, clear=True):
            c = EspnFantasyClient.from_env("hoh")
        assert c.league_id == "999"

    @pytest.mark.parametrize(
        "raw", ["ABC-123", "{ABC-123}", "  {ABC-123}  ", "{ABC-123", "ABC-123}"]
    )
    def test_swid_brace_wrapping_normalised(self, raw: str) -> None:
        """ESPN's cookie is brace-wrapped; users copy it either way."""
        assert _normalise_swid(raw) == "{ABC-123}"

    def test_swid_none_stays_none(self) -> None:
        assert _normalise_swid(None) is None
        assert _normalise_swid("") is None

    def test_team_id_resolved_from_swid(self) -> None:
        client = EspnFantasyClient(league_id="1", season=2026, swid="{ME}", espn_s2="x")
        league = {
            "settings": {"size": 10},
            "teams": [
                {"id": 1, "owners": ["{SOMEONE}"]},
                {"id": 4, "owners": ["{ME}"]},
            ],
        }
        with patch.object(client, "_get", return_value=league):
            assert client.get_my_team_id() == "4"

    def test_league_size_taken_from_settings(self) -> None:
        client = EspnFantasyClient(league_id="1", season=2026, league_size=12)
        with patch.object(client, "_get", return_value={"settings": {"size": 10}}):
            client.get_league()
        assert client.league_size == 10

    def test_league_size_fallback_to_teams_list(self) -> None:
        """If settings.size is missing, fall back to len(teams)."""
        client = EspnFantasyClient(league_id="1", season=2026, league_size=12)
        with patch.object(
            client, "_get", return_value={"teams": [{"id": 1}, {"id": 2}, {"id": 3}]}
        ):
            client.get_league()
        assert client.league_size == 3

    def test_no_swid_and_no_team_id_is_actionable(self) -> None:
        client = EspnFantasyClient(league_id="1", season=2026)
        with pytest.raises(EspnFantasyError, match="ESPN_TEAM_ID"):
            client.get_my_team_id()

    def test_from_env_parses_all_vars(self) -> None:
        env = {
            "ESPN_LEAGUE_ID": "999",
            "ESPN_SEASON": "2025",
            "ESPN_S2": "my_s2_cookie",
            "ESPN_SWID": "my_swid",
            "ESPN_TEAM_ID": "7",
            "ESPN_LEAGUE_SIZE": "10",
        }
        with patch.dict(os.environ, env, clear=True):
            c = EspnFantasyClient.from_env()
        assert c.league_id == "999"
        assert c.season == 2025
        assert c.espn_s2 == "my_s2_cookie"
        assert c.swid == "{my_swid}"
        assert c.team_id == "7"
        assert c.league_size == 10

    def test_from_env_defaults(self) -> None:
        """
        ESPN_LEAGUE_ID overrides the profile; the rest falls back to it.
        Cookies are never stored in a profile — they are credentials.
        """
        with patch.dict(os.environ, {"ESPN_LEAGUE_ID": "42"}, clear=True):
            c = EspnFantasyClient.from_env("ludopathy")
        assert c.league_id == "42"
        assert c.season == 2026
        assert c.espn_s2 is None
        assert c.swid is None
        assert c.team_id == "9"  # from the profile
        assert c.league_size == 10  # from the profile

    def test_from_env_empty_s2_becomes_none(self) -> None:
        """Empty ESPN_S2 string should be treated as None (public league)."""
        with patch.dict(os.environ, {"ESPN_LEAGUE_ID": "1", "ESPN_S2": ""}, clear=True):
            c = EspnFantasyClient.from_env()
        assert c.espn_s2 is None

    def test_credentials_available_only_needs_league_id(self) -> None:
        """Public leagues work with just ESPN_LEAGUE_ID."""
        with patch.dict(os.environ, {"ESPN_LEAGUE_ID": "123"}, clear=True):
            assert EspnFantasyClient.credentials_available() is True

    def test_team_id_returned_directly(self) -> None:
        """When team_id is set explicitly, get_my_team_id returns it without API call."""
        client = EspnFantasyClient(league_id="1", season=2026, team_id="42")
        assert client.get_my_team_id() == "42"


# ---------------------------------------------------------------------------
# SWID matching edge cases
# ---------------------------------------------------------------------------


class TestSwidMatching:
    def test_swid_match_is_case_insensitive(self) -> None:
        """SWID comparison should be case-insensitive."""
        client = EspnFantasyClient(
            league_id="1", season=2026, swid="{AbC-123}", espn_s2="x"
        )
        league = {
            "teams": [
                {"id": 5, "owners": ["{abc-123}"]},
            ],
        }
        with patch.object(client, "_get", return_value=league):
            assert client.get_my_team_id() == "5"

    def test_swid_not_found_raises(self) -> None:
        """SWID doesn't match any team owner."""
        client = EspnFantasyClient(
            league_id="1", season=2026, swid="{NOT-MINE}", espn_s2="x"
        )
        league = {
            "teams": [
                {"id": 1, "owners": ["{SOMEONE-ELSE}"]},
            ],
        }
        with (
            patch.object(client, "_get", return_value=league),
            pytest.raises(EspnFantasyError, match="No team.*NOT-MINE"),
        ):
            client.get_my_team_id()

    def test_swid_match_with_empty_owners(self) -> None:
        """A team with no owners list should not crash."""
        client = EspnFantasyClient(league_id="1", season=2026, swid="{ME}", espn_s2="x")
        league = {
            "teams": [
                {"id": 1},
                {"id": 2, "owners": ["{ME}"]},
            ],
        }
        with patch.object(client, "_get", return_value=league):
            assert client.get_my_team_id() == "2"

    def test_team_id_cached_after_first_resolve(self) -> None:
        """Once resolved via SWID, subsequent calls use the cached value."""
        client = EspnFantasyClient(league_id="1", season=2026, swid="{ME}", espn_s2="x")
        league = {"teams": [{"id": 7, "owners": ["{ME}"]}]}
        call_count = 0

        def counting_get(views):
            nonlocal call_count
            call_count += 1
            return league

        with patch.object(client, "_get", side_effect=counting_get):
            first = client.get_my_team_id()
            second = client.get_my_team_id()
        assert first == "7"
        assert second == "7"
        assert call_count == 1  # second call used cache, no API hit


# ---------------------------------------------------------------------------
# Player crosswalk
# ---------------------------------------------------------------------------


class TestPlayerCrosswalk:
    def test_crosswalk_loaded_once_and_cached(self) -> None:
        """_players() should load the crosswalk only once."""
        c = EspnFantasyClient(league_id="1", season=2026)
        assert c._player_map is None  # not loaded yet

        call_count = 0
        original_players = c._players

        def counting_players():
            nonlocal call_count
            call_count += 1
            return original_players()

        # Patch nflreadpy to control the crosswalk
        fake_frame = MagicMock()
        fake_frame.__getitem__ = MagicMock(return_value=fake_frame)
        fake_frame.itertuples = MagicMock(return_value=iter([]))

        fake_nfl = MagicMock()
        fake_nfl.load_ff_playerids.return_value = fake_frame

        with patch.dict("sys.modules", {"nflreadpy": fake_nfl}):
            c._players()
            c._players()

        assert c._player_map is not None  # now cached

    def test_crosswalk_graceful_failure(self) -> None:
        """If nflreadpy is unavailable, _players returns empty dict."""
        c = EspnFantasyClient(league_id="1", season=2026)
        with patch.dict("sys.modules", {"nflreadpy": None}):
            result = c._players()
        assert result == {}
        assert c._player_map == {}

    def test_crosswalk_nan_espn_id_skipped(self) -> None:
        """Rows with NaN espn_id should be skipped."""
        import pandas as pd

        c = EspnFantasyClient(league_id="1", season=2026)
        fake_frame = pd.DataFrame(
            {
                "espn_id": [100.0, float("nan"), 200.0],
                "gsis_id": ["00-0010000", "00-0020000", "00-0030000"],
                "name": ["Player A", "Player B", "Player C"],
                "position": ["QB", "RB", "WR"],
                "team": ["NE", "NYG", "DAL"],
            }
        )
        fake_nfl = MagicMock()
        fake_nfl.load_ff_playerids.return_value = fake_frame

        with patch.dict("sys.modules", {"nflreadpy": fake_nfl}):
            result = c._players()

        assert 100 in result
        assert 200 in result
        assert len(result) == 2  # NaN row skipped


# ---------------------------------------------------------------------------
# _get / HTTP behavior
# ---------------------------------------------------------------------------


class TestGetRequest:
    def test_url_construction(self, client: EspnFantasyClient) -> None:
        """_get should build the correct ESPN API URL."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"ok": True}).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            client._get(["mTeam", "mSettings"])

            req = mock_open.call_args[0][0]
            assert "seasons/2026" in req.full_url
            assert "leagues/123" in req.full_url
            assert "view=mTeam" in req.full_url
            assert "view=mSettings" in req.full_url

    def test_user_agent_header(self, client: EspnFantasyClient) -> None:
        """ESPN checks User-Agent; a missing one causes 403."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"ok": True}).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            client._get(["mTeam"])

            req = mock_open.call_args[0][0]
            assert "Mozilla" in req.get_header("User-agent")

    def test_cookie_header_for_private_league(
        self, private_client: EspnFantasyClient
    ) -> None:
        """Private leagues must send espn_s2 and SWID cookies."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"ok": True}).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            private_client._get(["mDraftDetail"])

            req = mock_open.call_args[0][0]
            cookie = req.get_header("Cookie")
            assert "espn_s2=test_s2_value" in cookie
            assert "SWID={ABC-123}" in cookie

    def test_no_cookie_header_for_public_league(
        self, client: EspnFantasyClient
    ) -> None:
        """Public leagues must not send a Cookie header."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"ok": True}).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            client._get(["mDraftDetail"])

            req = mock_open.call_args[0][0]
            assert req.get_header("Cookie") is None

    def test_league_history_wrapping_unwrapped(self, client: EspnFantasyClient) -> None:
        """leagueHistory-style responses come as a single-element list."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps([{"data": "payload"}]).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            result = client._get(["mTeam"])
            assert result == {"data": "payload"}

    def test_empty_list_response(self, client: EspnFantasyClient) -> None:
        """An empty list response should return the list as-is."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps([]).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            result = client._get(["mTeam"])
            assert result == []

    def test_non_list_response_returned_as_is(self, client: EspnFantasyClient) -> None:
        """A normal dict response is not unwrapped."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"key": "val"}).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            result = client._get(["mTeam"])
            assert result == {"key": "val"}


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------


class TestHttpErrors:
    def test_403_without_credentials_suggests_cookies(self) -> None:
        client = EspnFantasyClient(league_id="123", season=2026)
        err = urllib.error.HTTPError(
            url="http://test",
            code=403,
            msg="Forbidden",
            hdrs=email.message.Message(),
            fp=None,
        )
        msg = _explain_http_error(err, client)
        assert "ESPN_S2" in msg
        assert "ESPN_SWID" in msg

    def test_403_with_credentials_suggests_expired(self) -> None:
        client = EspnFantasyClient(
            league_id="123", season=2026, espn_s2="x", swid="{Y}"
        )
        err = urllib.error.HTTPError(
            url="http://test",
            code=403,
            msg="Forbidden",
            hdrs=email.message.Message(),
            fp=None,
        )
        msg = _explain_http_error(err, client)
        assert "expired" in msg.lower()

    def test_401_without_credentials_suggests_cookies(self) -> None:
        client = EspnFantasyClient(league_id="456", season=2026)
        err = urllib.error.HTTPError(
            url="http://test",
            code=401,
            msg="Unauthorized",
            hdrs=email.message.Message(),
            fp=None,
        )
        msg = _explain_http_error(err, client)
        assert "456" in msg
        assert "ESPN_S2" in msg

    def test_404_suggests_check_league_id(self) -> None:
        client = EspnFantasyClient(league_id="999", season=2025)
        err = urllib.error.HTTPError(
            url="http://test",
            code=404,
            msg="Not Found",
            hdrs=email.message.Message(),
            fp=None,
        )
        msg = _explain_http_error(err, client)
        assert "999" in msg
        assert "2025" in msg

    def test_500_returns_generic_message(self) -> None:
        client = EspnFantasyClient(league_id="123", season=2026)
        err = urllib.error.HTTPError(
            url="http://test",
            code=500,
            msg="Server Error",
            hdrs=email.message.Message(),
            fp=None,
        )
        msg = _explain_http_error(err, client)
        assert "500" in msg

    def test_http_403_raises_espn_fantasy_error(
        self, client: EspnFantasyClient
    ) -> None:
        """_get should wrap HTTP errors in EspnFantasyError."""
        err = urllib.error.HTTPError(
            url="http://test",
            code=403,
            msg="Forbidden",
            hdrs=email.message.Message(),
            fp=None,
        )
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(EspnFantasyError, match="403"),
        ):
            client._get(["mTeam"])

    def test_connection_error_raises_espn_fantasy_error(
        self, client: EspnFantasyClient
    ) -> None:
        """Network errors should be wrapped, not propagated."""
        with (
            patch("urllib.request.urlopen", side_effect=ConnectionError("timeout")),
            pytest.raises(EspnFantasyError, match="ESPN API error"),
        ):
            client._get(["mTeam"])


# ---------------------------------------------------------------------------
# fetch_all_picks with team_id=None (no team identity)
# ---------------------------------------------------------------------------


class TestNoTeamIdentity:
    def test_all_picks_not_mine_when_no_team_id(self) -> None:
        """Without team_id, no picks should be marked as 'mine'."""
        c = EspnFantasyClient(league_id="123", season=2026, league_size=12)
        c._player_map = dict(_PLAYER_MAP)
        with patch.object(c, "_get", return_value=_draft_payload()):
            picks = c.fetch_all_picks()
        assert all(not p["is_mine"] for p in picks)


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


class TestPollDraft:
    def test_stops_at_max_rounds(self) -> None:
        """Polling should stop when a pick reaches max_rounds."""
        from nfl_predict.espn_fantasy import poll_draft

        c = EspnFantasyClient(league_id="1", season=2026, team_id="1", league_size=2)
        c._player_map = {
            100: {"player_id": "", "name": "A", "position": "QB", "team": "NE"}
        }

        picks_batch = [
            [
                {
                    "overall_pick": 1,
                    "round": 3,
                    "pick_in_round": 1,
                    "player_name": "A",
                    "player_id": "",
                    "position": "QB",
                    "nfl_team": "NE",
                    "team_id": "1",
                    "is_mine": True,
                },
            ],
            [],
        ]
        call_idx = 0

        def fake_fetch(already_recorded=0):
            nonlocal call_idx
            batch = picks_batch[call_idx] if call_idx < len(picks_batch) else []
            call_idx += 1
            return batch

        recorded: list[dict] = []

        with (
            patch.object(c, "fetch_new_picks", side_effect=fake_fetch),
            patch("nfl_predict.espn_fantasy.time.sleep"),
        ):
            poll_draft(c, on_pick=recorded.append, interval=0, max_rounds=3)

        assert len(recorded) == 1
        assert recorded[0]["player_name"] == "A"

    def test_keyboard_interrupt_stops_polling(self) -> None:
        """Ctrl+C should stop the poll loop gracefully."""
        from nfl_predict.espn_fantasy import poll_draft

        c = EspnFantasyClient(league_id="1", season=2026)
        recorded: list[dict] = []
        # Should not raise
        with patch.object(c, "fetch_new_picks", side_effect=KeyboardInterrupt):
            poll_draft(c, on_pick=recorded.append, interval=0, max_rounds=20)

    def test_espn_error_during_poll_is_warning_not_fatal(self) -> None:
        """Transient ESPN errors should print a warning and continue."""
        from nfl_predict.espn_fantasy import poll_draft

        c = EspnFantasyClient(league_id="1", season=2026)

        call_count = 0

        def fetch_side_effect(already_recorded=0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise EspnFantasyError("transient error")
            return []  # second call returns empty to stop

        with (
            patch.object(c, "fetch_new_picks", side_effect=fetch_side_effect),
            patch("nfl_predict.espn_fantasy.time.sleep") as mock_sleep,
        ):
            # After the error, the loop sleeps and calls again; return empty to stop
            mock_sleep.side_effect = KeyboardInterrupt
            poll_draft(c, on_pick=lambda p: None, interval=0, max_rounds=20)

        assert call_count >= 1


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


class TestProviderSelection:
    def test_auto_prefers_espn(self) -> None:
        env = {
            "ESPN_LEAGUE_ID": "123",
            "NFL_FANTASY_USERNAME": "u",
            "NFL_FANTASY_PASSWORD": "p",
            "NFL_FANTASY_LEAGUE_ID": "9",
        }
        with patch.dict(os.environ, env, clear=True):
            assert available_providers() == ["espn", "nfl"]
            assert isinstance(make_client("auto"), EspnFantasyClient)

    def test_auto_still_prefers_espn_when_nfl_is_also_configured(self) -> None:
        """
        Every configured league carries an ESPN id, so ESPN is always
        available and auto always picks it. The deprecated NFL.com path is
        reachable only by asking for it explicitly.
        """
        env = {
            "NFL_FANTASY_USERNAME": "u",
            "NFL_FANTASY_PASSWORD": "p",
            "NFL_FANTASY_LEAGUE_ID": "9",
        }
        with patch.dict(os.environ, env, clear=True):
            client = make_client("auto")
        assert isinstance(client, EspnFantasyClient)

    def test_auto_falls_back_to_nfl_only_without_any_espn_league(self) -> None:
        from dataclasses import replace

        from nfl_predict import leagues

        anonymous = replace(leagues.LUDOPATHY, espn_league_id=None)
        env = {
            "NFL_FANTASY_USERNAME": "u",
            "NFL_FANTASY_PASSWORD": "p",
            "NFL_FANTASY_LEAGUE_ID": "9",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(leagues.PROFILES, {"ludopathy": anonymous}),
        ):
            from nfl_predict.nfl_fantasy import NflFantasyClient

            with pytest.warns(DeprecationWarning, match="deprecated"):
                client = make_client("auto")
            assert isinstance(client, NflFantasyClient)

    def test_a_configured_league_makes_espn_available_with_no_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert "espn" in available_providers()

    def test_nothing_configured_names_espn(self) -> None:
        from dataclasses import replace

        from nfl_predict import leagues

        anonymous = replace(leagues.LUDOPATHY, espn_league_id=None)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(leagues.PROFILES, {"ludopathy": anonymous}),
        ):
            assert available_providers() == []
            with pytest.raises(DraftSyncError) as exc:
                make_client("auto")
        assert "ESPN_LEAGUE_ID" in str(exc.value)

    def test_unknown_provider_rejected(self) -> None:
        with pytest.raises(DraftSyncError, match="Unknown provider"):
            make_client("yahoo")

    def test_explicit_provider_error_is_wrapped(self) -> None:
        """Callers catch DraftSyncError, so provider errors must surface as it."""
        from dataclasses import replace

        from nfl_predict import leagues

        anonymous = replace(leagues.LUDOPATHY, espn_league_id=None)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(leagues.PROFILES, {"ludopathy": anonymous}),
            pytest.raises(DraftSyncError),
        ):
            make_client("espn")

    def test_explicit_nfl_provider_warns(self) -> None:
        """Explicitly requesting the 'nfl' provider emits a deprecation warning."""
        env = {
            "NFL_FANTASY_USERNAME": "u",
            "NFL_FANTASY_PASSWORD": "p",
            "NFL_FANTASY_LEAGUE_ID": "9",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.warns(DeprecationWarning, match="nfl.*provider is deprecated"),
        ):
            make_client("nfl")

    def test_espn_only_env_returns_only_espn(self) -> None:
        """With only ESPN vars set, available_providers returns ['espn']."""
        with patch.dict(os.environ, {"ESPN_LEAGUE_ID": "123"}, clear=True):
            assert available_providers() == ["espn"]

    def test_make_client_explicit_espn(self) -> None:
        """Explicitly requesting 'espn' works."""
        with patch.dict(os.environ, {"ESPN_LEAGUE_ID": "42"}, clear=True):
            c = make_client("espn")
        assert isinstance(c, EspnFantasyClient)
        assert c.league_id == "42"

    def test_make_client_auto_with_only_espn(self) -> None:
        """Auto with only ESPN configured returns ESPN client."""
        with patch.dict(os.environ, {"ESPN_LEAGUE_ID": "77"}, clear=True):
            c = make_client("auto")
        assert isinstance(c, EspnFantasyClient)
        assert c.league_id == "77"


# ---------------------------------------------------------------------------
# Sync banner
# ---------------------------------------------------------------------------


class TestSyncBanner:
    """
    The sync endpoint used to set sync_count/sync_errors on the template
    context, but no template read them: a successful sync swapped the board
    with no confirmation, and picks ESPN reported that we could not match
    were dropped silently.
    """

    def test_reports_how_many_picks_landed(self) -> None:
        from nfl_predict.draft_api import _sync_banner

        message, tone = _sync_banner(3, [])
        assert message == "Synced 3 picks from ESPN."
        assert "blue" in tone

    def test_singular_pick(self) -> None:
        from nfl_predict.draft_api import _sync_banner

        message, _ = _sync_banner(1, [])
        assert message == "Synced 1 pick from ESPN."

    def test_unmatched_players_are_named_and_warn(self) -> None:
        from nfl_predict.draft_api import _sync_banner

        message, tone = _sync_banner(2, ["Player 'X' not found in available board."])
        assert "Synced 2 picks" in message
        assert "1 not matched" in message
        assert "Player 'X' not found" in message
        assert "yellow" in tone

    def test_all_picks_unmatched(self) -> None:
        from nfl_predict.draft_api import _sync_banner

        message, tone = _sync_banner(0, ["a failed", "b failed"])
        assert "Synced 0 picks" in message
        assert "2 not matched: a failed; b failed" in message
        assert "yellow" in tone


# ---------------------------------------------------------------------------
# Rosters (?view=mRoster)
# ---------------------------------------------------------------------------


def _roster_entry(
    player_id: int,
    slot: int,
    *,
    full_name: str | None = None,
    eligible: list[int] | None = None,
    injury: str | None = None,
    acquisition: str | None = None,
    stats: list[dict] | None = None,
) -> dict:
    """One ESPN roster entry, shaped as mRoster / mBoxscore return them."""
    player: dict = {"id": player_id}
    if full_name is not None:
        player["fullName"] = full_name
    if eligible is not None:
        player["eligibleSlots"] = eligible
    if injury is not None:
        player["injuryStatus"] = injury
    if stats is not None:
        player["stats"] = stats

    entry: dict = {
        "playerId": player_id,
        "lineupSlotId": slot,
        "playerPoolEntry": {"player": player},
    }
    if acquisition is not None:
        entry["acquisitionType"] = acquisition
    return entry


def _roster_payload() -> dict:
    """A minimal mRoster + mTeam response covering two teams."""
    return {
        "teams": [
            {
                "id": 3,
                "name": "My Team",
                "roster": {
                    "entries": [
                        _roster_entry(111, 2, injury="ACTIVE", acquisition="DRAFT"),
                        _roster_entry(222, 20, acquisition="ADD"),
                        _roster_entry(
                            -16033,
                            16,
                            full_name="Ravens D/ST",
                            eligible=[16, 20, 21],
                        ),
                    ]
                },
            },
            {
                "id": 7,
                "name": "Their Team",
                "roster": {
                    "entries": [
                        _roster_entry(
                            555,
                            17,
                            full_name="Cameron Dicker",
                            eligible=[17, 20, 21],
                        ),
                        _roster_entry(999, 21),
                    ]
                },
            },
        ]
    }


class TestFetchRosters:
    def test_every_rostered_player_returned(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value=_roster_payload()):
            rows = client.fetch_rosters()
        assert len(rows) == 5

    def test_crosswalk_resolves_gsis_id(self, client: EspnFantasyClient) -> None:
        """Rosters must join to the board on player_id, exactly as picks do."""
        with patch.object(client, "_get", return_value=_roster_payload()):
            rows = client.fetch_rosters()
        bijan = next(r for r in rows if r["espn_player_id"] == 111)
        assert bijan["player_id"] == "00-0011111"
        assert bijan["player_name"] == "Bijan Robinson"
        assert bijan["position"] == "RB"
        assert bijan["nfl_team"] == "ATL"

    def test_lineup_slot_ids_mapped_to_labels(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value=_roster_payload()):
            rows = client.fetch_rosters()
        slots = {r["espn_player_id"]: r["lineup_slot"] for r in rows}
        assert slots[111] == "RB"
        assert slots[222] == "BE"
        assert slots[-16033] == "DST"
        assert slots[555] == "K"
        assert slots[999] == "IR"

    def test_bench_and_ir_are_not_starters(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value=_roster_payload()):
            rows = client.fetch_rosters()
        starting = {r["espn_player_id"]: r["is_starter"] for r in rows}
        assert starting[111] is True
        assert starting[-16033] is True
        assert starting[222] is False  # bench
        assert starting[999] is False  # IR

    def test_missing_lineup_slot_is_not_a_starter(
        self, client: EspnFantasyClient
    ) -> None:
        payload = {"teams": [{"id": 3, "roster": {"entries": [{"playerId": 111}]}}]}
        with patch.object(client, "_get", return_value=payload):
            (row,) = client.fetch_rosters()
        assert row["is_starter"] is False
        assert row["lineup_slot"] == ""

    def test_dst_falls_back_to_espn_name_and_slots(
        self, client: EspnFantasyClient
    ) -> None:
        """A D/ST has no gsis id, so the crosswalk cannot identify it."""
        with patch.object(client, "_get", return_value=_roster_payload()):
            rows = client.fetch_rosters()
        dst = next(r for r in rows if r["espn_player_id"] == -16033)
        assert dst["player_name"] == "Ravens D/ST"
        assert dst["position"] == "DST"
        assert dst["player_id"] == ""

    def test_unknown_player_position_from_eligible_slots(
        self, client: EspnFantasyClient
    ) -> None:
        with patch.object(client, "_get", return_value=_roster_payload()):
            rows = client.fetch_rosters()
        kicker = next(r for r in rows if r["espn_player_id"] == 555)
        assert kicker["player_name"] == "Cameron Dicker"
        assert kicker["position"] == "K"

    def test_wholly_unknown_player_still_listed(
        self, client: EspnFantasyClient
    ) -> None:
        with patch.object(client, "_get", return_value=_roster_payload()):
            rows = client.fetch_rosters()
        unknown = next(r for r in rows if r["espn_player_id"] == 999)
        assert unknown["player_name"] == "ESPN player 999"
        assert unknown["position"] == ""

    def test_team_name_and_ownership(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value=_roster_payload()):
            rows = client.fetch_rosters()
        mine = [r for r in rows if r["is_mine"]]
        assert {r["espn_player_id"] for r in mine} == {111, 222, -16033}
        assert all(r["team_name"] == "My Team" for r in mine)
        assert next(r for r in rows if r["espn_player_id"] == 555)["is_mine"] is False

    def test_injury_and_acquisition_passed_through(
        self, client: EspnFantasyClient
    ) -> None:
        with patch.object(client, "_get", return_value=_roster_payload()):
            rows = client.fetch_rosters()
        bijan = next(r for r in rows if r["espn_player_id"] == 111)
        assert bijan["injury_status"] == "ACTIVE"
        assert bijan["acquisition_type"] == "DRAFT"
        assert (
            next(r for r in rows if r["espn_player_id"] == 999)["injury_status"] == ""
        )

    def test_requests_the_right_views(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value={}) as mock_get:
            client.fetch_rosters()
        assert mock_get.call_args[0][0] == ["mRoster", "mTeam"]
        assert mock_get.call_args[1]["scoringPeriodId"] is None

    def test_week_becomes_scoring_period(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value={}) as mock_get:
            client.fetch_rosters(week=4)
        assert mock_get.call_args[1]["scoringPeriodId"] == 4

    def test_empty_payload(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value={}):
            assert client.fetch_rosters() == []

    def test_team_without_a_roster(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value={"teams": [{"id": 3}]}):
            assert client.fetch_rosters() == []


# ---------------------------------------------------------------------------
# Weekly scoring (?view=mBoxscore)
# ---------------------------------------------------------------------------


def _stats(week: int, actual: float | None, projected: float | None) -> list[dict]:
    """ESPN stat rows for one scoring period, plus a season-total distractor."""
    rows = [{"scoringPeriodId": 0, "statSourceId": 0, "appliedTotal": 999.9}]
    if actual is not None:
        rows.append(
            {"scoringPeriodId": week, "statSourceId": 0, "appliedTotal": actual}
        )
    if projected is not None:
        rows.append(
            {"scoringPeriodId": week, "statSourceId": 1, "appliedTotal": projected}
        )
    return rows


def _boxscore_payload() -> dict:
    """A minimal mBoxscore response: one week-1 matchup, one week-2 matchup."""
    return {
        "teams": [{"id": 3, "name": "My Team"}, {"id": 7, "name": "Their Team"}],
        "schedule": [
            {
                "matchupPeriodId": 1,
                "home": {
                    "teamId": 3,
                    "totalPoints": 121.5,
                    "rosterForCurrentScoringPeriod": {
                        "entries": [
                            _roster_entry(111, 2, stats=_stats(1, 22.4, 15.1)),
                            _roster_entry(222, 20, stats=_stats(1, None, 14.0)),
                        ]
                    },
                },
                "away": {
                    "teamId": 7,
                    "totalPoints": 98.0,
                    "rosterForCurrentScoringPeriod": {
                        "entries": [_roster_entry(555, 17, stats=_stats(1, 9.0, None))]
                    },
                },
            },
            {
                "matchupPeriodId": 2,
                "home": {
                    "teamId": 3,
                    "totalPoints": 0,
                    "rosterForCurrentScoringPeriod": {
                        "entries": [_roster_entry(999, 4, stats=_stats(2, 5.0, 5.0))]
                    },
                },
                "away": {"teamId": 7, "totalPoints": 0},
            },
        ],
    }


class TestFetchBoxscore:
    def test_only_the_requested_week(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value=_boxscore_payload()):
            rows = client.fetch_boxscore(week=1)
        assert {r["espn_player_id"] for r in rows} == {111, 222, 555}
        assert all(r["week"] == 1 for r in rows)

    def test_actual_and_projected_points(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value=_boxscore_payload()):
            rows = client.fetch_boxscore(week=1)
        bijan = next(r for r in rows if r["espn_player_id"] == 111)
        assert bijan["actual_points"] == 22.4
        assert bijan["projected_points"] == 15.1

    def test_unplayed_game_has_no_actual(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value=_boxscore_payload()):
            rows = client.fetch_boxscore(week=1)
        jj = next(r for r in rows if r["espn_player_id"] == 222)
        assert jj["actual_points"] is None
        assert jj["projected_points"] == 14.0

    def test_missing_projection_is_none(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value=_boxscore_payload()):
            rows = client.fetch_boxscore(week=1)
        kicker = next(r for r in rows if r["espn_player_id"] == 555)
        assert kicker["actual_points"] == 9.0
        assert kicker["projected_points"] is None

    def test_season_totals_are_not_mistaken_for_a_week(
        self, client: EspnFantasyClient
    ) -> None:
        """stats[] carries season rows (scoringPeriodId 0) in the same list."""
        with patch.object(client, "_get", return_value=_boxscore_payload()):
            rows = client.fetch_boxscore(week=1)
        assert all(r["actual_points"] != 999.9 for r in rows)

    def test_identity_and_ownership_match_rosters(
        self, client: EspnFantasyClient
    ) -> None:
        with patch.object(client, "_get", return_value=_boxscore_payload()):
            rows = client.fetch_boxscore(week=1)
        bijan = next(r for r in rows if r["espn_player_id"] == 111)
        assert bijan["player_id"] == "00-0011111"
        assert bijan["position"] == "RB"
        assert bijan["lineup_slot"] == "RB"
        assert bijan["is_starter"] is True
        assert bijan["is_mine"] is True
        assert bijan["team_name"] == "My Team"
        assert next(r for r in rows if r["espn_player_id"] == 555)["is_mine"] is False

    def test_team_points_carried_through(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value=_boxscore_payload()):
            rows = client.fetch_boxscore(week=1)
        assert next(r for r in rows if r["team_id"] == "3")["team_points"] == 121.5
        assert next(r for r in rows if r["team_id"] == "7")["team_points"] == 98.0

    def test_completed_week_uses_matchup_period_roster(
        self, client: EspnFantasyClient
    ) -> None:
        """A finished week comes back under rosterForMatchupPeriod instead."""
        payload = {
            "schedule": [
                {
                    "matchupPeriodId": 1,
                    "home": {
                        "teamId": 3,
                        "rosterForMatchupPeriod": {
                            "entries": [
                                _roster_entry(111, 2, stats=_stats(1, 18.0, 12.0))
                            ]
                        },
                    },
                }
            ]
        }
        with patch.object(client, "_get", return_value=payload):
            (row,) = client.fetch_boxscore(week=1)
        assert row["actual_points"] == 18.0

    def test_falls_back_to_whole_schedule_when_periods_do_not_line_up(
        self, client: EspnFantasyClient
    ) -> None:
        """Multi-week matchup periods mean matchupPeriodId != scoringPeriodId."""
        payload = {
            "schedule": [
                {
                    "matchupPeriodId": 9,
                    "home": {
                        "teamId": 3,
                        "rosterForCurrentScoringPeriod": {
                            "entries": [
                                _roster_entry(111, 2, stats=_stats(1, 18.0, 12.0))
                            ]
                        },
                    },
                }
            ]
        }
        with patch.object(client, "_get", return_value=payload):
            rows = client.fetch_boxscore(week=1)
        assert [r["espn_player_id"] for r in rows] == [111]

    def test_fallback_does_not_double_count_a_player(
        self, client: EspnFantasyClient
    ) -> None:
        """The schedule-wide fallback can show one player in two matchups."""
        game = {
            "matchupPeriodId": 9,
            "home": {
                "teamId": 3,
                "rosterForCurrentScoringPeriod": {
                    "entries": [_roster_entry(111, 2, stats=_stats(1, 18.0, 12.0))]
                },
            },
        }
        with patch.object(client, "_get", return_value={"schedule": [game, game]}):
            rows = client.fetch_boxscore(week=1)
        assert len(rows) == 1

    def test_requests_the_right_views_and_week(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value={}) as mock_get:
            client.fetch_boxscore(week=6)
        assert mock_get.call_args[0][0] == ["mBoxscore", "mMatchupScore", "mTeam"]
        assert mock_get.call_args[1]["scoringPeriodId"] == 6

    def test_empty_schedule(self, client: EspnFantasyClient) -> None:
        with patch.object(client, "_get", return_value={}):
            assert client.fetch_boxscore(week=1) == []

    def test_bye_week_side_without_a_roster(self, client: EspnFantasyClient) -> None:
        payload = {"schedule": [{"matchupPeriodId": 1, "home": {"teamId": 3}}]}
        with patch.object(client, "_get", return_value=payload):
            assert client.fetch_boxscore(week=1) == []


# ---------------------------------------------------------------------------
# Roster helpers
# ---------------------------------------------------------------------------


class TestRosterHelpers:
    @pytest.mark.parametrize(
        ("eligible", "expected"),
        [
            ([2, 3, 23, 20, 21], "RB"),
            ([4, 3, 5, 23, 20, 21], "WR"),
            ([6, 5, 23, 20, 21], "TE"),
            ([0, 7, 20, 21], "QB"),
            ([16, 20, 21], "DST"),
            ([17, 20, 21], "K"),
            ([9, 11, 15, 20, 21], "DL"),
            ([10, 11, 15, 20, 21], "LB"),
            ([13, 14, 15, 20, 21], "DB"),
        ],
    )
    def test_position_from_eligible_slots(
        self, eligible: list[int], expected: str
    ) -> None:
        assert _position_from_slots(eligible) == expected

    def test_combination_slots_alone_identify_nothing(self) -> None:
        """FLEX and BE say where a player sits, not what he is."""
        assert _position_from_slots([23, 20, 21]) == ""

    def test_no_eligible_slots(self) -> None:
        assert _position_from_slots(None) == ""
        assert _position_from_slots([]) == ""

    def test_team_names_from_modern_payload(self) -> None:
        payload = {"teams": [{"id": 3, "name": "My Team"}]}
        assert _team_names(payload) == {"3": "My Team"}

    def test_team_names_from_legacy_location_nickname(self) -> None:
        """Seasons before 2023 split the name across two fields."""
        payload = {"teams": [{"id": 7, "location": "Team", "nickname": "Seven"}]}
        assert _team_names(payload) == {"7": "Team Seven"}

    def test_team_names_empty_payload(self) -> None:
        assert _team_names({}) == {}

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("pk", "K"), ("D/ST", "DST"), ("def", "DST"), ("wr", "WR"), ("", "")],
    )
    def test_normalise_position(self, raw: str, expected: str) -> None:
        assert _normalise_position(raw) == expected

    def test_week_points_ignores_other_weeks(self) -> None:
        player = {"stats": _stats(3, 11.0, 9.0) + _stats(4, 22.0, 18.0)}
        assert _week_points(player, 3) == (11.0, 9.0)
        assert _week_points(player, 4) == (22.0, 18.0)

    def test_week_points_without_stats(self) -> None:
        assert _week_points({}, 1) == (None, None)

    def test_week_points_ignores_null_totals(self) -> None:
        player = {
            "stats": [
                {"scoringPeriodId": 1, "statSourceId": 0, "appliedTotal": None},
                {"scoringPeriodId": 1, "statSourceId": 1, "appliedTotal": 8.0},
            ]
        }
        assert _week_points(player, 1) == (None, 8.0)


# ---------------------------------------------------------------------------
# Query parameters
# ---------------------------------------------------------------------------


class TestGetQueryParams:
    def test_extra_params_reach_the_url(self, client: EspnFantasyClient) -> None:
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"ok": True}).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            client._get(["mBoxscore"], scoringPeriodId=7)

        url = mock_open.call_args[0][0].full_url
        assert "view=mBoxscore" in url
        assert "scoringPeriodId=7" in url

    def test_none_params_are_omitted(self, client: EspnFantasyClient) -> None:
        """fetch_rosters() with no week must not ask for scoringPeriodId=None."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"ok": True}).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            client._get(["mRoster"], scoringPeriodId=None)

        assert "scoringPeriodId" not in mock_open.call_args[0][0].full_url


# ---------------------------------------------------------------------------
# Team abbreviations
# ---------------------------------------------------------------------------


class TestTeamNormalisation:
    @pytest.mark.parametrize(
        ("crosswalk", "nflverse"),
        [
            ("SFO", "SF"),
            ("GBP", "GB"),
            ("JAC", "JAX"),
            ("KCC", "KC"),
            ("LVR", "LV"),
            ("NEP", "NE"),
            ("NOS", "NO"),
            ("TBB", "TB"),
        ],
    )
    def test_mfl_style_codes_become_nflverse(
        self, crosswalk: str, nflverse: str
    ) -> None:
        assert _normalise_team(crosswalk) == nflverse

    @pytest.mark.parametrize("code", ["ATL", "MIN", "BUF", "LAC", "WAS", "NYG"])
    def test_shared_codes_pass_through(self, code: str) -> None:
        assert _normalise_team(code) == code

    @pytest.mark.parametrize("code", ["LAR", "RAM", "STL"])
    def test_rams_resolve_to_nflverses_la(self, code: str) -> None:
        """nflverse writes the Rams 'LA' throughout, never 'LAR'."""
        assert _normalise_team(code) == "LA"

    @pytest.mark.parametrize(("code", "expected"), [("OAK", "LV"), ("SDC", "LAC")])
    def test_relocated_franchises_resolve_to_the_current_holder(
        self, code: str, expected: str
    ) -> None:
        assert _normalise_team(code) == expected

    @pytest.mark.parametrize("code", ["FA", "FA*"])
    def test_free_agent_is_not_a_team(self, code: str) -> None:
        assert _normalise_team(code) == ""

    def test_empty_and_unknown_survive(self) -> None:
        assert _normalise_team("") == ""
        assert _normalise_team("ZZZ") == "ZZZ"

    def test_rosters_emit_nflverse_codes(self, client: EspnFantasyClient) -> None:
        """A roster's nfl_team must match the board's, or no join can work."""
        client._player_map = {
            111: {
                "player_id": "00-0011111",
                "name": "Brock Purdy",
                "position": "QB",
                "team": "SFO",
            }
        }
        payload = {"teams": [{"id": 3, "roster": {"entries": [_roster_entry(111, 0)]}}]}
        with patch.object(client, "_get", return_value=payload):
            (row,) = client.fetch_rosters()
        assert row["nfl_team"] == "SF"

    def test_picks_emit_nflverse_codes(self, client: EspnFantasyClient) -> None:
        client._player_map = {
            111: {
                "player_id": "00-0011111",
                "name": "Josh Jacobs",
                "position": "RB",
                "team": "GBP",
            }
        }
        payload = {
            "draftDetail": {
                "picks": [
                    {"teamId": 3, "playerId": 111, "roundId": 1, "roundPickNumber": 1}
                ]
            }
        }
        with patch.object(client, "_get", return_value=payload):
            (pick,) = client.fetch_all_picks()
        assert pick["nfl_team"] == "GB"
