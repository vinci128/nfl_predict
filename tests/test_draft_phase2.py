"""
Tests for Phase 2: draft_assistant (DraftState, mark_drafted,
suggest_best_available, render_board, save/load).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_predict.draft_assistant import (
    analyse_roster_needs,
    init_draft_state,
    load_state,
    mark_drafted,
    render_board,
    save_state,
    suggest_best_available,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_board(n_wr: int = 20, n_rb: int = 20, n_qb: int = 12) -> pd.DataFrame:
    """Build a minimal draft board mirroring build_draft_board() output."""
    rows = []
    rank = 1
    for i in range(n_qb):
        rows.append(
            {
                "overall_rank": rank,
                "tier": 1,
                "pos_rank": i + 1,
                "pos_tier": 1,
                "player_id": f"QB{i}",
                "player_name": f"QB Player {i}",
                "position": "QB",
                "team": "TM",
                "proj_p10": 200.0 - i * 5,
                "proj_p50": 350.0 - i * 10,
                "proj_p90": 500.0 - i * 15,
                "vor": 100.0 - i * 5,
                "replacement_baseline": 250.0,
                "projected_season": 2026,
            }
        )
        rank += 1
    for i in range(n_rb):
        rows.append(
            {
                "overall_rank": rank,
                "tier": 1 + i // 5,
                "pos_rank": i + 1,
                "pos_tier": 1,
                "player_id": f"RB{i}",
                "player_name": f"RB Player {i}",
                "position": "RB",
                "team": "TM",
                "proj_p10": 80.0 - i * 2,
                "proj_p50": 200.0 - i * 7,
                "proj_p90": 320.0 - i * 10,
                "vor": 80.0 - i * 4,
                "replacement_baseline": 120.0,
                "projected_season": 2026,
            }
        )
        rank += 1
    for i in range(n_wr):
        rows.append(
            {
                "overall_rank": rank,
                "tier": 1 + i // 5,
                "pos_rank": i + 1,
                "pos_tier": 1,
                "player_id": f"WR{i}",
                "player_name": f"WR Player {i}",
                "position": "WR",
                "team": "TM",
                "proj_p10": 60.0 - i * 2,
                "proj_p50": 180.0 - i * 7,
                "proj_p90": 280.0 - i * 10,
                "vor": 60.0 - i * 4,
                "replacement_baseline": 120.0,
                "projected_season": 2026,
            }
        )
        rank += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# init_draft_state
# ---------------------------------------------------------------------------


class TestInitDraftState:
    def test_available_equals_board_on_init(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "state.json")
        assert len(state.available) == len(board)

    def test_picks_empty_on_init(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "state.json")
        assert state.picks == []

    def test_current_pick_starts_at_1(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "state.json")
        assert state.current_pick == 1

    def test_league_size_stored(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, league_size=10, state_path=tmp_path / "s.json")
        assert state.league_size == 10


# ---------------------------------------------------------------------------
# mark_drafted
# ---------------------------------------------------------------------------


class TestMarkDrafted:
    def test_removes_player_from_available(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        initial_count = len(state.available)
        state = mark_drafted(state, "QB Player 0")
        assert len(state.available) == initial_count - 1

    def test_adds_to_picks(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        state = mark_drafted(state, "QB Player 0", drafter="other")
        assert len(state.picks) == 1
        assert state.picks[0].player_name == "QB Player 0"

    def test_my_pick_added_to_roster(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        state = mark_drafted(state, "RB Player 0", drafter="me")
        assert "RB Player 0" in state.my_roster.get("RB", [])

    def test_opponent_pick_not_in_my_roster(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        state = mark_drafted(state, "RB Player 0", drafter="team2")
        assert "RB Player 0" not in state.my_roster.get("RB", [])

    def test_current_pick_increments(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        state = mark_drafted(state, "QB Player 0")
        assert state.current_pick == 2

    def test_round_increments_after_league_size_picks(self, tmp_path: Path) -> None:
        board = _make_board(n_qb=15, n_rb=15, n_wr=15)
        state = init_draft_state(board, league_size=3, state_path=tmp_path / "s.json")
        state = mark_drafted(state, "QB Player 0")
        state = mark_drafted(state, "QB Player 1")
        state = mark_drafted(state, "QB Player 2")
        state = mark_drafted(state, "QB Player 3")  # pick 4 → round 2
        assert state.picks[-1].round == 2

    def test_raises_for_unknown_player(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        with pytest.raises(ValueError, match="not found"):
            mark_drafted(state, "Nonexistent Player XYZ")

    def test_case_insensitive_match(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        state = mark_drafted(state, "qb player 0")  # lowercase
        assert len(state.picks) == 1

    def test_partial_name_match(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        state = mark_drafted(state, "QB Player 0")
        assert state.picks[0].player_name == "QB Player 0"

    def test_raises_for_ambiguous_match(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        with pytest.raises(ValueError, match="Ambiguous"):
            mark_drafted(state, "QB Player")  # matches QB Player 0..11


# ---------------------------------------------------------------------------
# suggest_best_available
# ---------------------------------------------------------------------------


class TestSuggestBestAvailable:
    def test_returns_n_suggestions(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        suggestions = suggest_best_available(state, n=5)
        assert len(suggestions) <= 5

    def test_sorted_by_vor_without_needs(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        suggestions = suggest_best_available(state, n=10)
        vors = suggestions["vor"].tolist()
        assert vors == sorted(vors, reverse=True)

    def test_needs_filter_puts_position_first(self, tmp_path: Path) -> None:
        board = _make_board(n_qb=5, n_rb=10, n_wr=10)
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        suggestions = suggest_best_available(state, needs=["WR"], n=3)
        # At least the top suggestion should be a WR (highest-VOR WR > highest-VOR RB in our data)
        assert suggestions.iloc[0]["position"] == "WR"

    def test_empty_when_board_exhausted(self, tmp_path: Path) -> None:
        board = _make_board(n_qb=1, n_rb=0, n_wr=0)
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        state = mark_drafted(state, "QB Player 0")
        suggestions = suggest_best_available(state)
        assert suggestions.empty


# ---------------------------------------------------------------------------
# analyse_roster_needs
# ---------------------------------------------------------------------------


class TestAnalyseRosterNeeds:
    def test_all_positions_needed_on_empty_roster(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        needs = analyse_roster_needs(state)
        assert len(needs) > 0

    def test_filled_position_not_in_needs(self, tmp_path: Path) -> None:
        board = _make_board(n_qb=5)
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        # Draft 2 QBs (target depth = 2)
        state = mark_drafted(state, "QB Player 0", drafter="me")
        state = mark_drafted(state, "QB Player 1", drafter="me")
        needs = analyse_roster_needs(state)
        assert "QB" not in needs


# ---------------------------------------------------------------------------
# render_board
# ---------------------------------------------------------------------------


class TestRenderBoard:
    def test_returns_string(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        output = render_board(state, n=5)
        assert isinstance(output, str)

    def test_contains_pick_number(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        output = render_board(state)
        assert "Pick #1" in output

    def test_contains_my_roster_section(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        output = render_board(state)
        assert "MY ROSTER" in output

    def test_pick_number_updates_after_mark(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        state = mark_drafted(state, "QB Player 0")
        output = render_board(state)
        assert "Pick #2" in output

    def test_recent_picks_section_with_show_drafted(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "s.json")
        state = mark_drafted(state, "QB Player 0", drafter="me")
        output = render_board(state, show_drafted=True)
        assert "RECENT PICKS" in output
        assert "QB Player 0" in output


# ---------------------------------------------------------------------------
# save_state / load_state
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "state.json")
        save_state(state)
        assert (tmp_path / "state.json").exists()

    def test_roundtrip_preserves_league_size(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(
            board, league_size=10, state_path=tmp_path / "state.json"
        )
        save_state(state)
        loaded = load_state(tmp_path / "state.json")
        assert loaded.league_size == 10

    def test_roundtrip_preserves_picks(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "state.json")
        state = mark_drafted(state, "QB Player 0", drafter="me")
        state = mark_drafted(state, "RB Player 0", drafter="other")
        save_state(state)

        loaded = load_state(tmp_path / "state.json")
        assert len(loaded.picks) == 2
        assert loaded.picks[0].player_name == "QB Player 0"
        assert loaded.picks[0].drafter == "me"

    def test_roundtrip_preserves_available_count(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "state.json")
        state = mark_drafted(state, "QB Player 0")
        state = mark_drafted(state, "QB Player 1")
        save_state(state)

        loaded = load_state(tmp_path / "state.json")
        assert len(loaded.available) == len(board) - 2

    def test_roundtrip_preserves_my_roster(self, tmp_path: Path) -> None:
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "state.json")
        state = mark_drafted(state, "RB Player 0", drafter="me")
        save_state(state)

        loaded = load_state(tmp_path / "state.json")
        assert "RB Player 0" in loaded.my_roster.get("RB", [])

    def test_load_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_state(tmp_path / "nonexistent.json")

    def test_draft_continues_after_reload(self, tmp_path: Path) -> None:
        """Full round-trip: save after 3 picks, reload, draft 2 more."""
        board = _make_board()
        state = init_draft_state(board, state_path=tmp_path / "state.json")
        state = mark_drafted(state, "QB Player 0")
        state = mark_drafted(state, "QB Player 1")
        state = mark_drafted(state, "QB Player 2")
        save_state(state)

        state2 = load_state(tmp_path / "state.json")
        state2 = mark_drafted(state2, "QB Player 3")
        state2 = mark_drafted(state2, "QB Player 4")
        assert len(state2.picks) == 5
        assert state2.current_pick == 6
