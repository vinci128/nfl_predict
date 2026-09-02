"""
The autodraft queue is what stands in for you on a pick you miss.

Its whole job is to still be useful deep into a draft, so the tests are about
depth and balance rather than ordering alone.
"""

from __future__ import annotations

import pandas as pd

from nfl_predict.draft_queue import DEPTH_PER_TARGET, build_queue, write_queue
from nfl_predict.leagues import get_profile


def _board(per_position: int = 30) -> pd.DataFrame:
    rows = []
    rank = 0
    # Interleaved so no position is trivially first.
    for i in range(per_position):
        for pos in ("RB", "WR", "QB", "TE", "DST", "K"):
            rank += 1
            rows.append(
                {
                    "overall_rank": rank,
                    "player_name": f"{pos}{i}",
                    "position": pos,
                    "team": "ATL",
                    "proj_p50": 300 - rank,
                    "vor": 200 - rank,
                    "adp": rank,
                }
            )
    return pd.DataFrame(rows)


class TestDepth:
    def test_queue_is_deeper_than_the_roster(self) -> None:
        """One name per slot is not a queue — it empties in two rounds."""
        profile = get_profile("hoh")
        q = build_queue(_board(), profile)
        assert len(q) > profile.roster.roster_size

    def test_each_position_gets_several_alternatives(self) -> None:
        profile = get_profile("hoh")
        q = build_queue(_board(), profile)
        counts = q["position"].value_counts()
        assert counts["RB"] == profile.roster_targets["RB"] * DEPTH_PER_TARGET
        assert counts["WR"] == profile.roster_targets["WR"] * DEPTH_PER_TARGET

    def test_a_thin_target_still_gets_headroom(self) -> None:
        """TE targets 2; two names would be gone long before ESPN got there."""
        q = build_queue(_board(), get_profile("hoh"))
        assert (q["position"] == "TE").sum() >= 5

    def test_depth_caps_the_total(self) -> None:
        q = build_queue(_board(), get_profile("hoh"), depth=20)
        assert len(q) <= 20


class TestOrdering:
    def test_best_available_comes_first(self) -> None:
        q = build_queue(_board(), get_profile("hoh"))
        assert q.iloc[0]["vor"] == q["vor"].max()

    def test_skill_positions_are_ranked_by_vor(self) -> None:
        q = build_queue(_board(), get_profile("hoh"))
        skill = q[~q["position"].isin({"K", "DST"})]
        assert skill["vor"].is_monotonic_decreasing

    def test_kicker_and_defence_are_last(self) -> None:
        """Reaching one early wastes a pick — they are near-interchangeable."""
        q = build_queue(_board(), get_profile("hoh"))
        late = q[q["position"].isin({"K", "DST"})]["queue_rank"]
        assert late.min() > len(q) - 3

    def test_exactly_one_kicker_and_one_defence(self) -> None:
        q = build_queue(_board(), get_profile("hoh"))
        assert (q["position"] == "K").sum() == 1
        assert (q["position"] == "DST").sum() == 1

    def test_ranks_are_contiguous_from_one(self) -> None:
        q = build_queue(_board(), get_profile("hoh"))
        assert q["queue_rank"].tolist() == list(range(1, len(q) + 1))


class TestLeagueDifferences:
    def test_it_follows_the_league_it_is_built_for(self) -> None:
        """Royal Rumble rosters 14 to Hell or Highwater's 16."""
        hoh = build_queue(_board(), get_profile("hoh"))
        rumble = build_queue(_board(), get_profile("rumble"))
        assert len(rumble) < len(hoh)

    def test_an_idp_league_queues_defenders_not_a_defence(self) -> None:
        board = _board()
        board = pd.concat(
            [
                board,
                pd.DataFrame(
                    [
                        {
                            "overall_rank": 5,
                            "player_name": f"LB{i}",
                            "position": "LB",
                            "team": "KC",
                            "proj_p50": 250,
                            "vor": 150,
                            "adp": 40,
                        }
                        for i in range(10)
                    ]
                ),
            ]
        )
        q = build_queue(board, get_profile("ludopathy"))
        assert (q["position"] == "LB").sum() > 0
        assert (q["position"] == "DST").sum() == 0


class TestAlreadyDrafted:
    def test_drafted_players_are_left_out(self) -> None:
        board = _board()
        gone = {"RB0", "RB1", "WR0"}
        q = build_queue(board, get_profile("hoh"), drafted=gone)
        assert not set(q["player_name"]) & gone

    def test_the_queue_refills_from_below(self) -> None:
        """Removing names must not shorten the queue, only shift it down."""
        board = _board()
        full = build_queue(board, get_profile("hoh"))
        after = build_queue(board, get_profile("hoh"), drafted={"RB0", "RB1"})
        assert len(after) == len(full)


class TestExport:
    def test_writes_a_csv_named_for_the_league(self, tmp_path) -> None:
        profile = get_profile("rumble")
        path = write_queue(build_queue(_board(), profile), profile, tmp_path)
        assert path.name == "draft_queue_2026_rumble.csv"
        assert pd.read_csv(path)["queue_rank"].iloc[0] == 1

    def test_export_carries_what_you_need_at_the_table(self, tmp_path) -> None:
        profile = get_profile("hoh")
        out = pd.read_csv(write_queue(build_queue(_board(), profile), profile, tmp_path))
        for col in ("queue_rank", "player_name", "position", "vor", "adp"):
            assert col in out.columns


class TestEmptyBoard:
    def test_an_empty_board_yields_an_empty_queue(self) -> None:
        """A board that failed to build must not take the draft down with it."""
        empty = pd.DataFrame(
            {
                c: []
                for c in (
                    "overall_rank",
                    "player_name",
                    "position",
                    "team",
                    "proj_p50",
                    "vor",
                    "adp",
                )
            }
        )
        assert build_queue(empty, get_profile("hoh")).empty
