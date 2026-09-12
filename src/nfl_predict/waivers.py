"""Who to claim off waivers, and who to drop for them.

Written for the week after an elimination in Royal Rumble, when a whole
roster hits the pool at once — but it is not specific to that. The pool is
worth watching in an eight-team league regardless: on 2026-09-12, with
nobody yet eliminated, Patrick Mahomes, Bo Nix and Brock Purdy were all
unowned there. League size drives this far more than anything else, and
Royal Rumble is the shallowest of the three.

The recommendation is deliberately conservative in one respect. A claim
costs a roster spot, so it is only worth making when the incoming player
beats somebody you would actually have to drop — not merely when he is
better than nobody. Every suggestion therefore names the drop.
"""

from __future__ import annotations

# A claim has to clear this margin over the player it displaces. ESPN quotes
# projections to four decimals and the honest error on any one of them is
# points, not tenths; a swap that gains less than this is churn.
MIN_UPGRADE = 1.5

# Never suggested as a drop, whatever the projection says: a settled score is
# already banked, and an IR slot is not a roster spot you are competing for.
_UNDROPPABLE_SLOTS = frozenset({"IR"})


def _startable(profile) -> set[str]:
    """Fantasy positions this league actually starts."""
    slots = {p for p, n in profile.roster.starters.items() if n > 0}
    return slots | set(profile.roster.flex_positions)


def waiver_candidates(
    league: str | None = None,
    week: int | None = None,
    limit: int = 200,
    min_upgrade: float = MIN_UPGRADE,
):
    """Free agents worth claiming, each paired with who to drop for them.

    Returns ``(upgrades, my_roster, free_agents)``. ``upgrades`` is sorted by
    gain, each entry ``{add, add_proj, drop, drop_proj, gain, position}``.
    """
    from nfl_predict.espn_fantasy import EspnFantasyClient
    from nfl_predict.leagues import fantasy_position, get_profile
    from nfl_predict.predict_week import get_default_season_and_week

    profile = get_profile(league)
    if week is None:
        week = get_default_season_and_week()[1]

    client = EspnFantasyClient.from_env(profile.key)
    roster = [r for r in client.fetch_rosters(week=week) if r["is_mine"]]
    if not roster:
        raise RuntimeError(
            f"No team flagged as yours in {profile.name} — check ESPN_TEAM_ID."
        )
    free = client.fetch_free_agents(week=week, limit=limit)

    startable = _startable(profile)

    def norm(rows):
        out = []
        for r in rows:
            pos = fantasy_position(r["position"]) or r["position"]
            if pos not in startable:
                continue  # a position this league cannot start is not an upgrade
            out.append(
                {
                    "player_name": r["player_name"],
                    "position": pos,
                    "proj": r["projected_points"],
                    "injury_status": r.get("injury_status") or "",
                    "slot": r.get("lineup_slot", "FA"),
                    "owned": r.get("percent_owned"),
                }
            )
        return [x for x in out if x["proj"] is not None]

    mine, pool = norm(roster), norm(free)
    mine.sort(key=lambda x: x["proj"])

    upgrades = []
    taken: set[str] = set()
    for fa in sorted(pool, key=lambda x: -x["proj"]):
        # The drop has to be someone at a position the claim can cover, and
        # the worst such player still on the roster.
        worst = next(
            (
                m
                for m in mine
                if m["position"] == fa["position"]
                and m["player_name"] not in taken
                and m["slot"] not in _UNDROPPABLE_SLOTS
            ),
            None,
        )
        if worst is None:
            continue
        gain = fa["proj"] - worst["proj"]
        if gain < min_upgrade:
            continue
        upgrades.append(
            {
                "add": fa["player_name"],
                "add_proj": fa["proj"],
                "add_owned": fa["owned"],
                "add_injury": fa["injury_status"],
                "drop": worst["player_name"],
                "drop_proj": worst["proj"],
                "gain": round(gain, 1),
                "position": fa["position"],
            }
        )
        taken.add(worst["player_name"])

    upgrades.sort(key=lambda x: -x["gain"])
    return upgrades, mine, pool
