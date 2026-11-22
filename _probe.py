from nfl_predict.espn_fantasy import EspnFantasyClient, BENCH_SLOT_ID, IR_SLOT_ID, _position_from_slots

SLOT_NAMES = {0: "QB", 1: "QB?", 2: "RB", 3: "RB?", 4: "WR", 5: "WR?", 6: "TE", 7: "FLEX",
              8: "DST", 9: "K", 10: "P", 16: "D/ST?", 17: "K?", 20: "BE", 21: "BE",
              23: "FLEX?", 24: "ER", 25: "IR", 26: "LB", 27: "DL", 28: "DB", 29: "DP"}

for key in ("ludopathy", "hoh"):
    c = EspnFantasyClient.from_env(league=key)
    slots = c.fetch_roster_slots()
    print("=" * 78)
    print("LEGA:", key.upper())
    print("=" * 78)
    starters, bench = [], []
    for e in slots["entries"]:
        sid = e["lineup_slot_id"]
        tot = (e.get("actual_points") or 0) + (e.get("projected_points") or 0) * (0 if e.get("actual_points") else 1)
        row = "slot=%-3s %-22s %-10s act=%-6s proj=%-6s" % (
            sid, e["player_name"][:22], e.get("injury_status", "?"),
            e.get("actual_points"), round(e.get("projected_points") or 0, 1))
        if sid in (20, 21):
            bench.append(row)
        else:
            starters.append(row)
    print("-- TITOLARI --")
    for r in starters: print("  " + r)
    print("-- PANCHINA --")
    for r in bench: print("  " + r)
    print()
