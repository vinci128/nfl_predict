"""
League profiles — scoring rules and roster settings, one profile per league.

Everything downstream (fantasy point calculation, season models, VOR draft
board, draft UI) is scored *for a specific league*. Two leagues that differ
only in passing yardage produce different targets, so they need separately
trained models and separate artifact paths; a profile owns both the rules and
the paths its artifacts live under.

Add a league by appending a `LeagueProfile` to `PROFILES`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Stat aliases
# ---------------------------------------------------------------------------
#
# A canonical stat name maps to the raw column(s) that carry it. nflreadpy has
# renamed several of these over the years, and a few are genuinely the sum of
# per-source components (a fumble lost while rushing and one lost on a sack are
# both "fumbles lost").
#
# `sum` groups add their present members together. `first` groups are pure
# renames — only the first column that exists is used, so a release that ships
# both the old and new name cannot silently double-count.

_SUM_ALIASES: dict[str, tuple[str, ...]] = {
    "fumbles_lost": (
        "fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "sack_fumbles_lost",
    ),
    "two_point_conversions": (
        "passing_2pt_conversions",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
        "two_point_conversions",
    ),
    "return_tds": ("kick_return_tds", "punt_return_tds", "special_teams_tds"),
    # A blocked punt, PAT or FG is one ESPN category.
    "def_blocks": ("def_punt_blocks", "def_pat_blocks", "def_fg_blocks"),
}

_FIRST_ALIASES: dict[str, tuple[str, ...]] = {
    "passing_interceptions": ("passing_interceptions", "interceptions"),
    "fg_made": ("fg_made", "field_goals_made", "fgm"),
    "fg_long": ("fg_long", "field_goals_longest", "fg_longest"),
    "fg_missed": ("fg_missed",),
    "pat_made": ("pat_made", "extra_points_made", "xpmade"),
    # Offensive fumble-recovery TD (player recovers a fumble and scores).
    "fumble_recovery_tds": ("fumble_recovery_tds",),
    # Defensive TD — interception and fumble returns by a defender. Kept
    # distinct from fumble_recovery_tds so an IDP profile scoring both does
    # not pay twice for the same return.
    "def_tds": ("def_tds", "defense_tds"),
}


def stat_series(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Resolve a canonical stat name to a numeric Series, 0.0 where unavailable.

    Unknown names fall through to a plain column lookup, so any raw nflreadpy
    column can be scored without registering an alias for it first.
    """
    zero = pd.Series(0.0, index=df.index, dtype="float64")

    def col(raw: str) -> pd.Series | None:
        if raw in df.columns:
            return pd.to_numeric(df[raw], errors="coerce").fillna(0.0)
        return None

    if name in _SUM_ALIASES:
        total = zero.copy()
        for raw in _SUM_ALIASES[name]:
            s = col(raw)
            if s is not None:
                total = total + s
        return total

    for raw in _FIRST_ALIASES.get(name, (name,)):
        s = col(raw)
        if s is not None:
            return s
    return zero


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameBonus:
    """
    A per-game threshold bonus: award `points` when `stat` lands in [lo, hi).

    ESPN's "100-199 yard rushing game" style categories. These are only
    meaningful at weekly grain — applying them to a season total would pay the
    bonus once for the whole year, so scoring must run before aggregation.
    """

    stat: str
    lo: float
    points: float
    hi: float | None = None

    def award(self, df: pd.DataFrame) -> pd.Series:
        s = stat_series(df, self.stat)
        hit = s >= self.lo
        if self.hi is not None:
            hit &= s < self.hi
        return hit.astype("float64") * self.points


@dataclass(frozen=True)
class ScoringRules:
    """
    A league's scoring, as a data description rather than code.

    `per_unit` is a continuous rate (ESPN "Passing Yards 0.04"). `per_increment`
    is ESPN's bucketed form ("Every 10 passing yards = 1"), which awards whole
    points per completed increment and *floors* the remainder — 347 passing
    yards is 34 points, not 34.7. Leagues use one form or the other; scoring a
    stat under both would double-count it, which `__post_init__` rejects.
    """

    per_unit: Mapping[str, float] = field(default_factory=dict)
    per_increment: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    fg_buckets: Mapping[str, float] = field(default_factory=dict)
    bonuses: tuple[GameBonus, ...] = ()
    # Bonus categories ESPN scores that weekly player stats cannot express —
    # documented per profile so the gap is visible rather than silently zero.
    unmodelled: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        overlap = set(self.per_unit) & set(self.per_increment)
        if overlap:
            raise ValueError(
                f"stats scored both per-unit and per-increment: {sorted(overlap)}"
            )

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Fantasy points for each row. Rows are player-*games*, not seasons."""
        pts = pd.Series(0.0, index=df.index, dtype="float64")

        for stat, rate in self.per_unit.items():
            pts += stat_series(df, stat) * rate

        for stat, (step, award) in self.per_increment.items():
            pts += (stat_series(df, stat) / step).apply(math.floor) * award

        for bonus in self.bonuses:
            pts += bonus.award(df)

        pts += self._field_goal_points(df)
        return pts

    def _field_goal_points(self, df: pd.DataFrame) -> pd.Series:
        """
        Field goals, from distance buckets when present.

        Older feature tables carry only a made count and a longest-made
        distance. That fallback pays *every* make at the longest bucket's rate,
        which overstates a multi-kick game, so it is a last resort rather than
        an equivalent path.
        """
        if not self.fg_buckets:
            return pd.Series(0.0, index=df.index, dtype="float64")

        buckets = {b: stat_series(df, b) for b in self.fg_buckets}
        if sum(float(s.sum()) for s in buckets.values()) > 0:
            pts = pd.Series(0.0, index=df.index, dtype="float64")
            for bucket, award in self.fg_buckets.items():
                pts += buckets[bucket] * award
            return pts

        made = stat_series(df, "fg_made")
        long = stat_series(df, "fg_long")
        rate = pd.Series(0.0, index=df.index, dtype="float64")
        for lo, hi, bucket in _FG_FALLBACK_RANGES:
            award = self.fg_buckets.get(bucket)
            if award is None:
                continue
            band = (long >= lo) & (long < hi)
            rate[band] = award
        return made * rate


# Longest-made distance -> the bucket that distance falls in, for the fallback
# path above. Ordered low to high; the last range is open-ended.
_FG_FALLBACK_RANGES: tuple[tuple[float, float, str], ...] = (
    (0, 20, "fg_made_0_19"),
    (20, 30, "fg_made_20_29"),
    (30, 40, "fg_made_30_39"),
    (40, 50, "fg_made_40_49"),
    (50, 60, "fg_made_50_59"),
    (60, math.inf, "fg_made_60_"),
)


# ---------------------------------------------------------------------------
# Roster / draft configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RosterConfig:
    """Starting lineup and roster limits, as ESPN reports them."""

    league_size: int
    roster_size: int
    starters: Mapping[str, int]
    bench: int
    ir: int = 0
    flex_spots: int = 0
    # Positions the FLEX slot accepts, for replacement-level maths.
    flex_positions: tuple[str, ...] = ("RB", "WR", "TE")

    @property
    def positions(self) -> tuple[str, ...]:
        """Positions with at least one starting slot, board-ordered."""
        order = ["QB", "RB", "WR", "TE", "LB", "DL", "DB", "DST", "K"]
        started = {p for p, n in self.starters.items() if n > 0}
        ranked = [p for p in order if p in started]
        return tuple(ranked + sorted(started - set(ranked)))


@dataclass(frozen=True)
class LeagueProfile:
    """One league: its rules, its roster, and where its artifacts live."""

    key: str
    name: str
    season: int
    scoring: ScoringRules
    roster: RosterConfig
    positional_scarcity: Mapping[str, float]
    espn_league_id: str | None = None
    espn_team_id: str | None = None
    keepers_per_team: int = 0
    keepers_path: str | None = None
    dst_scoring: ScoringRules | None = None
    # How deep to go at each position by the end of the draft. Sums to the
    # roster size, so a draft that hits every target fills the roster exactly.
    roster_targets: Mapping[str, int] = field(default_factory=dict)
    # ESPN's own per-position maximums. A position at its cap is dropped from
    # the suggestion panel: past the cap another player there has no roster
    # value however his VOR ranks.
    roster_caps: Mapping[str, int] = field(default_factory=dict)
    # Positions whose VOR is not comparable across positions, capped in a
    # single suggestion panel. One kicker is a suggestion; five is a collapse.
    suggestion_limits: Mapping[str, int] = field(default_factory=lambda: {"K": 1})
    notes: tuple[str, ...] = ()

    # --- artifact locations -------------------------------------------------
    #
    # Scoring changes the training target, so models trained for one league are
    # not valid for another. Every generated artifact is namespaced by key.

    @property
    def features_path(self) -> Path:
        return Path("data") / "processed" / self.key / "player_week_features.parquet"

    @property
    def model_dir(self) -> Path:
        return Path("models") / self.key

    @property
    def registry_path(self) -> Path:
        return self.model_dir / "model_registry.json"

    def board_path(self, season: int, ext: str = ".csv") -> Path:
        return Path("outputs") / f"draft_board_{season}_{self.key}{ext}"

    @property
    def state_path(self) -> Path:
        return Path("outputs") / f"draft_state_{self.key}.json"

    def to_draft_settings(self):  # -> draft_board.DraftSettings
        """Build the VOR settings for this league. Imported lazily to avoid a cycle."""
        from nfl_predict.draft_board import DraftSettings

        s = self.roster.starters
        return DraftSettings(
            league_size=self.roster.league_size,
            qb_starters=s.get("QB", 0),
            rb_starters=s.get("RB", 0),
            wr_starters=s.get("WR", 0),
            te_starters=s.get("TE", 0),
            k_starters=s.get("K", 0),
            flex_spots=self.roster.flex_spots,
            scoring=self.key,
            positional_scarcity=dict(self.positional_scarcity),
            extra_starters={
                p: n for p, n in s.items() if p in ("LB", "DL", "DB", "DST") and n > 0
            },
        )


# ---------------------------------------------------------------------------
# Scoring: Ludopathy Bowl
# ---------------------------------------------------------------------------
#
# ESPN's settings summary lists only categories with a non-zero value, so a
# category absent from the page scores nothing. For this league that means no
# 50+ yard field goal value, no missed-FG penalty and no 2-point conversions —
# all three are genuinely worth zero here, not merely undisplayed.

_LUDOPATHY_SCORING = ScoringRules(
    per_increment={
        # "Every 10 yards = 1 point" — whole points per completed 10, remainder
        # dropped. Not the same as 0.1/yard: 347 yards scores 34, not 34.7.
        "passing_yards": (10.0, 1.0),
        "rushing_yards": (10.0, 1.0),
        "receiving_yards": (10.0, 1.0),
    },
    per_unit={
        "passing_tds": 4.0,
        "passing_interceptions": -2.0,
        "rushing_tds": 6.0,
        "receptions": 1.0,
        "receiving_tds": 6.0,
        "fumbles_lost": -2.0,
        "return_tds": 6.0,
        "fumble_recovery_tds": 6.0,
        "pat_made": 1.0,
        # IDP. ESPN's TK is solo tackles, TKA assists at half value.
        "def_tackles_solo": 1.0,
        "def_tackle_assists": 0.5,
        "def_pass_defended": 1.0,
        # ESPN applies these stat categories by stat id, not by roster slot, so
        # a linebacker's sack scores the same as a D/ST's. See notes below.
        "def_sacks": 1.0,
        "def_interceptions": 2.0,
        "def_fumbles_forced": 2.0,
        "def_fumbles": 2.0,
        "def_safeties": 2.0,
        "def_tds": 6.0,
    },
    fg_buckets={
        "fg_made_0_19": 3.0,
        "fg_made_20_29": 3.0,
        "fg_made_30_39": 3.0,
        "fg_made_40_49": 3.0,
        # 50+ carries no value in this league.
        "fg_made_50_59": 0.0,
        "fg_made_60_": 0.0,
    },
    bonuses=(
        GameBonus("passing_yards", lo=400, points=4.0),
        GameBonus("rushing_yards", lo=100, hi=200, points=2.0),
        GameBonus("rushing_yards", lo=200, points=3.0),
        GameBonus("receiving_yards", lo=100, hi=200, points=2.0),
        GameBonus("receiving_yards", lo=200, points=3.0),
    ),
    unmodelled=(
        "40+ yard TD pass bonus (PTD40, +2)",
        "50+ yard TD pass bonus (PTD50, +3)",
    ),
)

# ---------------------------------------------------------------------------
# Scoring: Hell or Highwater
# ---------------------------------------------------------------------------

_HOH_SCORING = ScoringRules(
    per_unit={
        "passing_yards": 0.04,
        "passing_tds": 4.0,
        "passing_interceptions": -2.0,
        "rushing_yards": 0.1,
        "rushing_tds": 6.0,
        "receiving_yards": 0.1,
        "receptions": 1.0,
        "receiving_tds": 6.0,
        "two_point_conversions": 2.0,
        "fumbles_lost": -2.0,
        "return_tds": 6.0,
        "fumble_recovery_tds": 6.0,
        "pat_made": 1.0,
        "fg_missed": -1.0,
    },
    fg_buckets={
        "fg_made_0_19": 3.0,
        "fg_made_20_29": 3.0,
        "fg_made_30_39": 3.0,
        "fg_made_40_49": 4.0,
        "fg_made_50_59": 5.0,
        "fg_made_60_": 6.0,
    },
)


# ---------------------------------------------------------------------------
# The leagues
# ---------------------------------------------------------------------------

# Team defence scores off a different stat table (team_stats plus points and
# yards allowed), so it gets its own rule set rather than sharing the player
# one. Both points-allowed and yards-allowed ladders are ordinary threshold
# bands, so they need no special machinery.

_HOH_DST_SCORING = ScoringRules(
    per_unit={
        "def_sacks": 1.0,
        "def_interceptions": 2.0,
        "def_fumbles": 2.0,
        "def_safeties": 2.0,
        "def_blocks": 2.0,
        "def_tds": 6.0,
        "special_teams_tds": 6.0,
    },
    bonuses=(
        GameBonus("points_allowed", lo=0, hi=1, points=5.0),
        GameBonus("points_allowed", lo=1, hi=7, points=4.0),
        GameBonus("points_allowed", lo=7, hi=14, points=3.0),
        GameBonus("points_allowed", lo=14, hi=18, points=1.0),
        # 18-27 allowed scores nothing.
        GameBonus("points_allowed", lo=28, hi=35, points=-1.0),
        GameBonus("points_allowed", lo=35, hi=46, points=-3.0),
        GameBonus("points_allowed", lo=46, points=-5.0),
        GameBonus("yards_allowed", lo=0, hi=100, points=5.0),
        GameBonus("yards_allowed", lo=100, hi=200, points=3.0),
        GameBonus("yards_allowed", lo=200, hi=300, points=2.0),
        # 300-349 allowed scores nothing.
        GameBonus("yards_allowed", lo=350, hi=400, points=-1.0),
        GameBonus("yards_allowed", lo=400, hi=450, points=-3.0),
        GameBonus("yards_allowed", lo=450, hi=500, points=-5.0),
        GameBonus("yards_allowed", lo=500, hi=550, points=-6.0),
        GameBonus("yards_allowed", lo=550, points=-7.0),
    ),
)


LUDOPATHY = LeagueProfile(
    key="ludopathy",
    name="Ludopathy Bowl",
    season=2026,
    scoring=_LUDOPATHY_SCORING,
    roster=RosterConfig(
        league_size=10,
        roster_size=21,
        starters={
            "QB": 1,
            "RB": 2,
            "WR": 2,
            "TE": 1,
            "LB": 3,
            "DL": 1,
            "K": 1,
        },
        bench=9,
        ir=3,
        flex_spots=1,
    ),
    # IDP starters are 4 of 12 slots against a shallow supply of startable
    # defenders, so they are not discounted the way a 1-QB league discounts QB.
    positional_scarcity={
        "QB": 0.7,
        "RB": 1.0,
        "WR": 1.0,
        "TE": 0.85,
        "LB": 1.0,
        "DL": 1.0,
        "K": 0.5,
    },
    espn_league_id="1773102615",
    espn_team_id="9",
    keepers_per_team=6,
    keepers_path="data/keepers_ludopathy_2026.txt",
    # 21 roster spots: 2+5+5+2+4+2+1.
    roster_targets={"QB": 2, "RB": 5, "WR": 5, "TE": 2, "LB": 4, "DL": 2, "K": 1},
    # ESPN maximums, except LB/DL which ESPN leaves unlimited — capped just
    # above the target so the panel cannot bury the roster in linebackers.
    roster_caps={"QB": 4, "RB": 8, "WR": 8, "TE": 3, "LB": 6, "DL": 4, "K": 1},
    notes=(
        "IDP: 3 LB + 1 DL start, 4 of 12 lineup slots.",
        "6 keepers per team (60 players) lock one hour before the draft.",
        "A 50+ yard field goal scores 0 — confirm with the league manager, "
        "it is unusual enough to look like a misconfiguration.",
        "Sacks/INTs/fumbles are scored for individual defenders on the "
        "assumption ESPN applies those categories by stat id rather than by "
        "roster slot. Verify against a real box score after week 1.",
    ),
)

HELL_OR_HIGHWATER = LeagueProfile(
    key="hoh",
    name="Hell or Highwater",
    season=2026,
    scoring=_HOH_SCORING,
    roster=RosterConfig(
        league_size=14,
        roster_size=16,
        starters={
            "QB": 1,
            "RB": 2,
            "WR": 2,
            "TE": 1,
            "DST": 1,
            "K": 1,
        },
        bench=7,
        ir=3,
        flex_spots=1,
    ),
    positional_scarcity={
        "QB": 0.7,
        "RB": 1.0,
        "WR": 1.0,
        "TE": 0.85,
        "DST": 0.5,
        "K": 0.5,
    },
    espn_league_id="581348581",
    espn_team_id="9",
    dst_scoring=_HOH_DST_SCORING,
    # 16 roster spots: 2+5+5+2+1+1.
    roster_targets={"QB": 2, "RB": 5, "WR": 5, "TE": 2, "DST": 1, "K": 1},
    roster_caps={"QB": 4, "RB": 8, "WR": 8, "TE": 3, "DST": 1, "K": 1},
    suggestion_limits={"K": 1, "DST": 1},
    notes=(
        "Standard passing yardage (0.04/yd), not the 0.1 the other league "
        "uses — QBs are worth far less here.",
        "14 teams and a 16-man roster: the player pool runs thin, replacement "
        "level is much lower than in a 10-team league.",
        "1 D/ST starter, no keepers.",
    ),
)

PROFILES: dict[str, LeagueProfile] = {p.key: p for p in (LUDOPATHY, HELL_OR_HIGHWATER)}

# Used when no league is named. Ludopathy is the league this repo's scoring was
# originally written for, so it keeps historical behaviour the default.
DEFAULT_LEAGUE = "ludopathy"

_ALIASES = {
    "ludopathy_bowl": "ludopathy",
    "ludopathybowl": "ludopathy",
    "hell_or_highwater": "hoh",
    "hellorhighwater": "hoh",
}


def get_profile(key: str | LeagueProfile | None = None) -> LeagueProfile:
    """
    Resolve a league profile by key.

    Falls back to $NFL_PREDICT_LEAGUE, then to DEFAULT_LEAGUE, so a shell can
    pin a league for a whole draft-day session without repeating the flag.
    """
    if isinstance(key, LeagueProfile):
        return key
    if key is None:
        key = os.environ.get("NFL_PREDICT_LEAGUE") or DEFAULT_LEAGUE

    norm = key.strip().lower().replace("-", "_").replace(" ", "_")
    norm = _ALIASES.get(norm, norm)
    if norm not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown league {key!r}. Known leagues: {known}")
    return PROFILES[norm]


def league_keys() -> list[str]:
    return sorted(PROFILES)


def with_season(profile: LeagueProfile, season: int) -> LeagueProfile:
    """A copy of `profile` pinned to a different season."""
    return replace(profile, season=season)


def describe(profile: LeagueProfile) -> str:
    """Human-readable summary, used by `nfl-predict leagues`."""
    r = profile.roster
    slots = "  ".join(f"{p}{n}" for p, n in r.starters.items() if n > 0)
    if r.flex_spots:
        slots += f"  FLEX{r.flex_spots}"
    lines = [
        f"{profile.name}  [{profile.key}]",
        f"  season          {profile.season}",
        f"  teams           {r.league_size}",
        f"  roster          {r.roster_size} ({r.bench} bench, {r.ir} IR)",
        f"  starters        {slots}",
        f"  positions       {', '.join(r.positions)}",
        f"  espn league id  {profile.espn_league_id or '-'}",
    ]
    if profile.keepers_per_team:
        lines.append(f"  keepers         {profile.keepers_per_team} per team")
    if profile.scoring.unmodelled:
        lines.append("  not modelled    " + "; ".join(profile.scoring.unmodelled))
    for note in profile.notes:
        lines.append(f"  note            {note}")
    return "\n".join(lines)


def score_positions(profile: LeagueProfile) -> tuple[str, ...]:
    """Positions this league needs projections for."""
    return profile.roster.positions


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------
#
# Raw roster positions are finer-grained than fantasy slots: ESPN's DL slot
# accepts DE/DT/NT, its LB slot accepts ILB/MLB/OLB. Map to the slot name so a
# board can rank one pool per slot.

_FANTASY_POSITION: dict[str, str] = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "K": "K",
    "PK": "K",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
    "LB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "OLB": "LB",
    "CB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "SAF": "DB",
    "DB": "DB",
    "DST": "DST",
    "D/ST": "DST",
}

IDP_POSITIONS = ("LB", "DL", "DB")


def fantasy_position(raw: str | float | None) -> str | None:
    """Map a raw roster position to its fantasy slot, or None if undraftable."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return _FANTASY_POSITION.get(str(raw).strip().upper())


def resolve_features_path(profile: str | LeagueProfile | None = None) -> Path:
    """
    Where this league's feature table lives.

    Falls back to the pre-league shared table when a league-specific one has
    not been built yet, so an existing checkout keeps working until the next
    `update-all`. The fallback is announced because that table was scored
    under whichever rules were current when it was written.
    """
    prof = get_profile(profile)
    if prof.features_path.exists():
        return prof.features_path
    legacy = Path("data") / "processed" / "player_week_features.parquet"
    if legacy.exists():
        print(
            f"  WARN no feature table for '{prof.key}' — falling back to {legacy}, "
            f"which may be scored under different rules. "
            f"Rebuild with: nfl-predict features --league {prof.key}"
        )
        return legacy
    return prof.features_path


def required_stat_columns(rules: ScoringRules) -> Sequence[str]:
    """Every canonical stat a rule set reads. Used to validate a feature table."""
    stats = set(rules.per_unit) | set(rules.per_increment) | set(rules.fg_buckets)
    stats.update(b.stat for b in rules.bonuses)
    return sorted(stats)
