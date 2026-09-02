"""
Tests verifying the three bug fixes:
  1. train_model.py: missing comma causing string concat in drop_exact list
  2. predict_week.py: hardcoded 2024 season start date
  3. train_model.py: metadata always saving position as "WR"
"""

import ast
import datetime
from pathlib import Path

from nfl_predict.predict_week import get_default_season_and_week

# Used by the AST-based tests to read source files directly
SRC = Path(__file__).parent.parent / "src"


# ---------------------------------------------------------------------------
# Bug 1: Missing comma between "fantasy_points_custom" and "fantasy_points_ppr"
# ---------------------------------------------------------------------------


def test_drop_exact_no_concatenated_string():
    """
    The two strings must appear as separate entries, never concatenated.
    Parses the source file with AST to check the list literally.
    """
    source = (SRC / "nfl_predict" / "train_model.py").read_text()
    tree = ast.parse(source)

    concatenated = "fantasy_points_customfantasy_points_ppr"
    found_concat = False
    found_custom = False
    found_ppr = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == concatenated:
                found_concat = True
            if node.value == "fantasy_points_custom":
                found_custom = True
            if node.value == "fantasy_points_ppr":
                found_ppr = True

    assert not found_concat, (
        "Found concatenated string 'fantasy_points_customfantasy_points_ppr' — "
        "missing comma between the two list entries."
    )
    assert found_custom, (
        "Expected 'fantasy_points_custom' as a standalone string in drop_exact."
    )
    assert found_ppr, (
        "Expected 'fantasy_points_ppr' as a standalone string in drop_exact."
    )


# ---------------------------------------------------------------------------
# Bug 2: Hardcoded 2024 season start date
# ---------------------------------------------------------------------------


def test_season_start_2024():
    """2024 opener was 2024-09-05 (first Thursday >= 5th in September 2024)."""
    season, week = get_default_season_and_week(today=datetime.date(2024, 9, 5))
    assert season == 2024
    assert week == 1


def test_season_start_2025():
    """
    2025 opener: first Thursday >= Sep 5 in 2025.
    Sep 1 2025 = Monday; Sep 4 = Thursday but day < 5; next = Sep 11.
    """
    d = datetime.date(2025, 9, 1)
    while d.day < 5 or d.weekday() != 3:
        d += datetime.timedelta(days=1)
    assert d == datetime.date(2025, 9, 11), "Sanity: 2025 opener should be Sep 11"

    season, week = get_default_season_and_week(today=d)
    assert season == 2025, f"Expected season 2025, got {season}"
    assert week == 1, f"Expected week 1, got {week}"


def test_season_not_stuck_at_2024_for_future_dates():
    """A date well into 2026 must not resolve to season 2024."""
    season, week = get_default_season_and_week(today=datetime.date(2026, 10, 1))
    assert season == 2026, f"Expected season 2026, got {season}"
    assert 1 <= week <= 22


def test_week_advances_correctly():
    """Week count should increase by 1 every 7 days from the opener."""
    opener_2024 = datetime.date(2024, 9, 5)
    _, week1 = get_default_season_and_week(today=opener_2024)
    _, week2 = get_default_season_and_week(
        today=opener_2024 + datetime.timedelta(days=7)
    )
    _, week3 = get_default_season_and_week(
        today=opener_2024 + datetime.timedelta(days=14)
    )
    assert week1 == 1
    assert week2 == 2
    assert week3 == 3


def test_january_resolves_to_previous_season():
    """January 2025 should map to the 2024 season (playoffs)."""
    season, week = get_default_season_and_week(today=datetime.date(2025, 1, 15))
    assert season == 2024, f"Expected season 2024, got {season}"


# ---------------------------------------------------------------------------
# Bug 3: Metadata always saves position as "WR"
# ---------------------------------------------------------------------------


def test_metadata_position_uses_variable():
    """
    In train_position_model(), the meta dict must use the `position` variable,
    not the hardcoded string "WR". Check via AST on the generalized function.
    """
    source = (SRC / "nfl_predict" / "train_model.py").read_text()
    tree = ast.parse(source)

    target_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train_position_model":
            target_func = node
            break

    assert target_func is not None, (
        "Function 'train_position_model' not found in train_model.py"
    )

    hardcoded_wr_in_meta = False
    for node in ast.walk(target_func):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=False):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "position"
                    and isinstance(value, ast.Constant)
                    and value.value == "WR"
                ):
                    hardcoded_wr_in_meta = True

    assert not hardcoded_wr_in_meta, (
        "In train_position_model(), meta['position'] is hardcoded to 'WR' "
        "instead of using the `position` variable."
    )


class TestBoardRecordsNaNHandling:
    """NaN is truthy in Jinja and renders as the literal string 'nan', so a
    missing value would slip past an `{% if row.col %}` guard in the draft
    templates instead of falling back to a placeholder."""

    def test_nan_becomes_none(self) -> None:
        import pandas as pd

        from nfl_predict.draft_api import _records

        df = pd.DataFrame(
            {
                "player_name": ["A", "B"],
                "proj_games_p50": [14.0, float("nan")],
                "adp": [float("nan"), 3.0],
            }
        )
        records = _records(df)
        assert records[1]["proj_games_p50"] is None
        assert records[0]["adp"] is None
        # Present values must survive untouched.
        assert records[0]["proj_games_p50"] == 14.0
        assert records[1]["adp"] == 3.0

    def test_no_nan_survives_any_column(self) -> None:
        import pandas as pd

        from nfl_predict.draft_api import _records

        df = pd.DataFrame({"a": [float("nan")], "b": [None], "c": ["x"]})
        for value in _records(df)[0].values():
            assert value is None or value == "x"


# ---------------------------------------------------------------------------
# Bug: a refresh died merging a stored parquet with a fresh nflverse load
# ---------------------------------------------------------------------------


class TestMergeDtypeAlignment:
    """
    nflverse changed jersey_number and draft_number from string to float
    between releases. Concatenating the stored parquet with a fresh load then
    produced an object column holding both, which pyarrow refuses to write —
    so `update-all` aborted on columns no model reads, leaving rosters,
    injuries, schedules and team stats stale.
    """

    @staticmethod
    def _merged(stored, fresh):
        import pandas as pd

        from nfl_predict.fetch_nfl_data import _align_dtypes

        a, b = _align_dtypes(stored, fresh)
        return pd.concat([a, b], ignore_index=True)

    def test_numeric_strings_merge_and_write(self, tmp_path):
        """The real failure: '9' stored, 69.0 fresh."""
        import pandas as pd

        stored = pd.DataFrame({"season": [2024], "jersey_number": ["9"]})
        fresh = pd.DataFrame({"season": [2025], "jersey_number": [69.0]})

        combined = self._merged(stored, fresh)

        assert pd.api.types.is_numeric_dtype(combined["jersey_number"])
        combined.to_parquet(tmp_path / "out.parquet", index=False)

    def test_values_survive_the_conversion(self):
        import pandas as pd

        stored = pd.DataFrame({"season": [2024, 2024], "n": ["9", "18"]})
        fresh = pd.DataFrame({"season": [2025], "n": [69.0]})

        assert self._merged(stored, fresh)["n"].tolist() == [9.0, 18.0, 69.0]

    def test_non_numeric_strings_fall_back_to_text(self, tmp_path):
        """A column that will not convert must not be silently NaN'd."""
        import pandas as pd

        stored = pd.DataFrame({"season": [2024], "code": ["ABC"]})
        fresh = pd.DataFrame({"season": [2025], "code": [12.0]})

        combined = self._merged(stored, fresh)

        assert combined["code"].tolist() == ["ABC", "12.0"]
        combined.to_parquet(tmp_path / "out.parquet", index=False)

    def test_nulls_do_not_trigger_the_string_fallback(self):
        """A missing jersey number is not a failed conversion."""
        import pandas as pd

        stored = pd.DataFrame({"season": [2024, 2024], "n": ["9", None]})
        fresh = pd.DataFrame({"season": [2025], "n": [69.0]})

        assert pd.api.types.is_numeric_dtype(self._merged(stored, fresh)["n"])

    def test_int_versus_float_is_left_alone(self):
        """Both numeric — concat widens these safely, so don't touch them."""
        import pandas as pd

        stored = pd.DataFrame({"season": [2024], "years_exp": [3.0]})
        fresh = pd.DataFrame({"season": [2025], "years_exp": [4]})

        assert self._merged(stored, fresh)["years_exp"].tolist() == [3.0, 4.0]

    def test_matching_dtypes_are_untouched(self):
        import pandas as pd

        stored = pd.DataFrame({"season": [2024], "name": ["J.Allen"]})
        fresh = pd.DataFrame({"season": [2025], "name": ["B.Robinson"]})

        assert self._merged(stored, fresh)["name"].tolist() == ["J.Allen", "B.Robinson"]

    def test_inputs_are_not_mutated(self):
        """The caller's frames must survive — _merge_and_save reuses them."""
        import pandas as pd

        stored = pd.DataFrame({"season": [2024], "n": ["9"]})
        fresh = pd.DataFrame({"season": [2025], "n": [69.0]})

        from nfl_predict.fetch_nfl_data import _align_dtypes

        _align_dtypes(stored, fresh)

        assert stored["n"].tolist() == ["9"]
        assert pd.api.types.is_object_dtype(stored["n"])
