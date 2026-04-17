"""
Tests verifying the three bug fixes:
  1. train_model.py: missing comma causing string concat in drop_exact list
  2. predict_week.py: hardcoded 2024 season start date
  3. train_model.py: metadata always saving position as "WR"
"""

import ast
import datetime
import sys
from pathlib import Path

from nfl_predict.predict_week import get_default_season_and_week

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))


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
