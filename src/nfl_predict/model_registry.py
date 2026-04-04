"""
Model registry for versioning trained CatBoost models.

Each trained model is saved as a versioned entry under models/versions/.
The registry (models/registry.json) tracks all versions, their metrics,
and which version is the current champion for each position.

The flat model path (models/{pos}_catboost.cbm) is always kept in sync
with the champion version, preserving backward compatibility.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
from pathlib import Path

from catboost import CatBoostRegressor

MODEL_DIR = Path("models")
VERSIONS_DIR = MODEL_DIR / "versions"
REGISTRY_PATH = MODEL_DIR / "registry.json"


def _short_hash(s: str, length: int = 8) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:length]


def _make_version_id(position: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h = _short_hash(f"{position}{ts}")
    return f"{position.lower()}_{ts}_{h}"


class ModelRegistry:
    """
    Tracks trained model versions and the current champion per position.

    Usage
    -----
    registry = ModelRegistry()
    version_id = registry.register("WR", model, meta)  # saves + auto-promotes
    registry.list_versions("WR")
    registry.compare("WR")
    registry.promote("wr_20250101_120000_abc12345", "WR")
    """

    def __init__(self) -> None:
        MODEL_DIR.mkdir(exist_ok=True, parents=True)
        VERSIONS_DIR.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if REGISTRY_PATH.exists():
            return json.loads(REGISTRY_PATH.read_text())
        return {}

    def _save(self, registry: dict) -> None:
        REGISTRY_PATH.write_text(json.dumps(registry, indent=2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        position: str,
        model: CatBoostRegressor,
        meta: dict,
        backtest_metrics: dict | None = None,
        auto_promote: bool = True,
    ) -> str:
        """
        Save a trained model as a new versioned entry and update the registry.

        Parameters
        ----------
        position        : e.g. "WR"
        model           : trained CatBoostRegressor
        meta            : dict with at minimum {feature_cols, cat_cols,
                          train_seasons, valid_season, valid_mae}
        backtest_metrics: optional dict from run_walk_forward_backtest()
        auto_promote    : if True, immediately promote this version to champion

        Returns
        -------
        version_id : str
        """
        version_id = _make_version_id(position)
        version_dir = VERSIONS_DIR / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "model.cbm"
        meta_path = version_dir / "meta.json"

        model.save_model(str(model_path))

        full_meta: dict = {
            **meta,
            "version_id": version_id,
            "position": position.upper(),
            "created_at": datetime.datetime.now().isoformat(),
            "model_path": str(model_path),
            "meta_path": str(meta_path),
        }
        if backtest_metrics:
            full_meta["backtest"] = backtest_metrics
        meta_path.write_text(json.dumps(full_meta, indent=2))

        # Update registry index
        registry = self._load()
        pos_entry = registry.setdefault(
            position.upper(), {"champion": None, "versions": []}
        )
        pos_entry["versions"].append(
            {
                "version_id": version_id,
                "created_at": full_meta["created_at"],
                "valid_season": meta.get("valid_season"),
                "valid_mae": meta.get("valid_mae"),
                "backtest_mae": backtest_metrics.get("mae")
                if backtest_metrics
                else None,
                "model_path": str(model_path),
                "meta_path": str(meta_path),
            }
        )
        self._save(registry)
        print(f"  Registered version {version_id}")

        if auto_promote:
            self.promote(version_id, position)

        return version_id

    def promote(self, version_id: str, position: str) -> None:
        """
        Make a versioned model the active champion.

        Copies model.cbm and meta.json to the flat models/ path so that
        existing code (predict_week, CLI) continues to work unchanged.
        """
        version_dir = VERSIONS_DIR / version_id
        src_model = version_dir / "model.cbm"
        src_meta = version_dir / "meta.json"

        if not src_model.exists():
            raise FileNotFoundError(f"Version {version_id} not found at {src_model}")

        pos = position.lower()
        shutil.copy2(src_model, MODEL_DIR / f"{pos}_catboost.cbm")
        shutil.copy2(src_meta, MODEL_DIR / f"{pos}_catboost_meta.json")

        registry = self._load()
        registry.setdefault(position.upper(), {"champion": None, "versions": []})[
            "champion"
        ] = version_id
        self._save(registry)
        print(f"  Promoted {version_id} → models/{pos}_catboost.cbm")

    def update_backtest(
        self, version_id: str, position: str, backtest_metrics: dict
    ) -> None:
        """Attach backtest results to an existing version entry."""
        registry = self._load()
        versions = registry.get(position.upper(), {}).get("versions", [])
        for v in versions:
            if v["version_id"] == version_id:
                v["backtest_mae"] = backtest_metrics.get("mae")
                break
        self._save(registry)

        # Also update the versioned meta.json
        meta_path = VERSIONS_DIR / version_id / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["backtest"] = backtest_metrics
            meta_path.write_text(json.dumps(meta, indent=2))

    def get_champion(self, position: str) -> dict | None:
        """Return the registry entry for the current champion, or None."""
        registry = self._load()
        pos_entry = registry.get(position.upper(), {})
        champion_id = pos_entry.get("champion")
        if not champion_id:
            return None
        return next(
            (
                v
                for v in pos_entry.get("versions", [])
                if v["version_id"] == champion_id
            ),
            None,
        )

    def list_versions(self, position: str | None = None) -> list[dict]:
        """Return all versions for a position (or all positions), newest first."""
        registry = self._load()
        if position:
            versions = registry.get(position.upper(), {}).get("versions", [])
        else:
            versions = [
                v for entry in registry.values() for v in entry.get("versions", [])
            ]
        return sorted(versions, key=lambda v: v["created_at"], reverse=True)

    def compare(self, position: str) -> None:
        """Print a formatted comparison table for all versions of a position."""
        versions = self.list_versions(position)
        champion = self.get_champion(position)
        champion_id = champion["version_id"] if champion else None

        print(f"\n{'=' * 72}")
        print(f"  Registered models — {position.upper()}")
        print(f"{'=' * 72}")
        print(f"{'Version ID':<42} {'Val MAE':>8} {'BT MAE':>8} {'Champ':>6}")
        print("-" * 72)
        for v in versions:
            star = "★" if v["version_id"] == champion_id else ""
            val_mae = (
                f"{v['valid_mae']:.4f}" if v.get("valid_mae") is not None else "    n/a"
            )
            bt_mae = (
                f"{v['backtest_mae']:.4f}"
                if v.get("backtest_mae") is not None
                else "    n/a"
            )
            print(f"{v['version_id']:<42} {val_mae:>8} {bt_mae:>8} {star:>6}")
        if not versions:
            print("  (no versions registered)")
        print()
