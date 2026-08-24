"""
Tests for the weekly data-refresh runner.

The real pipeline downloads NFL data and scores five positions, so every test
here injects its own steps. Nothing below touches the network or the models.
"""

from __future__ import annotations

import threading
import time

import pytest

from nfl_predict import weekly_api


@pytest.fixture(autouse=True)
def reset_state():
    """Each test starts from idle — the runner state is module-level."""
    weekly_api._update_state.update(
        status="idle", step="", started_at=None, finished_at=None, error=None
    )
    yield
    weekly_api._update_state.update(
        status="idle", step="", started_at=None, finished_at=None, error=None
    )


def _recorder(sink: list[str], label: str):
    """A zero-arg step that records it ran (a default-arg lambda is not one)."""

    def step() -> None:
        sink.append(label)

    return step


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestRunUpdate:
    def test_runs_every_step_in_order(self) -> None:
        seen: list[str] = []
        steps = [(f"step {i}", _recorder(seen, f"step {i}")) for i in range(3)]

        weekly_api._run_update(steps)

        assert seen == ["step 0", "step 1", "step 2"]
        assert weekly_api._update_state["status"] == "done"
        assert weekly_api._update_state["error"] is None

    def test_records_the_current_step(self) -> None:
        observed: list[str] = []

        def peek() -> None:
            observed.append(weekly_api._update_state["step"])

        weekly_api._run_update([("Fetching NFL data", peek), ("Predicting QB", peek)])

        assert observed == ["Fetching NFL data", "Predicting QB"]

    def test_failure_is_reported_not_swallowed(self) -> None:
        def boom() -> None:
            raise RuntimeError("nflverse is down")

        weekly_api._run_update([("Fetching NFL data", boom)])

        assert weekly_api._update_state["status"] == "error"
        assert "nflverse is down" in weekly_api._update_state["error"]
        assert "RuntimeError" in weekly_api._update_state["error"]

    def test_later_steps_are_skipped_after_a_failure(self) -> None:
        ran: list[str] = []

        def boom() -> None:
            raise ValueError("bad parquet")

        weekly_api._run_update(
            [("fetch", boom), ("predict", _recorder(ran, "predict"))]
        )

        assert ran == []
        assert weekly_api._update_state["status"] == "error"


class TestStartUpdate:
    def test_starts_a_run_and_reports_done(self) -> None:
        assert weekly_api.start_update([("noop", lambda: None)]) is True
        assert _wait_for(lambda: weekly_api._update_state["status"] == "done")

    def test_second_start_is_refused_while_running(self) -> None:
        release = threading.Event()

        assert weekly_api.start_update([("slow", release.wait)]) is True
        assert _wait_for(lambda: weekly_api._update_state["status"] == "running")

        # The pipeline rewrites the same parquet files from every step, so a
        # concurrent run would race on them.
        assert weekly_api.start_update([("other", lambda: None)]) is False

        release.set()
        assert _wait_for(lambda: weekly_api._update_state["status"] == "done")

    def test_a_finished_run_can_be_started_again(self) -> None:
        assert weekly_api.start_update([("noop", lambda: None)]) is True
        assert _wait_for(lambda: weekly_api._update_state["status"] == "done")
        assert weekly_api.start_update([("noop", lambda: None)]) is True


class TestUpdateStatus:
    def test_idle_status_has_no_elapsed(self) -> None:
        s = weekly_api.update_status()
        assert s["status"] == "idle"
        assert s["elapsed"] is None

    def test_running_status_reports_elapsed_seconds(self) -> None:
        weekly_api._update_state.update(status="running", started_at=time.time() - 42)
        assert weekly_api.update_status()["elapsed"] == pytest.approx(42, abs=1)

    def test_reports_prediction_file_count(self, monkeypatch) -> None:
        monkeypatch.setattr(weekly_api, "_latest_prediction_files", lambda: [])
        s = weekly_api.update_status()
        assert s["n_prediction_files"] == 0
        assert s["predictions_at"] is None


class TestSteps:
    def test_pipeline_covers_every_weekly_position(self) -> None:
        labels = [label for label, _ in weekly_api._update_steps()]
        for pos in weekly_api.UPDATE_POSITIONS:
            assert f"Predicting {pos}" in labels

    def test_pipeline_does_not_retrain(self) -> None:
        """Training is minutes slower and the weekly page does not need it."""
        labels = " ".join(label for label, _ in weekly_api._update_steps()).lower()
        assert "train" not in labels


class TestTimeago:
    @pytest.mark.parametrize(
        ("delta", "expected"),
        [(5, "5s ago"), (120, "2m ago"), (7200, "2h ago"), (172800, "2d ago")],
    )
    def test_formats_relative_age(self, delta: int, expected: str) -> None:
        assert weekly_api._timeago(time.time() - delta) == expected

    def test_missing_timestamp_reads_as_never(self) -> None:
        assert weekly_api._timeago(None) == "never"


class TestUpdateEndpoints:
    """
    Route wiring. `_update_steps` is patched out everywhere here — a real run
    downloads NFL data and rescores five positions.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from nfl_predict.api import app

        return TestClient(app)

    def test_post_starts_a_run_and_returns_the_panel(self, client, monkeypatch) -> None:
        release = threading.Event()
        monkeypatch.setattr(
            weekly_api, "_update_steps", lambda: [("Fetching NFL data", release.wait)]
        )

        r = client.post("/weekly/update")
        assert r.status_code == 200
        assert 'id="update-panel"' in r.text
        # While running the panel polls itself; that trigger is what advances it.
        assert 'hx-trigger="every 2s"' in r.text

        release.set()
        assert _wait_for(lambda: weekly_api._update_state["status"] == "done")

    def test_finished_panel_stops_polling(self, client, monkeypatch) -> None:
        monkeypatch.setattr(
            weekly_api, "_update_steps", lambda: [("noop", lambda: None)]
        )
        client.post("/weekly/update")
        assert _wait_for(lambda: weekly_api._update_state["status"] == "done")

        r = client.get("/weekly/update")
        assert 'hx-trigger="every 2s"' not in r.text
        assert "Refresh finished" in r.text

    def test_failure_is_shown_on_the_panel(self, client, monkeypatch) -> None:
        def boom() -> None:
            raise RuntimeError("nflverse is down")

        monkeypatch.setattr(weekly_api, "_update_steps", lambda: [("fetch", boom)])
        client.post("/weekly/update")
        assert _wait_for(lambda: weekly_api._update_state["status"] == "error")

        r = client.get("/weekly/update")
        assert "Refresh failed" in r.text
        assert "nflverse is down" in r.text

    def test_weekly_page_renders_the_panel(self, client) -> None:
        r = client.get("/weekly")
        assert r.status_code == 200
        assert 'id="update-panel"' in r.text
        assert "Refresh data" in r.text
