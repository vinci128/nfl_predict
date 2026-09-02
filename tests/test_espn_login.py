"""
`nfl-predict espn-login` stores the cookies a private league needs.

The point of the command is that the values never reach the terminal, the
scrollback, or a log — so these tests pin the handling, not just the parsing.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nfl_predict.cli import app

runner = CliRunner()

S2 = "AEB" + "x" * 300
SWID = "1A2B3C4D-5E6F-7788-99AA-BBCCDDEEFF00"


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ESPN_S2", raising=False)
    monkeypatch.delenv("ESPN_SWID", raising=False)
    return tmp_path / ".env"


def _run(monkeypatch: pytest.MonkeyPatch, s2: str = S2, swid: str = SWID, *args: str):
    answers = iter([s2, swid])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(answers))
    return runner.invoke(app, ["espn-login", "--no-check", *args])


class TestWriting:
    def test_writes_both_cookies(self, env: Path, monkeypatch) -> None:
        assert _run(monkeypatch).exit_code == 0
        body = env.read_text()
        assert f"ESPN_S2={S2}" in body
        assert f"ESPN_SWID={{{SWID}}}" in body

    def test_swid_is_brace_wrapped(self, env: Path, monkeypatch) -> None:
        """ESPN's cookie is brace-wrapped; accept it typed either way."""
        _run(monkeypatch, swid="{" + SWID + "}")
        assert f"ESPN_SWID={{{SWID}}}" in env.read_text()

    def test_replaces_existing_values_without_duplicating(
        self, env: Path, monkeypatch
    ) -> None:
        env.write_text("ESPN_S2=stale\nESPN_SWID=stale\nOTHER=keep\n")
        _run(monkeypatch)
        body = env.read_text()
        assert body.count("ESPN_S2=") == 1
        assert "stale" not in body

    def test_leaves_unrelated_settings_alone(self, env: Path, monkeypatch) -> None:
        env.write_text("# a comment\nNFL_PREDICT_LEAGUE=hoh\n")
        _run(monkeypatch)
        body = env.read_text()
        assert "# a comment" in body
        assert "NFL_PREDICT_LEAGUE=hoh" in body

    def test_creates_the_file_when_missing(self, env: Path, monkeypatch) -> None:
        assert not env.exists()
        _run(monkeypatch)
        assert env.exists()

    def test_file_is_owner_only(self, env: Path, monkeypatch) -> None:
        """It holds a live session token, so no other account should read it."""
        _run(monkeypatch)
        assert stat.S_IMODE(env.stat().st_mode) == 0o600


class TestSecrecy:
    def test_never_echoes_the_cookies(self, env: Path, monkeypatch) -> None:
        out = _run(monkeypatch).output
        assert S2 not in out
        assert SWID not in out

    def test_reports_only_their_length(self, env: Path, monkeypatch) -> None:
        out = _run(monkeypatch).output
        assert f"espn_s2 {len(S2)} chars" in out

    def test_uses_a_hidden_prompt(self, env: Path, monkeypatch) -> None:
        """input() would leave the token in the shell's scrollback."""
        called: list[str] = []

        def fake(prompt: str) -> str:
            called.append(prompt)
            return S2 if "s2" in prompt else SWID

        monkeypatch.setattr("getpass.getpass", fake)
        runner.invoke(app, ["espn-login", "--no-check"])
        assert len(called) == 2


class TestRefusesIncompleteInput:
    @pytest.mark.parametrize(("s2", "swid"), [("", SWID), (S2, ""), ("", "")])
    def test_both_cookies_are_required(
        self, env: Path, monkeypatch, s2: str, swid: str
    ) -> None:
        result = _run(monkeypatch, s2=s2, swid=swid)
        assert result.exit_code == 1
        assert not env.exists()

    def test_a_partial_entry_does_not_clobber_good_cookies(
        self, env: Path, monkeypatch
    ) -> None:
        env.write_text("ESPN_S2=good\nESPN_SWID={GOOD}\n")
        _run(monkeypatch, swid="")
        assert "good" in env.read_text()
