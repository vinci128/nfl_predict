"""
Provider-agnostic live draft sync.

``espn_fantasy.EspnFantasyClient`` is the sole supported provider (NFL.com
Fantasy has been folded into the ESPN app for the 2026 season).  The legacy
``nfl_fantasy.NflFantasyClient`` is retained for backward compatibility but
will emit a deprecation warning and is expected to fail at authentication.

Everything downstream — the ``nfl-sync`` CLI and the draft UI's sync button —
goes through here.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Protocol, runtime_checkable

PROVIDERS = ("espn",)
_DEPRECATED_PROVIDERS = ("nfl",)


class DraftSyncError(Exception):
    """Raised when no provider is configured, or a provider fails."""


@runtime_checkable
class DraftSyncClient(Protocol):
    """The surface both provider clients expose."""

    league_id: str

    def fetch_all_picks(self) -> list[dict]: ...

    def fetch_new_picks(self, already_recorded: int = 0) -> list[dict]: ...


def available_providers() -> list[str]:
    """Providers whose credentials are present in the environment."""
    from nfl_predict.espn_fantasy import EspnFantasyClient

    found = []
    if EspnFantasyClient.credentials_available():
        found.append("espn")
    # Legacy NFL.com provider — included only if ESPN is not available,
    # so users with only NFL_FANTASY_* vars get a clear deprecation path.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from nfl_predict.nfl_fantasy import NflFantasyClient

            if NflFantasyClient.credentials_available():
                found.append("nfl")
    except Exception:
        pass
    return found


def make_client(provider: str = "auto") -> DraftSyncClient:
    """
    Build a draft-sync client.

    ``auto`` prefers ESPN, since NFL.com Fantasy has been folded into the
    ESPN app and its OAuth2 endpoint is no longer functional.
    """
    from nfl_predict.espn_fantasy import EspnFantasyClient, EspnFantasyError

    provider = (provider or "auto").lower()

    if provider == "auto":
        found = available_providers()
        if not found:
            raise DraftSyncError(
                "No draft provider configured.\n"
                "  ESPN: set ESPN_LEAGUE_ID (plus ESPN_S2 and ESPN_SWID "
                "for a private league)\n\n"
                "NFL.com Fantasy has moved to ESPN for the 2026 season. "
                "If you previously used NFL_FANTASY_* env vars, migrate to "
                "ESPN credentials instead."
            )
        provider = found[0]

    if provider in _DEPRECATED_PROVIDERS:
        warnings.warn(
            f"The {provider!r} provider is deprecated: NFL.com Fantasy has "
            "moved to ESPN for the 2026 season.  The NFL.com API is no "
            "longer functional.  Set ESPN_LEAGUE_ID and use ESPN instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from nfl_predict.nfl_fantasy import NflFantasyClient, NflFantasyError

        try:
            return NflFantasyClient.from_env()
        except NflFantasyError as e:
            raise DraftSyncError(str(e)) from e

    if provider == "espn":
        try:
            return EspnFantasyClient.from_env()
        except EspnFantasyError as e:
            raise DraftSyncError(str(e)) from e

    raise DraftSyncError(
        f"Unknown provider {provider!r}. Choose one of: espn, or 'auto'."
    )


def poll_draft(
    client: DraftSyncClient,
    on_pick: Any,  # callable(pick_dict) -> None
    interval: int = 30,
    max_rounds: int = 20,
    initial_recorded: int = 0,
) -> None:
    """
    Poll for new picks every ``interval`` seconds, whichever provider is used.

    Provider errors are reported and retried rather than fatal — a transient
    outage mid-draft should not take the sync process down.
    """
    recorded = initial_recorded
    print(f"Polling draft (league {client.league_id}) every {interval}s…")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            try:
                for pick in client.fetch_new_picks(already_recorded=recorded):
                    on_pick(pick)
                    recorded += 1
                    if pick["round"] >= max_rounds:
                        print("Max rounds reached. Stopping poll.")
                        return
            except Exception as e:
                print(f"  Warning: {e}")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")
