"""
NFL.com Fantasy browser automation client using Playwright.

NOTE: Automating NFL.com may violate their Terms of Service.
Use only on your own account and at your own discretion.

The client intercepts network responses from the NFL.com React app to extract
clean JSON data, and uses UI automation for actions like setting lineups and
claiming waiver wire players.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

NFL_FANTASY_BASE = "https://fantasy.nfl.com"
NFL_ID_BASE = "https://id.nfl.com"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RosterPlayer:
    player_id: str
    name: str
    position: str
    nfl_team: str
    injury_status: str = "Active"
    lineup_slot: str = "BN"  # QB, RB, WR, TE, K, DEF, BN, IR
    projected_points: float = 0.0


@dataclass
class WaiverPlayer:
    player_id: str
    name: str
    position: str
    nfl_team: str
    injury_status: str = "Active"
    percent_owned: float = 0.0
    projected_points: float = 0.0
    news: str = ""


@dataclass
class TradeOffer:
    trade_id: str
    from_team_name: str
    players_giving: list[str] = field(default_factory=list)   # players you'd give
    players_receiving: list[str] = field(default_factory=list) # players you'd get
    expires: str = ""
    status: str = "pending"


@dataclass
class Matchup:
    week: int
    my_team_name: str
    my_score: float
    opponent_team_name: str
    opponent_score: float
    my_projected: float = 0.0
    opp_projected: float = 0.0


@dataclass
class DraftPick:
    pick_number: int
    round_number: int
    team_name: str
    player_name: str
    player_position: str
    player_nfl_team: str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class NFLFantasyClient:
    """
    Playwright-based client for NFL.com Fantasy operations.

    Handles authentication, data extraction via network interception,
    and UI-based actions (lineup, waivers, trades, draft).

    Usage:
        async with NFLFantasyClient(email, password, league_id, team_id) as client:
            roster = await client.get_roster()
    """

    def __init__(
        self,
        email: str,
        password: str,
        league_id: str,
        team_id: str,
        headless: bool = False,
    ):
        self.email = email
        self.password = password
        self.league_id = league_id
        self.team_id = team_id
        self.headless = headless

        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._api_responses: dict[str, Any] = {}

    async def __aenter__(self) -> "NFLFantasyClient":
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.stop()

    async def start(self) -> None:
        """Launch browser and log in to NFL.com."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        self._page = await self._context.new_page()

        # Intercept all JSON API responses so we can read them cleanly
        self._page.on("response", self._capture_api_response)

        await self._login()

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _capture_api_response(self, response) -> None:
        """Store JSON API responses keyed by URL path for later retrieval."""
        try:
            if "fantasy.nfl.com" in response.url and response.status == 200:
                ct = response.headers.get("content-type", "")
                if "json" in ct:
                    data = await response.json()
                    # Use path as key (strip query params)
                    path = response.url.split("?")[0]
                    self._api_responses[path] = data
        except Exception:
            pass

    async def _nav(self, url: str, wait: str = "domcontentloaded") -> None:
        """Navigate and wait for page load.

        Uses domcontentloaded (not networkidle) because NFL.com's React SPA
        continuously makes background API calls so networkidle never fires.
        """
        await self._page.goto(url, wait_until=wait, timeout=30_000)
        await asyncio.sleep(2)  # let React render after DOM is ready

    async def _wait_and_fill(self, selector: str, value: str) -> None:
        await self._page.wait_for_selector(selector, timeout=15_000)
        await self._page.fill(selector, value)

    async def _login(self) -> None:
        """
        Authenticate with NFL.com.

        Strategy: navigate to the fantasy homepage, find and click the Sign In
        button, then handle whatever login form NFL.com currently uses.
        Running with headless=False lets you see (and manually complete) any
        step the automation misses.
        """
        print(f"[NFLFantasy] Logging in as {self.email}...")

        # 1. Go to the fantasy homepage — this always works
        await self._nav(f"{NFL_FANTASY_BASE}", wait="domcontentloaded")
        await asyncio.sleep(2)

        current_url = self._page.url
        print(f"[NFLFantasy] Landed on: {current_url}")

        # 2. If not already on a login page, look for a Sign In button
        if "login" not in current_url and "signin" not in current_url:
            sign_in_selectors = [
                "a:has-text('Sign In')",
                "a:has-text('Log In')",
                "button:has-text('Sign In')",
                "button:has-text('Log In')",
                "[data-testid*='signin']",
                "[class*='signin']",
                "[class*='login']",
                "[href*='login']",
                "[href*='signin']",
            ]
            clicked = False
            for sel in sign_in_selectors:
                try:
                    btn = await self._page.query_selector(sel)
                    if btn:
                        await btn.click()
                        await asyncio.sleep(2)
                        clicked = True
                        print(f"[NFLFantasy] Clicked sign-in via: {sel}")
                        break
                except Exception:
                    continue

            if not clicked:
                print(
                    "[NFLFantasy] Could not find Sign In button automatically.\n"
                    "  → The browser is open — please click Sign In manually,\n"
                    "    then press Enter here to continue."
                )
                input("  Press Enter after clicking Sign In...")

        # 3. Wait for the login form to appear (any URL)
        await asyncio.sleep(2)
        current_url = self._page.url
        print(f"[NFLFantasy] Login form URL: {current_url}")

        # 4. Fill in email
        email_selector = "input[type='email'], input[name='email'], input[id*='email'], input[placeholder*='email' i]"
        try:
            await self._wait_and_fill(email_selector, self.email)
        except Exception:
            print(
                "[NFLFantasy] Could not find email field automatically.\n"
                "  → Please fill in your email in the browser, then press Enter."
            )
            input("  Press Enter after filling email...")

        # 5. Some NFL.com flows show email + password on one page,
        #    others show a "Next" / "Continue" button first.
        await asyncio.sleep(0.5)
        pw_visible = await self._page.query_selector(
            "input[type='password'], input[name='password']"
        )
        if not pw_visible:
            # Click Next/Continue
            for sel in ["button[type='submit']", "button:has-text('Next')", "button:has-text('Continue')"]:
                try:
                    btn = await self._page.query_selector(sel)
                    if btn:
                        await btn.click()
                        await asyncio.sleep(2)
                        break
                except Exception:
                    continue

        # 6. Fill password
        pw_selector = "input[type='password'], input[name='password'], input[id*='password']"
        try:
            await self._wait_and_fill(pw_selector, self.password)
        except Exception:
            print(
                "[NFLFantasy] Could not find password field automatically.\n"
                "  → Please fill in your password in the browser, then press Enter."
            )
            input("  Press Enter after filling password...")

        # 7. Submit
        for sel in ["button[type='submit']", "button:has-text('Sign In')", "button:has-text('Log In')", "input[type='submit']"]:
            try:
                btn = await self._page.query_selector(sel)
                if btn:
                    await btn.click()
                    break
            except Exception:
                continue

        # 8. Wait for redirect back to the fantasy site
        print("[NFLFantasy] Waiting for login to complete...")
        try:
            await self._page.wait_for_url("*fantasy*", timeout=25_000)
        except Exception:
            # Already there, or manual intervention needed
            pass

        await asyncio.sleep(3)
        print(f"[NFLFantasy] Login complete. Now at: {self._page.url}")

    async def _extract_from_page_state(self, js_expr: str) -> Any:
        """Evaluate JavaScript in the page context to extract React/Redux state."""
        try:
            return await self._page.evaluate(js_expr)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    async def get_roster(self) -> list[RosterPlayer]:
        """Fetch current team roster with lineup slots and injury status."""
        url = (
            f"{NFL_FANTASY_BASE}/league/{self.league_id}"
            f"/team/{self.team_id}/edit-lineup"
        )
        await self._nav(url)

        # Try to extract player data from the DOM
        # NFL.com renders roster in a table/list with data attributes
        players_raw = await self._page.evaluate("""
            () => {
                const results = [];

                // Look for player rows — NFL.com typically uses table rows or
                // list items with data-player-id attributes
                const rows = document.querySelectorAll(
                    '[data-player-id], .player-name-and-info, .playerNameAndInfo'
                );

                rows.forEach(el => {
                    const row = el.closest('tr, li, [class*="PlayerRow"], [class*="playerRow"], [class*="player-row"]') || el;
                    const pid = (
                        row.getAttribute('data-player-id') ||
                        el.getAttribute('data-player-id') ||
                        ''
                    );
                    const nameEl = row.querySelector(
                        '[class*="player-name"]:not([class*="team"]), ' +
                        '[class*="playerName"], .player-name a, .playerName a'
                    );
                    const name = nameEl ? nameEl.textContent.trim() : el.textContent.trim();
                    if (!name || name.length < 2) return;

                    const pos = (
                        row.querySelector('[class*="position"], [class*="Position"]')?.textContent?.trim() ||
                        row.querySelector('.position')?.textContent?.trim() || ''
                    );
                    const team = (
                        row.querySelector('[class*="team"]:not([class*="fantasy"])')?.textContent?.trim() || ''
                    );
                    const injury = (
                        row.querySelector('[class*="injury"], [class*="Injury"], .injury-status')?.textContent?.trim() || 'Active'
                    );
                    const slot = (
                        row.querySelector('[class*="slot"], [class*="Slot"], .lineup-slot')?.textContent?.trim() || 'BN'
                    );
                    const proj = parseFloat(
                        row.querySelector('[class*="proj"], [class*="Proj"]')?.textContent?.replace(/[^0-9.]/g,'') || '0'
                    );

                    results.push({ player_id: pid, name, position: pos, team, injury, slot, projected: proj });
                });

                return results;
            }
        """)

        players = []
        seen_names = set()
        for p in (players_raw or []):
            name = p.get("name", "").strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            players.append(
                RosterPlayer(
                    player_id=p.get("player_id", ""),
                    name=name,
                    position=p.get("position", ""),
                    nfl_team=p.get("team", ""),
                    injury_status=p.get("injury", "Active"),
                    lineup_slot=p.get("slot", "BN"),
                    projected_points=p.get("projected", 0.0),
                )
            )

        print(f"[NFLFantasy] Found {len(players)} roster players.")
        return players

    async def get_waiver_wire(
        self, position: Optional[str] = None, limit: int = 40
    ) -> list[WaiverPlayer]:
        """Get available players on the waiver wire, optionally filtered by position."""
        url = f"{NFL_FANTASY_BASE}/league/{self.league_id}/players"
        if position:
            # NFL.com position filter values (may vary by league settings)
            pos_map = {
                "QB": "1", "RB": "2", "WR": "3",
                "TE": "4", "K": "7", "DEF": "8",
            }
            pos_code = pos_map.get(position.upper(), "")
            if pos_code:
                url += f"?position={pos_code}&status=A"

        await self._nav(url)

        players_raw = await self._page.evaluate(f"""
            () => {{
                const results = [];
                const rows = document.querySelectorAll(
                    'tbody tr, [class*="playerRow"], [class*="PlayerRow"]'
                );
                rows.forEach(row => {{
                    const nameEl = row.querySelector(
                        '[class*="player-name"] a, [class*="playerName"] a, .playerName a'
                    );
                    if (!nameEl) return;
                    const name = nameEl.textContent.trim();
                    if (!name) return;

                    // Extract href for player_id
                    const href = nameEl.getAttribute('href') || '';
                    const pidMatch = href.match(/player\\/([0-9]+)/);
                    const pid = pidMatch ? pidMatch[1] : '';

                    const pos = row.querySelector('[class*="position"]')?.textContent?.trim() || '';
                    const team = row.querySelector('[class*="team"]:not([class*="fantasy"])')?.textContent?.trim() || '';
                    const injury = row.querySelector('[class*="injury"]')?.textContent?.trim() || 'Active';
                    const owned = parseFloat(
                        row.querySelector('[class*="owned"], [class*="Owned"]')?.textContent?.replace('%','') || '0'
                    );
                    const proj = parseFloat(
                        row.querySelector('[class*="proj"]')?.textContent?.replace(/[^0-9.]/g,'') || '0'
                    );
                    const news = row.querySelector('[class*="news"]')?.textContent?.trim() || '';

                    results.push({{ player_id: pid, name, position: pos, team, injury, owned, proj, news }});
                }});
                return results.slice(0, {limit});
            }}
        """)

        players = []
        for p in (players_raw or []):
            name = p.get("name", "").strip()
            if not name:
                continue
            players.append(
                WaiverPlayer(
                    player_id=p.get("player_id", ""),
                    name=name,
                    position=p.get("position", ""),
                    nfl_team=p.get("team", ""),
                    injury_status=p.get("injury", "Active"),
                    percent_owned=p.get("owned", 0.0),
                    projected_points=p.get("proj", 0.0),
                    news=p.get("news", ""),
                )
            )

        print(f"[NFLFantasy] Found {len(players)} available waiver wire players.")
        return players

    async def get_matchup(self) -> Optional[Matchup]:
        """Get current week matchup information."""
        url = (
            f"{NFL_FANTASY_BASE}/league/{self.league_id}"
            f"/team/{self.team_id}/matchup"
        )
        await self._nav(url)

        data = await self._page.evaluate("""
            () => {
                const getText = sel => document.querySelector(sel)?.textContent?.trim() || '';
                const getFloat = sel => parseFloat(getText(sel).replace(/[^0-9.]/g,'')) || 0;

                // Try various selectors NFL.com might use
                return {
                    week: getText('[class*="week"], [class*="Week"]').match(/\\d+/)?.[0] || '0',
                    my_team: getText('[class*="home"] [class*="team-name"], [class*="myTeam"] [class*="name"]'),
                    my_score: getFloat('[class*="home"] [class*="score"], [class*="homeScore"]'),
                    opp_team: getText('[class*="away"] [class*="team-name"], [class*="oppTeam"] [class*="name"]'),
                    opp_score: getFloat('[class*="away"] [class*="score"], [class*="awayScore"]'),
                    my_proj: getFloat('[class*="home"] [class*="proj"]'),
                    opp_proj: getFloat('[class*="away"] [class*="proj"]'),
                };
            }
        """)

        if not data:
            return None

        return Matchup(
            week=int(data.get("week", 0) or 0),
            my_team_name=data.get("my_team", "My Team"),
            my_score=data.get("my_score", 0.0),
            opponent_team_name=data.get("opp_team", "Opponent"),
            opponent_score=data.get("opp_score", 0.0),
            my_projected=data.get("my_proj", 0.0),
            opp_projected=data.get("opp_proj", 0.0),
        )

    async def get_trade_offers(self) -> list[TradeOffer]:
        """Get all pending incoming trade offers."""
        url = (
            f"{NFL_FANTASY_BASE}/league/{self.league_id}"
            f"/team/{self.team_id}/trades"
        )
        await self._nav(url)

        offers_raw = await self._page.evaluate("""
            () => {
                const offers = [];
                document.querySelectorAll('[class*="trade-offer"], [class*="tradeOffer"]').forEach(el => {
                    const tid = el.getAttribute('data-trade-id') || el.getAttribute('id') || '';
                    const from = el.querySelector('[class*="from-team"], [class*="fromTeam"]')?.textContent?.trim() || '';

                    const giving = Array.from(
                        el.querySelectorAll('[class*="give"] [class*="player"], [class*="giving"] [class*="player"]')
                    ).map(p => p.textContent.trim()).filter(Boolean);

                    const receiving = Array.from(
                        el.querySelectorAll('[class*="receive"] [class*="player"], [class*="receiving"] [class*="player"]')
                    ).map(p => p.textContent.trim()).filter(Boolean);

                    const expires = el.querySelector('[class*="expires"], [class*="Expires"]')?.textContent?.trim() || '';

                    if (from || giving.length || receiving.length) {
                        offers.push({ trade_id: tid, from_team: from, giving, receiving, expires });
                    }
                });
                return offers;
            }
        """)

        return [
            TradeOffer(
                trade_id=o.get("trade_id", ""),
                from_team_name=o.get("from_team", ""),
                players_giving=o.get("giving", []),
                players_receiving=o.get("receiving", []),
                expires=o.get("expires", ""),
            )
            for o in (offers_raw or [])
        ]

    async def get_league_teams(self) -> list[dict]:
        """Get all teams in the league with their names and standings."""
        url = f"{NFL_FANTASY_BASE}/league/{self.league_id}"
        await self._nav(url)

        teams = await self._page.evaluate("""
            () => {
                const teams = [];
                document.querySelectorAll('[class*="team-name"], [class*="teamName"]').forEach(el => {
                    const name = el.textContent.trim();
                    const href = el.closest('a')?.getAttribute('href') || '';
                    const idMatch = href.match(/team\\/([0-9]+)/);
                    if (name && idMatch) {
                        teams.push({ name, team_id: idMatch[1] });
                    }
                });
                return teams;
            }
        """)
        return teams or []

    # ------------------------------------------------------------------
    # Actions (destructive — require user confirmation in the agent)
    # ------------------------------------------------------------------

    async def set_lineup(self, starter_player_ids: list[str]) -> dict:
        """
        Set the starting lineup.

        NFL.com's lineup editor uses drag-and-drop, so this first attempts
        the "Auto Set Lineup" button, then falls back to direct slot clicks.
        Returns {'success': bool, 'message': str}.
        """
        url = (
            f"{NFL_FANTASY_BASE}/league/{self.league_id}"
            f"/team/{self.team_id}/edit-lineup"
        )
        await self._nav(url)

        # Try "Auto Set Lineup" button first
        auto_btn = await self._page.query_selector(
            "button:has-text('Auto Set'), button:has-text('Auto-Set'), "
            "[class*='auto-set'], [class*='autoSet']"
        )
        if auto_btn:
            await auto_btn.click()
            await asyncio.sleep(2)
            # Look for confirm/save button
            save_btn = await self._page.query_selector(
                "button:has-text('Save'), button:has-text('Submit'), "
                "button:has-text('Update')"
            )
            if save_btn:
                await save_btn.click()
                await asyncio.sleep(2)
                return {"success": True, "message": "Lineup set via Auto Set Lineup."}

        # Manual slot-by-slot lineup setting
        # This is site-specific and may need adjustment
        set_count = 0
        for player_id in starter_player_ids:
            try:
                # Find the player's bench slot and drag to starter slot
                # This is a simplified approach — actual implementation depends on UI
                bench_el = await self._page.query_selector(
                    f"[data-player-id='{player_id}'] [class*='move'], "
                    f"tr[data-player-id='{player_id}']"
                )
                if bench_el:
                    # Click the player to select, then choose slot
                    await bench_el.click()
                    set_count += 1
                    await asyncio.sleep(0.3)
            except Exception as e:
                print(f"[NFLFantasy] Warning setting player {player_id}: {e}")

        # Save the lineup
        save_btn = await self._page.query_selector(
            "button:has-text('Save'), button:has-text('Submit'), button:has-text('Update')"
        )
        if save_btn:
            await save_btn.click()
            await asyncio.sleep(2)

        return {
            "success": set_count > 0,
            "message": f"Attempted to set {set_count} players as starters.",
            "note": (
                "NFL.com uses drag-and-drop lineup editing. "
                "If auto-set wasn't available, manual slot placement may need review."
            ),
        }

    async def claim_waiver_player(
        self, add_player_id: str, drop_player_id: str
    ) -> dict:
        """
        Claim a player from waivers and drop a player from the roster.
        Returns {'success': bool, 'message': str}.
        """
        # Navigate to the add player page
        add_url = (
            f"{NFL_FANTASY_BASE}/league/{self.league_id}"
            f"/players/add-drop?addPlayerId={add_player_id}"
            f"&dropPlayerId={drop_player_id}"
        )
        await self._nav(add_url)

        # Try direct URL for add+drop confirmation
        # Look for submit/confirm button
        confirm_btn = await self._page.query_selector(
            "button:has-text('Add'), button:has-text('Submit'), "
            "button:has-text('Confirm'), input[type='submit']"
        )
        if confirm_btn:
            await confirm_btn.click()
            await asyncio.sleep(2)
            await asyncio.sleep(1)
            return {
                "success": True,
                "message": f"Claimed player {add_player_id}, dropped {drop_player_id}.",
            }

        # Fallback: navigate to waiver wire and find the player
        await self._nav(f"{NFL_FANTASY_BASE}/league/{self.league_id}/players")
        try:
            add_btn = await self._page.query_selector(
                f"[data-player-id='{add_player_id}'] button, "
                f"tr[data-player-id='{add_player_id}'] button"
            )
            if add_btn:
                await add_btn.click()
                await asyncio.sleep(1)

                # Select player to drop
                drop_el = await self._page.query_selector(
                    f"[data-player-id='{drop_player_id}'] input[type='radio'], "
                    f"[data-player-id='{drop_player_id}'] button"
                )
                if drop_el:
                    await drop_el.click()
                    await asyncio.sleep(0.5)

                    submit = await self._page.query_selector(
                        "button:has-text('Submit'), button:has-text('Confirm')"
                    )
                    if submit:
                        await submit.click()
                        await asyncio.sleep(2)
                        return {"success": True, "message": "Waiver claim submitted."}
        except Exception as e:
            return {"success": False, "message": f"Waiver claim failed: {e}"}

        return {
            "success": False,
            "message": "Could not complete waiver claim — selectors may need updating.",
        }

    async def accept_trade(self, trade_id: str) -> dict:
        """Accept a pending trade offer."""
        url = (
            f"{NFL_FANTASY_BASE}/league/{self.league_id}"
            f"/team/{self.team_id}/trades"
        )
        await self._nav(url)

        try:
            btn = await self._page.query_selector(
                f"[data-trade-id='{trade_id}'] button:has-text('Accept'), "
                f"#trade-{trade_id} button:has-text('Accept')"
            )
            if not btn:
                # Try finding by proximity to trade ID in text
                btn = await self._page.query_selector("button:has-text('Accept')")

            if btn:
                await btn.click()
                await asyncio.sleep(2)
                return {"success": True, "message": f"Trade {trade_id} accepted."}
        except Exception as e:
            return {"success": False, "message": f"Trade accept failed: {e}"}

        return {"success": False, "message": "Accept button not found."}

    async def decline_trade(self, trade_id: str) -> dict:
        """Decline a pending trade offer."""
        url = (
            f"{NFL_FANTASY_BASE}/league/{self.league_id}"
            f"/team/{self.team_id}/trades"
        )
        await self._nav(url)

        try:
            btn = await self._page.query_selector(
                f"[data-trade-id='{trade_id}'] button:has-text('Decline'), "
                f"[data-trade-id='{trade_id}'] button:has-text('Reject')"
            )
            if not btn:
                btn = await self._page.query_selector(
                    "button:has-text('Decline'), button:has-text('Reject')"
                )
            if btn:
                await btn.click()
                await asyncio.sleep(2)
                return {"success": True, "message": f"Trade {trade_id} declined."}
        except Exception as e:
            return {"success": False, "message": f"Trade decline failed: {e}"}

        return {"success": False, "message": "Decline button not found."}

    async def propose_trade(
        self,
        target_team_id: str,
        offer_player_ids: list[str],
        request_player_ids: list[str],
        message: str = "",
    ) -> dict:
        """Propose a trade to another team."""
        url = (
            f"{NFL_FANTASY_BASE}/league/{self.league_id}"
            f"/team/{self.team_id}/propose-trade?teamId={target_team_id}"
        )
        await self._nav(url)

        try:
            # Select players to offer
            for pid in offer_player_ids:
                checkbox = await self._page.query_selector(
                    f"[data-player-id='{pid}'] input[type='checkbox']"
                )
                if checkbox:
                    await checkbox.check()

            # Select players to request
            for pid in request_player_ids:
                checkbox = await self._page.query_selector(
                    f"[data-player-id='{pid}'] input[type='checkbox']"
                )
                if checkbox:
                    await checkbox.check()

            # Optional message
            if message:
                msg_input = await self._page.query_selector(
                    "textarea, input[type='text'][placeholder*='message']"
                )
                if msg_input:
                    await msg_input.fill(message)

            submit = await self._page.query_selector(
                "button:has-text('Send Trade'), button:has-text('Submit'), "
                "button:has-text('Propose')"
            )
            if submit:
                await submit.click()
                await asyncio.sleep(2)
                return {"success": True, "message": "Trade proposal sent."}
        except Exception as e:
            return {"success": False, "message": f"Trade proposal failed: {e}"}

        return {"success": False, "message": "Could not submit trade proposal."}

    # ------------------------------------------------------------------
    # Draft support
    # ------------------------------------------------------------------

    async def get_draft_board(self) -> list[dict]:
        """Get the current draft board (available players ranked by ADP)."""
        url = f"{NFL_FANTASY_BASE}/league/{self.league_id}/draftboard"
        await self._nav(url)

        board = await self._page.evaluate("""
            () => {
                const players = [];
                document.querySelectorAll('[class*="draftboard"] [class*="player"], tr[data-player-id]').forEach(row => {
                    const pid = row.getAttribute('data-player-id') || '';
                    const name = row.querySelector('[class*="player-name"], [class*="playerName"]')?.textContent?.trim() || '';
                    const pos = row.querySelector('[class*="position"]')?.textContent?.trim() || '';
                    const team = row.querySelector('[class*="team"]:not([class*="fantasy"])')?.textContent?.trim() || '';
                    const rank = parseInt(row.querySelector('[class*="rank"]')?.textContent?.trim() || '999');
                    const available = !row.classList.toString().includes('drafted') &&
                                      !row.querySelector('[class*="drafted"], [class*="picked"]');
                    if (name && available) {
                        players.push({ player_id: pid, name, position: pos, team, adp_rank: rank });
                    }
                });
                return players;
            }
        """)

        return board or []

    async def make_draft_pick(self, player_id: str) -> dict:
        """Draft a specific player (use during your draft turn)."""
        url = f"{NFL_FANTASY_BASE}/league/{self.league_id}/draftboard"
        await self._nav(url)

        try:
            # Find the draft button for this player
            draft_btn = await self._page.query_selector(
                f"[data-player-id='{player_id}'] button:has-text('Draft'), "
                f"tr[data-player-id='{player_id}'] button"
            )
            if draft_btn:
                await draft_btn.click()

                # Confirm if there's a confirmation dialog
                confirm = await self._page.query_selector(
                    "button:has-text('Confirm'), button:has-text('Yes')"
                )
                if confirm:
                    await confirm.click()

                await asyncio.sleep(2)
                return {"success": True, "message": f"Drafted player {player_id}."}
        except Exception as e:
            return {"success": False, "message": f"Draft pick failed: {e}"}

        return {"success": False, "message": "Draft button not found."}

    async def get_draft_turn_info(self) -> dict:
        """Check if it's our turn to pick in the draft."""
        url = f"{NFL_FANTASY_BASE}/league/{self.league_id}/draftboard"
        await self._nav(url)

        info = await self._page.evaluate("""
            () => ({
                is_my_turn: !!document.querySelector(
                    '[class*="my-turn"], [class*="myTurn"], [class*="on-the-clock"]'
                ),
                pick_number: parseInt(
                    document.querySelector('[class*="pick-number"], [class*="pickNumber"]')?.textContent || '0'
                ),
                time_remaining: document.querySelector(
                    '[class*="time"], [class*="countdown"], [class*="timer"]'
                )?.textContent?.trim() || '',
                current_pick_team: document.querySelector(
                    '[class*="on-clock"] [class*="team-name"], [class*="current-pick"] [class*="team"]'
                )?.textContent?.trim() || '',
            })
        """)

        return info or {}
