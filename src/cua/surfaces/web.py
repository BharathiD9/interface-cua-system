"""
Web surface adapter (Playwright).

This is the only file in the system that knows what a DOM is.

Perception is accessibility-shaped on purpose. We run one script that walks every frame and
reduces each control to (role, accessible name, value, region, bounds) -- the same tuple a
Windows UIA or macOS AX adapter would produce. We never hand raw markup to the model. In a
frameset with four levels of layout tables the markup is mostly noise, and more importantly a
model that learns to depend on markup produces locators that will not survive a template
change.

Accessible name derivation, in the order legacy apps actually make available:
  1. aria-label / aria-labelledby      (rare here, but free when present)
  2. value= on a button input           (how every button in MEMBERCORE is named)
  3. <label for>                        (rare in table-based forms)
  4. the text of the adjacent table cell (how every field in MEMBERCORE is named)
  5. placeholder, then title
Step 4 is the one that matters. Legacy table forms put the label in the previous <td> with no
programmatic association at all. A human reads it as the field's name; a screen reader often
does too. Reproducing that inference is what lets us have accessible-name locators on a surface
whose authors never thought about accessibility.
"""

from __future__ import annotations

import re
from typing import Any

from playwright.sync_api import Browser, BrowserContext, Frame, Locator, Page, sync_playwright

from cua.safety.policy import Allowlist, PolicyViolation, Redactor
from cua.schema import ActionType, LocatorStrategy, RiskClass, Target
from cua.surfaces.base import ActionResult, Control, Observation, Resolution

# Runs in the page. Returns one flat list of controls per frame, accessibility-shaped.
_EXTRACT_JS = r"""
() => {
  const roleOf = (el) => {
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'input') {
      if (['button','submit','reset','image'].includes(type)) return 'button';
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      if (type === 'password') return 'textbox';
      return 'textbox';
    }
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'button') return 'button';
    if (tag === 'a' && el.hasAttribute('href')) return 'link';
    if (/^h[1-6]$/.test(tag)) return 'heading';
    return null;
  };

  // Legacy-table label inference: the text of the nearest preceding cell.
  const adjacentLabel = (el) => {
    const cell = el.closest('td,th');
    if (!cell) return '';
    let prev = cell.previousElementSibling;
    while (prev) {
      const t = (prev.innerText || '').trim();
      if (t && t.length < 60 && !prev.querySelector('input,select,textarea,button')) return t;
      prev = prev.previousElementSibling;
    }
    // Fall back to a label cell in the row above (stacked forms).
    const row = cell.closest('tr');
    const prevRow = row && row.previousElementSibling;
    if (prevRow) {
      const idx = Array.from(row.children).indexOf(cell);
      const head = prevRow.children[idx];
      if (head) {
        const t = (head.innerText || '').trim();
        if (t && t.length < 60) return t;
      }
    }
    return '';
  };

  const nameOf = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria.trim();
    const labelledby = el.getAttribute('aria-labelledby');
    if (labelledby) {
      const ref = document.getElementById(labelledby);
      if (ref) return (ref.innerText || '').trim();
    }
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'input' && ['button','submit','reset'].includes(type)) {
      return (el.value || '').trim();
    }
    if (tag === 'button' || tag === 'a' || /^h[1-6]$/.test(tag)) {
      return (el.innerText || '').trim();
    }
    if (el.id) {
      const lab = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (lab) return (lab.innerText || '').trim();
    }
    const adj = adjacentLabel(el);
    if (adj) return adj;
    return (el.getAttribute('placeholder') || el.getAttribute('title') || '').trim();
  };

  // Nearest enclosing section heading -- our stand-in for a landmark region.
  const regionOf = (el) => {
    let node = el;
    while (node && node !== document.body) {
      const row = node.closest ? node.closest('table') : null;
      if (row) {
        const hdr = row.querySelector('.hdr');
        if (hdr) {
          const t = (hdr.innerText || '').trim();
          if (t) return t;
        }
      }
      node = node.parentElement;
    }
    const anyHdr = document.querySelector('.hdr');
    return anyHdr ? (anyHdr.innerText || '').trim() : '';
  };

  const out = [];
  const els = document.querySelectorAll('input,select,textarea,button,a[href],h1,h2,h3');
  els.forEach((el, i) => {
    const role = roleOf(el);
    if (!role) return;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return;
    const r = el.getBoundingClientRect();
    const isPassword = (el.getAttribute('type') || '').toLowerCase() === 'password';
    out.push({
      role,
      name: nameOf(el),
      value: isPassword ? null : (el.value !== undefined ? String(el.value || '') : ''),
      region: regionOf(el),
      bounds: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)],
      enabled: !el.disabled,
      index: i,
    });
  });
  return { controls: out, text: document.body ? document.body.innerText : '' };
}
"""


class WebSurface:
    """Playwright implementation of the Surface protocol."""

    def __init__(
        self,
        allowlist: Allowlist,
        redactor: Redactor | None = None,
        headless: bool = True,
        cdp_port: int = 9222,
        allow_coordinate_clicks: bool = False,
    ) -> None:
        self.allowlist = allowlist
        self.redactor = redactor or Redactor()
        self.headless = headless
        self.cdp_port = cdp_port
        # Off by default on web: this surface exposes a full accessibility tree, so a failed
        # ladder means the control is absent, not that we need to guess at pixels.
        self.allow_coordinate_clicks = allow_coordinate_clicks
        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._last_status: int | None = None

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._pw = sync_playwright().start()
        # Remote debugging is exposed so a human operator can attach to THIS session during
        # escalation rather than being handed a fresh one. See escalation/handoff.py.
        self._browser = self._pw.chromium.launch(
            headless=self.headless,
            args=[f"--remote-debugging-port={self.cdp_port}"],
        )
        self._context = self._browser.new_context(viewport={"width": 1280, "height": 900})
        self._page = self._context.new_page()
        self._page.on("response", self._track_status)

    def _track_status(self, response) -> None:
        try:
            if response.request.is_navigation_request():
                self._last_status = response.status
        except Exception:  # noqa: BLE001 - status tracking is best-effort telemetry
            pass

    def stop(self) -> None:
        for closer in (self._context, self._browser):
            try:
                if closer:
                    closer.close()
            except Exception:  # noqa: BLE001
                pass
        if self._pw:
            self._pw.stop()

    def session_handle(self) -> str | None:
        return f"http://127.0.0.1:{self.cdp_port}"

    # ---------------------------------------------------------------- frames

    def _frames(self) -> dict[str, Frame]:
        """Map frame-path string -> Frame. Top document is the empty string."""
        assert self._page is not None
        out: dict[str, Frame] = {"": self._page.main_frame}
        for f in self._page.frames:
            if f == self._page.main_frame:
                continue
            out[f.name or f.url.rsplit("/", 1)[-1]] = f
        return out

    def _frame_for(self, frame_path: list[str]) -> Frame:
        assert self._page is not None
        if not frame_path:
            return self._page.main_frame
        frames = self._frames()
        # We address frames by name rather than by index. Names are authored and stable;
        # index order changes whenever the layout is touched.
        target = frame_path[-1]
        if target in frames:
            return frames[target]
        raise PolicyViolation(f"frame {target!r} not found; available: {sorted(frames)}")

    # ---------------------------------------------------------------- perception

    def observe(self, screenshot: bool = False, screenshot_path: str | None = None) -> Observation:
        assert self._page is not None
        controls: list[Control] = []
        text_by_frame: dict[str, str] = {}

        for name, frame in self._frames().items():
            try:
                data: dict[str, Any] = frame.evaluate(_EXTRACT_JS)
            except Exception:  # noqa: BLE001 - a frame may be mid-navigation
                continue
            path = [name] if name else []
            text_by_frame[name] = self.redactor.scrub(data.get("text", "")) or ""
            for c in data.get("controls", []):
                b = c.get("bounds") or [0, 0, 0, 0]
                controls.append(
                    Control(
                        role=c["role"],
                        name=self.redactor.scrub(c.get("name") or "") or "",
                        value=self.redactor.scrub(c.get("value") or None),
                        frame_path=path,
                        region=c.get("region") or None,
                        bounds=(b[0], b[1], b[2], b[3]),
                        enabled=bool(c.get("enabled", True)),
                    )
                )

        shot = None
        if screenshot and screenshot_path:
            shot = self.screenshot(screenshot_path)

        return Observation(
            url=self._page.url,
            title=self._page.title(),
            text_by_frame=text_by_frame,
            controls=controls,
            http_status=self._last_status,
            screenshot_path=shot,
        )

    def screenshot(self, path: str) -> str | None:
        try:
            assert self._page is not None
            self._page.screenshot(path=path, full_page=False)
            return path
        except Exception:  # noqa: BLE001 - evidence capture must never break a run
            return None

    # ---------------------------------------------------------------- locator ladder

    def resolve(self, target: Target) -> Resolution:
        """Walk the ladder top-down. First candidate that matches EXACTLY ONE control wins.

        Ambiguity is treated as failure, not as 'take the first'. In a legacy app a locator
        that matches three controls is a locator that will eventually click the wrong one, and
        a loud failure now is cheaper than a silent wrong action in production.
        """
        frame = self._frame_for(target.frame_path)
        tried: list[str] = []

        for cand in target.candidates:  # already tier-sorted by the schema
            try:
                loc = self._build(frame, cand.strategy, cand.params)
                if loc is None:
                    tried.append(f"{cand.strategy.value}: unsupported")
                    continue
                count = loc.count()
                if count == 1:
                    return Resolution(
                        found=True,
                        tier=cand.tier,
                        strategy=cand.strategy.value,
                        handle=loc,
                        tried=tried,
                    )
                tried.append(f"{cand.strategy.value}: matched {count}")
            except Exception as exc:  # noqa: BLE001 - a bad candidate must not abort the ladder
                tried.append(f"{cand.strategy.value}: error {type(exc).__name__}")

        return Resolution(found=False, tried=tried)

    def _build(self, frame: Frame, strategy: LocatorStrategy, p: dict[str, Any]) -> Locator | None:
        if strategy is LocatorStrategy.ACCESSIBLE_NAME:
            return frame.get_by_role(p["role"], name=p["name"], exact=p.get("exact", True))

        if strategy is LocatorStrategy.LABEL_PROXIMITY:
            # The legacy-table case: control sitting in the cell after a cell of label text.
            label = p["label"].replace('"', '\\"')
            # "cell" targets the adjacent data cell itself rather than a control inside it.
            # This is how we extract OUTPUTS: in a legacy app a displayed value is a <td> next
            # to a <td> of label text, structurally identical to a form field next to its
            # label. Using one strategy for both means output extraction inherits the same
            # stability properties (and the same drift signal) as actions.
            kind = p.get("control", "textbox")
            if kind in {"cell", "text"}:
                nth = int(p.get("offset", 1))
                return frame.locator(
                    f'xpath=//td[normalize-space(.)="{label}"]/following-sibling::td[{nth}] | '
                    f'//th[normalize-space(.)="{label}"]/following-sibling::td[{nth}]'
                )
            control = {"textbox": "input", "combobox": "select"}.get(kind)
            # following-sibling::td[1] -- the IMMEDIATELY next cell only. Without the [1] this
            # matches every later cell in the row, and in a form row like
            #   | Member ID | <input> | <Search button> |
            # that is two controls, which our ambiguity rule (correctly) rejects.
            xpath = (
                f'xpath=//td[normalize-space(.)="{label}"]/following-sibling::td[1]'
                f"//{control} | "
                f'//th[normalize-space(.)="{label}"]/following-sibling::td[1]//{control}'
            )
            return frame.locator(xpath)

        if strategy is LocatorStrategy.TEXT:
            role = p.get("role")
            if role:
                return frame.get_by_role(role, name=p["text"], exact=p.get("exact", True))
            return frame.get_by_text(p["text"], exact=p.get("exact", False))

        if strategy is LocatorStrategy.STRUCTURAL:
            control = {"textbox": "input[type=text]", "button": "input[type=submit]",
                       "combobox": "select"}.get(p.get("control", "textbox"), "input")
            scope = frame.locator("table").filter(has_text=p["region"]) if p.get("region") else frame
            return scope.locator(control).nth(int(p.get("index", 0)))

        if strategy is LocatorStrategy.CSS:
            return frame.locator(p["selector"])

        if strategy is LocatorStrategy.COORDINATES:
            return None  # handled directly in act(); has no Locator representation

        return None

    # ---------------------------------------------------------------- action

    def act(
        self,
        action: ActionType,
        target: Target | None = None,
        value: str | None = None,
        risk: RiskClass = RiskClass.SAFE,
        attended: bool = False,
    ) -> ActionResult:
        """Every action in the system funnels through here, which is where policy is applied."""
        self.allowlist.check_action(action)
        self.allowlist.check_risk(risk, attended)
        assert self._page is not None

        if action is ActionType.NAVIGATE:
            if not value:
                return ActionResult(ok=False, error="navigate requires a URL")
            self.allowlist.check_url(value)
            self._page.goto(value, wait_until="load")
            return ActionResult(ok=True)

        if action is ActionType.WAIT_FOR:
            self._page.wait_for_timeout(int(value or 1000))
            return ActionResult(ok=True)

        if target is None:
            return ActionResult(ok=False, error=f"{action.value} requires a target")

        # Coordinate fallback. Deliberately NOT enabled on a surface that has a working
        # accessibility tree. The reasoning: if tiers 1-5 all matched nothing on a web page we
        # can fully query, the control genuinely is not on screen -- and clicking its last
        # known pixel is not a fallback, it is a blind click at whatever occupies that point
        # now. In a bank back office that is how you click "Confirm Transfer" because it landed
        # where "Search" used to be. Coordinates exist for surfaces with NO queryable tree
        # (the desktop case), which is why the tier is kept in the schema and recorded at
        # discovery time, but firing it here would trade a loud failure for a silent wrong
        # action. Enable explicitly per-surface when the tree is genuinely unavailable.
        coord = next(
            (c for c in target.candidates if c.strategy is LocatorStrategy.COORDINATES), None
        )
        res = self.resolve(target)
        if not res.found and coord and action is ActionType.CLICK and self.allow_coordinate_clicks:
            self._page.mouse.click(coord.params["x"], coord.params["y"])
            return ActionResult(
                ok=True, resolution=Resolution(found=True, tier=6, strategy="coordinates")
            )
        if not res.found:
            return ActionResult(
                ok=False,
                resolution=res,
                error=f"could not resolve {target.description!r}; tried: {'; '.join(res.tried)}",
            )

        loc: Locator = res.handle  # type: ignore[assignment]
        try:
            if action is ActionType.CLICK:
                loc.click()
            elif action is ActionType.TYPE:
                loc.fill(value or "")
            elif action is ActionType.SELECT:
                loc.select_option(value or "")
            elif action is ActionType.PRESS_KEY:
                loc.press(value or "Enter")
            elif action is ActionType.READ:
                pass
            else:
                return ActionResult(ok=False, error=f"unsupported action {action.value}")
        except Exception as exc:  # noqa: BLE001 - surfaced as a structured failure upstream
            return ActionResult(ok=False, resolution=res, error=f"{type(exc).__name__}: {exc}")

        # Legacy apps navigate on almost every action, and in a frameset the navigation happens
        # in the CHILD frame -- the top-level page load state is already "load" and returns
        # immediately, so waiting on the page alone observes stale frame content. We wait on
        # the frame we actually acted in. This race is the single most common source of
        # flaky legacy automation, which is why it gets handled here once rather than with
        # sleeps scattered through the replay engine.
        self._settle(target.frame_path)
        return ActionResult(ok=True, resolution=res)

    def _settle(self, frame_path: list[str], timeout_ms: int = 8000) -> None:
        assert self._page is not None
        try:
            self._page.wait_for_load_state("load", timeout=timeout_ms)
        except Exception:  # noqa: BLE001
            pass
        if frame_path:
            try:
                self._frame_for(frame_path).wait_for_load_state("load", timeout=timeout_ms)
            except Exception:  # noqa: BLE001 - frame may have been replaced by navigation
                pass
        # Frameset navigations swap the Frame object itself; a short settle lets the new
        # document attach before we enumerate frames again.
        self._page.wait_for_timeout(250)

    def read(self, target: Target) -> str | None:
        res = self.resolve(target)
        if not res.found:
            return None
        loc: Locator = res.handle  # type: ignore[assignment]
        try:
            text = loc.inner_text()
        except Exception:  # noqa: BLE001
            try:
                text = loc.input_value()
            except Exception:  # noqa: BLE001
                return None
        return self.redactor.scrub(re.sub(r"\s+", " ", text).strip())
