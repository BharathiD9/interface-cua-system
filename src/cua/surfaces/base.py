"""
The Surface abstraction -- the seam the brief asks about in 3.7.

Everything above this line (discovery loop, replay engine, artifact schema) is written against
this protocol and knows nothing about Playwright, the DOM, or a browser. Everything below it is
surface-specific. Adding a desktop surface means writing one more class here; it means zero
changes to the schema, the replay engine, or the escalation model.

The protocol is deliberately small and phrased in terms a human operator would recognise --
"what can I see", "click that control", "read that value" -- rather than in browser terms. That
phrasing is the whole point: it is the vocabulary that survives the jump to a native app.

PERCEPTION MODEL
We expose an accessibility-style snapshot, NOT raw markup. Each observable control is reduced
to (role, name, value, region, bounds). Three reasons:
  - It is the representation that exists on every surface we care about. A browser has ARIA; a
    Windows app has UIA; macOS has AX. Raw DOM exists on exactly one of the three.
  - It is far more stable than markup in legacy apps, where the table nesting gets rewritten
    but the button still says "Search".
  - It is compact enough to put in a model prompt without burning the context window on
    <table><tr><td><table> noise.
The escape hatch is screenshot + coordinates, which works when there is no queryable tree at
all. It is the lowest locator tier for a reason, but it exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cua.schema import ActionType, Target


@dataclass
class Control:
    """One observable, possibly interactive element, surface-agnostic."""

    role: str  # textbox, button, link, combobox, checkbox, text, heading
    name: str  # accessible name: label, button text, heading text
    value: str | None = None  # current value for inputs
    frame_path: list[str] = field(default_factory=list)
    region: str | None = None  # nearest enclosing section heading, for structural locators
    bounds: tuple[int, int, int, int] | None = None  # x, y, w, h -- for coordinate fallback
    enabled: bool = True

    def render(self) -> str:
        """Compact one-line form for a model prompt."""
        parts = [f"{self.role}"]
        if self.name:
            parts.append(f'"{self.name}"')
        if self.value:
            parts.append(f"value={self.value!r}")
        if not self.enabled:
            parts.append("(disabled)")
        return " ".join(parts)


@dataclass
class Observation:
    """A full perception of the surface at one instant."""

    url: str
    title: str
    # Visible text per frame. Used by text assertions and given to the model for context.
    text_by_frame: dict[str, str]
    controls: list[Control]
    http_status: int | None = None
    screenshot_path: str | None = None

    def render(self, max_text_chars: int = 1200) -> str:
        """What the model actually sees. Accessibility-style, not markup."""
        lines = [f"URL: {self.url}", f"TITLE: {self.title}"]
        if self.http_status and self.http_status >= 400:
            lines.append(f"HTTP STATUS: {self.http_status}")

        for frame, text in self.text_by_frame.items():
            label = frame or "(top document)"
            snippet = " ".join(text.split())[:max_text_chars]
            if snippet:
                lines.append(f"\n--- visible text in frame [{label}] ---\n{snippet}")

        lines.append("\n--- controls ---")
        by_frame: dict[str, list[Control]] = {}
        for c in self.controls:
            by_frame.setdefault(".".join(c.frame_path) or "(top)", []).append(c)
        for frame, controls in by_frame.items():
            lines.append(f"[frame: {frame}]")
            for c in controls:
                region = f"  (in section: {c.region})" if c.region else ""
                lines.append(f"  - {c.render()}{region}")
        return "\n".join(lines)


@dataclass
class Resolution:
    """The outcome of walking a locator ladder. The tier is the drift signal."""

    found: bool
    tier: int | None = None
    strategy: str | None = None
    handle: object | None = None  # surface-specific; callers must not introspect it
    tried: list[str] = field(default_factory=list)  # for debuggable failures
    ambiguous: bool = False  # matched more than one control -- treated as not-found


@dataclass
class ActionResult:
    ok: bool
    resolution: Resolution | None = None
    error: str | None = None


@runtime_checkable
class Surface(Protocol):
    """What any surface must provide. Web today; desktop is the same nine methods."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def observe(self, screenshot: bool = False) -> Observation:
        """Perceive current state."""

    def resolve(self, target: Target) -> Resolution:
        """Walk the locator ladder top-down; report which tier won."""

    def act(self, action: ActionType, target: Target | None, value: str | None) -> ActionResult:
        """Perform one action. Allowlist enforcement happens inside the adapter."""

    def read(self, target: Target) -> str | None:
        """Extract text from a control without acting on it."""

    def screenshot(self, path: str) -> str | None: ...

    def session_handle(self) -> str | None:
        """An identifier a human operator can attach to for takeover. This is what makes
        handoff real rather than 'open a fresh browser': the human drives THIS session."""
