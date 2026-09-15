"""
Guardrails: allowlist, risk policy, redaction.

DESIGN DECISION -- ONE CHOKE POINT.
The policy is enforced inside the surface adapter, not in the agent loop and not in the replay
engine. Both of those call the same adapter, so there is exactly one code path to audit and no
way to add a second path that forgets to check. An agent loop that politely asks permission is
a guardrail; an adapter that cannot perform a disallowed action is a control.

DESIGN DECISION -- REDACT AT THE OBSERVATION BOUNDARY.
Sensitive values are scrubbed when the surface is observed, before the text reaches the model,
the logs, the evidence files, or the artifact. The alternative -- scrubbing on write -- means
raw PII exists in memory and in the model transcript and we are one forgotten call site away
from persisting it. Redacting on read costs us a little fidelity and buys a much smaller blast
radius, which is the right trade in a regulated context.

LIMITS (stated plainly, because the brief asks for the limits of the model):
  - Pattern-based redaction catches formats we anticipated. A novel PII format gets through.
  - The allowlist is URL- and action-shaped. It cannot express "may read balances but not
    balances over $10,000" -- semantic policy would need a different mechanism.
  - Risk classification is recorded per step at discovery time. A step whose risk was
    misjudged during discovery stays misjudged until a human reviews the artifact. That is
    the main reason artifacts default to DRAFT and need approval before unattended replay.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from cua.schema import ActionType, RiskClass


class PolicyViolation(Exception):
    """Raised instead of performing a disallowed action. Never caught and continued past."""


@dataclass
class Allowlist:
    """What the agent may touch. Configurable per capability/tenant."""

    # Exact hosts (with port) the agent may navigate to or act within.
    allowed_hosts: set[str] = field(default_factory=set)
    # Regexes for permitted paths. Empty = any path on an allowed host.
    allowed_path_patterns: list[str] = field(default_factory=list)
    # Which action types are permitted at all on this surface.
    allowed_actions: set[ActionType] = field(
        default_factory=lambda: {
            ActionType.NAVIGATE,
            ActionType.CLICK,
            ActionType.TYPE,
            ActionType.SELECT,
            ActionType.READ,
            ActionType.WAIT_FOR,
            ActionType.PRESS_KEY,
        }
    )
    # Paths the agent may never touch, checked before the allow rules. The harness control
    # plane lives here: the agent must not be able to disable its own error injection.
    denied_path_patterns: list[str] = field(default_factory=lambda: [r"^/_harness"])
    # Highest risk class permitted to run unattended.
    max_unattended_risk: RiskClass = RiskClass.RISKY

    def check_url(self, url: str) -> None:
        parsed = urlparse(url)
        host = parsed.netloc
        path = parsed.path or "/"

        if parsed.scheme not in {"http", "https"}:
            raise PolicyViolation(f"scheme not permitted: {parsed.scheme!r}")
        if host not in self.allowed_hosts:
            raise PolicyViolation(
                f"host {host!r} is not on the allowlist {sorted(self.allowed_hosts)}"
            )
        for pattern in self.denied_path_patterns:
            if re.search(pattern, path):
                raise PolicyViolation(f"path {path!r} matches deny rule {pattern!r}")
        if self.allowed_path_patterns and not any(
            re.search(p, path) for p in self.allowed_path_patterns
        ):
            raise PolicyViolation(f"path {path!r} matches no allow rule")

    def check_action(self, action: ActionType) -> None:
        if action not in self.allowed_actions:
            raise PolicyViolation(f"action {action.value!r} is not permitted on this surface")

    def check_risk(self, risk: RiskClass, attended: bool) -> None:
        """IRREVERSIBLE always escalates, regardless of approval state.

        We chose block-and-escalate over flag-and-continue. In a bank back office an unwanted
        irreversible action (money moved, notice mailed, account closed) is dramatically more
        expensive than a run that stalls waiting for a person. The cost asymmetry decides it.
        """
        if risk is RiskClass.IRREVERSIBLE and not attended:
            raise PolicyViolation(
                "irreversible action requires human confirmation; escalating rather than acting"
            )
        order = {RiskClass.SAFE: 0, RiskClass.RISKY: 1, RiskClass.IRREVERSIBLE: 2}
        if not attended and order[risk] > order[self.max_unattended_risk]:
            raise PolicyViolation(
                f"risk {risk.value!r} exceeds unattended ceiling "
                f"{self.max_unattended_risk.value!r}"
            )


# ======================================================================================
# Redaction
# ======================================================================================

# Ordered most-specific first so a full SSN is not partially eaten by the number rule.
_REDACTION_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("EMAIL", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\b\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b")),
    ("DOB", re.compile(r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b")),
    ("ACCOUNT", re.compile(r"\b(?:acct|account)\s*#?\s*\d{6,}\b", re.I)),
]


class Redactor:
    """Scrubs sensitive values on the way out of the surface.

    Two mechanisms:
      1. Pattern rules for well-known regulated formats (above).
      2. Registered literal values -- anything the caller passed in marked sensitive=True.
         This catches values whose format we could not have anticipated, which is the gap
         pattern matching alone leaves open.
    """

    def __init__(self) -> None:
        self._literals: dict[str, str] = {}

    def register_secret(self, value: str, label: str = "SECRET") -> None:
        if value and len(value) >= 3:
            self._literals[value] = f"[REDACTED:{label}]"

    def scrub(self, text: str | None) -> str | None:
        if not text:
            return text
        out = text
        for literal, replacement in self._literals.items():
            out = out.replace(literal, replacement)
        for label, pattern in _REDACTION_RULES:
            out = pattern.sub(f"[REDACTED:{label}]", out)
        return out

    def scrub_dict(self, data: dict) -> dict:
        return {
            k: self.scrub(v) if isinstance(v, str) else (self.scrub_dict(v) if isinstance(v, dict) else v)
            for k, v in data.items()
        }


def default_allowlist(host: str) -> Allowlist:
    """The allowlist used by the demo runs. Deliberately narrow."""
    return Allowlist(
        allowed_hosts={host},
        allowed_path_patterns=[r"^/$", r"^/login$", r"^/nav", r"^/content"],
        denied_path_patterns=[r"^/_harness"],
    )
