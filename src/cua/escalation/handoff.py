"""
Human-in-the-loop escalation and control transfer.

THE CONTROL MODEL IS THE POINT.
The brief is explicit that a human must take over the SAME live session, not a fresh one, and
that there must be a way to know who is (or should be) in control. So control is modelled as an
explicit state machine with a single owner at any instant, persisted to disk so it is a
queryable fact rather than a convention held in one process's memory:

    AUTOMATION -> INTERVENTION_REQUESTED -> HUMAN -> RESUMING -> AUTOMATION
                                              |
                                              +-> ABANDONED

Only the current owner may act. Automation checks the lock before every action once an
intervention is open; the operator surface checks it before handing control back. Two actors
driving one browser session is the failure mode this prevents, and it is a nasty one: the
symptoms look like UI flakiness rather than a concurrency bug.

HOW THE HUMAN REACHES THE SAME SESSION.
The browser is launched with remote debugging exposed, and the escalation request carries that
endpoint as `session`. An operator attaches over CDP and drives the very same browser context --
same cookies, same session, same half-completed form. Nothing is replayed from the start, which
matters because re-running a flow that already submitted something is how you create duplicates.

WHAT IS MOCKED, AND WHAT IS NOT.
Mocked: the operator console UI. It is a small local web page listing open requests with a
take-control button. The brief explicitly permits this.
NOT mocked: the request payload, the control lock, the state transitions, the CDP endpoint that
makes same-session takeover real, and the record of what the human did. Those are the load-
bearing parts and they are all genuine.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class Controller(str, Enum):
    AUTOMATION = "automation"
    INTERVENTION_REQUESTED = "intervention_requested"
    HUMAN = "human"
    RESUMING = "resuming"
    ABANDONED = "abandoned"


# Legal transitions. Declared rather than implied so an illegal transition is a loud error
# instead of a silently corrupted session.
_TRANSITIONS: dict[Controller, set[Controller]] = {
    Controller.AUTOMATION: {Controller.INTERVENTION_REQUESTED},
    Controller.INTERVENTION_REQUESTED: {Controller.HUMAN, Controller.ABANDONED},
    Controller.HUMAN: {Controller.RESUMING, Controller.ABANDONED},
    Controller.RESUMING: {Controller.AUTOMATION, Controller.ABANDONED},
    Controller.ABANDONED: set(),
}


class ControlTransferError(RuntimeError):
    """Raised on an illegal transition or an action by a non-owner."""


@dataclass
class InterventionRequest:
    """Everything a human needs to act, carried with the request rather than looked up.

    An operator picking this up has no other context. If the payload is insufficient they have
    to go spelunking in logs, which is how a two-minute intervention becomes a twenty-minute one.
    """

    id: str
    capability: str
    version: int
    step: str | None
    intent: str | None
    reason: str
    screenshot: str | None
    session: str | None  # CDP endpoint -- how the human reaches THIS session
    evidence: str | None
    state: Controller = Controller.INTERVENTION_REQUESTED
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    human_actions: list[dict[str, Any]] = field(default_factory=list)
    resolution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["state"] = self.state.value
        return d


class EscalationSink:
    """Persists intervention requests and owns the control lock.

    File-backed rather than a queue or a service. The brief warns against building scaling
    infrastructure, and a directory of JSON files is genuinely sufficient at this scale while
    keeping the seam obvious: swap this class for one backed by a real queue and nothing else
    in the system changes.
    """

    def __init__(self, root: str | Path = "escalations") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, request_id: str) -> Path:
        return self.root / f"{request_id}.json"

    def raise_request(self, context: dict[str, Any]) -> str:
        request = InterventionRequest(
            id=f"esc-{uuid.uuid4().hex[:8]}",
            capability=context.get("capability", "unknown"),
            version=int(context.get("version", 0)),
            step=context.get("step"),
            intent=context.get("intent"),
            reason=context.get("reason", ""),
            screenshot=context.get("screenshot"),
            session=context.get("session"),
            evidence=context.get("evidence"),
        )
        self._write(request)
        return request.id

    def _write(self, request: InterventionRequest) -> None:
        self._path(request.id).write_text(json.dumps(request.to_dict(), indent=2), "utf-8")

    def load(self, request_id: str) -> InterventionRequest:
        data = json.loads(self._path(request_id).read_text("utf-8"))
        data["state"] = Controller(data["state"])
        return InterventionRequest(**data)

    def open_requests(self) -> list[InterventionRequest]:
        out = []
        for path in sorted(self.root.glob("esc-*.json")):
            data = json.loads(path.read_text("utf-8"))
            data["state"] = Controller(data["state"])
            request = InterventionRequest(**data)
            if request.state not in {Controller.ABANDONED, Controller.AUTOMATION}:
                out.append(request)
        return out

    # ------------------------------------------------------------------ transitions

    def _transition(self, request: InterventionRequest, to: Controller) -> InterventionRequest:
        if to not in _TRANSITIONS[request.state]:
            raise ControlTransferError(
                f"illegal control transfer {request.state.value} -> {to.value} for {request.id}"
            )
        request.state = to
        self._write(request)
        return request

    def take_control(self, request_id: str, operator: str) -> InterventionRequest:
        request = self.load(request_id)
        request = self._transition(request, Controller.HUMAN)
        request.human_actions.append(
            {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "operator": operator, "action": "took_control"}
        )
        self._write(request)
        return request

    def record_action(self, request_id: str, operator: str, description: str) -> None:
        """Capture what the human did. Required for audit in a regulated environment, and it
        is also the raw material for improving the capability: a human repeatedly performing
        the same manual step is a missing step in the artifact."""
        request = self.load(request_id)
        if request.state is not Controller.HUMAN:
            raise ControlTransferError(
                f"{request_id} is in {request.state.value}; only the controlling human may act"
            )
        request.human_actions.append(
            {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "operator": operator, "action": description}
        )
        self._write(request)

    def hand_back(self, request_id: str, operator: str, resolution: str) -> InterventionRequest:
        request = self.load(request_id)
        request = self._transition(request, Controller.RESUMING)
        request.human_actions.append(
            {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "operator": operator, "action": "handed_back", "resolution": resolution}
        )
        request.resolution = resolution
        self._write(request)
        return request

    def resume(self, request_id: str) -> InterventionRequest:
        """Automation reclaims control. Separate from hand_back so there is an explicit moment
        where automation re-observes the surface before acting: the human may have left the app
        on a different screen than the one the flow expects."""
        request = self.load(request_id)
        return self._transition(request, Controller.AUTOMATION)

    def abandon(self, request_id: str, reason: str) -> InterventionRequest:
        request = self.load(request_id)
        request.resolution = reason
        return self._transition(request, Controller.ABANDONED)

    def wait_for_handback(self, request_id: str, timeout_s: int = 300) -> InterventionRequest:
        """Block until a human hands control back, or give up.

        A timeout exists because an unattended run that waits forever on a human who went home
        is a hung browser session holding a login. On timeout the request is abandoned and the
        run reports ESCALATED rather than pretending to succeed.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            request = self.load(request_id)
            if request.state in {Controller.RESUMING, Controller.ABANDONED}:
                return request
            time.sleep(1.0)
        return self.abandon(request_id, f"no human response within {timeout_s}s")
