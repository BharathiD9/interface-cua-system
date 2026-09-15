"""
Deterministic replay: the production execution path.

No model is consulted here. Given an artifact and typed inputs, this executes the recorded
steps, evaluates declared conditions, verifies the checkpoint, and returns a structured result.

THE RESULT CONTRACT IS THREE-WAY, NOT TWO-WAY.
The glossary in the brief calls conflating business outcomes with failures "the most common
design mistake here," so the split is enforced by the type system rather than by convention:

  SUCCESS           the checkpoint held and every declared output was extracted.
  BUSINESS_OUTCOME  the app gave a legitimate answer that is not the happy path -- no such
                    member, permission denied, invalid input. The caller needs this. It is
                    not an error and it must not raise, retry, or page a human.
  FAILURE           something is actually wrong. Carries the step, what was expected, what was
                    observed, and a screenshot, because the only useful failure is a
                    debuggable one.
  ESCALATED         we cannot safely proceed and a human has been asked to take the session.

A caller can branch on status without parsing a message string, which is the whole point: the
capability is invoked by an agent, and an agent cannot reliably interpret prose.

WHERE DETERMINISM COMES FROM.
  1. Locator ladders resolved in tier order, with ambiguity treated as failure. The same page
     yields the same control every time, or we stop.
  2. Every step carries a post-condition. We never assume a click worked.
  3. Waits are condition-based (wait for the expected assertion to hold), never sleeps. A
     sleep is a guess about timing; an assertion is a statement about state.
  4. Conditions are evaluated in the artifact's declared order, first match wins. Ordering is
     data, so the behaviour is reviewable rather than emergent.

WHAT REPLAY DOES NOT DO.
It never improvises. If a step's locator does not resolve, replay fails or escalates -- it does
not ask a model to find something similar. Bounded LLM recovery is listed as a stretch goal in
the brief and is deliberately out of scope: the value of this path is that it is predictable,
and a fallback that re-introduces a model on the unhappy path quietly removes that property
exactly when it matters most.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cua.evidence import EvidenceWriter
from cua.safety.policy import PolicyViolation
from cua.schema import (
    ActionType,
    ApprovalState,
    Assertion,
    AssertionKind,
    Capability,
    Classification,
    ConditionHandler,
    ResponseType,
    Step,
)
from cua.surfaces.base import Observation


class ReplayStatus(str, Enum):
    SUCCESS = "success"
    BUSINESS_OUTCOME = "business_outcome"
    FAILURE = "failure"
    ESCALATED = "escalated"


@dataclass
class StepRecord:
    """Per-step telemetry. The resolved tier is the drift signal -- see ladder design."""

    step_id: str
    action: str
    resolved_tier: int | None = None
    resolved_strategy: str | None = None
    duration_ms: int = 0
    retries: int = 0
    note: str = ""


@dataclass
class ReplayResult:
    status: ReplayStatus
    capability_id: str
    capability_version: int
    outputs: dict[str, Any] = field(default_factory=dict)

    # BUSINESS_OUTCOME
    outcome_code: str | None = None
    outcome_detail: str | None = None

    # FAILURE -- enough to debug without re-running
    failed_step: str | None = None
    expected: str | None = None
    observed: str | None = None
    evidence_dir: str | None = None
    screenshot: str | None = None

    # ESCALATED
    escalation_id: str | None = None

    steps: list[StepRecord] = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """What a calling agent receives. Deliberately flat and JSON-safe."""
        base: dict[str, Any] = {
            "status": self.status.value,
            "capability": f"{self.capability_id}@v{self.capability_version}",
            "duration_ms": self.duration_ms,
        }
        if self.status is ReplayStatus.SUCCESS:
            base["outputs"] = self.outputs
        elif self.status is ReplayStatus.BUSINESS_OUTCOME:
            base["outcome"] = {"code": self.outcome_code, "detail": self.outcome_detail}
        elif self.status is ReplayStatus.FAILURE:
            base["error"] = {
                "step": self.failed_step,
                "expected": self.expected,
                "observed": self.observed,
                "evidence": self.evidence_dir,
                "screenshot": self.screenshot,
            }
        elif self.status is ReplayStatus.ESCALATED:
            base["escalation"] = {"id": self.escalation_id, "evidence": self.evidence_dir}
        base["steps"] = [
            {"id": s.step_id, "tier": s.resolved_tier, "strategy": s.resolved_strategy,
             "ms": s.duration_ms, "retries": s.retries}
            for s in self.steps
        ]
        return base


# ======================================================================================
# Assertions
# ======================================================================================


def render_template(text: str | None, params: dict[str, Any]) -> str | None:
    """Substitute {{param}} placeholders. Unknown placeholders are left intact on purpose:
    silently emptying one would turn a wiring bug into a mysteriously wrong action."""
    if not text:
        return text
    out = text
    for key, value in params.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


def evaluate(assertion: Assertion, obs: Observation, params: dict[str, Any]) -> tuple[bool, str]:
    """Evaluate one assertion against an observation. Returns (held, what_we_observed)."""
    scope = ".".join(assertion.frame_path)
    if assertion.frame_path:
        text = obs.text_by_frame.get(assertion.frame_path[-1], "")
    else:
        text = "\n".join(obs.text_by_frame.values())
    flat = " ".join(text.split())

    if assertion.kind is AssertionKind.TEXT_PRESENT:
        needle = render_template(assertion.params["text"], params) or ""
        return (needle.lower() in flat.lower(), f"frame[{scope}] text: {flat[:280]}")

    if assertion.kind is AssertionKind.TEXT_ABSENT:
        needle = render_template(assertion.params["text"], params) or ""
        return (needle.lower() not in flat.lower(), f"frame[{scope}] text: {flat[:280]}")

    if assertion.kind is AssertionKind.CONTROL_PRESENT:
        role = assertion.params.get("role")
        name = render_template(assertion.params.get("name"), params)
        found = any(
            c.role == role and (name is None or c.name == name)
            for c in obs.controls
            if not assertion.frame_path or c.frame_path == assertion.frame_path
        )
        return (found, f"controls: {[c.render() for c in obs.controls][:12]}")

    if assertion.kind is AssertionKind.URL_MATCHES:
        pattern = render_template(assertion.params["pattern"], params) or ""
        return (bool(re.search(pattern, obs.url)), f"url: {obs.url}")

    if assertion.kind is AssertionKind.HTTP_STATUS:
        want = int(assertion.params["status"])
        return (obs.http_status == want, f"http_status: {obs.http_status}")

    return (False, f"unknown assertion kind {assertion.kind}")


def describe(assertion: Assertion, params: dict[str, Any]) -> str:
    rendered = {k: render_template(v, params) if isinstance(v, str) else v
                for k, v in assertion.params.items()}
    return f"{assertion.kind.value}({rendered})"


def wait_for(
    assertion: Assertion, surface: Any, params: dict[str, Any], timeout_ms: int
) -> tuple[bool, Observation, str]:
    """Poll until the assertion holds or the timeout expires.

    Condition-based waiting rather than sleeping. A legacy app's response time varies with
    backend load, so any fixed sleep is either too short (flaky) or too long (slow at volume).
    """
    deadline = time.monotonic() + timeout_ms / 1000
    obs = surface.observe()
    held, observed = evaluate(assertion, obs, params)
    while not held and time.monotonic() < deadline:
        time.sleep(0.25)
        obs = surface.observe()
        held, observed = evaluate(assertion, obs, params)
    return held, obs, observed


# ======================================================================================
# Engine
# ======================================================================================


class ReplayEngine:
    def __init__(
        self,
        capability: Capability,
        surface: Any,
        evidence: EvidenceWriter,
        attended: bool = False,
        escalation_sink: Any | None = None,
    ) -> None:
        self.cap = capability
        self.surface = surface
        self.evidence = evidence
        self.attended = attended
        self.escalation_sink = escalation_sink
        self.steps: list[StepRecord] = []

    # ---------------------------------------------------------------- input validation

    def validate_inputs(self, supplied: dict[str, Any]) -> dict[str, Any]:
        """Validate against the declared contract BEFORE touching the surface.

        A capability is a typed function. Rejecting a malformed member ID here costs
        milliseconds; discovering it three screens in costs a browser session and leaves the
        app in a half-driven state.
        """
        resolved: dict[str, Any] = {}
        for spec in self.cap.inputs:
            if spec.name not in supplied:
                if spec.required:
                    raise ValueError(f"missing required input {spec.name!r}")
                continue
            value = str(supplied[spec.name])
            if spec.pattern and not re.fullmatch(spec.pattern, value):
                raise ValueError(
                    f"input {spec.name!r} does not match required pattern {spec.pattern!r}"
                )
            resolved[spec.name] = value
            if spec.sensitive:
                # Register before any action so the value can never reach a log or screenshot.
                self.surface.redactor.register_secret(value, spec.name.upper())

        unknown = set(supplied) - {s.name for s in self.cap.inputs}
        if unknown:
            raise ValueError(f"unknown inputs not in the capability contract: {sorted(unknown)}")
        return resolved

    def check_approval(self) -> None:
        """Gate unattended replay on approval state.

        A DRAFT capability has been recorded but not reviewed by a human. Its risk
        classifications and condition handlers are therefore unverified, and those are exactly
        the fields that decide whether an irreversible action gets blocked. Running it
        unattended would trust a review that never happened.
        """
        if self.cap.approval is ApprovalState.QUARANTINED:
            raise PolicyViolation(
                f"capability {self.cap.id} is quarantined and may not be replayed"
            )
        if self.cap.approval is ApprovalState.DRAFT and not self.attended:
            raise PolicyViolation(
                f"capability {self.cap.id} is in DRAFT and may only be replayed attended "
                "(--attended). Approve it after review to enable unattended replay."
            )

    # ---------------------------------------------------------------- conditions

    def applicable_conditions(self, step: Step | None) -> list[ConditionHandler]:
        by_id = {c.id: c for c in self.cap.conditions}
        # Capability-wide handlers first, in declared order, then step-specific ones. Order is
        # part of the artifact, so "which condition wins" is reviewable, not emergent.
        handlers = list(self.cap.conditions)
        if step:
            handlers += [by_id[cid] for cid in step.conditions if cid in by_id]
        return handlers

    def detect_condition(
        self, obs: Observation, step: Step | None, params: dict[str, Any]
    ) -> ConditionHandler | None:
        for handler in self.applicable_conditions(step):
            held, _ = evaluate(handler.detect, obs, params)
            if held:
                return handler
        return None

    # ---------------------------------------------------------------- main loop

    def run(self, inputs: dict[str, Any]) -> ReplayResult:
        started = time.monotonic()
        self.evidence.log(
            "replay_started",
            capability=self.cap.id,
            version=self.cap.version,
            approval=self.cap.approval.value,
            attended=self.attended,
        )

        def finish(result: ReplayResult) -> ReplayResult:
            result.duration_ms = int((time.monotonic() - started) * 1000)
            result.steps = self.steps
            result.evidence_dir = str(self.evidence.root)
            self.evidence.log("replay_finished", **result.to_dict())
            return result

        try:
            params = self.validate_inputs(inputs)
            self.check_approval()
        except (ValueError, PolicyViolation) as exc:
            self.evidence.log("replay_rejected", error=str(exc))
            return finish(
                ReplayResult(
                    status=ReplayStatus.FAILURE,
                    capability_id=self.cap.id,
                    capability_version=self.cap.version,
                    failed_step="<precondition>",
                    expected="valid inputs and an approval state permitting this run",
                    observed=str(exc),
                )
            )

        index = 0
        retries: dict[str, int] = {}
        while index < len(self.cap.steps):
            step = self.cap.steps[index]
            outcome = self._execute_step(step, params, retries)

            if outcome["kind"] == "advance":
                index += 1
                continue
            if outcome["kind"] == "retry":
                continue  # same index, counter already incremented
            return finish(outcome["result"])

        # All steps done. Verify the overall checkpoint before extracting anything.
        held, obs, observed = wait_for(self.cap.checkpoint, self.surface, params, 8000)
        if not held:
            # A checkpoint miss is often a condition in disguise; classify before failing.
            handler = self.detect_condition(obs, None, params)
            if handler:
                return finish(self._handle_condition(handler, None, params, obs))
            shot = self.evidence.screenshot_path("checkpoint-failed")
            self.surface.observe(screenshot=True, screenshot_path=shot)
            return finish(
                ReplayResult(
                    status=ReplayStatus.FAILURE,
                    capability_id=self.cap.id,
                    capability_version=self.cap.version,
                    failed_step="<checkpoint>",
                    expected=describe(self.cap.checkpoint, params),
                    observed=observed,
                    screenshot=shot,
                )
            )

        outputs, missing = self._extract_outputs(params)
        if missing:
            shot = self.evidence.screenshot_path("outputs-missing")
            self.surface.observe(screenshot=True, screenshot_path=shot)
            return finish(
                ReplayResult(
                    status=ReplayStatus.FAILURE,
                    capability_id=self.cap.id,
                    capability_version=self.cap.version,
                    failed_step="<outputs>",
                    expected=f"required outputs {missing}",
                    observed="checkpoint held but these outputs could not be located",
                    screenshot=shot,
                )
            )

        return finish(
            ReplayResult(
                status=ReplayStatus.SUCCESS,
                capability_id=self.cap.id,
                capability_version=self.cap.version,
                outputs=outputs,
            )
        )

    # ---------------------------------------------------------------- one step

    def _execute_step(
        self, step: Step, params: dict[str, Any], retries: dict[str, int]
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        record = StepRecord(step_id=step.id, action=step.action.value,
                            retries=retries.get(step.id, 0))

        value = render_template(step.value, params)
        try:
            result = self.surface.act(
                step.action, step.target, value, risk=step.risk, attended=self.attended
            )
        except PolicyViolation as exc:
            # A policy violation is never retried and never recovered from -- it is the system
            # working as designed. Risky/irreversible steps route to a human instead.
            self.evidence.log("policy_blocked", step=step.id, risk=step.risk.value, error=str(exc))
            return {"kind": "stop", "result": self._escalate(step, params, str(exc))}

        if result.resolution:
            record.resolved_tier = result.resolution.tier
            record.resolved_strategy = result.resolution.strategy
        record.duration_ms = int((time.monotonic() - t0) * 1000)
        self.steps.append(record)

        self.evidence.log(
            "step",
            step=step.id,
            action=step.action.value,
            intent=step.intent,
            ok=result.ok,
            tier=record.resolved_tier,
            strategy=record.resolved_strategy,
            ms=record.duration_ms,
        )

        obs = self.surface.observe()

        # Conditions are checked BEFORE the step's own post-condition. A "record not found"
        # page will also fail the expected assertion, and reporting it as an assertion failure
        # instead of a business outcome is precisely the mistake this system exists to avoid.
        handler = self.detect_condition(obs, step, params)
        if handler:
            outcome = self._handle_condition(handler, step, params, obs, action_succeeded=result.ok)
            if outcome is None:  # recovered; re-run this step
                retries[step.id] = retries.get(step.id, 0) + 1
                return {"kind": "retry"}
            if isinstance(outcome, str) and outcome == "continue":
                return {"kind": "advance"}
            return {"kind": "stop", "result": outcome}

        if not result.ok:
            shot = self.evidence.screenshot_path(f"{step.id}-unresolved")
            self.surface.observe(screenshot=True, screenshot_path=shot)
            return {
                "kind": "stop",
                "result": ReplayResult(
                    status=ReplayStatus.FAILURE,
                    capability_id=self.cap.id,
                    capability_version=self.cap.version,
                    failed_step=step.id,
                    expected=f"{step.action.value} on {step.target.description if step.target else value!r}",
                    observed=result.error or "action failed",
                    screenshot=shot,
                ),
            }

        if step.expect:
            held, obs, observed = wait_for(step.expect, self.surface, params, step.timeout_ms)
            if not held:
                handler = self.detect_condition(obs, step, params)
                if handler:
                    outcome = self._handle_condition(
                        handler, step, params, obs, action_succeeded=True
                    )
                    if outcome is None:
                        retries[step.id] = retries.get(step.id, 0) + 1
                        return {"kind": "retry"}
                    if outcome == "continue":
                        return {"kind": "advance"}
                    return {"kind": "stop", "result": outcome}

                shot = self.evidence.screenshot_path(f"{step.id}-expect-failed")
                self.surface.observe(screenshot=True, screenshot_path=shot)
                return {
                    "kind": "stop",
                    "result": ReplayResult(
                        status=ReplayStatus.FAILURE,
                        capability_id=self.cap.id,
                        capability_version=self.cap.version,
                        failed_step=step.id,
                        expected=describe(step.expect, params),
                        observed=observed,
                        screenshot=shot,
                    ),
                }

        return {"kind": "advance"}

    # ---------------------------------------------------------------- condition response

    def _handle_condition(
        self,
        handler: ConditionHandler,
        step: Step | None,
        params: dict[str, Any],
        obs: Observation,
        action_succeeded: bool = False,
    ) -> ReplayResult | str | None:
        """Apply a handler's declared response.

        Returns a ReplayResult to stop, the string "continue" to move to the next step, or
        None to re-run the current step after recovery.
        """
        self.evidence.log(
            "condition_detected",
            condition=handler.id,
            classify=handler.classify.value,
            response=handler.response.value,
            step=step.id if step else None,
        )

        if handler.response is ResponseType.RETURN_OUTCOME:
            detail = " ".join(
                " ".join(t.split()) for t in obs.text_by_frame.values()
            )[:300]
            return ReplayResult(
                status=ReplayStatus.BUSINESS_OUTCOME,
                capability_id=self.cap.id,
                capability_version=self.cap.version,
                outcome_code=handler.outcome_code or handler.id.upper(),
                outcome_detail=detail,
            )

        if handler.response is ResponseType.DISMISS_AND_CONTINUE:
            for recovery in handler.recovery_steps:
                self.evidence.log("recovery_step", condition=handler.id, step=recovery.id)
                res = self.surface.act(
                    recovery.action,
                    recovery.target,
                    render_template(recovery.value, params),
                    risk=recovery.risk,
                    attended=self.attended,
                )
                if not res.ok:
                    shot = self.evidence.screenshot_path(f"recovery-{handler.id}-failed")
                    self.surface.observe(screenshot=True, screenshot_path=shot)
                    return ReplayResult(
                        status=ReplayStatus.FAILURE,
                        capability_id=self.cap.id,
                        capability_version=self.cap.version,
                        failed_step=f"{handler.id}/{recovery.id}",
                        expected=f"recover from {handler.id}",
                        observed=res.error or "recovery step failed",
                        screenshot=shot,
                    )
            # Recovery cleared the obstruction. Do NOT blindly re-run the step that hit it:
            # the condition may have appeared AFTER that step's action already succeeded, in
            # which case re-running repeats a completed action against a screen that has moved
            # on. Getting this wrong is how automation double-submits a form.
            #
            # Decide by evidence, in order of strength:
            #   1. the step's own post-condition now holds     -> advance
            #   2. no post-condition, but the obstruction is gone and the action had already
            #      succeeded                                    -> advance
            #   3. otherwise                                    -> re-run the step
            if step is not None and step.expect is not None:
                held, _, _ = wait_for(step.expect, self.surface, params, 3000)
                if held:
                    self.evidence.log(
                        "recovery_satisfied_expectation",
                        condition=handler.id,
                        step=step.id,
                        note="post-condition held after recovery; advancing",
                    )
                    return "continue"
                return None

            if step is not None and action_succeeded:
                cleared, _ = evaluate(handler.detect, self.surface.observe(), params)
                if not cleared:
                    self.evidence.log(
                        "recovery_cleared_obstruction",
                        condition=handler.id,
                        step=step.id,
                        note=(
                            "action had already succeeded and the condition is gone; advancing "
                            "rather than repeating a completed action"
                        ),
                    )
                    return "continue"
            return None

        if handler.response is ResponseType.RETRY_STEP:
            if step is None:
                return "continue"
            done = sum(1 for s in self.steps if s.step_id == step.id)
            if done > handler.max_retries:
                shot = self.evidence.screenshot_path(f"{step.id}-retries-exhausted")
                self.surface.observe(screenshot=True, screenshot_path=shot)
                return ReplayResult(
                    status=ReplayStatus.FAILURE,
                    capability_id=self.cap.id,
                    capability_version=self.cap.version,
                    failed_step=step.id,
                    expected=f"condition {handler.id} to clear within {handler.max_retries} retries",
                    observed="condition persisted",
                    screenshot=shot,
                )
            time.sleep(handler.backoff_ms / 1000)
            return None

        if handler.response is ResponseType.ESCALATE:
            return self._escalate(step, params, f"condition {handler.id}: {handler.description}")

        # FAIL
        shot = self.evidence.screenshot_path(f"{handler.id}-hard-failure")
        self.surface.observe(screenshot=True, screenshot_path=shot)
        return ReplayResult(
            status=ReplayStatus.FAILURE,
            capability_id=self.cap.id,
            capability_version=self.cap.version,
            failed_step=step.id if step else "<condition>",
            expected="no hard-failure condition present",
            observed=f"{handler.id}: {handler.description}",
            screenshot=shot,
        )

    def _escalate(self, step: Step | None, params: dict[str, Any], reason: str) -> ReplayResult:
        shot = self.evidence.screenshot_path("escalation")
        self.surface.observe(screenshot=True, screenshot_path=shot)
        context = {
            "capability": self.cap.id,
            "version": self.cap.version,
            "step": step.id if step else None,
            "intent": step.intent if step else None,
            "reason": reason,
            "screenshot": shot,
            "session": self.surface.session_handle(),
            "evidence": str(self.evidence.root),
        }
        self.evidence.log("escalation_raised", **context)

        escalation_id = None
        if self.escalation_sink is not None:
            escalation_id = self.escalation_sink.raise_request(context)

        return ReplayResult(
            status=ReplayStatus.ESCALATED,
            capability_id=self.cap.id,
            capability_version=self.cap.version,
            escalation_id=escalation_id,
            failed_step=step.id if step else None,
            observed=reason,
            screenshot=shot,
        )

    # ---------------------------------------------------------------- outputs

    def _extract_outputs(self, params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        outputs: dict[str, Any] = {}
        missing: list[str] = []
        for field_spec in self.cap.outputs:
            raw = self.surface.read(field_spec.source)
            if raw is None:
                if field_spec.required:
                    missing.append(field_spec.name)
                continue
            value = raw
            if field_spec.extract_pattern:
                match = re.search(field_spec.extract_pattern, raw)
                if not match:
                    if field_spec.required:
                        missing.append(field_spec.name)
                    continue
                value = match.group(1) if match.groups() else match.group(0)
            outputs[field_spec.name] = value
            self.evidence.log("output_extracted", name=field_spec.name, value=value)
        return outputs, missing
