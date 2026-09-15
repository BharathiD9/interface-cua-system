"""
The recorder: turns a successful discovery run into a capability artifact.

THE CENTRAL DECISION HERE: THE MODEL DOES NOT WRITE LOCATORS.

The model's tool calls identify a control the way a human would describe it -- "the textbox
labelled Member ID in the workspace frame". It never emits a selector. The recorder then
synthesizes a full ladder of candidates from the observation that was in front of the model at
that moment, and VERIFIES each candidate against the live surface before writing it down.

Two reasons this is better than letting the model emit selectors:

  1. Quality. A model writing CSS produces a plausible-looking selector it cannot test. The
     recorder produces six candidates and empirically checks which ones uniquely resolve, right
     now, on the real page. What lands in the artifact is verified, not guessed.

  2. Separation of concerns. The model is good at "what should happen next" and bad at "what
     will still be true in three months". Locator durability is an engineering property, so an
     engineering component owns it. If we later swap the model, or drop the model entirely for
     a human-authored flow, the locator strategy is unchanged.

Verification also prunes: a candidate that matches zero or several controls at record time is
dropped with a logged reason rather than written into the artifact as dead weight. Coordinates
are the exception -- always kept as the final rung even though they are the least durable,
because on a surface with no queryable tree they are the only thing that works.
"""

from __future__ import annotations

from typing import Any

from cua.schema import (
    ActionType,
    Assertion,
    AssertionKind,
    Capability,
    LocatorCandidate,
    LocatorStrategy,
    RiskClass,
    Step,
    Target,
)
from cua.surfaces.base import Control, Observation


def find_control(obs: Observation, frame: str, role: str, name: str) -> Control | None:
    """Locate the control the model referred to, within the observation it was shown."""
    frame_key = [frame] if frame else []
    exact = [
        c
        for c in obs.controls
        if c.frame_path == frame_key and c.role == role and c.name.strip() == name.strip()
    ]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return exact[0]  # ambiguous by name; the ladder verification will catch it
    loose = [
        c
        for c in obs.controls
        if c.frame_path == frame_key and c.role == role and name.lower() in c.name.lower()
    ]
    return loose[0] if len(loose) == 1 else None


def synthesize_ladder(control: Control) -> list[LocatorCandidate]:
    """Build every candidate this control supports, tier order handled by the schema."""
    cands: list[LocatorCandidate] = []

    if control.name:
        cands.append(
            LocatorCandidate(
                strategy=LocatorStrategy.ACCESSIBLE_NAME,
                params={"role": control.role, "name": control.name},
                rationale=(
                    "Role plus accessible name. Most durable tier: survives markup rewrites and "
                    "has a direct equivalent in desktop accessibility APIs (UIA/AX)."
                ),
            )
        )
        if control.role in {"textbox", "combobox"}:
            cands.append(
                LocatorCandidate(
                    strategy=LocatorStrategy.LABEL_PROXIMITY,
                    params={"label": control.name, "control": control.role},
                    rationale=(
                        "Control in the table cell adjacent to its label text. Legacy table "
                        "forms have no <label for>, but the visual association is what the app "
                        "authors maintain and what operators rely on."
                    ),
                )
            )
        cands.append(
            LocatorCandidate(
                strategy=LocatorStrategy.TEXT,
                params={"role": control.role, "text": control.name},
                rationale="Visible text match. Breaks if the label is reworded or re-branded.",
            )
        )

    if control.region:
        cands.append(
            LocatorCandidate(
                strategy=LocatorStrategy.STRUCTURAL,
                params={"region": control.region, "control": control.role, "index": 0},
                rationale=(
                    f"Nth {control.role} within the {control.region!r} section. Positional, so "
                    "it breaks if a field is inserted -- kept only as a late fallback."
                ),
            )
        )

    if control.bounds:
        x, y, w, h = control.bounds
        cands.append(
            LocatorCandidate(
                strategy=LocatorStrategy.COORDINATES,
                params={"x": x + w // 2, "y": y + h // 2, "viewport": [1280, 900]},
                rationale=(
                    "Screenshot-space centre point. Least durable rung -- breaks on any layout "
                    "shift -- but retained because it is the only strategy that works on a "
                    "surface exposing no queryable tree at all."
                ),
            )
        )

    return cands


def verify_ladder(
    surface: Any,
    frame_path: list[str],
    candidates: list[LocatorCandidate],
    description: str,
) -> tuple[list[LocatorCandidate], list[str]]:
    """Keep only candidates that uniquely resolve right now. Returns (kept, pruned_reasons)."""
    kept: list[LocatorCandidate] = []
    pruned: list[str] = []

    for cand in candidates:
        if cand.strategy is LocatorStrategy.COORDINATES:
            kept.append(cand)  # cannot be verified by resolution; always retained last
            continue
        probe = Target(frame_path=frame_path, candidates=[cand], description=description)
        res = surface.resolve(probe)
        if res.found:
            kept.append(cand)
        else:
            reason = res.tried[0] if res.tried else "no match"
            pruned.append(f"{cand.strategy.value} ({reason})")

    return kept, pruned


def build_target(
    surface: Any,
    obs: Observation,
    frame: str,
    role: str,
    name: str,
    description: str,
) -> tuple[Target | None, dict[str, Any]]:
    """Full pipeline: find the control, synthesize a ladder, verify it, return a Target."""
    control = find_control(obs, frame, role, name)
    if control is None:
        return None, {"error": f"no control {role}:{name!r} in frame {frame!r}"}

    frame_path = [frame] if frame else []
    ladder = synthesize_ladder(control)
    kept, pruned = verify_ladder(surface, frame_path, ladder, description)

    if not kept:
        return None, {"error": "no locator candidate resolved", "pruned": pruned}

    target = Target(frame_path=frame_path, candidates=kept, description=description)
    return target, {
        "kept": [c.strategy.value for c in kept],
        "pruned": pruned,
        "best_tier": min(c.tier for c in kept),
    }


def infer_checkpoint(obs: Observation, frame: str) -> Assertion:
    """Derive a success condition from the final screen.

    We prefer the section heading over arbitrary body text: headings are the most stable
    identifying text on a screen, and asserting on a heading says 'we reached the right screen'
    rather than 'some expected words appeared somewhere'.
    """
    frame_key = [frame] if frame else []
    headings = [
        c.name
        for c in obs.controls
        if c.frame_path == frame_key and c.role == "heading" and c.name
    ]
    regions = [c.region for c in obs.controls if c.frame_path == frame_key and c.region]
    text = headings[0] if headings else (regions[0] if regions else None)

    if text:
        return Assertion(
            kind=AssertionKind.TEXT_PRESENT,
            frame_path=frame_key,
            params={"text": text},
            description=f"Reached the {text!r} screen.",
        )
    return Assertion(
        kind=AssertionKind.URL_MATCHES,
        params={"pattern": ".*"},
        description="Fallback checkpoint: no stable heading found on the final screen.",
    )


def classify_risk(action: ActionType, control_name: str, url: str) -> RiskClass:
    """Conservative default risk classification, recorded per step for human review.

    Heuristic and deliberately cautious: we would rather over-classify and make a reviewer
    downgrade it than under-classify and let an irreversible action run unattended. The
    artifact ships as DRAFT precisely so a human confirms these before unattended replay.
    """
    if action in {ActionType.READ, ActionType.WAIT_FOR, ActionType.NAVIGATE}:
        return RiskClass.SAFE

    name = (control_name or "").lower()
    irreversible_markers = (
        "transfer", "submit application", "close account", "delete", "post ",
        "disburse", "wire", "send notice", "approve", "authorize",
    )
    if any(m in name for m in irreversible_markers):
        return RiskClass.IRREVERSIBLE

    if action in {ActionType.TYPE, ActionType.SELECT}:
        return RiskClass.SAFE  # filling a field changes nothing until it is submitted
    if action is ActionType.CLICK:
        # A click that merely navigates is safe; we cannot tell them apart from the name
        # alone, so anything that looks like a form submission is RISKY.
        return RiskClass.RISKY if "submit" in name or "save" in name else RiskClass.SAFE
    return RiskClass.SAFE


def parameterize(value: str, inputs: dict[str, str]) -> str:
    """Replace concrete values with {{param}} templates.

    This is what makes a recording reusable rather than a one-off transcript: the member ID the
    model happened to type during discovery becomes a parameter the calling agent supplies.
    Longest values first so a short value that is a substring of a longer one cannot corrupt it.
    """
    out = value
    for name, concrete in sorted(inputs.items(), key=lambda kv: -len(str(kv[1]))):
        if concrete and str(concrete) in out:
            out = out.replace(str(concrete), f"{{{{{name}}}}}")
    return out


def default_conditions() -> list:
    """The error taxonomy for MEMBERCORE, expressed as data.

    Ships with every capability recorded against this app. In production this would be a
    per-app library that a recorder attaches by app id, and a tenant overlay could override
    individual detectors when a tenant's build words a message differently.

    Order matters -- first match wins -- so specific detectors precede general ones.
    """
    from cua.schema import Classification, ConditionHandler, ResponseType

    return [
        ConditionHandler(
            id="session_timeout",
            description="Session expired mid-flow. Listed first: it can appear on any screen "
            "and must not be mistaken for a business outcome.",
            detect=Assertion(
                kind=AssertionKind.TEXT_PRESENT,
                frame_path=["workspace"],
                params={"text": "SESSION EXPIRED"},
            ),
            classify=Classification.HARD_FAILURE,
            response=ResponseType.ESCALATE,
            outcome_code=None,
        ),
        ConditionHandler(
            id="app_error",
            description="Application returned an error page. Not recoverable by retrying the "
            "same step; a human needs to see it.",
            detect=Assertion(
                kind=AssertionKind.TEXT_PRESENT,
                frame_path=["workspace"],
                params={"text": "SYSTEM ERROR"},
            ),
            classify=Classification.HARD_FAILURE,
            response=ResponseType.FAIL,
        ),
        ConditionHandler(
            id="member_not_found",
            description="No record for the supplied member ID. A legitimate answer the caller "
            "needs, NOT a failure -- conflating these is the classic design mistake.",
            detect=Assertion(
                kind=AssertionKind.TEXT_PRESENT,
                frame_path=["workspace"],
                params={"text": "No member record found"},
            ),
            classify=Classification.BUSINESS_OUTCOME,
            response=ResponseType.RETURN_OUTCOME,
            outcome_code="MEMBER_NOT_FOUND",
        ),
        ConditionHandler(
            id="permission_denied",
            description="Record is restricted and needs supervisor elevation. A business "
            "outcome the caller must handle, and the natural escalation trigger.",
            detect=Assertion(
                kind=AssertionKind.TEXT_PRESENT,
                frame_path=["workspace"],
                params={"text": "Permission denied"},
            ),
            classify=Classification.BUSINESS_OUTCOME,
            response=ResponseType.RETURN_OUTCOME,
            outcome_code="PERMISSION_DENIED",
        ),
        ConditionHandler(
            id="validation_error",
            description="The app rejected our input. Returned to the caller so it can correct "
            "and re-invoke, rather than retried blindly with the same bad value.",
            detect=Assertion(
                kind=AssertionKind.TEXT_PRESENT,
                frame_path=["workspace"],
                params={"text": "is required"},
            ),
            classify=Classification.BUSINESS_OUTCOME,
            response=ResponseType.RETURN_OUTCOME,
            outcome_code="VALIDATION_ERROR",
        ),
        ConditionHandler(
            id="validation_format",
            description=(
                "The app rejected the input's format. Kept distinct from validation_error: a "
                "missing value is a caller bug, a malformed one is usually bad upstream data, "
                "and a caller may reasonably handle the two differently. Detectors stay "
                "specific to one message rather than being broadened to catch both -- a vague "
                "detector is one that eventually matches something it should not."
            ),
            detect=Assertion(
                kind=AssertionKind.TEXT_PRESENT,
                frame_path=["workspace"],
                params={"text": "must be numeric"},
                description="format validation message on the inquiry screen",
            ),
            classify=Classification.BUSINESS_OUTCOME,
            response=ResponseType.RETURN_OUTCOME,
            outcome_code="INVALID_INPUT",
        ),
        ConditionHandler(
            id="compliance_interstitial",
            description="Known interstitial between us and the target screen. Recoverable: "
            "acknowledge it and carry on. Recovery is an explicit, reviewable step rather "
            "than a blind 'click whatever button is present'.",
            detect=Assertion(
                kind=AssertionKind.TEXT_PRESENT,
                frame_path=["workspace"],
                params={"text": "REVIEW NOTICE"},
            ),
            classify=Classification.RECOVERABLE,
            response=ResponseType.DISMISS_AND_CONTINUE,
            recovery_steps=[
                Step(
                    id="recover_ack_notice",
                    intent="Acknowledge the compliance notice to reach the account summary.",
                    action=ActionType.CLICK,
                    risk=RiskClass.SAFE,
                    target=Target(
                        frame_path=["workspace"],
                        description="the 'Acknowledge and Continue' button",
                        candidates=[
                            LocatorCandidate(
                                strategy=LocatorStrategy.ACCESSIBLE_NAME,
                                params={"role": "button", "name": "Acknowledge and Continue"},
                                rationale="Button text is the accessible name on this surface.",
                            )
                        ],
                    ),
                )
            ],
        ),
    ]


def assemble_capability(
    *,
    capability_id: str,
    name: str,
    description: str,
    entry_url: str,
    base_app_id: str,
    steps: list[Step],
    inputs: list,
    outputs: list,
    checkpoint: Assertion,
    model: str,
    run_id: str,
) -> Capability:
    from cua.schema import Provenance

    # Provenance must never overstate where an artifact came from. A mock-planner run exercises
    # the identical loop, tools, and recorder, which makes it genuinely useful for testing
    # replay without burning API calls -- but it discovered nothing, and an artifact that
    # claimed otherwise would be an artifact that lies to its reviewer. The approval gate is
    # built on provenance, so this field has to be trustworthy.
    is_real_discovery = bool(model) and model != "mock"
    provenance_notes = (
        "Locator ladders synthesized by the recorder and verified against the live surface at "
        "record time. The model named controls; it did not author selectors."
    )
    if not is_real_discovery:
        provenance_notes = (
            "RECORDED BY MOCK PLANNER -- no model was involved in any decision. Produced by "
            "replaying a scripted action sequence through the same loop, tools, and recorder, "
            "to exercise replay without API calls. Not evidence of a discovery run."
        )

    return Capability(
        id=capability_id,
        name=name,
        description=description,
        base_app_id=base_app_id,
        entry_url=entry_url,
        surface_kind="legacy_web",
        inputs=inputs,
        outputs=outputs,
        steps=steps,
        checkpoint=checkpoint,
        conditions=default_conditions(),
        provenance=Provenance(
            base_app_id=base_app_id,
            model=model,
            discovery_run_id=run_id,
            recorded_by="llm_discovery" if is_real_discovery else "human_authored",
            notes=provenance_notes,
        ),
    )
