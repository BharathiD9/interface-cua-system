"""
The capability artifact schema.

This is the contract between three parties:
  - the DISCOVERY loop, which writes it (an LLM figured out how to do the task once),
  - the REPLAY engine, which executes it with no model in the decision loop,
  - the CALLING AGENT, which invokes it by name with typed args and consumes typed outputs.

Four decisions shape everything below.

1. LOCATORS ARE A LADDER, NOT A STRING.
   A legacy app has no test IDs, so any single locator is a guess. We record an ordered list
   of candidates per step, each tagged with the strategy that produced it and the recorder's
   rationale. Replay walks the ladder top-down and records WHICH TIER RESOLVED. That last
   part matters more than it looks: if a capability starts resolving on tier 3 when it used to
   resolve on tier 1, the UI drifted, and we know before it breaks outright.

   Tier order is by expected stability, not convenience:
     ACCESSIBLE_NAME  - role + accessible name. Survives markup rewrites. Works on desktop
                        via UIA/AX, which is why it is tier 1 rather than CSS.
     LABEL_PROXIMITY  - the control in the cell/next to text "Member ID". Legacy table forms
                        have no <label for>, but the visual association is stable.
     TEXT             - visible text of a button/link.
     STRUCTURAL       - nth control of a type within a named region. Brittle but often the
                        only option in a nested-table layout.
     CSS              - last resort. Present because sometimes it is genuinely the most
                        stable thing available; ranked last because in legacy apps it usually
                        is not.
     COORDINATES      - screenshot-space fallback. The only tier that works on a surface with
                        no queryable tree at all.

2. FRAME PATH IS PART OF THE TARGET, NOT THE LOCATOR.
   "Which document am I looking in" and "which control within it" are separate concerns.
   Keeping them separate is what lets the same schema describe a desktop app later: frame_path
   becomes window/pane path, and the locator itself is unchanged. This is the seam the brief
   asks about in 3.7.

3. THE ERROR TAXONOMY LIVES IN THE ARTIFACT, NOT THE ENGINE.
   Hardcoding "if page says 'not found' return NOT_FOUND" into the replay engine means every
   new tenant's wording needs an engine change. Instead each capability carries a list of
   ConditionHandlers: a detector, a classification, and a response. The engine is a dumb
   interpreter of those. Two consequences we want: the taxonomy is reviewable by a human in
   the same file as the steps, and a tenant overlay can override one handler's detector
   without forking the whole capability.

4. OUTPUTS ARE DECLARED, TYPED, AND EXTRACTED BY LOCATOR.
   A capability that returns "whatever text was on the page" is not callable by an agent.
   Every output names its type and the locator that produces it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0.0"


# ======================================================================================
# Locators
# ======================================================================================


class LocatorStrategy(str, Enum):
    """Ordered by expected stability on a legacy surface. Lower tier = tried first."""

    ACCESSIBLE_NAME = "accessible_name"
    LABEL_PROXIMITY = "label_proximity"
    TEXT = "text"
    STRUCTURAL = "structural"
    CSS = "css"
    COORDINATES = "coordinates"


STRATEGY_TIER: dict[LocatorStrategy, int] = {
    LocatorStrategy.ACCESSIBLE_NAME: 1,
    LocatorStrategy.LABEL_PROXIMITY: 2,
    LocatorStrategy.TEXT: 3,
    LocatorStrategy.STRUCTURAL: 4,
    LocatorStrategy.CSS: 5,
    LocatorStrategy.COORDINATES: 6,
}


class LocatorCandidate(BaseModel):
    """One way to find a control. Several of these form a ladder."""

    strategy: LocatorStrategy
    # Strategy-specific payload. Kept as a dict so adding a surface (desktop) does not
    # require changing this class -- the surface adapter interprets its own strategies.
    #   accessible_name: {"role": "textbox", "name": "Member ID"}
    #   label_proximity: {"label": "Member ID", "control": "textbox"}
    #   text:            {"text": "Search", "role": "button"}
    #   structural:      {"region": "MEMBER INQUIRY", "control": "textbox", "index": 0}
    #   css:             {"selector": "form[action='/content/search'] input[type=text]"}
    #   coordinates:     {"x": 118, "y": 242, "viewport": [1280, 800]}
    params: dict[str, Any]
    rationale: str = Field(
        description="Why the recorder believes this candidate is stable. Reviewed by a human."
    )

    @property
    def tier(self) -> int:
        return STRATEGY_TIER[self.strategy]


class Target(BaseModel):
    """Where to act: which document/window, then which control inside it."""

    # Named frames from outermost to innermost. Empty = top document.
    # On a desktop surface this becomes the window/pane path; the locator ladder is unchanged.
    frame_path: list[str] = Field(default_factory=list)
    candidates: list[LocatorCandidate] = Field(min_length=1)
    description: str = Field(description="Human-readable: 'the Member ID search box'.")

    @field_validator("candidates")
    @classmethod
    def _sorted_by_tier(cls, v: list[LocatorCandidate]) -> list[LocatorCandidate]:
        return sorted(v, key=lambda c: c.tier)


# ======================================================================================
# Assertions (checkpoints and condition detectors are the same primitive)
# ======================================================================================


class AssertionKind(str, Enum):
    TEXT_PRESENT = "text_present"
    TEXT_ABSENT = "text_absent"
    CONTROL_PRESENT = "control_present"
    URL_MATCHES = "url_matches"
    HTTP_STATUS = "http_status"


class Assertion(BaseModel):
    kind: AssertionKind
    frame_path: list[str] = Field(default_factory=list)
    # text_present/absent: {"text": "MEMBER ACCOUNT SUMMARY"}  (supports {{param}} templating)
    # control_present:     {"role": "button", "name": "Open Sub-Account"}
    # url_matches:         {"pattern": "/content/member/\\d+"}
    # http_status:         {"status": 500}
    params: dict[str, Any]
    description: str = ""


# ======================================================================================
# Parameters and outputs -- the callable contract
# ======================================================================================


class ParamType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    MONEY = "money"


class InputParam(BaseModel):
    name: str
    type: ParamType
    required: bool = True
    description: str
    example: str | None = None
    pattern: str | None = Field(default=None, description="Regex the value must match.")
    # If true the value is redacted everywhere: logs, evidence, transcripts, artifacts.
    # Redaction happens at the observation boundary, before anything reaches the model.
    sensitive: bool = False


class OutputField(BaseModel):
    name: str
    type: ParamType
    description: str
    # Where to read it from. Same Target machinery as actions -- one locator concept, reused.
    source: Target
    # Optional regex with one capture group, applied to the extracted text.
    # e.g. "\\$([\\d,]+\\.\\d{2})" to pull 4,182.55 out of "$4,182.55".
    extract_pattern: str | None = None
    required: bool = True


# ======================================================================================
# Steps
# ======================================================================================


class ActionType(str, Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    READ = "read"
    WAIT_FOR = "wait_for"
    PRESS_KEY = "press_key"


class RiskClass(str, Enum):
    """Drives the guardrail. See safety/policy.py.

    SAFE        - read-only or trivially reversible. Runs unattended.
    RISKY       - writes state but reversible/low-blast-radius. Runs unattended only if the
                  capability is APPROVED; otherwise requires confirmation.
    IRREVERSIBLE- moves money, sends notices, closes accounts. NEVER runs unattended in this
                  implementation; always escalates for human confirmation. We chose block-and-
                  escalate over flag-and-continue because in a regulated context an unwanted
                  irreversible action is far more costly than a stalled run.
    """

    SAFE = "safe"
    RISKY = "risky"
    IRREVERSIBLE = "irreversible"


class Step(BaseModel):
    id: str = Field(description="Stable within the artifact; referenced by failures and logs.")
    intent: str = Field(description="Why this step exists, in plain language. For reviewers.")
    action: ActionType
    target: Target | None = Field(
        default=None, description="None only for NAVIGATE, which targets a URL."
    )
    # Literal value or a {{param_name}} template resolved from invocation inputs.
    value: str | None = None
    risk: RiskClass = RiskClass.SAFE
    # Post-condition. If absent we assume the click worked -- which is exactly the mistake the
    # brief's glossary warns about, so the recorder is prompted to emit one for every step.
    expect: Assertion | None = None
    timeout_ms: int = 10_000
    # Condition handler ids that apply specifically after this step, in addition to the
    # capability-wide ones. Lets a step opt into a handler that would be noise elsewhere.
    conditions: list[str] = Field(default_factory=list)


# ======================================================================================
# Condition handlers -- the error taxonomy, declarative
# ======================================================================================


class Classification(str, Enum):
    """The three-way split the brief calls the most common design mistake to get wrong."""

    BUSINESS_OUTCOME = "business_outcome"  # legitimate answer: "no such member"
    RECOVERABLE = "recoverable"  # dismiss an interstitial, retry a slow load
    HARD_FAILURE = "hard_failure"  # stop, surface a debuggable error


class ResponseType(str, Enum):
    RETURN_OUTCOME = "return_outcome"  # end the run, report the outcome code to the caller
    DISMISS_AND_CONTINUE = "dismiss_and_continue"  # perform recovery_steps, resume the flow
    RETRY_STEP = "retry_step"  # re-run the current step
    ESCALATE = "escalate"  # hand to a human operator on the live session
    FAIL = "fail"  # hard stop with evidence


class ConditionHandler(BaseModel):
    """Detect a runtime condition and say what to do about it.

    Checked after every step (plus any step-specific ones). Order matters: first match wins,
    so specific handlers should be listed before general ones.
    """

    id: str
    description: str
    detect: Assertion
    classify: Classification
    response: ResponseType
    # For BUSINESS_OUTCOME: the code the caller receives. Part of the public contract, so it
    # is declared here rather than derived from the message text.
    outcome_code: str | None = None
    # For DISMISS_AND_CONTINUE: what to do to clear it before resuming.
    recovery_steps: list[Step] = Field(default_factory=list)
    # For RETRY_STEP.
    max_retries: int = 2
    backoff_ms: int = 1500


# ======================================================================================
# The capability
# ======================================================================================


class ApprovalState(str, Enum):
    DRAFT = "draft"  # replayable only in attended mode
    APPROVED = "approved"  # replayable unattended
    QUARANTINED = "quarantined"  # failed too often; blocked pending review


class Provenance(BaseModel):
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    recorded_by: Literal["llm_discovery", "human_authored", "tenant_override"] = "llm_discovery"
    model: str | None = None
    discovery_run_id: str | None = None
    # Which app variant this was recorded against. Two tenants on the same vendor product
    # share base_app_id and differ by variant_id -- that is the multi-tenant reuse hook.
    base_app_id: str
    variant_id: str | None = None
    notes: str = ""


class StabilityRecord(BaseModel):
    """Feeds the approval gate. Updated by replay, not by the recorder."""

    total_replays: int = 0
    successes: int = 0
    business_outcomes: int = 0
    hard_failures: int = 0
    last_replayed_at: datetime | None = None
    # Tier the locators resolved on most recently, per step. Rising tiers = drift warning.
    last_resolved_tiers: dict[str, int] = Field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        # Business outcomes count as successes for stability purposes: "no such member" means
        # the capability worked correctly. Conflating them would quarantine healthy artifacts.
        if self.total_replays == 0:
            return 0.0
        return (self.successes + self.business_outcomes) / self.total_replays


class Capability(BaseModel):
    """A recorded flow, callable by an AI agent."""

    schema_version: str = SCHEMA_VERSION
    id: str = Field(description="Stable slug, e.g. 'membercore.lookup_savings_balance'.")
    version: int = Field(default=1, description="Bumped on any edit to steps or contract.")
    name: str
    description: str = Field(description="What an agent reads to decide whether to call this.")

    # --- target ---
    base_app_id: str
    entry_url: str
    surface_kind: Literal["web", "legacy_web", "desktop"] = "legacy_web"

    # --- callable contract ---
    inputs: list[InputParam] = Field(default_factory=list)
    outputs: list[OutputField] = Field(default_factory=list)

    # --- the flow ---
    steps: list[Step]
    checkpoint: Assertion = Field(
        description="Overall success condition. Verified before outputs are extracted."
    )
    conditions: list[ConditionHandler] = Field(default_factory=list)

    # --- governance ---
    approval: ApprovalState = ApprovalState.DRAFT
    stability: StabilityRecord = Field(default_factory=StabilityRecord)
    provenance: Provenance

    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the inputs -- this is what we hand a tool-calling agent."""
        type_map = {
            ParamType.STRING: "string",
            ParamType.INTEGER: "integer",
            ParamType.NUMBER: "number",
            ParamType.BOOLEAN: "boolean",
            ParamType.MONEY: "string",
        }
        props: dict[str, Any] = {}
        for p in self.inputs:
            spec: dict[str, Any] = {"type": type_map[p.type], "description": p.description}
            if p.pattern:
                spec["pattern"] = p.pattern
            if p.example:
                spec["examples"] = [p.example]
            props[p.name] = spec
        return {
            "type": "object",
            "properties": props,
            "required": [p.name for p in self.inputs if p.required],
        }

    def outcome_codes(self) -> list[str]:
        """Every business outcome a caller must be prepared to handle. Public contract."""
        return sorted({c.outcome_code for c in self.conditions if c.outcome_code})


# ======================================================================================
# Tenant overlays -- design hook for 3.7, deliberately small
# ======================================================================================


class TenantOverlay(BaseModel):
    """A narrow, reviewable diff against a base capability.

    The multi-tenant answer is NOT "re-record per tenant" and NOT "one artifact fits all".
    It is: one base capability per vendor product, plus a small overlay per tenant that may
    only override things that vary by branding/config -- never the shape of the contract.

    Deliberately cannot change inputs, outputs, or step order. If a tenant needs a different
    flow, that is a different capability, and forcing that distinction is the point: it keeps
    overlays reviewable and stops them from silently becoming forks.
    """

    base_capability_id: str
    base_version: int
    tenant_id: str
    variant_id: str
    entry_url: str | None = None
    # step_id -> replacement Target (branding changed a button's label, say)
    target_overrides: dict[str, Target] = Field(default_factory=dict)
    # condition_id -> replacement ConditionHandler (tenant's app words "not found" differently)
    condition_overrides: dict[str, ConditionHandler] = Field(default_factory=dict)
    # step_id -> replacement Assertion
    expect_overrides: dict[str, Assertion] = Field(default_factory=dict)
    notes: str = ""

    def apply(self, base: Capability) -> Capability:
        merged = base.model_copy(deep=True)
        if self.entry_url:
            merged.entry_url = self.entry_url
        for step in merged.steps:
            if step.id in self.target_overrides:
                step.target = self.target_overrides[step.id]
            if step.id in self.expect_overrides:
                step.expect = self.expect_overrides[step.id]
        merged.conditions = [self.condition_overrides.get(c.id, c) for c in merged.conditions]
        merged.provenance.variant_id = self.variant_id
        merged.provenance.recorded_by = "tenant_override"
        # Stability is per-variant. Inheriting the base's record would let one tenant's health
        # vouch for another's, which is exactly the failure mode overlays are meant to avoid.
        merged.stability = StabilityRecord()
        return merged
