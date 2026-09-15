"""
The discovery loop: observe -> decide -> act, driven by an LLM, against a live surface.

STRUCTURE
The model is given tools that speak in human terms (click the button named X in frame Y), never
in selector terms. Each turn we hand it a fresh accessibility-shaped observation and the goal;
it calls one tool; we execute it through the Surface (which enforces the allowlist); we record
what happened. Stopping conditions are max steps, wall-clock timeout, an explicit `done`, or an
explicit `stuck` -- the last of which is the escalation trigger rather than an error.

WHY ONE TOOL CALL PER TURN
We deliberately do not let the model batch actions. In a legacy app almost every action
navigates, which invalidates every locator it might have planned against the previous screen.
Forcing a re-observation between actions costs tokens and buys correctness.

WHY `stuck` IS A TOOL
Giving the model an explicit way to say "I cannot safely proceed" is what makes escalation a
designed path rather than an exception handler. A model with no way to give up will invent an
action instead, which in a bank back office is the dangerous failure mode.

MOCK PLANNER
`MockPlanner` replays a hardcoded action sequence through the identical loop, tools, recorder
and artifact writer. It exists so the plumbing can be tested without API spend, and so the repo
is runnable by a reviewer who has no key. It is NOT a substitute for the real run -- the brief
requires a genuine LLM-driven run, and `--planner llm` is the default for exactly that reason.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from cua.agent import recorder
from cua.evidence import EvidenceWriter
from cua.safety.policy import PolicyViolation
from cua.schema import (
    ActionType,
    Assertion,
    AssertionKind,
    Capability,
    InputParam,
    OutputField,
    ParamType,
    Step,
    Target,
)
from cua.surfaces.base import Observation

def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader. Avoids a dependency for something this small, and keeps keys out
    of shell history -- the repo never contains .env, only .env.example."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    except FileNotFoundError:
        pass


_load_dotenv()

MODEL = os.environ.get("CUA_MODEL", "claude-sonnet-5")

TOOLS: list[dict[str, Any]] = [
    {
        "name": "click",
        "description": "Click a control. Identify it by the role and name shown in the observation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "frame": {"type": "string", "description": "Frame name, or '' for top document."},
                "role": {"type": "string", "enum": ["button", "link", "checkbox", "radio"]},
                "name": {"type": "string", "description": "Exact name from the observation."},
                "intent": {"type": "string", "description": "Why, in one short sentence."},
            },
            "required": ["frame", "role", "name", "intent"],
        },
    },
    {
        "name": "type_text",
        "description": "Type into a textbox. Clears existing content first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "frame": {"type": "string"},
                "name": {"type": "string", "description": "Name of the textbox."},
                "text": {"type": "string"},
                "intent": {"type": "string"},
            },
            "required": ["frame", "name", "text", "intent"],
        },
    },
    {
        "name": "select_option",
        "description": "Choose an option in a dropdown by its value.",
        "input_schema": {
            "type": "object",
            "properties": {
                "frame": {"type": "string"},
                "name": {"type": "string"},
                "value": {"type": "string"},
                "intent": {"type": "string"},
            },
            "required": ["frame", "name", "value", "intent"],
        },
    },
    {
        "name": "read_value",
        "description": (
            "Record a displayed value as an OUTPUT of this capability. Use for any data the "
            "goal asks you to retrieve. Identify it by the label text next to the value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "frame": {"type": "string"},
                "label": {"type": "string", "description": "Label text next to the value."},
                "output_name": {"type": "string", "description": "snake_case output name."},
                "intent": {"type": "string"},
            },
            "required": ["frame", "label", "output_name", "intent"],
        },
    },
    {
        "name": "done",
        "description": "The goal is complete and the success screen is visible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "success_text": {
                    "type": "string",
                    "description": "Text on the current screen that proves the goal was reached.",
                },
            },
            "required": ["summary", "success_text"],
        },
    },
    {
        "name": "stuck",
        "description": (
            "You cannot safely proceed: the screen is unexpected, a permission is missing, or "
            "continuing would risk an irreversible action. Prefer this over guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "what_a_human_should_do": {"type": "string"},
            },
            "required": ["reason", "what_a_human_should_do"],
        },
    },
]

SYSTEM_PROMPT = """\
You operate a legacy back-office banking application by driving its user interface, exactly as \
a human operator would. You are shown an accessibility-style observation of the current screen \
after every action: the frames present, the visible text in each, and the controls with their \
roles and names.

Rules:
- Take ONE action per turn, then wait for the new observation. Almost every action navigates, \
so anything you planned against the previous screen may no longer exist.
- Refer to controls only by the role and name shown in the observation. Never invent a name, \
and never write a CSS selector or XPath -- the system builds robust locators for you.
- Read the visible text before acting. If an error, a denial, or an unexpected notice is on \
screen, deal with what is actually there rather than what you expected.
- If the goal asks you to retrieve data, use read_value to record it as a typed output.
- Call done only when the screen itself proves the goal is met, and quote text from it.
- If you cannot proceed safely, call stuck. Guessing is worse than stopping: this is real \
financial software and a wrong action may be irreversible.
- This is a synthetic test environment with fake data. Do not enter real personal information.
"""



def derive_post_condition(obs: Observation, frame_path: list[str]) -> Assertion | None:
    """Derive a per-step post-condition from the screen the action actually produced.

    The brief's glossary defines a checkpoint as the thing that confirms you reached the state
    you expected "rather than assuming the click worked" -- so every step needs one, not just
    the flow as a whole. Without per-step checkpoints a mis-click is not detected until the
    final assertion, and the resulting error blames the wrong step.

    We use the screen's section heading. In this class of application every screen is titled
    (MEMBER INQUIRY, MEMBER ACCOUNT SUMMARY, REVIEW NOTICE), the heading is the most stable
    text on the page, and it is exactly what a human operator uses to confirm where they are.
    Deriving it from the observation rather than asking the model keeps it deterministic: the
    same run always produces the same assertion.

    Returns None when no heading is visible, in which case the step simply carries no
    post-condition rather than one we invented.
    """
    frame_key = frame_path[-1] if frame_path else ""
    text = obs.text_by_frame.get(frame_key, "")
    for line in (ln.strip() for ln in text.splitlines()):
        # A heading here is a short, fully upper-case line. Digits and spaces are allowed
        # (e.g. "MEMBERCORE 4.2"); anything lower-case is body copy, not a heading.
        if 3 <= len(line) <= 60 and line == line.upper() and any(c.isalpha() for c in line):
            return Assertion(
                kind=AssertionKind.TEXT_PRESENT,
                frame_path=list(frame_path),
                params={"text": line},
                description=f"screen heading {line!r} confirms the step landed where expected",
            )
    return None

@dataclass
class DiscoveryResult:
    success: bool
    capability: Capability | None = None
    steps_taken: int = 0
    stop_reason: str = ""
    stuck_context: dict[str, Any] | None = None
    outputs: dict[str, str] = field(default_factory=dict)


class Planner(Protocol):
    def next_action(self, goal: str, observation: str, history: list[dict]) -> dict: ...


class LLMPlanner:
    """Anthropic tool-use planner. The real discovery path."""

    def __init__(self, model: str = MODEL, max_tokens: int = 1024) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = []
        self.last_tool_use_id: str | None = None

    def next_action(self, goal: str, observation: str, history: list[dict]) -> dict:
        if not self.messages:
            self.messages.append(
                {"role": "user", "content": f"GOAL: {goal}\n\nCURRENT SCREEN:\n{observation}"}
            )
        else:
            # Tool results must be returned in the tool_result block the API expects.
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": self.last_tool_use_id,
                            "content": f"{history[-1]['result']}\n\nNEW SCREEN:\n{observation}",
                        }
                    ],
                }
            )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "tool_use":
                self.last_tool_use_id = block.id
                return {"tool": block.name, "args": block.input}

        text = " ".join(b.text for b in response.content if b.type == "text")
        return {
            "tool": "stuck",
            "args": {
                "reason": f"Model returned no tool call: {text[:300]}",
                "what_a_human_should_do": "Review the screen and continue manually.",
            },
        }


class GroqPlanner:
    """Groq (Llama) planner, OpenAI-compatible tool calling.

    Why a second provider exists at all: the brief makes the model choice ours to defend, and
    a discovery loop that only works against one vendor's API is a weaker design than one where
    the planner is swappable. The Planner protocol is three lines wide -- goal, observation,
    history in; a tool call out -- so a provider is a leaf dependency, not a structural one.

    The interesting difference from Anthropic is not the wire format, it is the reliability of
    tool calling. Llama models are more likely to return prose instead of a tool call, or to
    call a tool with a control name that is not on screen. We handle the first case by asking
    once more with an explicit nudge, and the second is caught by the recorder, which validates
    every candidate locator against the live page and fails the step if none resolve. That
    validation is not a Groq-specific workaround -- it is the same check that protects us from
    any model hallucinating a control, which is exactly why it belongs in the recorder rather
    than in a provider adapter.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 1024) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Put it in .env (see .env.example) or export it."
            )
        try:
            from openai import OpenAI  # Groq speaks the OpenAI protocol
        except ModuleNotFoundError as exc:  # pragma: no cover - environment issue, not logic
            raise RuntimeError(
                "The 'openai' package is required for the Groq planner. "
                "Run: pip install -r requirements.txt"
            ) from exc

        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        # Default chosen for the free developer plan. Groq rotates model IDs and moves models
        # behind Enterprise plans without notice, so treat this as a default, not a
        # commitment: scripts/list_models.py prints what a given key can actually reach.
        self.model = model or os.environ.get("CUA_MODEL", "openai/gpt-oss-120b")
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._last_call_id: str | None = None
        self._tools = [_to_openai_tool(t) for t in TOOLS]

    def next_action(self, goal: str, observation: str, history: list[dict]) -> dict:
        if len(self.messages) == 1:
            self.messages.append(
                {"role": "user", "content": f"GOAL: {goal}\n\nCURRENT SCREEN:\n{observation}"}
            )
        else:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": self._last_call_id,
                    "content": f"{history[-1]['result']}\n\nNEW SCREEN:\n{observation}",
                }
            )

        action = self._ask()
        if action is None:
            # One retry with an explicit instruction. Llama models sometimes narrate a plan
            # instead of calling a tool; a single nudge recovers it far more often than not,
            # and if it does not we stop rather than guess.
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "You replied with text instead of calling a tool. Call exactly one "
                        "tool now, using only control names visible in the observation above."
                    ),
                }
            )
            action = self._ask()

        if action is None:
            return {
                "tool": "stuck",
                "args": {
                    "reason": "Model returned prose rather than a tool call, twice.",
                    "what_a_human_should_do": "Review the current screen and continue manually.",
                },
            }
        return action

    def _ask(self) -> dict | None:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self.messages,
            tools=self._tools,
            tool_choice="auto",
            temperature=0,  # discovery should be as reproducible as a sampled model allows
        )
        message = response.choices[0].message
        self.messages.append(message.model_dump(exclude_none=True))

        calls = message.tool_calls or []
        if not calls:
            return None
        call = calls[0]  # one action per turn; the loop re-observes before the next
        self._last_call_id = call.id
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            return None
        return {"tool": call.function.name, "args": args}


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate our Anthropic-shaped tool definition to OpenAI/Groq shape.

    The tool CONTRACT is defined once in TOOLS and translated per provider, rather than
    maintained twice. If the two ever drifted, the two providers would be driving subtly
    different systems and comparing them would be meaningless.
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


class MockPlanner:
    """Fixed action sequence. For testing the loop without API spend."""

    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)
        self.i = 0

    def next_action(self, goal: str, observation: str, history: list[dict]) -> dict:
        if self.i >= len(self.script):
            return {"tool": "stuck", "args": {"reason": "mock script exhausted",
                                              "what_a_human_should_do": "extend the script"}}
        action = self.script[self.i]
        self.i += 1
        return action


def run_discovery(
    *,
    surface: Any,
    goal: str,
    entry_url: str,
    capability_id: str,
    base_app_id: str,
    planner: Planner,
    evidence: EvidenceWriter,
    declared_inputs: dict[str, str],
    sensitive_inputs: set[str] | None = None,
    max_steps: int = 25,
    timeout_s: int = 300,
) -> DiscoveryResult:
    """Run the observe -> decide -> act loop and record the result as a capability.

    `sensitive_inputs` names the parameters whose concrete values are secrets (credentials).
    They are parameterized exactly like any other input, so the artifact stores {{name}} and
    never the value, and the resulting InputParam is marked sensitive=True so replay knows to
    source it from a secret store and keep it out of logs.
    """
    sensitive_inputs = sensitive_inputs or set()
    evidence.log("goal", goal=goal, entry_url=entry_url, inputs=list(declared_inputs))

    steps: list[Step] = []
    outputs: list[OutputField] = []
    collected: dict[str, str] = {}
    history: list[dict] = []
    started = time.monotonic()

    # Step 0 is always the navigation to the entry point. Recording it explicitly means the
    # artifact is self-contained: replay does not depend on ambient browser state.
    surface.act(ActionType.NAVIGATE, value=entry_url)
    steps.append(
        Step(
            id="step_00",
            intent="Open the application entry point.",
            action=ActionType.NAVIGATE,
            value=entry_url,
        )
    )

    obs: Observation = surface.observe(
        screenshot=True, screenshot_path=evidence.screenshot_path("start")
    )
    evidence.log("observed", url=obs.url, controls=len(obs.controls),
                 screenshot=obs.screenshot_path)

    for turn in range(max_steps):
        if time.monotonic() - started > timeout_s:
            evidence.log("stopped", reason="timeout", seconds=round(time.monotonic() - started))
            return DiscoveryResult(False, steps_taken=len(steps), stop_reason="timeout")

        rendered = obs.render()
        try:
            decision = planner.next_action(goal, rendered, history)
        except Exception as exc:  # noqa: BLE001 - a planner failure must be evidence, not a crash
            evidence.log("planner_error", error=f"{type(exc).__name__}: {exc}")
            return DiscoveryResult(False, steps_taken=len(steps),
                                   stop_reason=f"planner error: {exc}")

        tool, args = decision["tool"], decision["args"]
        evidence.log("decision", turn=turn, tool=tool, args=args)

        # ---- terminal tools -------------------------------------------------------
        if tool == "done":
            checkpoint = recorder.infer_checkpoint(obs, "workspace")
            if args.get("success_text"):
                from cua.schema import Assertion, AssertionKind

                proposed = args["success_text"]

                # A checkpoint must hold for EVERY valid invocation, not just the one we
                # happened to record. Asked to quote proof of success, a model naturally
                # quotes what is on screen -- which includes this run's member name and this
                # run's balance. Accepting that verbatim produces a capability that succeeds
                # for member 12345 and fails the checkpoint for everyone else: the flow works,
                # the assertion does not, and the failure looks like a bug in the app.
                #
                # So the recorder VALIDATES the proposal rather than trusting it. If it
                # contains a value we extracted as an output, or a value supplied as an input,
                # it is run-specific and we fall back to the derived screen heading, which is
                # invariant across invocations. Same principle as locator synthesis: the model
                # identifies WHAT proves success; the recorder decides how to assert it.
                run_specific = [v for v in collected.values() if v and v in proposed]
                run_specific += [
                    v for v in declared_inputs.values() if v and str(v) in proposed
                ]
                if run_specific:
                    evidence.log(
                        "checkpoint_rejected",
                        proposed=proposed,
                        reason="contains run-specific values; would only hold for this input",
                        offending=run_specific,
                        substituted=checkpoint.params.get("text"),
                    )
                else:
                    checkpoint = Assertion(
                        kind=AssertionKind.TEXT_PRESENT,
                        frame_path=["workspace"],
                        params={"text": proposed},
                        description=f"Model-verified success marker: {proposed!r}",
                    )
            surface.observe(screenshot=True, screenshot_path=evidence.screenshot_path("done"))
            evidence.log("done", summary=args.get("summary"), checkpoint=checkpoint.params)

            cap = recorder.assemble_capability(
                capability_id=capability_id,
                name=capability_id.split(".")[-1].replace("_", " ").title(),
                description=goal,
                entry_url=entry_url,
                base_app_id=base_app_id,
                steps=steps,
                inputs=[
                    InputParam(
                        name=k,
                        type=ParamType.STRING,
                        description=f"Value supplied for {k}.",
                        # A secret's example value is itself a secret, so it is never stored.
                        example=None if k in sensitive_inputs else v,
                        sensitive=k in sensitive_inputs,
                    )
                    for k, v in declared_inputs.items()
                ],
                outputs=outputs,
                checkpoint=checkpoint,
                model=getattr(planner, "model", "mock"),
                run_id=evidence.run_id,
            )
            return DiscoveryResult(True, capability=cap, steps_taken=len(steps),
                                   stop_reason="done", outputs=collected)

        if tool == "stuck":
            shot = evidence.screenshot_path("stuck")
            surface.observe(screenshot=True, screenshot_path=shot)
            evidence.log("stuck", **args, screenshot=shot)
            return DiscoveryResult(
                False,
                steps_taken=len(steps),
                stop_reason="stuck",
                stuck_context={**args, "screenshot": shot, "url": obs.url,
                               "step_index": len(steps)},
            )

        # ---- acting tools ---------------------------------------------------------
        step_id = f"step_{len(steps):02d}"
        result_note = ""

        try:
            if tool == "read_value":
                target, meta = recorder.build_target(
                    surface, obs, args["frame"], "text", args["label"],
                    f"the value labelled {args['label']!r}",
                )
                # Reading a displayed value uses label-proximity against the adjacent cell.
                from cua.schema import LocatorCandidate, LocatorStrategy

                target = Target(
                    frame_path=[args["frame"]] if args["frame"] else [],
                    description=f"the value labelled {args['label']!r}",
                    candidates=[
                        LocatorCandidate(
                            strategy=LocatorStrategy.LABEL_PROXIMITY,
                            params={"label": args["label"], "control": "cell"},
                            rationale="Value sits in the table cell adjacent to its label. "
                            "Same strategy used for form fields, so outputs inherit the same "
                            "stability properties and drift signal as actions.",
                        )
                    ],
                )
                value = surface.read(target)
                if value is None:
                    result_note = f"Could not read a value labelled {args['label']!r}."
                else:
                    collected[args["output_name"]] = value
                    outputs.append(
                        OutputField(
                            name=args["output_name"],
                            type=ParamType.MONEY if "$" in value else ParamType.STRING,
                            description=f"Value displayed next to {args['label']!r}.",
                            source=target,
                            extract_pattern=r"\$?([\d,]+\.\d{2})" if "$" in value else None,
                        )
                    )
                    steps.append(
                        Step(id=step_id, intent=args.get("intent", "Read a value."),
                             action=ActionType.READ, target=target)
                    )
                    result_note = f"Read {args['output_name']} = {value}"
                evidence.log("read", label=args["label"], name=args["output_name"],
                             value=collected.get(args["output_name"]))

            else:
                role = {"click": args.get("role", "button"), "type_text": "textbox",
                        "select_option": "combobox"}[tool]
                control_name = args["name"]
                target, meta = recorder.build_target(
                    surface, obs, args["frame"], role, control_name,
                    f"the {role} named {control_name!r}",
                )
                if target is None:
                    result_note = f"Could not find {role} named {control_name!r}. {meta}"
                    evidence.log("resolve_failed", tool=tool, args=args, meta=meta)
                else:
                    evidence.log("ladder", step=step_id, control=control_name, **meta)
                    action = {"click": ActionType.CLICK, "type_text": ActionType.TYPE,
                              "select_option": ActionType.SELECT}[tool]
                    raw_value = args.get("text") or args.get("value")
                    risk = recorder.classify_risk(action, control_name, obs.url)

                    res = surface.act(action, target, raw_value, risk=risk, attended=False)
                    if not res.ok:
                        result_note = f"Action failed: {res.error}"
                        evidence.log("action_failed", step=step_id, error=res.error)
                    else:
                        # Concrete values become {{param}} templates here -- this is what turns
                        # a transcript into a reusable capability, and it is also how secrets
                        # stay out of the artifact.
                        recorded_value = (
                            recorder.parameterize(raw_value, declared_inputs)
                            if raw_value
                            else None
                        )
                        # Defence in depth: parameterization is the mechanism that keeps
                        # credentials out, but if a secret ever reaches this point un-templated
                        # we refuse to record it rather than writing it to disk. A capability
                        # with a redacted step is debuggable; a leaked credential is not
                        # retractable.
                        if recorded_value and surface.redactor.scrub(recorded_value) != recorded_value:
                            evidence.log(
                                "secret_not_parameterized",
                                step=step_id,
                                warning="value matched a registered secret and was redacted "
                                "rather than recorded; declare it as a --secret parameter",
                            )
                            recorded_value = surface.redactor.scrub(recorded_value)

                        # Observe the screen the action produced, and record what we saw as
                        # this step's post-condition. Replay asserts it before moving on.
                        post = surface.observe()
                        steps.append(
                            Step(
                                id=step_id,
                                intent=args.get("intent", tool),
                                action=action,
                                target=target,
                                value=recorded_value,
                                risk=risk,
                                expect=derive_post_condition(post, target.frame_path if target else []),
                            )
                        )
                        result_note = f"{tool} succeeded (locator tier {res.resolution.tier})."
                        evidence.log("acted", step=step_id, tool=tool, risk=risk.value,
                                     tier=res.resolution.tier if res.resolution else None)

        except PolicyViolation as exc:
            # A blocked action is evidence of the guardrail working, not a bug. We tell the
            # model plainly so it can choose a different path or call stuck.
            result_note = f"BLOCKED BY POLICY: {exc}"
            evidence.log("policy_block", tool=tool, args=args, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            result_note = f"Error: {type(exc).__name__}: {exc}"
            evidence.log("error", tool=tool, error=result_note)

        history.append({"tool": tool, "args": args, "result": result_note})
        obs = surface.observe(
            screenshot=True, screenshot_path=evidence.screenshot_path(f"t{turn:02d}")
        )
        evidence.log("observed", turn=turn, url=obs.url, controls=len(obs.controls))

    evidence.log("stopped", reason="max_steps", max_steps=max_steps)
    return DiscoveryResult(False, steps_taken=len(steps), stop_reason="max_steps")
