#!/usr/bin/env python3
"""Run an LLM-driven discovery pass against the target app and save a capability artifact.

  python3 scripts/discover.py --goal "look up member 12345 and read their savings balance" \
      --param member_id=12345 --capability-id membercore.lookup_savings_balance

Add --planner mock to exercise the identical loop with a scripted action sequence and no API
calls -- useful for verifying the plumbing, and so a reviewer without a key can run something.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cua.agent.discovery import (  # noqa: E402
    GroqPlanner,
    LLMPlanner,
    MockPlanner,
    run_discovery,
)
from cua.evidence import EvidenceWriter  # noqa: E402
from cua.safety.policy import Redactor, default_allowlist  # noqa: E402
from cua.surfaces.web import WebSurface  # noqa: E402

# Scripted sequence for --planner mock. Mirrors what we expect the model to do, so the mock
# exercises exactly the same tools, recorder and artifact path as the real run.
MOCK_SCRIPT = [
    {"tool": "type_text", "args": {"frame": "", "name": "Operator ID", "text": "op-demo",
                                   "intent": "Sign on to reach the member screens."}},
    {"tool": "type_text", "args": {"frame": "", "name": "Passcode", "text": "demo-passcode",
                                   "intent": "Complete the sign-on form."}},
    {"tool": "click", "args": {"frame": "", "role": "button", "name": "Sign On",
                               "intent": "Submit the sign-on form."}},
    {"tool": "type_text", "args": {"frame": "workspace", "name": "Member ID", "text": "12345",
                                   "intent": "Enter the member number to look up."}},
    {"tool": "click", "args": {"frame": "workspace", "role": "button", "name": "Search",
                               "intent": "Run the member inquiry."}},
    {"tool": "read_value", "args": {"frame": "workspace", "label": "Savings",
                                    "output_name": "savings_balance",
                                    "intent": "Record the savings balance as the output."}},
    {"tool": "done", "args": {"summary": "Reached the account summary and read the balance.",
                              "success_text": "MEMBER ACCOUNT SUMMARY"}},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", required=True)
    ap.add_argument("--entry-url", default="http://127.0.0.1:5051/")
    ap.add_argument("--capability-id", required=True)
    ap.add_argument("--base-app-id", default="membercore-4.2")
    ap.add_argument("--param", action="append", default=[],
                    help="name=value, repeatable. Values become {{name}} templates.")
    ap.add_argument("--secret", action="append", default=[],
                    help="name=value for credentials. Parameterized like --param, but the "
                         "value is registered with the redactor and marked sensitive, so it "
                         "never reaches the model, the logs, or the artifact.")
    ap.add_argument(
        "--planner",
        choices=["anthropic", "groq", "mock"],
        default="groq",
        help=(
            "Which planner drives discovery. anthropic and groq are real model runs; "
            "mock replays a scripted sequence through the same loop for testing replay "
            "without API spend, and marks the artifact as not model-discovered."
        ),
    )
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--headed", action="store_true", help="Show the browser window.")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    params = dict(p.split("=", 1) for p in args.param)
    secrets = dict(p.split("=", 1) for p in args.secret)
    overlap = set(params) & set(secrets)
    if overlap:
        print(f"error: {sorted(overlap)} declared as both --param and --secret")
        return 2
    all_inputs = {**secrets, **params}

    host = args.entry_url.split("//", 1)[1].split("/", 1)[0]
    allowlist = default_allowlist(host)
    redactor = Redactor()
    # Registering here means the value is scrubbed at the observation boundary -- before it
    # can reach the model prompt, the evidence log, or the artifact.
    for name, value in secrets.items():
        redactor.register_secret(value, name.upper())

    evidence = EvidenceWriter.create("discovery", redactor=redactor)
    surface = WebSurface(allowlist, redactor, headless=not args.headed)
    surface.start()

    try:
        planner = {
            "mock": lambda: MockPlanner(MOCK_SCRIPT),
            "groq": GroqPlanner,
            "anthropic": LLMPlanner,
        }[args.planner]()
        result = run_discovery(
            surface=surface,
            goal=args.goal,
            entry_url=args.entry_url,
            capability_id=args.capability_id,
            base_app_id=args.base_app_id,
            planner=planner,
            evidence=evidence,
            declared_inputs=all_inputs,
            sensitive_inputs=set(secrets),
            max_steps=args.max_steps,
        )
    finally:
        surface.stop()

    print(f"\nrun id      : {evidence.run_id}")
    print(f"evidence    : {evidence.root}")
    print(f"steps taken : {result.steps_taken}")
    print(f"stop reason : {result.stop_reason}")

    if not result.success:
        if result.stuck_context:
            print("\nAgent became stuck -- this is the escalation path, not a crash:")
            print(json.dumps(result.stuck_context, indent=2))
        return 1

    cap = result.capability
    assert cap is not None
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cap.id}.v{cap.version}.json"
    path.write_text(cap.model_dump_json(indent=2), encoding="utf-8")
    evidence.write_text("capability.json", cap.model_dump_json(indent=2))

    print(f"\ncapability  : {path}")
    print(f"inputs      : {[(p.name, 'sensitive' if p.sensitive else 'plain') for p in cap.inputs]}")
    print(f"outputs     : {[o.name for o in cap.outputs]}")
    print(f"outcomes    : {cap.outcome_codes()}")
    print(f"collected   : {result.outputs}")
    print(f"approval    : {cap.approval.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
