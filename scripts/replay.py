"""
Replay a saved capability. This is the production execution path.

    python3 scripts/replay.py --capability artifacts/<id>.v1.json --param member_id=12345

Add --inject <flag> to exercise the error taxonomy against the target app's harness.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request
from urllib.parse import urlparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cua.escalation.handoff import EscalationSink  # noqa: E402
from cua.evidence import EvidenceWriter  # noqa: E402
from cua.replay.engine import ReplayEngine, ReplayStatus  # noqa: E402
from cua.safety.policy import Redactor, default_allowlist  # noqa: E402
from cua.schema import ActionType, Capability  # noqa: E402
from cua.surfaces.web import WebSurface  # noqa: E402


def inject(base_url: str, flag: str, state: str = "on") -> None:
    """Flip a failure flag on the target app.

    Note this goes through the harness endpoint directly, NOT through the surface -- the agent
    itself is denied /_harness by the allowlist. The test harness and the automation are
    separate actors on purpose: a system that can disable its own failure injection cannot be
    trusted to demonstrate that it handles failures.
    """
    url = f"{base_url.rstrip('/')}/_harness/inject/{flag}/{state}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        print(f"injected {flag}={state}: {resp.read().decode()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capability", required=True, help="Path to a saved capability JSON.")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--secret", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument(
        "--attended",
        action="store_true",
        help="Run with a human available. Required for DRAFT capabilities and for risky steps.",
    )
    ap.add_argument("--inject", help="Failure flag to inject before replaying (see target app).")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="Replay N times and report stability.")
    args = ap.parse_args()

    cap = Capability.model_validate_json(pathlib.Path(args.capability).read_text())
    params = dict(p.split("=", 1) for p in args.param)
    params.update(dict(s.split("=", 1) for s in args.secret))

    base_url = f"{urlparse(cap.entry_url).scheme}://{urlparse(cap.entry_url).netloc}"
    if args.inject:
        inject(base_url, args.inject)

    statuses: list[str] = []
    last: dict = {}

    for run in range(args.repeat):
        redactor = Redactor()
        evidence = EvidenceWriter.create("replay", redactor=redactor)
        surface = WebSurface(
            default_allowlist(urlparse(cap.entry_url).netloc),
            redactor,
            headless=not args.headed,
        )
        surface.start()
        try:
            engine = ReplayEngine(
                cap,
                surface,
                evidence,
                attended=args.attended,
                escalation_sink=EscalationSink(),
            )
            result = engine.run(params)
        finally:
            surface.stop()

        last = result.to_dict()
        statuses.append(result.status.value)
        evidence.write_json("result.json", last)

        if args.repeat > 1:
            print(f"run {run + 1}/{args.repeat}: {result.status.value}")

    print()
    print(json.dumps(last, indent=2))
    print()
    print(f"evidence: {last.get('error', {}).get('evidence') or evidence.root}")

    if args.repeat > 1:
        # Stability signal. Business outcomes count as successes: "no such member" means the
        # capability worked correctly, and treating it as flakiness would quarantine a
        # healthy artifact.
        good = sum(1 for s in statuses if s in {"success", "business_outcome"})
        print(f"stability: {good}/{len(statuses)} ({good / len(statuses):.0%})  {statuses}")

    return 0 if last["status"] in {"success", "business_outcome"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
