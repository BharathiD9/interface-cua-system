# Computer-Use Automation System

An LLM discovers how to drive a legacy application UI once; the run is recorded as a typed,
versioned **capability artifact**; that artifact is then **replayed deterministically** with no
model in the decision loop. Built for the case where a back-office banking application exposes
no API and the only way in is to drive the UI the way a human operator would.

Submission for the interface.ai take-home. Design rationale is in `REPORT.md`.

## Status

| Component | State |
|---|---|
| Target app (hostile legacy surface) | done |
| Capability artifact schema | done |
| Surface abstraction + Playwright adapter | done |
| Safety guardrails (allowlist, redaction) | done |
| LLM discovery loop | done |
| Deterministic replay engine | in progress |
| Human-in-the-loop escalation | in progress |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env          # add your ANTHROPIC_API_KEY
```

## Run the target application

`MEMBERCORE 4.2` is a deliberately hostile stand-in for legacy credit-union back-office
software: frameset, nested layout tables, no test IDs, inline handlers.

```bash
python3 target_app/app.py        # serves http://127.0.0.1:5051
```

Sign on with any non-empty operator ID and passcode. Seeded members:

| Member ID | Behaviour |
|---|---|
| `12345` | clean happy path |
| `45678` | dormant account |
| `22222` | permission denied (escalation trigger) |
| `33333` | compliance interstitial (recoverable condition) |
| anything else | record not found (business outcome) |

Runtime failures can be injected without editing code:

```bash
curl http://127.0.0.1:5051/_harness/inject/slow/on
curl http://127.0.0.1:5051/_harness/inject/session_timeout/on
curl http://127.0.0.1:5051/_harness/inject/app_error/on
curl http://127.0.0.1:5051/_harness/reset
```

## Verify your setup

```bash
python3 scripts/verify_setup.py
```

## Demo path

**Discovery** — a real LLM-driven run against the live surface:

```bash
python3 scripts/discover.py \
  --goal "look up member 12345 and read their current savings balance" \
  --param member_id=12345 \
  --secret operator_id=op-demo --secret operator_passcode=demo-passcode \
  --capability-id membercore.lookup_savings_balance
```

Writes `artifacts/membercore.lookup_savings_balance.v1.json` and an evidence directory
under `evidence/discovery-*/` containing a JSONL run log and per-step screenshots.

Add `--headed` to watch the browser. Add `--planner mock` to run the identical loop with a
scripted action sequence and no API calls — useful for verifying setup, and so this repo is
runnable without a key. The mock is not a substitute for the real run: the brief requires a
genuine LLM-driven discovery pass, and `--planner llm` is the default.

### `--param` vs `--secret`

Both become `{{name}}` templates in the artifact. `--secret` additionally registers the value
with the redactor, so it is scrubbed at the observation boundary and at the evidence-write
boundary, and the resulting input parameter is marked `sensitive`. Credential values appear
in neither the artifact nor the logs — replay supplies them at invocation time.

**Replay** — commands land here once the replay engine is complete.
