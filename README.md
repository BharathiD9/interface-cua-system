# Computer-Use Automation System

An LLM discovers how to drive a legacy application UI once; the run is recorded as a typed,
versioned **capability artifact**; that artifact is then **replayed deterministically** with no
model in the decision loop. Built for the case where a back-office banking application exposes
no API and the only way in is to drive the UI the way a human operator would.

Submission for the interface.ai take-home. Design rationale and trade-offs: **[REPORT.md](REPORT.md)**.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env          # add your GROQ_API_KEY
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
| `22222` | permission denied |
| `33333` | compliance interstitial (recoverable condition) |
| anything else | record not found (business outcome) |

Runtime failures can be injected without editing code:

```bash
curl http://127.0.0.1:5051/_harness/inject/session_timeout/on
curl http://127.0.0.1:5051/_harness/inject/app_error/on
curl http://127.0.0.1:5051/_harness/reset
```

## Verify your setup

```bash
python3 scripts/verify_setup.py
```

## Demo path

### 1. Discovery — a real LLM-driven run against the live surface

```bash
PYTHONPATH=src python3 scripts/discover.py \
  --goal "look up member 12345 and read their current savings balance" \
  --capability-id membercore.lookup_savings_balance \
  --param member_id=12345 \
  --secret operator_id=op1 --secret operator_passcode=demo \
  --headed
```

Writes `artifacts/membercore.lookup_savings_balance.v1.json` plus an evidence directory with a
JSONL run log and per-turn screenshots. `--planner mock` runs the identical loop with a scripted
sequence and no API calls; artifacts so produced are marked as **not** model-discovered.

### 2. Replay — deterministic, no model in the loop

```bash
PYTHONPATH=src python3 scripts/replay.py \
  --capability artifacts/membercore.lookup_savings_balance.v1.json \
  --param member_id=12345 \
  --secret operator_id=op1 --secret operator_passcode=demo \
  --attended
```

`--attended` is required while the capability is `DRAFT`; unattended replay is gated on approval.

### 3. The whole outcome taxonomy

Same artifact, different inputs:

| Input | Result |
|---|---|
| `12345`, `33333`, `45678` | `success` — $4,182.55 / $760.00 / $12.00 |
| `99999` | `business_outcome` `MEMBER_NOT_FOUND` |
| `22222` | `business_outcome` `PERMISSION_DENIED` |
| `abc` | `business_outcome` `INVALID_INPUT` |
| `--inject app_error` | `failure` with step, expectation, observation, screenshot |
| `--inject session_timeout` | `escalated`, carrying the live session handle |

`33333` is worth watching: a compliance interstitial is detected, dismissed by a declared
recovery step, and the flow resumes to success.

## `--param` vs `--secret`

Both become `{{name}}` templates in the artifact. `--secret` additionally registers the value
with the redactor, so it is scrubbed at the observation boundary and never reaches the artifact,
the logs, or the model. Replay supplies credentials at invocation time.

## Evidence

`/evidence/` holds one folder per run: a JSONL event log, screenshots, and the result contract.
Discovery folders also contain the emitted capability.
