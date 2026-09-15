# Design Write-Up

An LLM discovers how to drive a legacy UI once. The run is recorded as a typed, versioned
capability artifact. That artifact is replayed deterministically with no model in the decision
loop, and escalates to a human when it cannot safely proceed.

Measured on the demo flow: discovery takes 7 model round-trips and ~31 seconds. Replaying the
same flow takes **1.8 seconds and zero model calls**. That ratio is the entire argument for
record-once / replay-many.

---

## 1. Architecture

Single Python process, four layers, one dependency direction.

```
scripts/discover.py                scripts/replay.py
        |                                   |
   agent/discovery.py  ---------->  replay/engine.py
   agent/recorder.py                escalation/handoff.py
        |                                   |
        +---------> surfaces/base.py <------+        the seam
                           |
                    surfaces/web.py  (Playwright)
                    safety/policy.py (enforced here)
```

**Key decision: perception is accessibility-shaped, never markup.** Every control is reduced to
`(role, accessible name, value, region, bounds)` — the same tuple Windows UIA or macOS AX would
give. The model never sees HTML. In a frameset with four levels of layout tables the markup is
mostly noise, and a model that learns to depend on markup produces locators that die at the next
template change. This also makes the prompt small enough to stay cheap.

The target app has no `<label for>` anywhere, so accessible names are derived the way a human
reads them: from the text in the preceding table cell. Reproducing that inference is what makes
name-based targeting possible on a surface whose authors never considered accessibility.

**Key decision: guardrails live in the surface adapter, not the agent loop.** Discovery and
replay both call the same adapter, so there is one code path to audit and no way to add a second
that forgets to check. An agent loop that politely asks permission is a guardrail; an adapter
that *cannot* perform a disallowed action is a control.

**Trade-offs.** Single process, no queue, no service boundaries — the brief warns against
building scaling infrastructure, and a directory of JSON files is genuinely sufficient here. The
seams are placed so that swapping the escalation sink for a real queue changes nothing else.
Playwright over a screenshot-only CUA SDK because the accessibility tree is both more reliable
and far cheaper than pixels, with coordinates retained in the schema as the escape hatch for
surfaces that have no tree.

**Model choice.** Groq running `openai/gpt-oss-120b`, behind a three-line `Planner` protocol
with an Anthropic implementation alongside it. This was not academic: the originally chosen model
was moved behind an Enterprise plan mid-project and started returning 404. Because the planner is
a leaf dependency, the fix was one line in a config file. The tool *contract* is defined once and
translated per provider, so the two cannot drift.

---

## 2. Artifact schema

`src/cua/schema.py`. A `Capability` is a callable function with a typed contract: `inputs`,
`outputs`, `steps`, a `checkpoint`, and a list of `conditions`. `input_schema()` emits JSON
Schema directly, so an agent can discover and invoke it by name with typed args.

**Locators are a ladder, not a string.** A legacy app has no test IDs, so any single locator is a
guess. Each step carries an ordered list of candidates — accessible name, label proximity, text,
structural, CSS, coordinates — tier-ordered by expected stability, each with the recorder's
written rationale. The recorder validates every candidate against the live page at record time
and prunes the ones that don't resolve, so the ladder contains only locators proven to work.

Replay records **which tier actually resolved**. That is the drift signal: a step that resolved
on tier 1 last month and resolves on tier 3 today is degrading before it breaks. On the demo
flow buttons resolve on tier 1 (their `value` attribute gives a real accessible name) and form
fields on tier 2 (label proximity), which is a precise, reproducible measurement of how the
legacy surface defeats standard targeting.

**Frame path is separate from the locator.** "Which document" and "which control inside it" are
different concerns. That separation is the seam that makes the desktop story credible: frame path
becomes window/pane path and the ladder is untouched.

**The error taxonomy lives in the artifact.** Condition handlers are data — a detector, a
classification, a response — and the engine is a dumb interpreter of them. Hardcoding "if the page
says 'not found'" into the engine means every tenant's wording needs an engine change. As data,
the taxonomy sits in the same reviewable file as the steps, and a tenant overlay can replace one
detector without forking the flow.

**The model names controls; the recorder authors selectors.** The model chooses *what* to act on;
locator synthesis is deterministic. Locator quality therefore does not vary with sampling. The
same rule governs checkpoints — see §3.

---

## 3. Determinism & error handling

Determinism comes from four properties:

1. **Tier-ordered ladders, ambiguity treated as failure.** A locator matching three controls is
   one that will eventually click the wrong control. Failing loudly now beats a silent wrong
   action in production.
2. **Per-step post-conditions.** Every step asserts the screen it produced (`step_05` asserts it
   reached `MEMBER ACCOUNT SUMMARY`). Without them, a mis-click surfaces at the final checkpoint
   and blames the wrong step.
3. **Condition-based waiting, never sleeps.** A sleep is a guess about timing; an assertion is a
   statement about state.
4. **Declared condition order, first match wins.** Ordering is data, so behaviour is reviewable
   rather than emergent. `session_timeout` is listed first deliberately: a dead session makes
   every later assertion fail for the same underlying reason, and detecting it early is what keeps
   the error message honest.

**The result contract is three-way**, because the brief's glossary calls conflating outcomes with
failures the most common design mistake here:

| Status | Meaning |
|---|---|
| `success` | checkpoint held, declared outputs extracted |
| `business_outcome` | a legitimate answer with a typed code — `MEMBER_NOT_FOUND`, `PERMISSION_DENIED`, `INVALID_INPUT`. Not an error; must not raise, retry, or page anyone |
| `failure` | genuinely wrong: step, expected, observed, screenshot |
| `escalated` | cannot safely proceed; a human has been given the live session |

A caller branches on `status` without parsing prose — which matters, because the caller is an
agent.

Measured behaviour, one recording:

| Input | Result |
|---|---|
| `12345` / `33333` / `45678` | `success` — $4,182.55 / $760.00 / $12.00 |
| `99999` | `business_outcome` `MEMBER_NOT_FOUND` |
| `22222` | `business_outcome` `PERMISSION_DENIED` |
| `abc` | `business_outcome` `INVALID_INPUT` |
| injected app error | `failure` at `step_05`, with screenshot |
| injected session timeout | `escalated`, with a live session handle |

Member `33333` is the interesting one: a compliance interstitial is detected, the declared
recovery step dismisses it, and the flow resumes to success — a recoverable condition handled
without a human and without a model.

**Three bugs found by testing inputs other than the recorded one.** All three are worth naming
because they are the failure modes this class of system actually has:

- *A checkpoint that encoded run-specific data.* Asked to quote proof of success, the model quoted
  `Savings $4,182.55` — the balance from the discovery run. The capability then worked for member
  12345 and failed the checkpoint for everyone else: flow correct, assertion wrong, and the
  failure looks like an app bug. The recorder now validates the proposed checkpoint against
  collected outputs and supplied inputs, substitutes the derived screen heading when it is
  run-specific, and logs the rejection. Caught live on a real Groq run; the log line is in
  `/evidence/`.
- *A blind coordinate click.* After recovering from the interstitial, the engine re-ran a step
  whose control no longer existed; tiers 1–5 correctly matched nothing and the coordinate fallback
  clicked a bare pixel. It passed by luck. Coordinate clicks are now disabled on any surface with
  a working accessibility tree — if the tree says the control is absent, it is absent, and
  clicking its last known pixel is how you hit "Confirm Transfer" where "Search" used to be.
- *Recovery that re-ran completed actions.* Recovery now advances when the step's post-condition
  holds, or when the obstruction cleared and the action had already succeeded. Getting this wrong
  is how automation double-submits a form.

**UI drift**, secondarily: rising resolution tiers are the early signal, and the stability record
(with business outcomes counted as successes, since "no such member" means the capability worked)
feeds the approval gate.

---

## 4. Heterogeneity & multi-tenant

**Surface abstraction.** The `Surface` protocol is nine methods phrased in operator vocabulary —
what can I see, click that control, read that value. The seam between "how we perceive and act"
and "the recorded flow" is exactly this protocol: everything above it is surface-agnostic.

- *Legacy web* is what is already implemented — frameset, nested layout tables, no test IDs.
- *Desktop* needs one new class. `frame_path` becomes a window/pane path; role and accessible name
  come from UIA or AX instead of ARIA; the ladder, schema, replay engine, error taxonomy and
  escalation model are unchanged. This is precisely why tier 1 is accessible name rather than CSS:
  CSS does not exist on a native app, and a schema built around it would have to be rewritten.
- *No queryable tree at all* falls to the coordinate tier, which is recorded but gated off where a
  tree exists.

**Multi-tenant reuse.** One base capability per vendor product, plus a narrow `TenantOverlay` per
institution. An overlay may override the entry URL, per-step targets (branding changed a button's
label), condition detectors (a tenant's app words "not found" differently), and expectations. It
**cannot** change inputs, outputs, or step order. If a tenant needs a different flow, that is a
different capability — forcing that distinction is the point, because it stops overlays from
quietly becoming forks. Stability is tracked per variant, so one tenant's health never vouches for
another's.

**Drift detection across tenants** uses the same signal as within one: the resolved-tier record.
A tenant whose steps start resolving lower, or whose detectors stop matching, surfaces as a
measurable change rather than a support ticket.

---

## 5. Escalation & handoff

**Detecting stuck.** Three triggers: the discovery loop calls `stuck`; a replay hits a condition
classified `ESCALATE`; or the policy blocks a risky or irreversible step. All three route through
one path.

**Control is an explicit state machine** with a single owner at any instant, persisted to disk so
it is a queryable fact rather than a convention in one process's memory:

```
AUTOMATION -> INTERVENTION_REQUESTED -> HUMAN -> RESUMING -> AUTOMATION
                                          +----> ABANDONED
```

Illegal transitions are rejected, and a non-owner attempting to act is rejected. Two actors
driving one browser session is the failure this prevents, and it is nasty because the symptoms
look like UI flakiness rather than a concurrency bug.

**Same session, not a fresh one.** The browser is launched with remote debugging exposed, and the
intervention request carries that endpoint. An operator attaches over CDP and drives the very same
context — same cookies, same session, same half-completed form. Nothing replays from the start,
which matters because re-running a flow that already submitted something creates duplicates.

**Handing back** is deliberately two steps — `hand_back` then `resume` — so there is an explicit
moment where automation re-observes before acting. The human may have left the app on a different
screen than the flow expects. Every human action is recorded, which is both an audit requirement
and useful signal: a human repeatedly performing the same manual step is a missing step in the
artifact.

A timeout exists because an unattended run waiting forever on an operator who went home is a hung
session holding a login; on timeout the request is abandoned and the run reports `escalated` rather
than pretending to succeed.

**Mocked:** the operator console UI. **Not mocked:** the request payload, the control lock, the
transitions, the CDP endpoint that makes same-session takeover real, and the record of what the
human did.

---

## 6. Safety

**Allowlist**, enforced inside the surface adapter: permitted scheme, host, path patterns, and
action types. Deny rules are checked before allow rules. The test harness that injects failures
lives under `/_harness`, which is explicitly denied — a system that can disable its own failure
injection cannot be trusted to demonstrate that it handles failures.

**Risk classes.** `SAFE` runs unattended. `RISKY` requires approval. `IRREVERSIBLE` always
escalates for human confirmation, regardless of approval state. Block-and-escalate rather than
flag-and-continue, on cost asymmetry: in a bank back office an unwanted irreversible action is
dramatically more expensive than a stalled run.

**Approval gate.** Artifacts are born `DRAFT` and may only replay attended. A draft's risk
classifications and condition handlers are unreviewed — and those are the fields that decide
whether an irreversible action gets blocked. Running one unattended would trust a review that
never happened. Three real bugs found after a green first run is the empirical case for this gate.

**Redaction at the observation boundary.** Sensitive values are scrubbed when the surface is read,
before anything reaches the model, the logs, the evidence, or the artifact. Scrubbing on write
would mean raw PII lives in memory and in the model transcript, one forgotten call site away from
being persisted. Two mechanisms: pattern rules for regulated formats (SSN, card, email, phone,
DOB, account numbers) and registered literals for every value declared `sensitive`, which catches
formats we could not have anticipated. Credentials are parameterised as `{{operator_passcode}}`;
the artifact never contains the value.

**Provenance cannot lie.** A mock-planner run records `human_authored` with a note stating plainly
that no model was involved. The approval gate is built on provenance, so an artifact that
overstated its origin would undermine every control above.

**Limits.** Pattern redaction catches anticipated formats only. The allowlist is URL- and
action-shaped and cannot express semantic policy ("may read balances under $10,000"). Risk
classification is recorded at discovery time and stays wrong until a human reviews it — which is
what the approval gate is for. Condition detectors are string matches: specific enough to avoid
false positives, but a tenant that rewords a message needs an overlay.

---

## 7. Cuts

**Deliberately cut, with the seam left real:**

- *Operator console UI.* The handoff mechanism, control lock, and CDP session handle are real; the
  page an operator clicks is not built. The brief permits this explicitly.
- *Desktop surface.* Designed for (§4), not implemented. The abstraction is proven by the fact
  that nothing above `Surface` mentions a browser.
- *Multi-tenant overlays.* `TenantOverlay` and its merge logic exist and are typed; no second
  tenant variant was built to demonstrate it end to end.
- *Bounded LLM fallback on replay failure.* Listed as a stretch goal; deliberately skipped. The
  value of the replay path is that it is predictable, and re-introducing a model on the unhappy
  path removes that property exactly when it matters most.
- *Queues, services, workers.* Not built, per the brief's explicit warning.

**What I would build next, in order:**

1. **A second app variant** to exercise overlays for real — the same flow with a rebranded button
   and a differently worded "not found". That is the highest-value unproven claim in this design.
2. **Multi-run stability scoring wired to the approval gate**, so `DRAFT → APPROVED` is earned by
   measured replays across several inputs rather than granted by hand. `--repeat` already collects
   the data.
3. **A real operator console** — request list, live view, take-control and hand-back.
4. **Tier-regression alerting**: today the resolved tier is recorded but nothing watches it. The
   signal is more valuable than the log.
5. **The write flow** (open a sub-account through to confirmation), which exercises the
   `IRREVERSIBLE` path end to end. The target app already supports it.

**Known weakness I would fix first:** condition detectors are substring matches on screen text.
They work, they are reviewable, and they are why `INVALID_INPUT` needed a second handler when the
app turned out to have two different validation messages. A structured detector (message region +
code) would be more robust, at the cost of assuming more about the surface than a legacy app
reliably provides.
