"""
MEMBERCORE 4.2 - a deliberately hostile stand-in for legacy credit-union back-office software.

Why we built our own target instead of using a public demo site:
  1. We can INJECT runtime failures on demand (not-found, validation, permission denial,
     session timeout, interstitials, transient slowness). The brief asks for evidence of a
     replay hitting an exceptional state; a public site cannot give us that reproducibly.
  2. We can make the surface genuinely hostile: framesets, table-based layout, no test IDs,
     non-semantic markup, inline onclick handlers. This is the "no clean DOM" case.
  3. No terms-of-service or rate-limit concerns, and no real credentials or PII.

Deliberate hostility (all intentional, do not "fix"):
  - <frameset> with named frames; content lives two frames deep.
  - Layout tables nested 3-4 deep, spacer GIF-era markup.
  - No id= or data-testid= on any interactive control.
  - Buttons are <input type=button> with inline JS, not <a href>.
  - Class names are presentational (td1, td2, hdr) and reused everywhere.

The ONE concession: form controls carry accessible names via <label for> equivalents in the
form of adjacent table cells, and buttons have value= text. That is exactly the situation in
real legacy apps -- there is no test ID, but there IS an accessible name a human operator
reads. Our locator strategy is built on that fact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from flask import Flask, redirect, render_template, request, session, url_for

app = Flask(__name__)
app.secret_key = "not-a-secret-local-demo-only"  # noqa: S105 - local demo, never real


# --------------------------------------------------------------------------------------
# Seed data. Entirely synthetic. No real PII, no real account numbers.
# --------------------------------------------------------------------------------------


@dataclass
class Member:
    member_id: str
    name: str
    status: str
    savings_balance: str
    checking_balance: str
    # Behaviours we use to exercise the error taxonomy:
    restricted: bool = False  # -> permission denial on detail view
    interstitial: bool = False  # -> a "review notice" dialog before detail renders
    sub_accounts: list[str] = field(default_factory=list)


MEMBERS: dict[str, Member] = {
    "12345": Member("12345", "Dana Whitfield", "Active", "4,182.55", "911.20"),
    "22222": Member("22222", "Corey Almeda", "Active", "18,004.10", "2,340.00", restricted=True),
    "33333": Member("33333", "Priya Raghunathan", "Active", "760.00", "88.15", interstitial=True),
    "45678": Member("45678", "Marcus Oyelaran", "Dormant", "12.00", "0.00"),
}

# Injection flags are set at runtime by the harness (scripts/inject.py) or by query param.
# This models "the app is having a bad day" without us editing code between runs.
INJECT: dict[str, bool] = {
    "slow": False,  # transient slowness on the detail screen
    "session_timeout": False,  # next content request bounces to login
    "app_error": False,  # hard 500 on the detail screen
}


def _slow_if_injected() -> None:
    if INJECT["slow"] or request.args.get("inject") == "slow":
        time.sleep(6)


def _timed_out() -> bool:
    if INJECT["session_timeout"] or request.args.get("inject") == "session_timeout":
        INJECT["session_timeout"] = False  # one-shot, so resume after handoff can succeed
        session.pop("user", None)
        return True
    return not session.get("user")


# --------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------


@app.route("/")
def root():
    if not session.get("user"):
        return render_template("login.html")
    return render_template("frameset.html")


@app.route("/login", methods=["POST"])
def login():
    user = request.form.get("operator", "").strip()
    pw = request.form.get("passcode", "").strip()
    if user and pw:
        session["user"] = user
        return redirect(url_for("root"))
    return render_template("login.html", error="Operator ID and passcode are required.")


@app.route("/nav")
def nav():
    return render_template("nav.html")


@app.route("/content")
def content():
    """Default content pane: the member search screen."""
    if _timed_out():
        return render_template("timeout.html")
    return render_template("search.html")


@app.route("/content/search", methods=["POST"])
def do_search():
    if _timed_out():
        return render_template("timeout.html")

    member_id = request.form.get("memberid", "").strip()

    # Validation error: an expected business condition, not a crash.
    if not member_id:
        return render_template("search.html", error="Member ID is required.")
    if not member_id.isdigit():
        return render_template("search.html", error="Member ID must be numeric.")

    member = MEMBERS.get(member_id)
    if member is None:
        # "Record not found" -- a legitimate business outcome the caller needs.
        return render_template("search.html", notfound=member_id)

    return redirect(url_for("member_detail", member_id=member_id))


@app.route("/content/member/<member_id>")
def member_detail(member_id: str):
    if _timed_out():
        return render_template("timeout.html")
    if INJECT["app_error"] or request.args.get("inject") == "app_error":
        return render_template("apperror.html"), 500

    member = MEMBERS.get(member_id)
    if member is None:
        return render_template("search.html", notfound=member_id)

    if member.restricted and not session.get("elevated"):
        # Permission denial -- business outcome, and the classic escalation trigger.
        return render_template("denied.html", member=member)

    if member.interstitial and not session.get(f"ack_{member_id}"):
        # Unexpected-but-known interstitial. Recoverable: dismiss and continue.
        return render_template("interstitial.html", member=member)

    _slow_if_injected()
    return render_template("member.html", member=member)


@app.route("/content/member/<member_id>/ack", methods=["POST"])
def ack_interstitial(member_id: str):
    session[f"ack_{member_id}"] = True
    return redirect(url_for("member_detail", member_id=member_id))


@app.route("/content/member/<member_id>/subaccount", methods=["GET", "POST"])
def sub_account(member_id: str):
    if _timed_out():
        return render_template("timeout.html")
    member = MEMBERS.get(member_id)
    if member is None:
        return render_template("search.html", notfound=member_id)

    if request.method == "GET":
        return render_template("subaccount.html", member=member)

    nickname = request.form.get("nickname", "").strip()
    acct_type = request.form.get("accttype", "")
    deposit = request.form.get("deposit", "").strip()

    errors = []
    if not nickname:
        errors.append("Nickname is required.")
    if acct_type not in {"SAV", "MMK", "CD"}:
        errors.append("Select an account type.")
    try:
        if float(deposit or "0") < 25:
            errors.append("Opening deposit must be at least $25.00.")
    except ValueError:
        errors.append("Opening deposit must be a number.")

    if errors:
        return render_template("subaccount.html", member=member, errors=errors)

    member.sub_accounts.append(nickname)
    return render_template(
        "confirm.html", member=member, nickname=nickname, acct_type=acct_type, deposit=deposit
    )


# --------------------------------------------------------------------------------------
# Test-harness control plane. Used by the replay evidence runs to inject failures.
# Deliberately separate from the operator-facing surface.
# --------------------------------------------------------------------------------------


@app.route("/_harness/inject/<flag>/<state>", methods=["POST", "GET"])
def set_inject(flag: str, state: str):
    if flag not in INJECT:
        return {"error": f"unknown flag {flag}", "known": list(INJECT)}, 400
    INJECT[flag] = state.lower() in {"1", "true", "on", "yes"}
    return {"flag": flag, "state": INJECT[flag]}


@app.route("/_harness/reset", methods=["POST", "GET"])
def reset():
    for k in INJECT:
        INJECT[k] = False
    for m in MEMBERS.values():
        m.sub_accounts.clear()
    session.clear()
    return {"reset": True}


if __name__ == "__main__":
    app.run(port=5051, debug=False)
