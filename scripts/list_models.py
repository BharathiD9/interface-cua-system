"""Print the models your GROQ_API_KEY can actually reach.

Groq rotates model IDs and moves models between free and Enterprise tiers, so a hardcoded
default goes stale. Run this before blaming the code for a 404.
"""
import json
import os
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cua.agent.discovery import _load_dotenv  # noqa: E402

_load_dotenv()
key = os.environ.get("GROQ_API_KEY")
if not key:
    sys.exit("GROQ_API_KEY not set. Add it to .env (see .env.example).")

req = urllib.request.Request(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {key}"},
)
with urllib.request.urlopen(req, timeout=20) as resp:
    data = json.load(resp)

models = sorted(m["id"] for m in data.get("data", []))
chat = [m for m in models if not any(x in m for x in ("whisper", "tts", "guard", "embed"))]

print(f"{len(models)} models reachable with this key.\n")
print("Chat / tool-calling candidates:")
for m in chat:
    print("  ", m)
print("\nUse one with:  CUA_MODEL=<id>  in .env")
