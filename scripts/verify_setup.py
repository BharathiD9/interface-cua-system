"""Verify the environment is wired up correctly before running discovery."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

ok = True
def check(label, fn):
    global ok
    try:
        fn(); print(f"  OK    {label}")
    except Exception as e:
        ok = False; print(f"  FAIL  {label}: {type(e).__name__}: {e}")

print("Checking imports...")
check("cua.schema", lambda: __import__("cua.schema", fromlist=["Capability"]))
check("cua.surfaces.base", lambda: __import__("cua.surfaces.base", fromlist=["Surface"]))
check("cua.surfaces.web", lambda: __import__("cua.surfaces.web", fromlist=["WebSurface"]))
check("cua.safety.policy", lambda: __import__("cua.safety.policy", fromlist=["Allowlist"]))

print("Checking browser...")
def browser():
    from playwright.sync_api import sync_playwright
    p = sync_playwright().start(); b = p.chromium.launch(); b.close(); p.stop()
check("chromium launches", browser)

print("Checking target app on http://127.0.0.1:5051 ...")
def app():
    import urllib.request
    with urllib.request.urlopen("http://127.0.0.1:5051/", timeout=5) as r:
        assert r.status == 200
check("target app responding", app)

print()
print("ALL GOOD - ready for the discovery run." if ok else "Some checks failed. See above.")
sys.exit(0 if ok else 1)
