"""Run all original examples and check exit codes."""
import subprocess
import sys

examples = [
    "examples/basic_inference.py",
    "examples/evolution_demo.py",
    "examples/load_checkpoint.py",
    "examples/test_wired.py",
]

ok = True
for path in examples:
    r = subprocess.run(
        ["python3", path], capture_output=True, timeout=300)
    status = "OK" if r.returncode == 0 else "FAILED"
    if r.returncode != 0:
        ok = False
        print(f"{path}: {status} (exit {r.returncode})")
        print(r.stdout.decode()[-200:])
        print(r.stderr.decode()[-200:])
    else:
        print(f"{path}: {status}")

print("ALL ORIGINAL EXAMPLES PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
