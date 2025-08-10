import os
import sys
import time
import argparse
import subprocess
import pandas as pd

"""
Periodically aggregate partial rewire-sweep outputs and run interim analysis
until a target number of rewire levels is available (or timeout).
"""


def run(cmd: list[str]) -> int:
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="runs/rewire_sweep_dense")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--interval", type=int, default=60, help="Seconds between checks")
    ap.add_argument("--until_levels", type=int, default=4, help="Stop when this many rewire levels available")
    ap.add_argument("--max_checks", type=int, default=120, help="Max iterations before exit")
    args = ap.parse_args()

    checks = 0
    while checks < args.max_checks:
        checks += 1
        print(f"\n[monitor] Iteration {checks}")
        # Aggregate
        rc = run([args.python, "scripts/aggregate_rewire_partial.py", "--dir", args.dir])
        if rc != 0:
            print("[monitor] Aggregation failed; retrying next cycle.")
        # Analyze
        rc = run([args.python, "scripts/analyze_rewire_current.py", "--dir", args.dir])
        # Count rewire levels
        thr_cur = os.path.join(args.dir, "rewire_sweep_thresholds_current.csv")
        n_levels = 0
        if os.path.isfile(thr_cur):
            try:
                df = pd.read_csv(thr_cur)
                n_levels = int(df["RewireP"].nunique()) if "RewireP" in df.columns else 0
            except Exception:
                pass
        print(f"[monitor] Levels available: {n_levels}")
        if n_levels >= args.until_levels:
            print("[monitor] Target level count reached; exiting monitor.")
            sys.exit(0)
        time.sleep(max(1, args.interval))

    print("[monitor] Max checks reached; exiting.")


if __name__ == "__main__":
    main()
