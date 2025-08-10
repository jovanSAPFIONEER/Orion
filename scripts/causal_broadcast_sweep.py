#!/usr/bin/env python3
"""
causal_broadcast_sweep.py

Runs a sweep over broadcast_gain to probe causal sensitivity of CAI.
For each gain, runs overnight_full_run with CAI JSON dumping, evaluates with
CV-calibrated evaluator, and aggregates AUC/ECE into a CSV and a quick plot.

Usage (Windows PowerShell example):
    python scripts/causal_broadcast_sweep.py --out runs/causal_bcast_sweep \
            --gains 0.0,0.1,0.22,0.3 --n_mask 80 --n_blink 60 --n_cb 48 --n_dual 60 --boots 600
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable or "python"


def run_cmd(args: List[str]) -> None:
    proc = subprocess.run(args, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(args)} (code {proc.returncode})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Base output directory for sweep")
    p.add_argument("--gains", default="0.0,0.1,0.22,0.3", help="Comma-separated broadcast gains")
    p.add_argument("--n_mask", type=int, default=80)
    p.add_argument("--n_blink", type=int, default=60)
    p.add_argument("--n_cb", type=int, default=48)
    p.add_argument("--n_dual", type=int, default=60)
    p.add_argument("--boots", type=int, default=600)
    args = p.parse_args()

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    gains = [float(x.strip()) for x in args.gains.split(",") if len(x.strip()) > 0]
    rows = []

    for g in gains:
        tag = f"gain_{str(g).replace('.', 'p')}"
        outdir = out_base / tag
        cai_dir = outdir / "cai"
        outdir.mkdir(parents=True, exist_ok=True)
        # 1) Run experiment with given broadcast_gain
        cmd = [
            PY, str(ROOT / "overnight_full_run.py"),
            "--out", str(outdir),
            "--n_mask", str(args.n_mask),
            "--n_blink", str(args.n_blink),
            "--n_cb", str(args.n_cb),
            "--n_dual", str(args.n_dual),
            "--boots", str(args.boots),
            "--dump_cai_json",
            "--cai_dir", str(cai_dir),
            "--broadcast_gain", str(g),
        ]
        run_cmd(cmd)
        # 2) Evaluate with CV-calibrated evaluator
        cv_metrics = outdir / "cai_cv_eval.json"
        cmd_cv = [
            PY, str(ROOT / "scripts" / "eval_cai_cv.py"),
            "--cai_dir", str(cai_dir),
            "--outfile", str(cv_metrics),
        ]
        run_cmd(cmd_cv)
        m = json.loads(cv_metrics.read_text())
        rows.append({
            "broadcast_gain": g,
            "n": int(m.get("n", 0)),
            "AUC": float(m.get("AUC", float("nan"))),
            "ECE": float(m.get("ECE", float("nan"))),
            "pos_rate": float(m.get("pos_rate", float("nan"))),
            "prob_mean": float(m.get("prob_mean", float("nan"))),
            "prob_std": float(m.get("prob_std", float("nan"))),
        })

    df = pd.DataFrame(rows).sort_values("broadcast_gain").reset_index(drop=True)
    csv_path = out_base / "broadcast_gain_sweep.csv"
    df.to_csv(csv_path, index=False)

    # Quick plot
    plt.figure(figsize=(5.5, 4))
    plt.plot(df["broadcast_gain"], df["AUC"], marker="o")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Broadcast gain")
    plt.ylabel("AUC (CV-calibrated)")
    plt.title("CAI vs. broadcast gain")
    plt.tight_layout()
    fig_path = out_base / "broadcast_gain_sweep.png"
    plt.savefig(fig_path, dpi=160)
    print("Saved:", csv_path, fig_path)


if __name__ == "__main__":
    main()
