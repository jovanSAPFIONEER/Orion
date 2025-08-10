import os, glob, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Aggregates runs/*_backward_harsh pattern threshold results into one CSV and adds
# a near-threshold ignition latency estimate (mean IgnLat_mean for SOAs within ±1 of logistic threshold).
# Produces: pattern_compare_thresholds.csv and a scatter plot threshold vs sync.

def collect(run_glob: str = 'runs/*_backward_harsh'):
    rows = []
    for path in glob.glob(run_glob):
        thr_csv = os.path.join(path, 'multi_size_thresholds.csv')
        det_csv = os.path.join(path, 'multi_size_masking_detailed.csv')
        if not (os.path.isfile(thr_csv) and os.path.isfile(det_csv)):
            continue
        try:
            df_thr = pd.read_csv(thr_csv)
            df_det = pd.read_csv(det_csv)
        except Exception as e:
            print(f'[warn] failed to read {path}: {e}')
            continue
        if df_thr.empty: continue
        # Assume single N row per file (current design)
        r = df_thr.iloc[0].to_dict()
        th = r.get('Threshold50_logistic', np.nan)
        near_lat = np.nan
        if not np.isnan(th):
            # rows within ±1 SOA of threshold
            band = df_det[np.abs(df_det['SOA'] - th) <= 1.0]
            if not band.empty:
                # Weight by Trials if available
                if 'Trials' in band.columns:
                    w = band['Trials'].fillna(0).values.astype(float)
                    if w.sum() == 0: w = np.ones_like(w)
                    lat_vals = band['IgnLat_mean'].values.astype(float)
                    if np.isfinite(lat_vals).any():
                        # Weighted mean ignoring NaN
                        mask = np.isfinite(lat_vals)
                        if mask.any():
                            near_lat = float(np.average(lat_vals[mask], weights=w[mask]))
                else:
                    lat_vals = band['IgnLat_mean'].values.astype(float)
                    if np.isfinite(lat_vals).any():
                        near_lat = float(np.nanmean(lat_vals))
        r['NearThresh_IgnLat'] = near_lat
        r['RunDir'] = os.path.basename(path)
        rows.append(r)
    return pd.DataFrame(rows)

if __name__ == '__main__':
    df = collect()
    if df.empty:
        print('No pattern runs found matching glob.')
        raise SystemExit(0)
    out_csv = 'runs/pattern_compare_thresholds.csv'
    os.makedirs('runs', exist_ok=True)
    df.to_csv(out_csv, index=False)
    print('Wrote', out_csv)
    # Scatter: threshold vs sync
    if 'Threshold50_logistic' in df.columns and 'Sync_mean' in df.columns:
        plt.figure(figsize=(5.2,4.4))
        x = df['Sync_mean'].values
        y = df['Threshold50_logistic'].values
        for i, row in df.iterrows():
            plt.scatter(row['Sync_mean'], row['Threshold50_logistic'], s=55)
            label = str(row.get('Pattern', row.get('RunDir','?')))
            plt.text(row['Sync_mean']+0.002, row['Threshold50_logistic']+0.015, label, fontsize=8)
        plt.xlabel('Sync_mean')
        plt.ylabel('Threshold50_logistic')
        plt.title('Masking Threshold vs Synchrony')
        # Optional correlation
        if np.isfinite(x).all() and np.isfinite(y).all() and len(x) >= 2:
            xm, ym = x - x.mean(), y - y.mean()
            corr = float((xm*ym).sum() / (math.sqrt((xm**2).sum() * (ym**2).sum()) + 1e-12))
            plt.suptitle(f'Threshold vs Sync (r={corr:.3f})', fontsize=9, y=0.97)
        plt.tight_layout()
        png = 'runs/pattern_threshold_vs_sync.png'
        pdf = 'runs/pattern_threshold_vs_sync.pdf'
        plt.savefig(png, dpi=300)
        plt.savefig(pdf)
        plt.close()
        print('Saved scatter:', png, pdf)
    # Also print concise table
    cols = ['Pattern','Threshold50_logistic','Threshold50_lo','Threshold50_hi','Sync_mean','NearThresh_IgnLat']
    for c in cols:
        if c not in df.columns:
            print('Missing column', c)
    present = [c for c in cols if c in df.columns]
    print(df[present].to_string(index=False))
