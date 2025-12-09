#!/usr/bin/env python3
"""Generate diagnostics (CSV + plots) from a walk-forward results.json.

Usage:
  python scripts/generate_walkforward_diagnostics.py <results_json_path> [<out_dir>]

The script writes `diagnostics.csv`, `sharpe_hist.png`, and `final_value_bar.png`
into the specified output directory (defaults to the results.json parent dir).
"""
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_walkforward_diagnostics.py <results_json_path> [<out_dir>]")
        sys.exit(2)
    results_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.dirname(results_path)
    os.makedirs(out_dir, exist_ok=True)

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for seed_entry in data.get('seeds', []):
        seed = seed_entry.get('seed')
        folds = seed_entry.get('folds', [])
        for i, fold in enumerate(folds):
            metrics = fold.get('metrics', {})
            rows.append({
                'seed': seed,
                'fold_index': i,
                'sharpe': metrics.get('sharpe', 0.0),
                'final_value': metrics.get('final_value', 0.0),
            })

    if not rows:
        print('No fold data found in results.json')
        sys.exit(0)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'diagnostics.csv')
    df.to_csv(csv_path, index=False)
    print('Wrote', csv_path)

    # Sharpe histogram
    plt.figure(figsize=(6,4))
    df['sharpe'].hist(bins=20)
    plt.title('Sharpe Distribution (folds)')
    plt.xlabel('Sharpe')
    plt.ylabel('Count')
    hist_path = os.path.join(out_dir, 'sharpe_hist.png')
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print('Wrote', hist_path)

    # Final value bar per fold (grouped by seed)
    try:
        piv = df.pivot(index='fold_index', columns='seed', values='final_value')
    except Exception:
        piv = df.set_index(['seed','fold_index'])['final_value'].unstack(fill_value=0)

    plt.figure(figsize=(10,4))
    piv.plot(kind='bar', width=0.8)
    plt.title('Final Portfolio Value per Fold (by seed)')
    plt.xlabel('Fold Index')
    plt.ylabel('Final Value')
    plt.legend(title='seed')
    val_path = os.path.join(out_dir, 'final_value_bar.png')
    plt.tight_layout()
    plt.savefig(val_path)
    plt.close()
    print('Wrote', val_path)

    print('Diagnostics generation complete.')


if __name__ == '__main__':
    main()
