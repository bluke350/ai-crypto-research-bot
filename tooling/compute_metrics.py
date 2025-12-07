"""Small helper to print CSV head and compute final return + max drawdown.
Usage: python tooling/compute_metrics.py results/eval_nav_sim_2k_stochastic.csv
"""
from __future__ import annotations
import sys
import pandas as pd

def max_drawdown(series: pd.Series) -> float:
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    return float(dd.min())

def main():
    if len(sys.argv) < 2:
        print('usage: compute_metrics.py <csv>')
        return
    path = sys.argv[1]
    from src.utils.io import load_prices_csv
    df = load_prices_csv(path, dedupe='first')
    print('head:')
    print(df.head(12).to_string(index=False))
    nav = df['nav']
    initial = float(nav.iloc[0])
    final = float(nav.iloc[-1])
    final_return = (final / initial - 1.0) * 100.0
    mdd = max_drawdown(nav) * 100.0
    print('\nmetrics:')
    print(f'initial_nav: {initial:.2f}')
    print(f'final_nav:   {final:.2f}')
    print(f'final_return: {final_return:.4f}%')
    print(f'max_drawdown: {mdd:.4f}%')

if __name__ == '__main__':
    main()
