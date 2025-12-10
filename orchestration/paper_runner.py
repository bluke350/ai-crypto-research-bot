from __future__ import annotations

import json
import os
import uuid
from datetime import timezone
from src.utils.time import now_iso
from pathlib import Path
from typing import Optional

import pandas as pd

from src.training.inference import ModelWrapper
from src.training.adapter import inference_backtest
from src.execution.order_executor import PaperOrderExecutor
from src.execution.simulators import ExchangeSimulator
from tooling.structured_logging import setup_structured_logger


def run_paper(checkpoint_path: str,
              prices_csv: Optional[str] = None,
              out_root: str = "experiments/artifacts",
              pair: str = "XBT/USD",
              sim_seed: Optional[int] = None,
              steps: Optional[int] = None,
              **executor_kwargs):
    """Run a paper-mode replay/eval using a checkpoint and a prices CSV.

    Writes result.json and summary.json into a new run directory under out_root.
    """
    run_id = str(uuid.uuid4())
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # configure per-run structured logger
    try:
        run_logger = setup_structured_logger(f"{__name__}.{run_id}", file_path=str(out_dir / 'run.log'))
    except Exception:
        run_logger = setup_structured_logger(__name__)

    # load prices
    if prices_csv:
        if not os.path.exists(prices_csv):
            raise FileNotFoundError(prices_csv)
        from src.utils.io import load_prices_csv
        df = load_prices_csv(prices_csv, dedupe='first')
        # try to locate timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        else:
            first = df.columns[0]
            df[first] = pd.to_datetime(df[first])
            df = df.set_index(first)
        if 'close' not in df.columns:
            raise ValueError('prices CSV must contain close column')
        prices = pd.DataFrame({'close': df['close'].astype(float)}, index=df.index)
        prices.index.name = None
    else:
        # minimal synthetic series
        idx = pd.date_range('2020-01-01', periods=120, freq='T')
        import numpy as _np

        rng = _np.random.default_rng(0)
        returns = rng.normal(scale=0.001, size=len(idx))
        price = 100.0 + _np.cumsum(returns)
        prices = pd.DataFrame({'close': price}, index=idx)
        prices.index.name = None

    # load model (allow simple coef-based ckpt)
    wrapper = ModelWrapper.from_file(checkpoint_path)

    # create simulator and executor
    sim = ExchangeSimulator(pair=pair, seed=sim_seed)
    executor = PaperOrderExecutor(simulator=sim)

    # run inference->backtest using executor
    result = inference_backtest(checkpoint_path, prices, sim, executor=executor, method='vol_norm')

    # serialize and save
    serial = {}
    pnl = result.get('pnl')
    if pnl is not None:
        try:
            serial['pnl'] = [[str(idx), float(v)] for idx, v in zip(pnl.index.astype(str).to_list(), pnl.tolist())]
        except Exception:
            serial['pnl'] = []
    executions = result.get('executions')
    if executions is not None:
        try:
            serial['executions'] = [dict(r) for r in executions]
        except Exception:
            serial['executions'] = []
    # attach executor fills for audit
    try:
        serial.setdefault('executions', [])
        serial['executions'] = list(serial['executions']) + list(executor.fills)
    except Exception:
        pass

    with open(out_dir / 'result.json', 'w', encoding='utf-8') as fh:
        json.dump(serial, fh, indent=2)

    summary = {'run_id': run_id, 'created_at': now_iso(), 'n_executions': len(serial.get('executions', []))}
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    return str(out_dir)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--prices-csv', default=None)
    p.add_argument('--out-root', default='experiments/artifacts')
    p.add_argument('--pair', default='XBT/USD')
    p.add_argument('--sim-seed', type=int, default=None)
    args = p.parse_args()
    print('running paper runner...')
    print(run_paper(args.ckpt, prices_csv=args.prices_csv, out_root=args.out_root, pair=args.pair, sim_seed=args.sim_seed))
