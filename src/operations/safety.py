from __future__ import annotations

import logging
import os
from typing import Dict, Any

import glob
import pandas as pd

LOG = logging.getLogger(__name__)


def _compute_state_from_exec_logs(artifacts_root: str = "experiments/artifacts") -> Dict[str, Any]:
    """Scan exec_log parquet/csv files under artifacts and compute a naive exposure and realized pnl.

    This is a best-effort aggregator for PoC. It sums filled sizes by side and computes
    notional exposure = sum(net_size * avg_fill_price). Realized P&L is not computed
    precisely here; we approximate daily_loss as negative realized pnl if present.
    """
    exec_files = glob.glob(os.path.join(artifacts_root, "**", "exec_log.parquet"), recursive=True)
    exec_files += glob.glob(os.path.join(artifacts_root, "**", "exec_log.csv"), recursive=True)
    net_position = 0.0
    notional = 0.0
    realized_pnl = 0.0

    for f in exec_files:
        try:
            if f.lower().endswith('.parquet'):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)
        except Exception:
            LOG.exception("failed to read exec log %s", f)
            continue

        # expected columns: side, filled_size, avg_fill_price, filled_size
        for _, r in df.iterrows():
            try:
                side = str(r.get('side', '')).lower()
                size = float(r.get('filled_size') or r.get('filled', r.get('size', 0.0)) or 0.0)
                price = float(r.get('avg_fill_price') or r.get('avg_price') or 0.0)
                if side in ('buy', 'b', 'long'):
                    net = size
                elif side in ('sell', 's', 'short'):
                    net = -size
                else:
                    # unknown side: skip
                    continue
                net_position += net
                notional += net * price
                # if a realized_pnl field exists, aggregate it
                if 'realized_pnl' in r.index:
                    try:
                        realized_pnl += float(r.get('realized_pnl') or 0.0)
                    except Exception:
                        pass
            except Exception:
                continue

    state = {
        'current_exposure': float(abs(notional)),
        'daily_loss': float(-realized_pnl) if realized_pnl < 0 else 0.0,
    }
    return state


def check_limits(state: Dict[str, Any] | None = None, cfg: Dict[str, Any] | None = None) -> bool:
    """Check runtime limits. If `state` is None or missing keys, attempt to infer state from exec logs.

    cfg may contain `max_exposure` and `max_daily_loss` to override defaults.
    """
    try:
        if state is None:
            state = {}

        # if missing keys attempt to compute from artifacts
        if 'current_exposure' not in state or 'daily_loss' not in state:
            inferred = _compute_state_from_exec_logs((cfg or {}).get('artifacts_dir', 'experiments/artifacts'))
            # merge inferred values if missing
            for k, v in inferred.items():
                state.setdefault(k, v)

        current_exposure = float(state.get('current_exposure', 0.0))
        daily_loss = float(state.get('daily_loss', 0.0))
        max_exposure = float((cfg or {}).get('max_exposure', 10.0))
        max_daily_loss = float((cfg or {}).get('max_daily_loss', 1000.0))

        if abs(current_exposure) > max_exposure:
            LOG.warning('exposure %s exceeds max %s', current_exposure, max_exposure)
            return False
        if daily_loss > max_daily_loss:
            LOG.warning('daily loss %s exceeds max %s', daily_loss, max_daily_loss)
            return False
        return True
    except Exception:
        LOG.exception('safety check failed')
        return False
