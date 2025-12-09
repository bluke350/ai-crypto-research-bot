from __future__ import annotations

import logging
from typing import Tuple, Dict, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.persistence import db as pdb

LOG = logging.getLogger(__name__)


def should_promote(candidate_metrics: Dict[str, Any], db_url: str | None = None, min_improve: float = 0.10, max_drawdown_increase: float = 0.10) -> Tuple[bool, str]:
    """Decide whether to promote a candidate based on historical best metrics in experiments DB.

    `candidate_metrics` expected keys: `sharpe`, optional `final_value`, optional `drawdown`.
    Returns (bool, reason).
    """
    db_url = db_url or "sqlite:///experiments/registry.db"
    try:
        engine = create_engine(db_url, echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        # find best historic sharpe metric and its run_id
        best_row = session.query(pdb.Metric).filter(pdb.Metric.name == "sharpe").order_by(pdb.Metric.value.desc()).first()
        if best_row is None:
            return True, "no baseline (first candidate)"

        best_run_id = best_row.run_id
        baseline_metrics = {m.name: m.value for m in session.query(pdb.Metric).filter(pdb.Metric.run_id == best_run_id).all()}

        baseline_sharpe = float(baseline_metrics.get("sharpe", 0.0))
        cand_sharpe = float(candidate_metrics.get("sharpe", 0.0))

        # require sharpe improvement
        if baseline_sharpe <= 0 and cand_sharpe > 0:
            return True, f"baseline non-positive ({baseline_sharpe}), candidate positive ({cand_sharpe})"
        if cand_sharpe <= baseline_sharpe * (1.0 + float(min_improve)):
            return False, f"sharpe {cand_sharpe} not improved by {min_improve*100:.0f}% over baseline {baseline_sharpe}"

        # optional final_value check
        if "final_value" in candidate_metrics and "final_value" in baseline_metrics:
            cand_val = float(candidate_metrics.get("final_value", 0.0))
            base_val = float(baseline_metrics.get("final_value", 0.0))
            if cand_val < base_val:
                return False, f"final_value decreased: candidate {cand_val} < baseline {base_val}"

        # optional drawdown check (if metrics present)
        if "drawdown" in candidate_metrics and "drawdown" in baseline_metrics:
            cand_dd = float(candidate_metrics.get("drawdown", 0.0))
            base_dd = float(baseline_metrics.get("drawdown", 0.0))
            # allow small relative increase in drawdown
            if cand_dd > base_dd * (1.0 + float(max_drawdown_increase)):
                return False, f"drawdown increased too much: candidate {cand_dd} vs baseline {base_dd}"

        return True, f"improved sharpe from {baseline_sharpe} to {cand_sharpe}"
    except Exception as e:
        LOG.exception("promotion check error: %s", e)
        return False, "error during promotion check"
