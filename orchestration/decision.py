from __future__ import annotations

from typing import Dict, Any, Optional


def simple_rule_decision(summary: Dict[str, Any], thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Returns a decision dict: {approved: bool, reason: str}

    thresholds can include: max_drawdown, min_sharpe, min_expected_return
    """
    t = thresholds or {}
    max_dd = float(summary.get("max_drawdown") or 1e9)
    if "max_drawdown" in t and max_dd > float(t["max_drawdown"]):
        return {"approved": False, "reason": f"drawdown {max_dd} > {t['max_drawdown']}"}

    if "min_sharpe" in t:
        sharpe = summary.get("sharpe")
        if sharpe is None or float(sharpe) < float(t["min_sharpe"]):
            return {"approved": False, "reason": f"sharpe {sharpe} < {t['min_sharpe']}"}

    if "min_expected_return" in t:
        avg = summary.get("avg_step_return")
        if avg is None or float(avg) < float(t["min_expected_return"]):
            return {"approved": False, "reason": f"avg_return {avg} < {t['min_expected_return']}"}

    return {"approved": True, "reason": "passed simple thresholds"}


def ai_policy_decision(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder for an AI policy decision. For now we call the simple rule.

    In future this function can load a small model or call a hosted policy evaluator.
    """
    # try to load a trained policy model from `RUN_POLICY_MODEL` env var or default `models/policy.pkl`
    import os
    import warnings

    model_path = os.environ.get("RUN_POLICY_MODEL", "models/policy.pkl")
    try:
        import joblib

        art = joblib.load(model_path)
        model = art.get("model")
        features = art.get("features") or []
        if model is None or not features:
            return simple_rule_decision(summary, thresholds={"max_drawdown": 500.0, "min_sharpe": 0.2})
        # build feature vector
        vec = [float(summary.get(k) or 0.0) for k in features]
        pred = model.predict([vec])[0]
        approved = bool(pred)
        return {"approved": approved, "reason": "ai_model_predict"}
    except Exception as e:
        warnings.warn(f"ai_policy load/predict failed: {e}")
        return simple_rule_decision(summary, thresholds={"max_drawdown": 500.0, "min_sharpe": 0.2})
