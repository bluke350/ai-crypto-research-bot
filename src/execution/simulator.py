from typing import Dict, Any, List, Optional
from src.execution.order_models import Order
from src.execution.fee_models import compute_fee
from src.execution.slippage import sqrt_impact_slippage
from src.execution.latency import LatencyModel
from src.execution.cost_models import FeeModel, SlippageModel, LatencySampler
import random
import numpy as _np


class Simulator:
    def __init__(
        self,
        rules: Dict[str, Any] = None,
        fee_model: Optional[FeeModel] = None,
        slippage_model: Optional[SlippageModel] = None,
        latency_model: Optional[LatencySampler] = None,
        seed: Optional[int] = None,
        # explicit simulator parameters (prefer these over `rules` dict)
        partial_fill_fraction: float = 1.0,
        partial_fill_slices: int = 1,
        book_depth: float = 1.0,
        slippage_k: float = 0.1,
        slippage_daily_vol: float = 0.02,
        maker_bps: int = 16,
        taker_bps: int = 26,
        fixed_fee_pct: Optional[float] = None,
        fixed_slippage_pct: Optional[float] = None,
    ):
        # backwards-compatible rules mapping: callers may still pass a `rules` dict;
        # explicit args take precedence, then rules are used as a fallback.
        self.rules = rules or {}
        self.orders: List[Order] = []
        self.fills: List[Dict[str, Any]] = []
        self.fee_model = fee_model
        self.slippage_model = slippage_model
        # explicit simulator behavior params
        self.partial_fill_fraction = float(self.rules.get("partial_fill_fraction", partial_fill_fraction))
        self.partial_fill_slices = int(self.rules.get("partial_fill_slices", partial_fill_slices))
        self.book_depth = float(self.rules.get("book_depth", book_depth))
        self.slippage_k = float(self.rules.get("slippage_k", slippage_k))
        self.slippage_daily_vol = float(self.rules.get("slippage_daily_vol", slippage_daily_vol))
        self.maker_bps = int(self.rules.get("maker_bps", maker_bps))
        self.taker_bps = int(self.rules.get("taker_bps", taker_bps))
        self.fixed_fee_pct = (self.rules.get("fixed_fee_pct") if self.rules.get("fixed_fee_pct") is not None else fixed_fee_pct)
        self.fixed_slippage_pct = (self.rules.get("fixed_slippage_pct") if self.rules.get("fixed_slippage_pct") is not None else fixed_slippage_pct)
        # deterministic seed for tests/training when provided
        if seed is not None:
            random.seed(seed)
            _np.random.seed(seed)
        self.latency = latency_model or LatencyModel(seed=seed)
        self.stats = {"fees": 0.0, "slippage": 0.0, "latencies_ms": []}

    def place_order(self, order: Order, market_price: Optional[float] = None, is_maker: bool = False) -> Dict[str, Any]:
        """Place an order and return a simulated fill dict.

        This is still a simplified immediate-fill simulator but applies fee and slippage models.
        """
        self.orders.append(order)
        price = market_price if market_price is not None else (order.price or 0.0)

        # support partial fills via explicit simulator attributes (fraction 0..1)
        partial_frac = float(getattr(self, "partial_fill_fraction", 1.0))
        filled_size = order.size * partial_frac
        filled_notional = abs(price * filled_size)

        # slippage as fraction (allow tuning via provided SlippageModel or rules)
        if self.slippage_model is not None:
            slippage_pct = float(self.slippage_model.estimate_pct(filled_notional))
        else:
            # support a fixed slippage percentage override for deterministic cost modeling
            if getattr(self, "fixed_slippage_pct", None) is not None:
                slippage_pct = float(self.fixed_slippage_pct)
            else:
                daily_vol = getattr(self, "slippage_daily_vol", 0.02)
                k = getattr(self, "slippage_k", 0.1)
                slippage_pct = sqrt_impact_slippage(filled_notional, daily_vol=daily_vol, k=k)
                # reduce impact if book depth is provided (deeper book -> lower slippage)
                book_depth = float(getattr(self, "book_depth", 1.0))
                if book_depth > 1.0:
                    slippage_pct = slippage_pct / (book_depth ** 0.5)

        # optional stochastic slippage: scale by lognormal draw around a base
        if "stochastic_slippage_sigma" in self.rules:
            base = slippage_pct
            if base == 0.0:
                base = float(self.rules.get("stochastic_base_slippage_pct", 0.0))
            mu = float(self.rules.get("stochastic_slippage_mu", 0.0))
            sigma = float(self.rules.get("stochastic_slippage_sigma", 0.0))
            if sigma > 0.0:
                draw = float(_np.random.lognormal(mean=mu, sigma=sigma))
                slippage_pct = base * draw
        slippage_amt = price * slippage_pct

        # adjust fill price by slippage in the adverse direction
        fill_price = price + (slippage_amt if filled_size > 0 else -slippage_amt)
        # latency sampling: support both LatencyModel.sample() and LatencySampler.sample_ms()
        try:
            # prefer sample_ms if present
            if hasattr(self.latency, 'sample_ms'):
                latency_ms = float(self.latency.sample_ms())
            else:
                latency_ms = float(self.latency.sample())
        except Exception:
            latency_ms = 0.0

        # compute fee based on executed notional (filled_notional)
        # support a fixed fee percent override (e.g., 0.00075)
        if getattr(self, "fixed_fee_pct", None) is not None:
            fee = abs(filled_notional) * float(self.fixed_fee_pct)
        else:
            # prefer provided FeeModel instance, otherwise fallback to configured maker/taker bps
            if self.fee_model is not None:
                fee = float(self.fee_model.compute(filled_notional, is_maker=is_maker))
            else:
                fee = compute_fee(filled_notional, order.side, is_maker=is_maker,
                                  maker_bps=getattr(self, "maker_bps", 16),
                                  taker_bps=getattr(self, "taker_bps", 26))

        # support multi-slice fills to emulate partial fills arriving over time
        slices = int(getattr(self, "partial_fill_slices", 1))
        if slices <= 1:
            fill = {
                "order_id": order.order_id,
                "filled_size": filled_size,
                "avg_fill_price": fill_price,
                "notional": filled_notional,
                "fee": fee,
                "slippage": slippage_amt,
                "latency_ms": latency_ms,
                "status": "filled",
            }
            self.fills.append(fill)
            # update stats
            self.stats["fees"] += fee
            self.stats["slippage"] += slippage_amt
            self.stats["latencies_ms"].append(latency_ms)
            return fill

        # if slices > 1, return a single aggregated fill but simulate slice-level stats
        per_slice = filled_size / slices if slices > 0 else filled_size
        total_fee = 0.0
        total_slippage = 0.0
        slice_records = []
        for i in range(slices):
            slice_notional = abs(price * per_slice)
            slice_fee = compute_fee(slice_notional, order.side, is_maker=is_maker,
                                     maker_bps=getattr(self, "maker_bps", 16),
                                     taker_bps=getattr(self, "taker_bps", 26))
            slice_slippage_pct = sqrt_impact_slippage(slice_notional, daily_vol=getattr(self, "slippage_daily_vol", 0.02), k=getattr(self, "slippage_k", 0.1))
            if getattr(self, "book_depth", 1.0) > 1.0:
                slice_slippage_pct = slice_slippage_pct / (getattr(self, "book_depth", 1.0) ** 0.5)
            slice_slippage_amt = price * slice_slippage_pct
            total_fee += slice_fee
            total_slippage += slice_slippage_amt
            slice_records.append({"size": per_slice, "fee": slice_fee, "slippage": slice_slippage_amt})

        agg_fill = {
            "order_id": order.order_id,
            "filled_size": filled_size,
            "avg_fill_price": fill_price,
            "notional": filled_notional,
            "fee": total_fee,
            "slippage": total_slippage,
            "latency_ms": latency_ms,
            "status": "filled",
            "slices": slice_records,
        }
        self.fills.append(agg_fill)

        # update stats with aggregates
        self.stats["fees"] += total_fee
        self.stats["slippage"] += total_slippage
        self.stats["latencies_ms"].append(latency_ms)
        return agg_fill

    def cancel_order(self, order_id: str) -> bool:
        before = len(self.orders)
        self.orders = [o for o in self.orders if o.order_id != order_id]
        return len(self.orders) < before
