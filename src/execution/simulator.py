from typing import Dict, Any, List, Optional
from src.execution.order_models import Order
from src.execution.fee_models import compute_fee
from src.execution.slippage import sqrt_impact_slippage
from src.execution.latency import LatencyModel
import random
import numpy as _np


class Simulator:
    def __init__(self, rules: Dict[str, Any] = None, latency_model: Optional[LatencyModel] = None, seed: Optional[int] = None):
        self.rules = rules or {}
        self.orders: List[Order] = []
        self.fills: List[Dict[str, Any]] = []
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

        # support partial fills via rules (fraction 0..1)
        partial_frac = float(self.rules.get("partial_fill_fraction", 1.0))
        filled_size = order.size * partial_frac
        filled_notional = abs(price * filled_size)

        # slippage as fraction (allow tuning via rules) — compute from executed notional
        daily_vol = self.rules.get("slippage_daily_vol", 0.02)
        k = self.rules.get("slippage_k", 0.1)
        slippage_pct = sqrt_impact_slippage(filled_notional, daily_vol=daily_vol, k=k)
        # reduce impact if book depth is provided (deeper book -> lower slippage)
        book_depth = float(self.rules.get("book_depth", 1.0))
        if book_depth > 1.0:
            slippage_pct = slippage_pct / (book_depth ** 0.5)
        slippage_amt = price * slippage_pct

        # adjust fill price by slippage in the adverse direction
        fill_price = price + (slippage_amt if filled_size > 0 else -slippage_amt)
        latency_ms = self.latency.sample()

        # compute fee based on executed notional (filled_notional)
        fee = compute_fee(filled_notional, order.side, is_maker=is_maker,
                          maker_bps=self.rules.get("maker_bps", 16),
                          taker_bps=self.rules.get("taker_bps", 26))

        # support multi-slice fills to emulate partial fills arriving over time
        slices = int(self.rules.get("partial_fill_slices", 1))
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
                                     maker_bps=self.rules.get("maker_bps", 16),
                                     taker_bps=self.rules.get("taker_bps", 26))
            slice_slippage_pct = sqrt_impact_slippage(slice_notional, daily_vol=daily_vol, k=k)
            if book_depth > 1.0:
                slice_slippage_pct = slice_slippage_pct / (book_depth ** 0.5)
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
