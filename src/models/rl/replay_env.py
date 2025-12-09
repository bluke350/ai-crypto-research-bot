"""Deterministic historical-price replay environment.

This environment steps through a provided price history (pandas DataFrame
with 'timestamp' and 'close') deterministically. It uses the project's
`Simulator` to execute fills so behavior is consistent with backtesting.

Observation: [price, position]
Action: desired absolute position (float)
Reward: change in NAV (cash + position*price) after price move
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import uuid
import math

import numpy as np
import pandas as pd

from src.execution.simulator import Simulator
from src.execution.order_models import Order


class ReplayEnv:
    def __init__(self, prices: pd.DataFrame, pair: str = "XBT/USD", initial_cash: float = 100000.0, seed: Optional[int] = None,
                 fee_model: Optional[object] = None, slippage_model: Optional[object] = None, latency_model: Optional[object] = None):
        if "timestamp" not in prices.columns or "close" not in prices.columns:
            raise ValueError("prices must contain 'timestamp' and 'close' columns")
        self.prices = prices.sort_values("timestamp").reset_index(drop=True)
        self.pair = pair
        self.initial_cash = float(initial_cash)
        self.seed = seed
        self.fee_model = fee_model
        self.slippage_model = slippage_model
        self.latency_model = latency_model

        self.sim: Optional[Simulator] = None
        self.position = 0.0
        self.cash = self.initial_cash
        self.idx = 0

    def reset(self) -> np.ndarray:
        # create fresh deterministic simulator for every episode
        self.sim = Simulator(seed=self.seed, fee_model=self.fee_model, slippage_model=self.slippage_model, latency_model=self.latency_model)
        self.position = 0.0
        self.cash = float(self.initial_cash)
        self.idx = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        price = float(self.prices.at[self.idx, "close"])
        return np.array([price, float(self.position)], dtype=np.float32)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take an action (desired absolute position) and step one historical point.

        Returns (obs, reward, done, info).
        """
        desired = float(np.array(action).ravel()[0])
        price = float(self.prices.at[self.idx, "close"])
        delta = desired - self.position
        info: Dict[str, Any] = {}

        if abs(delta) > 1e-12:
            side = "buy" if delta > 0 else "sell"
            order = Order(order_id=str(uuid.uuid4()), pair=self.pair, side=side, size=abs(delta), price=None)
            fill = self.sim.place_order(order, market_price=price, is_maker=False)
            filled = float(fill.get("filled_size", 0.0))
            avg_price = float(fill.get("avg_fill_price", price))
            fee = float(fill.get("fee", 0.0))
            if side == "buy":
                self.position += filled
                self.cash -= filled * avg_price + fee
            else:
                self.position -= filled
                self.cash += filled * avg_price - fee
            info.update({"filled": filled, "avg_price": avg_price, "fee": fee})

        # compute NAV after the trade but before the next price move
        prev_nav = self.cash + self.position * price

        # advance price pointer and pick the next price for this timestep
        self.idx += 1
        done = self.idx >= len(self.prices)

        # use the upcoming price (self.idx) as the next price if available
        next_price = float(self.prices.at[self.idx, "close"]) if not done else price
        nav = self.cash + self.position * next_price
        reward = float(nav - prev_nav)

        obs = self._obs() if not done else np.array([next_price, float(self.position)], dtype=np.float32)
        return obs, reward, bool(done), info
