"""A tiny Gym-like wrapper that adapts the project's Simulator to an RL env.

This wrapper is intentionally minimal: observations are [price, position]
and actions are 1-dimensional continuous values representing a desired
position (absolute size). The environment translates an action into a
trade (difference between desired and current position), calls
Simulator.place_order to get a fill, and updates cash/position.

Reward is the change in net asset value (cash + position*price) after the
trade and a simulated price move. This is sufficient for testing and
local experiments; replace with a richer market simulation for research.
"""
from __future__ import annotations

import math
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.execution.order_models import Order
from src.execution.simulator import Simulator


class SimulatorEnv:
    def __init__(self, pair: str = "XBT/USD", init_price: float = 100.0, init_cash: float = 100000.0,
                 slippage_rules: Optional[Dict[str, Any]] = None, max_steps: int = 200,
                 fee_model: Optional[object] = None, slippage_model: Optional[object] = None, latency_model: Optional[object] = None):
        self.pair = pair
        self.init_price = float(init_price)
        self.init_cash = float(init_cash)
        self.max_steps = int(max_steps)
        self.slippage_rules = slippage_rules or {}
        self.fee_model = fee_model
        self.slippage_model = slippage_model
        self.latency_model = latency_model

        self.sim: Optional[Simulator] = None
        self.price: float = self.init_price
        self.position: float = 0.0
        self.cash: float = self.init_cash
        self.step_count = 0

    def reset(self):
        # construct Simulator with structured cost models if provided
        # map any legacy slippage_rules into explicit simulator kwargs
        sim_kwargs = {}
        if self.slippage_rules:
            allowed = (
                "partial_fill_fraction",
                "partial_fill_slices",
                "book_depth",
                "slippage_k",
                "slippage_daily_vol",
                "maker_bps",
                "taker_bps",
                "fixed_fee_pct",
                "fixed_slippage_pct",
            )
            for k in allowed:
                if k in self.slippage_rules:
                    sim_kwargs[k] = self.slippage_rules[k]
        self.sim = Simulator(**sim_kwargs, fee_model=self.fee_model, slippage_model=self.slippage_model, latency_model=self.latency_model)
        self.price = float(self.init_price)
        self.position = 0.0
        self.cash = float(self.init_cash)
        self.step_count = 0
        return self._obs()

    def _obs(self):
        # observation: price and current position
        return np.array([self.price, float(self.position)], dtype=np.float32)

    def _simulate_price_move(self):
        # small random walk in log-price
        sigma = 0.001  # tiny volatility for short tests
        dx = np.random.normal(loc=0.0, scale=sigma)
        self.price = float(self.price * math.exp(dx))

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take an action (desired absolute position) and return (obs, reward, done, info).

        action: array-like or float, desired absolute position in base units
        """
        desired = float(np.array(action).ravel()[0])
        # compute trade needed (signed)
        delta = desired - self.position
        info: Dict[str, Any] = {}

        if abs(delta) > 0.0:
            side = "buy" if delta > 0 else "sell"
            size = abs(delta)
            order = Order(order_id=str(uuid.uuid4()), pair=self.pair, side=side, size=size, price=None)
            fill = self.sim.place_order(order, market_price=self.price, is_maker=False)
            filled = float(fill.get("filled_size", 0.0))
            avg_price = float(fill.get("avg_fill_price", self.price))
            fee = float(fill.get("fee", 0.0))
            # update position and cash
            if side == "buy":
                self.position += filled
                self.cash -= filled * avg_price + fee
            else:
                self.position -= filled
                self.cash += filled * avg_price - fee
            info.update({"filled": filled, "avg_price": avg_price, "fee": fee})

        prev_nav = self.cash + self.position * self.price
        # simulate a price move after the trade
        self._simulate_price_move()
        nav = self.cash + self.position * self.price
        reward = nav - prev_nav

        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self._obs(), float(reward), bool(done), info
try:
    import gym
    from gym import spaces


    class SimpleTradingEnv(gym.Env):
        """A tiny trading env that wraps a price series and exposes actions: -1,0,1"""
        def __init__(self, prices: np.ndarray):
            super().__init__()
            self.prices = prices
            self.pos = 0.0
            self.t = 0
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
            self.action_space = spaces.Discrete(3)  # sell, hold, buy

        def reset(self):
            self.t = 0
            self.pos = 0.0
            return np.array([self.prices[self.t]], dtype=np.float32)

        def step(self, action):
            # map action to position change
            if action == 0:
                delta = -1
            elif action == 1:
                delta = 0
            else:
                delta = 1
            reward = -abs(delta) * 0.0
            self.t += 1
            done = self.t >= len(self.prices) - 1
            obs = np.array([self.prices[self.t]], dtype=np.float32)
            return obs, reward, done, {}
except Exception:
    # gym not available — SimpleTradingEnv is optional helper only
    pass
