from __future__ import annotations

import os
from typing import Any, Dict, List

from src.execution.order_models import Order
from src.execution.simulators import ExchangeSimulator
from src.execution.clients import ExchangeClientWrapper
from tooling.structured_logging import setup_structured_logger
import logging
import time


class OrderExecutor:
    """Abstract order executor interface.

    Implementations should provide `execute(order, market_price, is_maker=False)`
    which returns a dict similar to ExchangeSimulator.place_order: {filled_size, avg_fill_price, fee}
    """

    def execute(self, order: Order, market_price: float, is_maker: bool = False) -> Dict[str, Any]:
        raise NotImplementedError()


class PaperOrderExecutor(OrderExecutor):
    """Paper-mode executor that routes orders to an `ExchangeSimulator` and records fills."""

    def __init__(self, simulator: ExchangeSimulator | None = None):
        self.sim = simulator or ExchangeSimulator()
        self.fills: List[Dict[str, Any]] = []

    def execute(self, order: Order, market_price: float, is_maker: bool = False) -> Dict[str, Any]:
        fill = self.sim.place_order(order, market_price=market_price, is_maker=is_maker)
        # annotate fill with order metadata for traceability
        out = dict(fill)
        out.update({
            "order_id": order.order_id,
            "pair": order.pair,
            "side": order.side,
            "requested_size": order.size,
        })
        self.fills.append(out)
        return out


class LiveOrderExecutor(OrderExecutor):
    """Placeholder live executor.

    This class should be implemented to integrate with a real exchange client. For safety in
    this repository it currently raises NotImplementedError to avoid accidental live orders.
    """

    def __init__(self, *, dry_run: bool = True):
        self.dry_run = bool(dry_run)
        # expose fills for audit parity with PaperOrderExecutor
        self.fills: List[Dict[str, Any]] = []

    def execute(self, order: Order, market_price: float, is_maker: bool = False) -> Dict[str, Any]:
        if self.dry_run:
            # do not send any live requests in dry_run; return zero-fill placeholder
            out = {"filled_size": 0.0, "avg_fill_price": float(market_price), "fee": 0.0}
            out['order_id'] = order.order_id
            out['pair'] = order.pair
            out['side'] = order.side
            out['requested_size'] = order.size
            try:
                self.fills.append(out)
            except Exception:
                pass
            return out
        raise NotImplementedError("LiveOrderExecutor.execute must be implemented to send orders to an exchange")


class LiveAdapterOrderExecutor(LiveOrderExecutor):
    """Adapter that wraps a provided exchange client object.

    The exchange client is expected to implement a `place_order(order)` method
    or similar. This adapter will call `client.place_order` when not in dry_run.
    Use environment variables or secure vaults for credentials; this adapter
    does not manage credentials itself.
    """

    def __init__(self, client: object | None = None, *, dry_run: bool = True, max_retries: int = 3, backoff_seconds: float = 0.5):
        super().__init__(dry_run=dry_run)
        self.client = client
        # ensure fills exists even when client is not provided
        if not hasattr(self, 'fills'):
            self.fills: List[Dict[str, Any]] = []
        # retry/backoff settings
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)
        self.last_error: str | None = None
        # structured logger
        self.logger = setup_structured_logger("live_adapter_executor")
        # Optionally attempt to construct a real client when explicitly enabled via env vars.
        # This is a cautious, opt-in flow: both ENABLE_REAL_LIVE_EXECUTOR and CONFIRM_LIVE
        # must be set to '1' and credentials must be provided in env vars.
        try:
            provider = os.environ.get('LIVE_EXECUTOR_PROVIDER', '').lower()
            enable_real = os.environ.get('ENABLE_REAL_LIVE_EXECUTOR', '0') == '1'
            confirm = os.environ.get('CONFIRM_LIVE', '0') == '1'
            if client is None and enable_real and confirm and provider:
                key = os.environ.get('EXCHANGE_API_KEY')
                secret = os.environ.get('EXCHANGE_API_SECRET')
                if not key or not secret:
                    # credentials not provided; remain in dry-run
                    self.logger.info("missing_credentials_for_live_executor", extra={"provider": provider})
                else:
                    # try provider-specific client creation via wrapper
                    try:
                        wrapper = ExchangeClientWrapper.from_provider(provider, key, secret)
                        self.client = wrapper
                        self.dry_run = False
                        self.logger.info("live_client_constructed", extra={"provider": provider})
                    except Exception as e:
                        # unable to build a real client; remain in dry_run
                        self.logger.warning("failed_build_client", extra={"provider": provider, "err": str(e)})
        except Exception as e:
            # conservative: never raise during construction
            try:
                self.logger.exception("unexpected_error_live_executor_init", extra={"err": str(e)})
            except Exception:
                pass

    def execute(self, order: Order, market_price: float, is_maker: bool = False) -> Dict[str, Any]:
        if self.dry_run or self.client is None:
            # safe: do not call external services
            out = {"filled_size": 0.0, "avg_fill_price": float(market_price), "fee": 0.0}
            out['order_id'] = order.order_id
            out['pair'] = order.pair
            out['side'] = order.side
            out['requested_size'] = order.size
            try:
                self.fills.append(out)
            except Exception:
                pass
            return out
        # attempt to call client with retries/backoff
        attempt = 0
        last_exc = None
        while attempt <= self.max_retries:
            try:
                if isinstance(self.client, ExchangeClientWrapper):
                    res = self.client.place_order(order, market_price=market_price, is_maker=is_maker)
                elif hasattr(self.client, 'place_order'):
                    res = self.client.place_order(order, market_price=market_price, is_maker=is_maker)
                elif hasattr(self.client, 'submit_order'):
                    res = self.client.submit_order(order, price=market_price)
                else:
                    raise NotImplementedError("exchange client does not implement a known place_order/submit_order API")

                # normalize and record fill
                out = dict(res)
                out.update({
                    "order_id": out.get('order_id', order.order_id),
                    "pair": out.get('pair', order.pair),
                    "side": out.get('side', order.side),
                    "requested_size": out.get('requested_size', order.size),
                })
                try:
                    self.fills.append(out)
                except Exception:
                    pass
                return out
            except Exception as e:
                last_exc = e
                self.last_error = str(e)
                try:
                    self.logger.error("place_order_failed", extra={"err": str(e), "attempt": attempt, "order_id": order.order_id})
                except Exception:
                    pass
                attempt += 1
                if attempt > self.max_retries:
                    break
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

        # if we reach here, all retries failed — return safe dry-run zero-fill and record error
        out = {"filled_size": 0.0, "avg_fill_price": float(market_price), "fee": 0.0}
        out['order_id'] = order.order_id
        out['pair'] = order.pair
        out['side'] = order.side
        out['requested_size'] = order.size
        out['error'] = self.last_error
        try:
            self.fills.append(out)
        except Exception:
            pass
        return out
