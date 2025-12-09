from __future__ import annotations
from typing import Literal


def compute_fee(notional: float, side: Literal['buy', 'sell'], is_maker: bool = False, maker_bps: float = 16, taker_bps: float = 26) -> float:
    """Compute fee amount in currency units given a trade notional.

    Fees expressed in basis points (bps). Default maker/taker from Kraken example.

    Parameters
    ----------
    notional: float
        Price * filled_size.
    side: 'buy'|'sell'
    is_maker: bool
    maker_bps: float
    taker_bps: float

    Returns
    -------
    float
        Fee amount (positive number representing cost).
    """
    bps = maker_bps if is_maker else taker_bps
    return abs(notional) * (bps / 10000.0)
