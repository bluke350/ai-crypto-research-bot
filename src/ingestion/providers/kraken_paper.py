from typing import Dict, Any
import uuid
import os
import pandas as pd
from src.execution.simulator import Simulator


class PaperBroker:
    def __init__(self, run_id: str = None, simulator: Simulator = None, out_dir: str = "experiments/runs"):
        self.sim = simulator or Simulator()
        self.run_id = run_id or str(uuid.uuid4())
        self.out_dir = os.path.join(out_dir, self.run_id)
        os.makedirs(self.out_dir, exist_ok=True)
        self.exec_log = []

    def place_order(self, order: Any, market_price: float = None) -> Dict[str, Any]:
        fill = self.sim.place_order(order, market_price=market_price)
        rec = dict(fill)
        rec["order_id"] = order.order_id
        self.exec_log.append(rec)
        pd.DataFrame(self.exec_log).to_parquet(os.path.join(self.out_dir, "exec_log.parquet"), index=False)
        return fill

    def cancel_order(self, order_id: str) -> bool:
        return self.sim.cancel_order(order_id)

    def get_exec_log(self) -> pd.DataFrame:
        if self.exec_log:
            return pd.DataFrame(self.exec_log)
        return pd.DataFrame()
import os
import pandas as pd
from typing import Dict, Any
from uuid import uuid4


class PaperBroker:
    """Simple paper broker shim that records order requests to a CSV/Parquet log and returns simulated fills.

    This is intentionally minimal and should be replaced by the full simulator for realistic behavior.
    """

    def __init__(self, run_id: str, artifacts_root: str = "experiments/artifacts"):
        self.run_id = run_id
        self.artifacts_root = artifacts_root
        self.log = []
        os.makedirs(os.path.join(self.artifacts_root, self.run_id), exist_ok=True)

    def place_order(self, pair: str, side: str, size: float, price: float = None, type: str = "limit") -> Dict[str, Any]:
        order_id = str(uuid4())
        rec = {"order_id": order_id, "pair": pair, "side": side, "size": size, "price": price, "type": type}
        self.log.append(rec)
        # naive immediate fill for paper
        fill = {**rec, "filled_size": size, "avg_fill_price": price or 0.0, "status": "filled"}
        # persist
        df = pd.DataFrame(self.log)
        path = os.path.join(self.artifacts_root, self.run_id, "exec_log.parquet")
        df.to_parquet(path, index=False)
        return fill
