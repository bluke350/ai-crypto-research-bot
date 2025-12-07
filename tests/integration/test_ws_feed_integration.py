import threading
import time
import json
from pathlib import Path

import pytest

import orchestration.paper_live as paper_live


class FakeKrakenWSClient:
    last_instance = None

    def __init__(self, out_root: str = "data/raw"):
        import asyncio

        self.msg_queue = asyncio.Queue()
        self.out_root = out_root
        self._running = False
        FakeKrakenWSClient.last_instance = self

    async def start(self):
        # start hook called in the client's event loop; set running flag
        self._running = True

    async def stop(self):
        self._running = False

    async def feed_message(self, msg: dict):
        await self.msg_queue.put(msg)


@pytest.mark.integration
def test_feed_messages_produce_paper_executions(tmp_path, monkeypatch):
    # patch ModelWrapper to produce a non-zero target so orders are generated
    class DummyWrapper:
        @classmethod
        def from_file(cls, path: str):
            return cls()

        def predicted_to_targets(self, prices, method='vol_norm', **kwargs):
            idx = prices.index[2:]
            import pandas as pd

            # always request small non-zero target
            return pd.Series([0.5] * len(idx), index=idx, name='target')

    monkeypatch.setattr(paper_live, 'ModelWrapper', DummyWrapper)
    # patch KrakenWSClient used by paper_live
    monkeypatch.setattr(paper_live, 'KrakenWSClient', FakeKrakenWSClient)

    # run the live runner in a background thread so we can inject messages
    def runner():
        paper_live.run_live(checkpoint_path='fake', prices_csv=None, out_root=str(tmp_path), use_ws=True, max_ticks=10, paper_mode=True)

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    # wait for fake client instance to be created by runner
    timeout = 5.0
    start = time.time()
    fake = None
    while time.time() - start < timeout:
        fake = FakeKrakenWSClient.last_instance
        if fake is not None:
            break
        time.sleep(0.05)

    assert fake is not None, "FakeKrakenWSClient instance was not created"

    # push a few trade messages into the client's queue using its event loop
    loop = getattr(fake, '_loop', None)
    assert loop is not None, "fake client loop not attached"

    # schedule messages
    import asyncio

    for i in range(5):
        msg = {'type': 'trade', 'price': 100.0 + i, 'timestamp': i}
        fut = asyncio.run_coroutine_threadsafe(fake.msg_queue.put(msg), loop)
        fut.result(timeout=1.0)
        time.sleep(0.01)

    # wait for runner to finish
    t.join(timeout=10.0)
    # read result.json
    # find the single run folder under tmp_path
    runs = list(Path(str(tmp_path)).iterdir())
    assert runs, "no run artifacts produced"
    run_dir = runs[0]
    res = json.loads((run_dir / 'result.json').read_text())
    assert 'executions' in res
    # ensure at least one paper execution present (fills recorded)
    assert any(float(e.get('filled_size', 0)) >= 0.0 for e in res['executions'])
