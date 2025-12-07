import asyncio
import types
import pytest
from typing import Optional


@pytest.mark.asyncio
async def test_run_live_ws_reconnect_exit(monkeypatch, tmp_path):
    # simulate websockets.connect raising OSError so the live loop hits reconnect logic
    fake_websockets = types.SimpleNamespace()

    class _FakeConnectCtx:
        def __init__(self, exc: Optional[Exception] = None):
            self._exc = exc

        async def __aenter__(self):
            # Simulate connect raising at enter-time
            if self._exc is not None:
                raise self._exc
            return None

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def fake_connect(uri, ping_interval=None):
        # return an async context manager instance that raises when entered
        return _FakeConnectCtx(OSError("connection failed"))

    fake_websockets.connect = fake_connect
    monkeypatch.setitem(__import__('sys').modules, 'websockets', fake_websockets)

    from tooling.run_paper_live import run_live_ws

    # run with max_retries=1 to exit soon
    try:
        await run_live_ws('XBT/USD', 10000.0, short=5, long=20, dry_run=True, max_retries=1, max_backoff=1)
    except Exception as e:
        pytest.fail(f"run_live_ws raised unexpected exception: {e}")
