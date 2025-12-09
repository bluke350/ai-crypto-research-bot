import asyncio
import pytest
from unittest.mock import patch

from src.ingestion.providers.kraken_ws import KrakenWSClient


@pytest.mark.asyncio
async def test_checkpoint_save_resilience(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))

    # patch _save_checkpoints to raise an exception to simulate filesystem errors
    with patch.object(client, "_save_checkpoints", side_effect=Exception("disk error")):
        await client.start(start_connect_loop=False)
        await client.feed_message({"type": "trade", "pair": "XBT/USD", "timestamp": 1609459200, "price": 29000.0, "size": 0.001, "seq": 1})
        # allow consumer to run
        await asyncio.sleep(0.1)
        # stopping should not raise despite save failures
        await client.stop()

    # If we reach here without exceptions, resilience is working
    assert True
