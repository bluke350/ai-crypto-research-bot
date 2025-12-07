import os
import json
from src.ingestion.providers.kraken_ws import KrakenWSClient


def test_checkpoints_save_and_load_roundtrip(tmp_path):
    out = tmp_path / "data"
    client = KrakenWSClient(out_root=str(out))
    # set a checkpoint and save
    client.checkpoints = {"XBT/USD": 42}
    client._save_checkpoints()
    # ensure file exists
    assert os.path.exists(client.checkpoint_file)
    # load into new instance
    client2 = KrakenWSClient(out_root=str(out))
    assert client2.checkpoints.get("XBT/USD") == 42


def test_load_checkpoints_handles_invalid_json(tmp_path):
    out = tmp_path / "data"
    os.makedirs(str(out), exist_ok=True)
    ck = os.path.join(str(out), "_ws_checkpoints.json")
    with open(ck, 'w', encoding='utf-8') as fh:
        fh.write("not a json")
    client = KrakenWSClient(out_root=str(out))
    # invalid content should fall back to empty dict
    assert client.checkpoints == {}
