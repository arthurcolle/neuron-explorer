"""The real HTTP handlers exercised with a small random Qwen model, not model-quality evidence."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
pytest.importorskip("mlx.core")
from fastapi.testclient import TestClient
from test_mlx_runtime import tiny_runtime


class TinyTokenizer:
    eos_token_id = 127
    eos_token_ids = {127}
    unk_token_id = None

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[1 + ord(c) % 120 for c in text] or [1])

    def apply_chat_template(self, messages, **kwargs):
        return self("|".join(m["content"] for m in messages)).input_ids

    def decode(self, ids, **kwargs):
        return "".join(chr(32 + int(i) % 95) for i in ids)

    def convert_tokens_to_ids(self, token):
        return 127


@pytest.fixture
def client(tmp_path, monkeypatch):
    import server
    runtime = tiny_runtime(tmp_path)
    runtime._install_hooks()
    runtime.tok = TinyTokenizer()
    runtime.resid_p50 = {i: 1.0 for i in range(4)}
    runtime.status = "ready"
    runtime._steer_idx = False
    runtime._lens_idx = {s: np.zeros((4, 256, 12), dtype=np.int32) for s in ("up", "down")}
    runtime._vocab = [runtime.tok.decode([i]) for i in range(128)]
    scale = tmp_path / "scale.npz"
    np.savez(scale, pos=np.ones((4, 256)), neg=-np.ones((4, 256)), tokens=100)
    monkeypatch.setattr(server, "rt", runtime)
    monkeypatch.setattr(server, "NO_MODEL", False)
    monkeypatch.setattr(server, "ACT_SCALE", str(scale))
    monkeypatch.setattr(server, "lens_index", lambda: runtime._lens_idx)
    monkeypatch.setattr(server, "_act", {})
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as c:
        yield c


def test_reads_and_steered_probability_contract(client):
    response = client.post("/api/reads", json={"layer": 1, "n": 3, "text": "abc", "chat": False})
    assert response.status_code == 200
    assert len(response.json()["units"]) == 3
    body = {"text": "abc", "chat": False, "k": 5, "targets": ["a"],
            "neurons": [{"layer": 1, "n": 3, "dose": 0.8}]}
    response = client.post("/api/agent/logprobs", json=body)
    assert response.status_code == 200
    result = response.json()
    assert len(result["baseline"]["top"]) == 5
    assert len(result["intervened"]["top"]) == 5
    assert result["kl"] >= -1e-5
    assert result["rose"] and result["fell"]


def test_three_column_steering_and_head_ablations(client):
    response = client.post("/api/agent/steer", json={"layer": 1, "n": 3, "prompt": "abc", "max_new": 3})
    assert response.status_code == 200
    assert {"up", "baseline", "down"} <= response.json().keys()
    response = client.post("/api/agent/activations", json={"layer": 1, "n": 3, "text": "abc",
                           "heads": [{"layer": 0, "head": "all", "dose": 0}]})
    assert response.status_code == 200
    assert response.json()["heads"] == [{"layer": 0, "head": "all", "scale": 0.0}]
    assert len(response.json()["tokens"]) == 3


def test_native_numpy_fires_contract(client):
    response = client.get("/api/fires", params={"text": "abc", "limit": 5})
    assert response.status_code == 200
    assert response.json()["ready"]
    assert len(response.json()["results"]) == 5


def test_streaming_contract_and_localhost_guard(client):
    response = client.post("/api/steer", json={"layer": 1, "n": 3, "dose": 0.8, "prompt": "abc", "max_new": 2})
    assert response.status_code == 200
    assert '"row": null' in response.text
    assert '"error"' not in response.text
    assert client.get("/api/status", headers={"origin": "https://untrusted.example"}).status_code == 403
