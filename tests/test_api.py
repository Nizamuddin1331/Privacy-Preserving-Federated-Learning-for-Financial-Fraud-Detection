import pytest
from fastapi.testclient import TestClient

from app.main import app

SMALL = {"n_synthetic": 4000, "fraud_rate": 0.03, "rounds": 2, "n_clients": 3, "local_epochs": 1}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # triggers lifespan -> creates tables
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_dataset_info_falls_back_to_synthetic(client):
    r = client.get("/api/dataset/info", params={"n_synthetic": 3000})
    j = r.json()
    assert r.status_code == 200 and j["source"] == "synthetic" and j["n_rows"] == 3000 and len(j["head"]) == 8


def test_synthetic_generator_and_validation(client):
    assert client.post("/api/dataset/synthetic", json={"n_samples": 3000, "fraud_rate": 0.02}).json()["n_rows"] == 3000
    assert client.post("/api/dataset/synthetic", json={"n_samples": 10}).status_code == 422


def test_partition_preview(client):
    r = client.get("/api/dataset/partition-preview", params={"n_clients": 4, "strategy": "dirichlet", "n_synthetic": 4000})
    assert r.status_code == 200 and len(r.json()["clients"]) == 4
    assert client.get("/api/dataset/partition-preview", params={"strategy": "nope", "n_synthetic": 4000}).status_code == 422


def test_launch_poll_and_fetch_results(client):
    r = client.post("/api/experiments", json={**SMALL, "mode": "E"})
    assert r.status_code == 202
    eid = r.json()["ids"][0]
    st = client.get(f"/api/experiments/{eid}").json()      # background task has finished inside TestClient
    assert st["status"] == "completed" and st["progress"] == 1.0 and st["epsilon"] > 0
    res = client.get(f"/api/experiments/{eid}/results").json()
    assert len(res["history"]) == 2 and res["config"]["secure_aggregation"] is True
    assert any(e["id"] == eid for e in client.get("/api/experiments").json())


def test_validation_and_not_found(client):
    assert client.post("/api/experiments", json={"mode": "Z"}).status_code == 422
    assert client.post("/api/experiments", json={"rounds": 0}).status_code == 422
    assert client.post("/api/experiments", json={**SMALL, "secure_aggregation": True, "aggregator": "median"}).status_code == 422
    assert client.get("/api/experiments/99999").status_code == 404


def test_suite_comparison_convergence_privacy_report(client):
    r = client.post("/api/experiments/suite", json=SMALL)
    assert len(r.json()["ids"]) == 5
    comp = client.get("/api/metrics/comparison").json()
    assert [row["mode"] for row in comp] == list("ABCDE")
    assert comp[3]["epsilon"] > 0 and comp[0]["epsilon"] is None
    conv = client.get("/api/metrics/convergence").json()
    assert len(conv) == 5 and len(conv[0]["history"]) == 2
    assert len(client.get("/api/metrics/privacy").json()) >= 2
    assert any(x["secure"] for x in client.get("/api/metrics/secure-aggregation").json())
    assert "Mode comparison" in client.get("/api/metrics/report").text


def test_robustness_grid_and_delete(client):
    r = client.post("/api/experiments/robustness", json={"base": SMALL, "attacks": ["sign_flip"], "n_malicious": 1})
    ids = r.json()["ids"]
    assert len(ids) == 5                                   # clean + 3 aggregators + fedavg under secagg
    rows = client.get("/api/metrics/robustness").json()
    assert {x["aggregator"] for x in rows} >= {"fedavg", "median", "trimmed_mean"}
    assert client.delete(f"/api/experiments/{ids[0]}").status_code == 204
    assert client.get(f"/api/experiments/{ids[0]}").status_code == 404
