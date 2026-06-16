from __future__ import annotations

from fastapi.testclient import TestClient

from tests.api.conftest import AUTH_A, upload_dataset


def test_forecast_job_lifecycle(client: TestClient, api_transactions_csv: bytes):
    dataset_id = upload_dataset(client, api_transactions_csv)

    run_resp = client.post("/forecast/run", json={"dataset_id": dataset_id}, headers=AUTH_A)
    assert run_resp.status_code == 202
    job = run_resp.json()
    assert job["status"] == "succeeded"
    job_id = job["job_id"]

    results_resp = client.get(f"/forecast/{job_id}/results", headers=AUTH_A)
    assert results_resp.status_code == 200
    result = results_resp.json()
    assert result["n_series"] > 0
    assert result["horizon_days"] == 14
    assert all(len(series["points"]) == 14 for series in result["series"])

    job_resp = client.get(f"/jobs/{job_id}", headers=AUTH_A)
    assert job_resp.status_code == 200
    job_body = job_resp.json()
    assert job_body["module"] == "forecast"
    assert job_body["status"] == "succeeded"
    assert job_body["result_summary"]["n_series"] == result["n_series"]


def test_forecast_run_is_idempotent_without_force(client: TestClient, api_transactions_csv: bytes):
    dataset_id = upload_dataset(client, api_transactions_csv)

    first = client.post("/forecast/run", json={"dataset_id": dataset_id}, headers=AUTH_A)
    second = client.post("/forecast/run", json={"dataset_id": dataset_id}, headers=AUTH_A)

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]

    forced = client.post(
        "/forecast/run", json={"dataset_id": dataset_id, "force": True}, headers=AUTH_A
    )
    assert forced.status_code == 202
    assert forced.json()["job_id"] != first.json()["job_id"]


def test_basket_job_lifecycle(client: TestClient, api_transactions_csv: bytes):
    dataset_id = upload_dataset(client, api_transactions_csv)

    run_resp = client.post("/basket/run", json={"dataset_id": dataset_id}, headers=AUTH_A)
    assert run_resp.status_code == 202
    job = run_resp.json()
    assert job["status"] == "succeeded"

    results_resp = client.get(f"/basket/{job['job_id']}/results", headers=AUTH_A)
    assert results_resp.status_code == 200
    result = results_resp.json()
    assert result["level"] == "product"
    assert isinstance(result["rules"], list)


def test_customer_job_lifecycle_and_artifact(client: TestClient, api_transactions_csv: bytes):
    dataset_id = upload_dataset(client, api_transactions_csv)

    run_resp = client.post("/customer/run", json={"dataset_id": dataset_id}, headers=AUTH_A)
    assert run_resp.status_code == 202
    job = run_resp.json()
    assert job["status"] == "succeeded"
    job_id = job["job_id"]

    results_resp = client.get(f"/customer/{job_id}/results", headers=AUTH_A)
    assert results_resp.status_code == 200
    result = results_resp.json()
    assert result["n_customers"] > 0
    assert len(result["records_preview"]) == result["n_customers"]

    # Customer jobs always produce a `customers.csv` artifact.
    artifact_resp = client.get(f"/jobs/{job_id}/artifact", headers=AUTH_A)
    assert artifact_resp.status_code == 200
    artifact = artifact_resp.json()
    assert artifact["url"]
    assert artifact["expires_at"]


def test_jobs_list_and_filter(client: TestClient, api_transactions_csv: bytes):
    dataset_id = upload_dataset(client, api_transactions_csv)
    client.post("/customer/run", json={"dataset_id": dataset_id}, headers=AUTH_A)
    client.post("/basket/run", json={"dataset_id": dataset_id}, headers=AUTH_A)

    all_jobs = client.get("/jobs", headers=AUTH_A)
    assert all_jobs.status_code == 200
    modules = {j["module"] for j in all_jobs.json()}
    assert {"customer", "basket"} <= modules

    customer_jobs = client.get("/jobs", params={"module": "customer"}, headers=AUTH_A)
    assert customer_jobs.status_code == 200
    assert all(j["module"] == "customer" for j in customer_jobs.json())
    assert len(customer_jobs.json()) >= 1


_SHORT_CSV = (
    "order_id,order_date,sku,customerid,qty,price,category\n"
    + "\n".join(
        f"O{i},2024-01-0{(i % 5) + 1},ITEM-A,CUST-{i % 3},1,5.0,General" for i in range(1, 21)
    )
    + "\n"
).encode("utf-8")


def test_forecast_job_fails_with_insufficient_history(client: TestClient):
    dataset_id = upload_dataset(client, _SHORT_CSV)

    run_resp = client.post("/forecast/run", json={"dataset_id": dataset_id}, headers=AUTH_A)
    assert run_resp.status_code == 202
    job = run_resp.json()
    assert job["status"] == "failed"

    results_resp = client.get(f"/forecast/{job['job_id']}/results", headers=AUTH_A)
    assert results_resp.status_code == 409
    assert results_resp.json()["error"]["code"] == "conflict"

    job_resp = client.get(f"/jobs/{job['job_id']}", headers=AUTH_A)
    assert "Not enough history" in job_resp.json()["error"]


def test_run_on_unknown_dataset_returns_404(client: TestClient):
    resp = client.post("/forecast/run", json={"dataset_id": "does-not-exist"}, headers=AUTH_A)
    assert resp.status_code == 404
