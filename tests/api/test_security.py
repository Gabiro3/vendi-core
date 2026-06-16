from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from tests.api.conftest import AUTH_A, AUTH_B, upload_dataset


def test_missing_token_returns_401(client: TestClient):
    resp = client.get("/me")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_invalid_token_returns_401(client: TestClient):
    resp = client.get("/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_me_returns_org_and_role(client: TestClient):
    resp = client.get("/me", headers=AUTH_A)
    assert resp.status_code == 200
    body = resp.json()
    assert body["org_id"] == "org-a"
    assert body["role"] == "owner"


def test_cross_org_dataset_access_returns_404(client: TestClient, api_transactions_csv: bytes):
    dataset_id = upload_dataset(client, api_transactions_csv, headers=AUTH_A)

    resp = client.get(f"/datasets/{dataset_id}", headers=AUTH_B)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_cross_org_job_access_returns_404(client: TestClient, api_transactions_csv: bytes):
    dataset_id = upload_dataset(client, api_transactions_csv, headers=AUTH_A)
    run_resp = client.post("/customer/run", json={"dataset_id": dataset_id}, headers=AUTH_A)
    job_id = run_resp.json()["job_id"]

    resp = client.get(f"/jobs/{job_id}", headers=AUTH_B)
    assert resp.status_code == 404

    # Org B can't even start a job against org A's dataset.
    run_as_b = client.post("/customer/run", json={"dataset_id": dataset_id}, headers=AUTH_B)
    assert run_as_b.status_code == 404


def test_oversized_upload_rejected(client: TestClient):
    client.app.dependency_overrides[get_settings] = lambda: Settings(
        sync_jobs=True, max_upload_mb=0
    )
    resp = client.post(
        "/datasets",
        files={"file": ("big.csv", b"transaction_id,date,product_id,quantity,unit_price\n", "text/csv")},
        headers=AUTH_A,
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


def test_non_csv_content_type_rejected(client: TestClient):
    resp = client.post(
        "/datasets",
        files={"file": ("data.bin", b"\x00\x01\x02\x03", "application/zip")},
        headers=AUTH_A,
    )
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_media_type"


def test_binary_content_rejected_despite_csv_content_type(client: TestClient):
    resp = client.post(
        "/datasets",
        files={"file": ("data.csv", b"PK\x03\x04binarycontent", "text/csv")},
        headers=AUTH_A,
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"


def test_empty_file_rejected(client: TestClient):
    resp = client.post("/datasets", files={"file": ("empty.csv", b"", "text/csv")}, headers=AUTH_A)
    assert resp.status_code == 400


def test_unknown_field_in_request_body_rejected(client: TestClient, api_transactions_csv: bytes):
    dataset_id = upload_dataset(client, api_transactions_csv, headers=AUTH_A)

    resp = client.post(
        "/forecast/run", json={"dataset_id": dataset_id, "bogus_field": True}, headers=AUTH_A
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
