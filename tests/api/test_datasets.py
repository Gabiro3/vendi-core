from __future__ import annotations

from fastapi.testclient import TestClient

from tests.api.conftest import AUTH_A, upload_dataset


def test_upload_returns_validation_report(client: TestClient, api_transactions_csv: bytes):
    resp = client.post(
        "/datasets",
        files={"file": ("transactions.csv", api_transactions_csv, "text/csv")},
        data={"name": "My Dataset"},
        headers=AUTH_A,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "ready"
    report = body["validation_report"]
    assert report["rows_kept"] > 0
    assert report["rows_dropped"] == 0
    assert report["has_customer_id"] is True


def test_list_get_delete_dataset(client: TestClient, api_transactions_csv: bytes):
    dataset_id = upload_dataset(client, api_transactions_csv)

    list_resp = client.get("/datasets", headers=AUTH_A)
    assert list_resp.status_code == 200
    ids = [d["id"] for d in list_resp.json()]
    assert dataset_id in ids

    get_resp = client.get(f"/datasets/{dataset_id}", headers=AUTH_A)
    assert get_resp.status_code == 200
    detail = get_resp.json()
    assert detail["id"] == dataset_id
    assert detail["status"] == "ready"
    assert detail["row_count"] == detail["validation_report"]["rows_kept"]

    delete_resp = client.delete(f"/datasets/{dataset_id}", headers=AUTH_A)
    assert delete_resp.status_code == 204

    missing_resp = client.get(f"/datasets/{dataset_id}", headers=AUTH_A)
    assert missing_resp.status_code == 404
    assert missing_resp.json()["error"]["code"] == "not_found"


def test_get_unknown_dataset_returns_404(client: TestClient):
    resp = client.get("/datasets/does-not-exist", headers=AUTH_A)
    assert resp.status_code == 404
