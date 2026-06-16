"""FastAPI `TestClient` wiring for API integration tests.

`SYNC_JOBS=true` is set before `app.main` is ever imported so that
`POST /<module>/run` runs the worker in-process (no Redis needed) and the
job is `succeeded`/`failed` by the time the request returns.

Auth is faked at the `get_current_user` dependency: the real `_get_bearer_token`
sub-dependency still runs (so "missing/invalid token" -> 401 is exercised for
real), but the token string is just looked up in `FAKE_USERS` instead of being
verified as a JWT - this avoids any real Supabase/JWT calls in tests.
"""

from __future__ import annotations

import os

os.environ["SYNC_JOBS"] = "true"
os.environ.setdefault("SUPABASE_JWT_SECRET", "test-secret")

from collections.abc import Iterator  # noqa: E402

import pytest  # noqa: E402
from fastapi import Depends, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.deps import (  # noqa: E402
    CurrentUser,
    _get_bearer_token,
    get_current_user,
    get_db_client,
    get_redis_pool,
    get_service_client,
)
from app.main import create_app  # noqa: E402
from tests.api.fakes import FakeStore, FakeSupabaseClient  # noqa: E402

FAKE_USERS: dict[str, CurrentUser] = {
    "org-a-token": CurrentUser(
        user_id="user-a", email="a@example.com", org_id="org-a", role="owner", access_token="org-a-token"
    ),
    "org-b-token": CurrentUser(
        user_id="user-b", email="b@example.com", org_id="org-b", role="owner", access_token="org-b-token"
    ),
}

AUTH_A = {"Authorization": "Bearer org-a-token"}
AUTH_B = {"Authorization": "Bearer org-b-token"}


async def _fake_get_current_user(token: str = Depends(_get_bearer_token)) -> CurrentUser:
    user = FAKE_USERS.get(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


@pytest.fixture()
def store() -> FakeStore:
    fake_store = FakeStore()
    fake_store.tables["memberships"] = [
        {"user_id": "user-a", "org_id": "org-a", "role": "owner"},
        {"user_id": "user-b", "org_id": "org-b", "role": "owner"},
    ]
    return fake_store


@pytest.fixture()
def fake_client(store: FakeStore) -> FakeSupabaseClient:
    return FakeSupabaseClient(store)


@pytest.fixture()
def client(fake_client: FakeSupabaseClient) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_current_user] = _fake_get_current_user
    app.dependency_overrides[get_db_client] = lambda: fake_client
    app.dependency_overrides[get_service_client] = lambda: fake_client
    app.dependency_overrides[get_redis_pool] = lambda: None

    with TestClient(app) as test_client:
        yield test_client


def upload_dataset(client: TestClient, csv_bytes: bytes, headers: dict[str, str] = AUTH_A) -> str:
    """Upload a CSV and assert it lands in `ready`, returning its `dataset_id`."""
    resp = client.post(
        "/datasets",
        files={"file": ("transactions.csv", csv_bytes, "text/csv")},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "ready", body
    return body["dataset_id"]
