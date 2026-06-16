"""DB helpers wrapping the Supabase Postgrest client.

Every function takes an already-scoped `Client`:
- For user-facing routes, `app.deps.get_db_client` returns a client carrying
  the caller's JWT, so Postgres RLS enforces org isolation.
- For the worker, `app.deps.get_service_client` returns the service-role
  client, which **bypasses RLS**. Every function below that touches `jobs` or
  `datasets` therefore requires an explicit `org_id` filter regardless of
  which client is passed — never rely on RLS alone.
"""

from __future__ import annotations

from typing import Any

from supabase import Client

# `select("*")` returns DB-internal bookkeeping columns that aren't part of
# the API response schemas (which use `extra="forbid"`). Strip them before
# a row is unpacked into a Pydantic model.
_INTERNAL_FIELDS = ("updated_at",)


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in _INTERNAL_FIELDS}


# ---------------------------------------------------------------------------
# memberships
# ---------------------------------------------------------------------------


def list_memberships(client: Client, user_id: str) -> list[dict[str, Any]]:
    resp = client.table("memberships").select("org_id, role").eq("user_id", user_id).execute()
    return resp.data or []


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


def create_dataset(client: Client, org_id: str, name: str) -> dict[str, Any]:
    """Insert a placeholder `datasets` row (status `validating`, no storage
    paths yet). The caller fills in `storage_path`/`parquet_path`/etc. via
    `update_dataset` once the dataset id is known and files are uploaded.
    """
    resp = (
        client.table("datasets")
        .insert({"org_id": org_id, "name": name, "status": "validating"})
        .execute()
    )
    return resp.data[0]


def update_dataset(client: Client, dataset_id: str, org_id: str, **fields: Any) -> dict[str, Any]:
    resp = (
        client.table("datasets")
        .update(fields)
        .eq("id", dataset_id)
        .eq("org_id", org_id)
        .execute()
    )
    if not resp.data:
        raise LookupError("dataset not found")
    return resp.data[0]


def get_dataset(client: Client, dataset_id: str, org_id: str) -> dict[str, Any] | None:
    resp = (
        client.table("datasets")
        .select("*")
        .eq("id", dataset_id)
        .eq("org_id", org_id)
        .limit(1)
        .execute()
    )
    return _public(resp.data[0]) if resp.data else None


def list_datasets(client: Client, org_id: str) -> list[dict[str, Any]]:
    resp = (
        client.table("datasets")
        .select("*")
        .eq("org_id", org_id)
        .order("created_at", desc=True)
        .execute()
    )
    return [_public(row) for row in resp.data or []]


def delete_dataset(client: Client, dataset_id: str, org_id: str) -> None:
    client.table("datasets").delete().eq("id", dataset_id).eq("org_id", org_id).execute()


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def create_job(
    client: Client, org_id: str, dataset_id: str, module: str, params: dict[str, Any]
) -> dict[str, Any]:
    resp = (
        client.table("jobs")
        .insert(
            {
                "org_id": org_id,
                "dataset_id": dataset_id,
                "module": module,
                "params": params,
                "status": "queued",
            }
        )
        .execute()
    )
    return resp.data[0]


def get_job_by_id(client: Client, job_id: str) -> dict[str, Any] | None:
    """Worker-only: look up a job by primary key with no `org_id` filter (the
    org isn't known until after this lookup). Every subsequent query for this
    job must go through `update_job`/`get_job` with the returned `org_id`.
    """
    resp = client.table("jobs").select("*").eq("id", job_id).limit(1).execute()
    return resp.data[0] if resp.data else None


def get_job(client: Client, job_id: str, org_id: str) -> dict[str, Any] | None:
    resp = (
        client.table("jobs")
        .select("*")
        .eq("id", job_id)
        .eq("org_id", org_id)
        .limit(1)
        .execute()
    )
    return _public(resp.data[0]) if resp.data else None


def list_jobs(
    client: Client,
    org_id: str,
    module: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    query = client.table("jobs").select("*").eq("org_id", org_id)
    if module:
        query = query.eq("module", module)
    if status:
        query = query.eq("status", status)
    resp = query.order("created_at", desc=True).execute()
    return [_public(row) for row in resp.data or []]


def update_job(client: Client, job_id: str, org_id: str, **fields: Any) -> dict[str, Any]:
    resp = (
        client.table("jobs")
        .update(fields)
        .eq("id", job_id)
        .eq("org_id", org_id)
        .execute()
    )
    if not resp.data:
        raise LookupError("job not found")
    return resp.data[0]


def find_existing_succeeded_job(
    client: Client, org_id: str, dataset_id: str, module: str, params: dict[str, Any]
) -> dict[str, Any] | None:
    """Used for idempotency: a prior succeeded run with identical params can
    be returned instead of recomputing (unless the caller passes `force=true`).
    """
    resp = (
        client.table("jobs")
        .select("*")
        .eq("org_id", org_id)
        .eq("dataset_id", dataset_id)
        .eq("module", module)
        .eq("status", "succeeded")
        .order("created_at", desc=True)
        .execute()
    )
    for job in resp.data or []:
        if job.get("params") == params:
            return job
    return None
