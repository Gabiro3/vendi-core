"""Supabase Storage helpers.

Storage paths are always server-generated as `{org_id}/{dataset_id}/...` —
never derived from client-supplied filenames or ids.
"""

from __future__ import annotations

from supabase import Client


def raw_csv_path(org_id: str, dataset_id: str) -> str:
    return f"{org_id}/{dataset_id}/raw.csv"


def parquet_path(org_id: str, dataset_id: str) -> str:
    return f"{org_id}/{dataset_id}/transactions.parquet"


def job_artifact_path(org_id: str, dataset_id: str, job_id: str, filename: str) -> str:
    return f"{org_id}/{dataset_id}/jobs/{job_id}/{filename}"


def upload_bytes(client: Client, bucket: str, path: str, data: bytes, content_type: str) -> None:
    client.storage.from_(bucket).upload(
        path, data, {"content-type": content_type, "x-upsert": "true"}
    )


def download_bytes(client: Client, bucket: str, path: str) -> bytes:
    return client.storage.from_(bucket).download(path)


def create_signed_url(client: Client, bucket: str, path: str, expires_in: int = 3600) -> str:
    resp = client.storage.from_(bucket).create_signed_url(path, expires_in)
    return resp.get("signedURL") or resp["signed_url"]


def delete_paths(client: Client, bucket: str, paths: list[str]) -> None:
    if paths:
        client.storage.from_(bucket).remove(paths)
