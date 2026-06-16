"""Fakes for the Supabase Postgrest + Storage surface used by `app/db.py` and
`app/storage.py`.

`FakeSupabaseClient` is an in-memory stand-in: `.table(name)` returns a tiny
query builder supporting the `select/insert/update/delete/eq/order/limit/execute`
chains actually used by `app/db.py`, and `.storage.from_(bucket)` returns an
in-memory file store supporting `upload/download/create_signed_url/remove`.

This does not simulate Postgres RLS - `app/db.py` always filters by `org_id`
explicitly, and that's what these fakes exercise.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


class FakeResponse:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data


class FakeStore:
    """Shared in-memory tables, keyed by table name."""

    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {
            "organizations": [],
            "memberships": [],
            "datasets": [],
            "jobs": [],
        }

    def rows(self, table_name: str) -> list[dict[str, Any]]:
        return self.tables.setdefault(table_name, [])


class FakeQueryBuilder:
    def __init__(self, store: FakeStore, table_name: str) -> None:
        self._store = store
        self._table_name = table_name
        self._op: str | None = None
        self._payload: dict[str, Any] | None = None
        self._filters: list[tuple[str, Any]] = []
        self._order_col: str | None = None
        self._order_desc = False
        self._limit_n: int | None = None
        self._columns: list[str] | None = None

    def select(self, columns: str = "*", **_kwargs: Any) -> FakeQueryBuilder:
        self._op = "select"
        self._columns = None if columns.strip() == "*" else [c.strip() for c in columns.split(",")]
        return self

    def insert(self, payload: dict[str, Any]) -> FakeQueryBuilder:
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload: dict[str, Any]) -> FakeQueryBuilder:
        self._op = "update"
        self._payload = payload
        return self

    def delete(self) -> FakeQueryBuilder:
        self._op = "delete"
        return self

    def eq(self, column: str, value: Any) -> FakeQueryBuilder:
        self._filters.append((column, value))
        return self

    def order(self, column: str, desc: bool = False) -> FakeQueryBuilder:
        self._order_col = column
        self._order_desc = desc
        return self

    def limit(self, n: int) -> FakeQueryBuilder:
        self._limit_n = n
        return self

    def _matches(self, row: dict[str, Any]) -> bool:
        return all(row.get(col) == val for col, val in self._filters)

    def execute(self) -> FakeResponse:
        rows = self._store.rows(self._table_name)

        if self._op == "insert":
            assert self._payload is not None
            new_row = {"id": str(uuid.uuid4()), "created_at": _now(), "updated_at": _now()}
            new_row.update(self._payload)
            rows.append(new_row)
            return FakeResponse([dict(new_row)])

        matches = [row for row in rows if self._matches(row)]

        if self._op == "update":
            assert self._payload is not None
            for row in matches:
                row.update(self._payload)
                row["updated_at"] = _now()
            return FakeResponse([dict(row) for row in matches])

        if self._op == "delete":
            for row in matches:
                rows.remove(row)
            return FakeResponse([dict(row) for row in matches])

        result = matches
        if self._order_col is not None:
            result = sorted(
                result, key=lambda row: row.get(self._order_col), reverse=self._order_desc
            )
        if self._limit_n is not None:
            result = result[: self._limit_n]
        if self._columns is not None:
            result = [{col: row.get(col) for col in self._columns} for row in result]
        return FakeResponse([dict(row) for row in result])


class FakeStorageBucket:
    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files

    def upload(self, path: str, data: bytes, _options: dict[str, Any] | None = None) -> None:
        self._files[path] = data

    def download(self, path: str) -> bytes:
        if path not in self._files:
            raise FileNotFoundError(f"No such object: {path}")
        return self._files[path]

    def create_signed_url(self, path: str, expires_in: int) -> dict[str, str]:
        return {"signedURL": f"https://fake.storage.local/{path}?expires_in={expires_in}"}

    def remove(self, paths: list[str]) -> None:
        for path in paths:
            self._files.pop(path, None)


class FakeStorageAPI:
    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files

    def from_(self, _bucket: str) -> FakeStorageBucket:
        return FakeStorageBucket(self._files)


class FakePostgrest:
    def auth(self, _token: str) -> None:
        return None


class FakeSupabaseClient:
    """Stands in for `supabase.Client` wherever `app/db.py`/`app/storage.py`
    only use `.table()`, `.storage`, and `.postgrest.auth()`.
    """

    def __init__(self, store: FakeStore, files: dict[str, bytes] | None = None) -> None:
        self.store = store
        self.files: dict[str, bytes] = files if files is not None else {}
        self.storage = FakeStorageAPI(self.files)
        self.postgrest = FakePostgrest()

    def table(self, name: str) -> FakeQueryBuilder:
        return FakeQueryBuilder(self.store, name)
