"""FastAPI dependency providers: settings, Supabase clients, auth, org scoping.

`get_current_user` is the auth dependency referenced throughout the spec. It
verifies the JWT, resolves the caller's organization (via `X-Org-Id` when the
user belongs to more than one), and returns a `CurrentUser`. Integration tests
override this dependency with a fake user/org (see `tests/api/conftest.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from arq import ArqRedis
from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import Client, create_client

from app import db
from app.config import Settings, get_settings
from app.security import AuthError, authenticated_user_from_claims, decode_token

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    """The authenticated caller, scoped to a single organization."""

    user_id: str
    email: str | None
    org_id: str
    role: str
    access_token: str


@lru_cache(maxsize=1)
def get_service_client() -> Client:
    """Service-role client. Bypasses RLS — worker use only, never per-request
    user data without an explicit org_id filter.
    """
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def get_user_client(access_token: str) -> Client:
    """A Supabase client that forwards the caller's JWT so Postgrest requests
    are subject to Row-Level Security.
    """
    settings = get_settings()
    client = create_client(settings.supabase_url, settings.supabase_anon_key)
    client.postgrest.auth(access_token)
    return client


async def _get_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return credentials.credentials


async def get_current_user(
    token: str = Depends(_get_bearer_token),
    settings: Settings = Depends(get_settings),
    x_org_id: str | None = Header(default=None, alias="X-Org-Id"),
) -> CurrentUser:
    try:
        claims = decode_token(token, settings)
        auth_user = authenticated_user_from_claims(claims)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    client = get_user_client(token)
    memberships = db.list_memberships(client, auth_user.user_id)
    if not memberships:
        raise HTTPException(status_code=403, detail="User has no organization membership")

    if x_org_id:
        membership = next((m for m in memberships if m["org_id"] == x_org_id), None)
        if membership is None:
            raise HTTPException(
                status_code=403, detail="Not a member of the requested organization"
            )
    elif len(memberships) == 1:
        membership = memberships[0]
    else:
        raise HTTPException(
            status_code=400,
            detail="User belongs to multiple organizations; specify the X-Org-Id header",
        )

    return CurrentUser(
        user_id=auth_user.user_id,
        email=auth_user.email,
        org_id=membership["org_id"],
        role=membership["role"],
        access_token=token,
    )


def get_db_client(current_user: CurrentUser = Depends(get_current_user)) -> Client:
    """RLS-aware Postgrest client scoped to the current user."""
    return get_user_client(current_user.access_token)


async def get_redis_pool(
    request: Request, settings: Settings = Depends(get_settings)
) -> ArqRedis | None:
    """`None` under `SYNC_JOBS=true` (jobs run in-process; no queue needed).
    Otherwise the pool created at startup, or 503 if unavailable.
    """
    if settings.sync_jobs:
        return None
    pool = getattr(request.app.state, "redis_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="Job queue unavailable")
    return pool
