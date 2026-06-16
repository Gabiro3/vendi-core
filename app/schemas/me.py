"""Schema for `GET /me`."""

from __future__ import annotations

from app.schemas.common import APIModel


class MembershipRead(APIModel):
    org_id: str
    role: str


class MeResponse(APIModel):
    user_id: str
    email: str | None
    org_id: str
    role: str
    memberships: list[MembershipRead]
