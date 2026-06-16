"""`GET /me`."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from supabase import Client

from app import db
from app.deps import CurrentUser, get_current_user, get_db_client
from app.schemas.me import MembershipRead, MeResponse

router = APIRouter(tags=["me"])


@router.get("/me", response_model=MeResponse)
async def get_me(
    current_user: CurrentUser = Depends(get_current_user),
    db_client: Client = Depends(get_db_client),
) -> MeResponse:
    memberships = db.list_memberships(db_client, current_user.user_id)
    return MeResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        org_id=current_user.org_id,
        role=current_user.role,
        memberships=[MembershipRead(**m) for m in memberships],
    )
