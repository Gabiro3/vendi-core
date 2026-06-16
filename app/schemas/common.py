"""Shared base classes for all API schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class APIModel(BaseModel):
    """Base for all request/response models.

    `extra="forbid"` on request bodies rejects unknown fields (TECHNICAL_SPEC
    §7.3); applying it uniformly to response models too keeps schemas honest
    about their exact shape.
    """

    model_config = ConfigDict(extra="forbid")


class ErrorDetail(APIModel):
    code: str
    message: str


class ErrorResponse(APIModel):
    error: ErrorDetail
