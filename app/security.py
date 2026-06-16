"""JWT verification for Supabase Auth tokens.

Supports both HS256 (legacy Supabase projects, shared `SUPABASE_JWT_SECRET`)
and JWKS-based asymmetric verification (RS256/ES256, `SUPABASE_JWKS_URL`).
Validates `exp` and `aud` ("authenticated"). Extracts `sub` as the user id.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jwt
from jwt import PyJWKClient

from app.config import Settings


class AuthError(Exception):
    """Raised when a token is missing, malformed, expired, or invalid."""


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str | None
    claims: dict


@lru_cache(maxsize=1)
def _jwks_client(jwks_url: str) -> PyJWKClient:
    return PyJWKClient(jwks_url)


def decode_token(token: str, settings: Settings) -> dict:
    """Decode and verify a Supabase JWT, returning its claims.

    Raises `AuthError` on any failure: missing token, bad signature, expired,
    wrong audience, or a server missing the required verification config.
    """
    if not token:
        raise AuthError("Missing token")

    try:
        if settings.supabase_jwks_url:
            signing_key = _jwks_client(settings.supabase_jwks_url).get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience="authenticated",
                options={"require": ["exp", "sub", "aud"]},
            )
        else:
            if not settings.supabase_jwt_secret:
                raise AuthError("Server is not configured for JWT verification")
            claims = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                options={"require": ["exp", "sub", "aud"]},
            )
    except jwt.PyJWTError as exc:
        raise AuthError(f"Invalid token: {exc}") from exc

    return claims


def authenticated_user_from_claims(claims: dict) -> AuthenticatedUser:
    sub = claims.get("sub")
    if not sub:
        raise AuthError("Token missing 'sub' claim")
    return AuthenticatedUser(user_id=str(sub), email=claims.get("email"), claims=claims)
