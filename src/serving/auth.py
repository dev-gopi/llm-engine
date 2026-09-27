"""OIDC/JWT authentication, tenant binding, RBAC and tenant quotas."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

try:
    import jwt
except ImportError:  # pragma: no cover
    jwt = None


@dataclass(frozen=True)
class AuthPrincipal:
    subject: str
    tenant_id: str = "default"
    roles: tuple[str, ...] = ("user",)
    scopes: tuple[str, ...] = ()
    claims: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def has_role(self, *roles: str) -> bool:
        return bool(set(self.roles).intersection(roles))


@dataclass(frozen=True)
class OIDCConfig:
    enabled: bool = False
    issuer: str | None = None
    audience: str | None = None
    jwks_url: str | None = None
    algorithms: tuple[str, ...] = ("RS256",)
    hs256_secret: str | None = None
    tenant_claim: str = "tenant_id"
    roles_claim: str = "roles"
    scopes_claim: str = "scope"


class OIDCAuthenticator:
    def __init__(self, config: OIDCConfig) -> None:
        self.config = config
        self._jwks_client = None
        if config.enabled and jwt is None:
            raise RuntimeError("PyJWT is required when OIDC authentication is enabled")
        if config.enabled and config.jwks_url:
            self._jwks_client = jwt.PyJWKClient(config.jwks_url)

    def authenticate(self, token: str) -> AuthPrincipal:
        if not self.config.enabled:
            raise RuntimeError("OIDC authentication is disabled")
        if not token:
            raise ValueError("missing bearer token")
        if self._jwks_client is not None:
            key = self._jwks_client.get_signing_key_from_jwt(token).key
        elif self.config.hs256_secret:
            key = self.config.hs256_secret
        else:
            raise RuntimeError("OIDC requires jwks_url or hs256_secret")
        options = {"verify_aud": self.config.audience is not None, "verify_iss": self.config.issuer is not None}
        claims = jwt.decode(
            token,
            key=key,
            algorithms=list(self.config.algorithms),
            audience=self.config.audience,
            issuer=self.config.issuer,
            options=options,
        )
        subject = str(claims.get("sub") or "").strip()
        if not subject:
            raise ValueError("OIDC token is missing sub")
        tenant = str(claims.get(self.config.tenant_claim) or "default")
        raw_roles = claims.get(self.config.roles_claim, ("user",))
        if isinstance(raw_roles, str):
            roles = tuple(part for part in raw_roles.replace(",", " ").split() if part)
        else:
            roles = tuple(str(x) for x in raw_roles)
        raw_scopes = claims.get(self.config.scopes_claim, "")
        scopes = tuple(raw_scopes.split()) if isinstance(raw_scopes, str) else tuple(str(x) for x in raw_scopes)
        return AuthPrincipal(subject=subject, tenant_id=tenant, roles=roles or ("user",), scopes=scopes, claims=dict(claims))


class TenantQuotaLimiter:
    """Fixed-window per-tenant request quota; zero disables the limit."""
    def __init__(self, requests_per_minute: int = 0) -> None:
        if requests_per_minute < 0:
            raise ValueError("requests_per_minute cannot be negative")
        self.limit = requests_per_minute
        self._windows: dict[str, tuple[int, int]] = {}

    def allow(self, tenant: str, now: float | None = None) -> bool:
        if self.limit <= 0:
            return True
        minute = int((time.time() if now is None else now) // 60)
        current_minute, count = self._windows.get(tenant, (minute, 0))
        if current_minute != minute:
            current_minute, count = minute, 0
        if count >= self.limit:
            self._windows[tenant] = (current_minute, count)
            return False
        self._windows[tenant] = (current_minute, count + 1)
        return True


class RBACPolicy:
    """Small role hierarchy used by admin/tenant control-plane endpoints."""
    hierarchy = {
        "user": 10,
        "developer": 20,
        "tenant_admin": 30,
        "admin": 100,
    }

    @classmethod
    def allowed(cls, principal: AuthPrincipal, required: str) -> bool:
        target = cls.hierarchy.get(required, 10**9)
        return max((cls.hierarchy.get(role, 0) for role in principal.roles), default=0) >= target


__all__ = ["AuthPrincipal", "OIDCConfig", "OIDCAuthenticator", "TenantQuotaLimiter", "RBACPolicy"]
