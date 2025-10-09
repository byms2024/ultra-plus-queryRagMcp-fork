from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

# Module-level overrides so callers can configure without env vars
_BASE_URL_OVERRIDE: Optional[str] = None
_BASIC_USER_OVERRIDE: Optional[str] = None
_BASIC_PASS_OVERRIDE: Optional[str] = None


def set_base_url(base_url: str) -> None:
    """Set the base URL programmatically (overrides env)."""
    global _BASE_URL_OVERRIDE
    _BASE_URL_OVERRIDE = base_url.rstrip("/") if base_url else None


def set_basic_auth(username: Optional[str], password: Optional[str]) -> None:
    """Set Basic Auth programmatically (overrides env)."""
    global _BASIC_USER_OVERRIDE, _BASIC_PASS_OVERRIDE
    _BASIC_USER_OVERRIDE = username
    _BASIC_PASS_OVERRIDE = password


def get_base_url() -> str:
    """Return the base URL for the external NPS API."""
    if _BASE_URL_OVERRIDE:
        return _BASE_URL_OVERRIDE
    base = os.getenv("NPS_API_BASE_URL", "http://127.0.0.1:7777").rstrip("/")
    return base


def get_basic_auth() -> Optional[httpx.BasicAuth]:
    """Return BasicAuth if username/password are set, preferring overrides."""
    user = _BASIC_USER_OVERRIDE if _BASIC_USER_OVERRIDE is not None else os.getenv("NPS_BASIC_USER")
    pwd = _BASIC_PASS_OVERRIDE if _BASIC_PASS_OVERRIDE is not None else os.getenv("NPS_BASIC_PASS")
    if user and pwd:
        return httpx.BasicAuth(user, pwd)
    return None


async def get_json(path: str, params: Optional[Dict[str, Any]] = None, use_basic: bool = False):
    """Perform a GET and return JSON payload from the NPS API."""
    base = get_base_url()
    url = f"{base}{path}"
    auth = get_basic_auth() if use_basic else None
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(timeout=timeout, auth=auth) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


