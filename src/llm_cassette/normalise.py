"""Request normalisation and deterministic hashing."""

from __future__ import annotations

import hashlib
import json
import math

from llm_cassette.errors import CassetteError
from llm_cassette.types import JSONValue

_REDACTED: JSONValue = "<redacted>"
_VOLATILE_EXACT = frozenset(
    {"id", "created", "createdat", "updatedat", "completedat", "startedat"}
)
_SECRET_EXACT = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "proxy_authorization",
        "password",
        "secret",
        "access_token",
        "refresh_token",
        "id_token",
        "auth_token",
        "x_api_key",
        "token",
    }
)


def normalise_request(value: object) -> JSONValue:
    """Return JSON with volatile fields removed and secrets redacted.

    Keys named ``id``, ``created``, ``created_at``, ``updated_at``,
    ``started_at``, and ``completed_at`` are dropped, as are keys that end
    in ``timestamp`` or ``request_id``. Hyphens and underscores are
    ignored for that match. Secret names such as ``api_key``,
    ``authorization``, ``password``, ``secret``, and ``token`` are replaced
    with ``<redacted>``. ``max_tokens`` is left unchanged.
    """

    return _walk(ensure_json(value), drop_volatile=True)


def redact_secrets(value: JSONValue) -> JSONValue:
    """Replace secret fields and keep every other value."""

    return _walk(value, drop_volatile=False)


def request_hash(value: object) -> str:
    """Return the SHA-256 hex digest of the normalised JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_json(value: object) -> str:
    """Return the canonical JSON string used for hashing and strict matches."""

    try:
        return json.dumps(
            normalise_request(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CassetteError("cannot hash a non-JSON value") from exc


def ensure_json(value: object) -> JSONValue:
    """Copy ``value`` into JSON, converting tuples to lists."""

    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CassetteError("cassette numbers must be finite")
        return value
    if isinstance(value, dict):
        copied: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CassetteError(
                    f"JSON object keys must be strings, got {type(key).__name__}"
                )
            copied[key] = ensure_json(item)
        return copied
    if isinstance(value, (list, tuple)):
        return [ensure_json(item) for item in value]
    raise CassetteError(f"cannot record a value of type {type(value).__name__}")


def clone_json(value: JSONValue) -> JSONValue:
    """Return a deep copy of a JSON value."""

    if isinstance(value, dict):
        return {key: clone_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_json(item) for item in value]
    return value


def _walk(value: JSONValue, *, drop_volatile: bool) -> JSONValue:
    if isinstance(value, dict):
        walked: dict[str, JSONValue] = {}
        for key, item in value.items():
            if drop_volatile and _is_volatile_key(key):
                continue
            if _is_secret_key(key):
                walked[key] = _REDACTED
                continue
            walked[key] = _walk(item, drop_volatile=drop_volatile)
        return walked
    if isinstance(value, list):
        return [_walk(item, drop_volatile=drop_volatile) for item in value]
    return value


def _is_volatile_key(key: str) -> bool:
    compact = key.lower().replace("-", "").replace("_", "")
    if compact in _VOLATILE_EXACT:
        return True
    return compact.endswith(("timestamp", "timestamps", "requestid"))


def _is_secret_key(key: str) -> bool:
    name = key.lower().replace("-", "_")
    if name in _SECRET_EXACT:
        return True
    if name.endswith(("_api_key", "_apikey", "_secret", "_password", "_access_token")):
        return True
    if name.startswith("secret_"):
        return True
    return name.endswith("_token") and "max_token" not in name
