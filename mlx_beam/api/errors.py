"""Errors in the shape OpenAI clients parse."""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status = status
        self.type = type
        self.param = param
        self.code = code

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "param": self.param,
                "code": self.code,
            }
        }


def unsupported(what: str, param: str | None = None) -> ApiError:
    return ApiError(f"{what} is not supported", param=param, code="unsupported")


def missing_extra(feature: str, extra: str) -> ApiError:
    """The feature lives outside the core (vision and audio as their own
    package, structured output behind a guard). No install line until the
    extra exists: this release provides none of them."""
    return ApiError(
        f"{feature} needs the '{extra}' extra, which this release does not "
        "provide yet",
        code="extra_not_installed",
    )


def object_field(body: dict, key: str, param: str | None = None) -> dict:
    """``body[key]`` as a JSON object; a missing or null field is ``{}``,
    anything else is the client's mistake, not the server's."""
    value = body.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ApiError(f"{key} must be a JSON object", param=param or key)
    return value


def text_field(part: dict, key: str, where: str) -> str:
    value = part.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ApiError(f"{where}.{key} must be text", param=where.split("[")[0])
    return value


def bool_field(body: dict, key: str, default: bool = False) -> bool:
    """A JSON boolean; "false" is not False, and 1 is not true."""
    value = body.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ApiError(f"{key} must be true or false", param=key)
    return value
