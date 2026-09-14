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
    return ApiError(
        f"{feature} needs the '{extra}' extra, which is not installed",
        code="extra_not_installed",
    )
