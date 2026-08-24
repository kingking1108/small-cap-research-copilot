from unittest.mock import MagicMock, patch

import pytest

from research_copilot.observability import get_langfuse_handler


@pytest.mark.parametrize(
    "public_key,secret_key",
    [("", ""), ("pk-test", ""), ("", "sk-test")],
)
def test_returns_none_when_not_fully_configured(
    monkeypatch: pytest.MonkeyPatch, public_key: str, secret_key: str
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", secret_key)

    assert get_langfuse_handler() is None


@patch("research_copilot.observability.CallbackHandler")
def test_returns_handler_when_both_keys_configured(
    mock_handler_cls: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    mock_handler_cls.return_value = MagicMock()

    handler = get_langfuse_handler()

    assert handler is mock_handler_cls.return_value
    mock_handler_cls.assert_called_once()
