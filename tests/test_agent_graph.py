from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from research_copilot.agent.graph import (
    MAX_TOOL_CALLS,
    REPORT_PROMPT,
    SYSTEM_PROMPT,
    _build_call_model,
    _trim_stale_tool_messages,
    build_report_graph,
)
from research_copilot.report.schema import ResearchReport


def _llm_returning(response: AIMessage) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = response
    return llm


@patch("research_copilot.agent.graph.get_chat_model")
def test_call_model_uses_tool_bound_llm_below_the_limit(mock_get_chat_model: MagicMock) -> None:
    with_tools_response = AIMessage(content="direct answer")
    with_tools = _llm_returning(with_tools_response)
    with_tools.bind_tools.return_value = with_tools
    without_tools = _llm_returning(AIMessage(content="unused"))
    mock_get_chat_model.side_effect = [with_tools, without_tools]

    call_model = _build_call_model()
    result = call_model({"messages": [HumanMessage(content="Wie hoch war der Umsatz?")]})

    with_tools.invoke.assert_called_once()
    without_tools.invoke.assert_not_called()
    assert result["messages"] == [with_tools_response]


@patch("research_copilot.agent.graph.get_chat_model")
def test_call_model_prepends_system_prompt_when_missing(mock_get_chat_model: MagicMock) -> None:
    with_tools = _llm_returning(AIMessage(content="answer"))
    with_tools.bind_tools.return_value = with_tools
    without_tools = _llm_returning(AIMessage(content="unused"))
    mock_get_chat_model.side_effect = [with_tools, without_tools]

    call_model = _build_call_model()
    call_model({"messages": [HumanMessage(content="hi")]})

    invoked_messages = with_tools.invoke.call_args[0][0]
    assert isinstance(invoked_messages[0], SystemMessage)
    assert invoked_messages[0].content == SYSTEM_PROMPT


@patch("research_copilot.agent.graph.get_chat_model")
def test_call_model_switches_to_tool_free_llm_at_max_calls(
    mock_get_chat_model: MagicMock,
) -> None:
    with_tools = _llm_returning(AIMessage(content="should not be reached"))
    with_tools.bind_tools.return_value = with_tools
    forced_answer = AIMessage(content="Ich konnte es nicht finden.")
    without_tools = _llm_returning(forced_answer)
    mock_get_chat_model.side_effect = [with_tools, without_tools]

    call_model = _build_call_model()
    messages = [HumanMessage(content="Frage")]
    messages += [
        ToolMessage(content=f"result {i}", tool_call_id=str(i)) for i in range(MAX_TOOL_CALLS)
    ]

    result = call_model({"messages": messages})

    without_tools.invoke.assert_called_once()
    with_tools.invoke.assert_not_called()
    assert result["messages"] == [forced_answer]


def _chunky_content(n: int) -> str:
    return "\n\n---\n\n".join(f"[Source: file.pdf, S. {i}]\nchunk {i}" for i in range(n))


def test_trim_stale_tool_messages_caps_earlier_rounds_only() -> None:
    stale = ToolMessage(content=_chunky_content(12), tool_call_id="1")
    latest = ToolMessage(content=_chunky_content(12), tool_call_id="2")
    messages = [
        HumanMessage(content="Frage"),
        AIMessage(content="", tool_calls=[{"name": "search_filings", "args": {}, "id": "1"}]),
        stale,
        AIMessage(content="", tool_calls=[{"name": "search_filings", "args": {}, "id": "2"}]),
        latest,
    ]

    trimmed = _trim_stale_tool_messages(messages)

    trimmed_stale = trimmed[2]
    assert isinstance(trimmed_stale, ToolMessage)
    assert trimmed_stale.content.count("chunk ") == 4
    assert "8 weitere" in trimmed_stale.content
    assert trimmed[4] is latest
    assert trimmed[4].content == latest.content


def test_trim_stale_tool_messages_leaves_short_content_untouched() -> None:
    messages = [
        HumanMessage(content="Frage"),
        AIMessage(content="", tool_calls=[{"name": "get_stock_price", "args": {}, "id": "1"}]),
        ToolMessage(content="[Source: Yahoo Finance (NA9.DE)]\nshort result", tool_call_id="1"),
        AIMessage(content="answer"),
    ]

    trimmed = _trim_stale_tool_messages(messages)

    assert trimmed[2].content == messages[2].content


@patch("research_copilot.agent.tools.yf.Ticker")
@patch("research_copilot.agent.graph.get_chat_model")
def test_generate_report_sees_the_full_conversation(
    mock_get_chat_model: MagicMock, mock_ticker: MagicMock
) -> None:
    """generate_report gets the whole message history, including the
    agent's own AIMessages - tried restricting this to just Human/Tool
    messages for latency, but a token benchmark showed savings of well
    under 1% (tool payloads dominate the context, not the agent's prose),
    so it wasn't worth the added indirection. See README."""
    import pandas as pd

    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [1.0, 1.0]})

    fixed_report = ResearchReport(company="Test AG", summary="ok")
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = fixed_report
    with_tools = MagicMock()
    with_tools.bind_tools.return_value = with_tools
    final_prose = AIMessage(content="final prose the report node should also see")
    with_tools.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[{"name": "get_stock_price", "args": {"ticker": "NA9.DE"}, "id": "1"}],
        ),
        final_prose,
    ]
    without_tools = MagicMock()
    mock_get_chat_model.side_effect = [structured_llm, with_tools, without_tools]
    structured_llm.with_structured_output.return_value = structured_llm

    graph = build_report_graph()
    initial_question = HumanMessage(content="Erstelle einen Report zu Nagarro")
    result = graph.invoke({"messages": [initial_question]})

    assert result["report"] is fixed_report
    invoked_messages = structured_llm.invoke.call_args[0][0]
    assert isinstance(invoked_messages[0], SystemMessage)
    assert invoked_messages[0].content == REPORT_PROMPT
    assert invoked_messages[1] is initial_question
    assert any(isinstance(m, ToolMessage) for m in invoked_messages)
    assert invoked_messages[-1] is final_prose
