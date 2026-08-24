from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from research_copilot.agent.graph import MAX_TOOL_CALLS, SYSTEM_PROMPT, _build_call_model


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
