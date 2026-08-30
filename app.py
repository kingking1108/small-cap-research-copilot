"""Minimal Streamlit demo frontend for the Small-Cap Research Copilot.

Standalone by design: duplicates the small amount of streaming/status glue
also used by cli.py (research_copilot/cli.py::_run_with_status) rather than
factoring it into a shared module, so this file has zero coupling to the
CLI's internals and can't conflict with changes made there.

Run with:
    streamlit run app.py
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from research_copilot.agent.graph import build_agent_graph, build_report_graph
from research_copilot.observability import get_langfuse_handler
from research_copilot.report.verify import find_unverified_claims

st.set_page_config(page_title="Small-Cap Research Copilot", page_icon="🔎")


def _tool_call_status(message: AIMessage) -> str:
    call = message.tool_calls[0]
    if call["name"] == "get_stock_price":
        return f"Rufe Kursdaten ab: {call['args'].get('ticker', '')}..."
    query = call["args"].get("query", "")
    return f"Suche: {query}..." if query else "Suche in Filings..."


def _run_with_status(graph, initial_state: dict, config: dict, final_step_label: str) -> dict:
    """Runs the graph via `.stream()` (not `.invoke()`) so the status widget
    can show what the agent is actually doing round by round - which tool,
    which query - instead of a blank spinner for however long a multi-round
    question takes."""
    messages = list(initial_state.get("messages", []))
    report = None
    with st.status("Denke nach...", expanded=False) as status:
        for update in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_output in update.items():
                if "messages" in node_output:
                    messages.extend(node_output["messages"])
                if "report" in node_output:
                    report = node_output["report"]

                if node_name == "agent":
                    latest = node_output["messages"][-1]
                    if latest.tool_calls:
                        status.update(label=_tool_call_status(latest))
                    else:
                        status.update(label=final_step_label)
                elif node_name == "tools":
                    status.update(label="Werte Ergebnisse aus...")
        status.update(label="Fertig", state="complete")
    return {"messages": messages, "report": report}


def _render_report(research_report, messages: list) -> None:
    st.markdown(f"## {research_report.company}")
    st.markdown(research_report.summary)

    if research_report.key_facts:
        st.markdown("### Key Facts")
        for fact in research_report.key_facts:
            page = f", S. {fact.page}" if fact.page is not None else ""
            st.markdown(f"- {fact.claim} _(Quelle: {fact.source}{page})_")

    if research_report.open_questions:
        st.markdown("### Open Questions")
        for question in research_report.open_questions:
            st.markdown(f"- {question}")

    unverified = find_unverified_claims(research_report, messages)
    if unverified:
        lines = []
        for claim in unverified:
            page = f", S. {claim.page}" if claim.page is not None else ""
            lines.append(f"- {claim.claim!r} → zitierte Quelle {claim.source!r}{page}")
        st.warning(
            "**Nicht verifizierte Zitate** (Quelle/Seite wurde vom Agenten nie "
            "tatsächlich abgerufen):\n\n" + "\n".join(lines)
        )


st.title("🔎 Small-Cap Research Copilot")
st.caption(
    "Agentische RAG-Recherche über Amadeus Fire, Nagarro, Hypoport, "
    "SUSS MicroTec und Deutsche Beteiligungs AG."
)

mode = st.sidebar.radio("Modus", ["Frage stellen", "Report erstellen"])
st.sidebar.caption(
    "**Frage stellen**: freie Konversation mit Zitaten.\n\n"
    "**Report erstellen**: strukturierter, zitat-geprüfter Report zu einem "
    "Unternehmen."
)

handler = get_langfuse_handler()

if mode == "Frage stellen":
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    question = st.chat_input("Frage zur Watchlist stellen...")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            agent = build_agent_graph()
            config = {"callbacks": [handler], "run_name": question} if handler else {}
            result = _run_with_status(
                agent,
                {"messages": [HumanMessage(content=question)]},
                config,
                final_step_label="Formuliere Antwort...",
            )
            answer = result["messages"][-1].content
            st.markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))

else:
    topic = st.text_input("Unternehmen / Thema", placeholder="z.B. Nagarro")
    if st.button("Report erstellen", disabled=not topic):
        graph = build_report_graph()
        run_name = f"report: {topic}"
        config = {"callbacks": [handler], "run_name": run_name} if handler else {}
        prompt = f"Erstelle einen Research-Report zu: {topic}"
        try:
            result = _run_with_status(
                graph,
                {"messages": [HumanMessage(content=prompt)]},
                config,
                final_step_label="Erstelle strukturierten Report...",
            )
        except Exception as exc:
            # The hosted model is not fully deterministic (see README) and
            # can return output that fails ResearchReport's structured-
            # output validation - fail visibly instead of an unhandled
            # traceback filling the page.
            st.error(f"Report-Generierung fehlgeschlagen: {exc}")
            st.stop()

        research_report = result.get("report")
        if research_report is None:
            st.error("Konnte keinen strukturierten Report erzeugen.")
        else:
            _render_report(research_report, result["messages"])
