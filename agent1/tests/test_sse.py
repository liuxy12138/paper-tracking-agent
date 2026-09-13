from __future__ import annotations

import json

from competitive_research_agent.pipeline import CompetitiveResearchAgent
from competitive_research_agent.webapp import _sse_message
from competitive_research_agent.workflow import emit_workflow_event


def test_sse_message_uses_named_event_and_preserves_unicode():
    message = _sse_message({"event": "token", "content": "竞品资料"})

    assert message.startswith("event: token\n")
    assert message.endswith("\n\n")
    payload = json.loads(message.split("data: ", 1)[1])
    assert payload == {"event": "token", "content": "竞品资料"}


def test_agent_stream_bridges_workflow_events_and_result():
    agent = object.__new__(CompetitiveResearchAgent)

    def fake_ask(question, thread_id=None, user_id=None):
        emit_workflow_event({"event": "node", "node": "summary", "status": "start"})
        emit_workflow_event({"event": "token", "content": "answer"})
        return {"answer": "answer"}

    agent.ask = fake_ask

    events = list(agent.ask_stream("question", thread_id="thread", user_id="user"))

    assert [event["event"] for event in events] == ["node", "token", "result", "done"]
    assert events[2]["result"]["answer"] == "answer"
