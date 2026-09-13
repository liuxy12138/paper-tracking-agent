from __future__ import annotations

from competitive_research_agent.config import AgentConfig
from competitive_research_agent.workflow import LangGraphResearchWorkflow


def test_normalize_plan_payload_coerces_object_steps_to_strings():
    workflow = LangGraphResearchWorkflow.__new__(LangGraphResearchWorkflow)
    workflow.config = AgentConfig()
    default_plan = {
        "objective": "answer question",
        "steps": ["plan", "retrieve", "summarize"],
        "search_queries": ["original query"],
        "tool_calls": [{"name": "search_knowledge_base", "args": {"query": "q", "top_k": 6}}],
        "answer_format": "grounded answer",
        "memory_summary": [],
    }
    raw_plan = {
        "objective": "answer question",
        "steps": [
            {"description": "Search for AI coding assistants."},
            {"name": "Fetch external evidence"},
            "Compile the final answer",
        ],
        "search_queries": [{"queries": ["coding assistant pricing", "enterprise governance"]}],
        "tool_calls": [
            {"name": "search_knowledge_base", "args": {"query": "coding assistant pricing", "top_k": 6}},
            {"name": "", "args": "ignored"},
        ],
        "answer_format": "markdown",
    }

    normalized = workflow._normalize_plan_payload(raw_plan, default_plan)

    assert normalized["steps"] == [
        "Search for AI coding assistants.",
        "Fetch external evidence",
        "Compile the final answer",
    ]
    assert normalized["search_queries"] == ["coding assistant pricing", "enterprise governance"]
    assert normalized["tool_calls"] == [
        {"name": "search_knowledge_base", "args": {"query": "coding assistant pricing", "top_k": 6}},
        {"name": "search_knowledge_base", "args": {"query": "enterprise governance", "top_k": 6}},
    ]


def test_normalize_reflection_payload_coerces_retry_focus_list_to_string():
    workflow = LangGraphResearchWorkflow.__new__(LangGraphResearchWorkflow)
    fallback = {
        "passed": True,
        "score": 0.6,
        "issues": [],
        "should_retry": False,
        "retry_focus": "",
    }
    raw_reflection = {
        "passed": False,
        "score": "0.4",
        "issues": ["Need more evidence", {"description": "Missing source support"}],
        "should_retry": True,
        "retry_focus": [
            "Provide more detailed evidence",
            "especially on current applications and potential future ones.",
        ],
    }

    normalized = workflow._normalize_reflection_payload(raw_reflection, fallback)

    assert normalized["passed"] is False
    assert normalized["should_retry"] is True
    assert normalized["score"] == 0.4
    assert normalized["issues"] == ["Need more evidence", "Missing source support"]
    assert normalized["retry_focus"] == (
        "Provide more detailed evidence especially on current applications and potential future ones."
    )


class AvailableReflectionLlm:
    is_available = True


def _reflection_workflow() -> LangGraphResearchWorkflow:
    workflow = LangGraphResearchWorkflow.__new__(LangGraphResearchWorkflow)
    workflow.config = AgentConfig()
    workflow.llm = AvailableReflectionLlm()
    return workflow


def test_deterministic_reflection_score_uses_evidence_sources_citations_and_scores():
    workflow = _reflection_workflow()
    state = {
        "final_answer": "结论一 [Competitor Report A]；结论二 [Competitor Report B]。",
        "retrieval_results": [
            {"title": "Competitor Report A", "source": "a.pdf", "score": 0.8},
            {"title": "Competitor Report A", "source": "a.pdf", "score": 0.6},
            {"title": "Competitor Report B", "source": "b.pdf", "score": 0.7},
        ],
    }

    score, breakdown = workflow._deterministic_reflection_score(state)

    assert score == 0.925
    assert breakdown["evidence_count_score"] == 1.0
    assert breakdown["source_count_score"] == 1.0
    assert breakdown["citation_coverage"] == 1.0
    assert breakdown["retrieval_score"] == 0.7


def test_reflection_decision_fuses_rule_and_llm_scores():
    workflow = _reflection_workflow()
    state = {
        "retry_count": 0,
        "final_answer": "结论一 [Competitor Report A]；结论二 [Competitor Report B]。",
        "retrieval_results": [
            {"title": "Competitor Report A", "source": "a.pdf", "score": 0.8},
            {"title": "Competitor Report A", "source": "a.pdf", "score": 0.6},
            {"title": "Competitor Report B", "source": "b.pdf", "score": 0.7},
        ],
    }

    reflection, should_retry = workflow._finalize_reflection_decision(
        state,
        {"score": 0.4, "issues": [], "should_retry": True, "retry_focus": ""},
    )

    assert reflection["score"] == 0.7413
    assert reflection["llm_score"] == 0.4
    assert reflection["deterministic_score"] == 0.925
    assert reflection["passed"] is True
    assert reflection["retry_recommended"] is False
    assert reflection["llm_retry_recommended"] is True
    assert should_retry is False


def test_reflection_decision_retries_low_rule_score_and_honors_round_limit():
    workflow = _reflection_workflow()
    state = {
        "retry_count": 0,
        "final_answer": "单一来源结论 [Competitor Report A]。",
        "retrieval_results": [
            {"title": "Competitor Report A", "source": "a.pdf", "score": 1.0},
        ],
    }
    reflection, should_retry = workflow._finalize_reflection_decision(
        state,
        {"score": 1.0, "issues": [], "should_retry": False, "retry_focus": ""},
    )

    assert reflection["score"] > workflow.config.graph.reflection_retry_score_threshold
    assert reflection["passed"] is False
    assert reflection["retry_recommended"] is True
    assert reflection["score_breakdown"]["quality_gate_failures"] == [
        "evidence_count",
        "source_count",
    ]
    assert should_retry is True
    assert "independent sources" in reflection["retry_focus"]

    state["retry_count"] = workflow.config.graph.max_reflection_rounds
    reflection, should_retry = workflow._finalize_reflection_decision(
        state,
        {"score": 1.0, "issues": [], "should_retry": False, "retry_focus": ""},
    )
    assert reflection["retry_recommended"] is True
    assert reflection["retry_blocked_by_limit"] is True
    assert should_retry is False


def test_tool_calls_fill_missing_query_args_for_common_tools():
    workflow = LangGraphResearchWorkflow.__new__(LangGraphResearchWorkflow)
    workflow.config = AgentConfig()
    workflow._latest_question = lambda state: "What are the application scenarios of AI coding assistants?"
    state = {
        "plan": {
            "tool_calls": [
                {"name": "search_knowledge_base", "args": {}},
            ]
        },
        "rewritten_queries": [],
        "reflection": {},
        "retry_count": 0,
    }

    tool_calls = workflow._tool_calls_from_plan(state)

    assert tool_calls[0]["args"]["query"] == "What are the application scenarios of AI coding assistants?"
    assert tool_calls[0]["args"]["top_k"] == workflow.config.search.semantic_top_k


def test_normalize_plan_payload_replaces_duplicate_semantic_calls_with_search_queries():
    workflow = LangGraphResearchWorkflow.__new__(LangGraphResearchWorkflow)
    workflow.config = AgentConfig()
    default_plan = {
        "objective": "competitive brief",
        "steps": ["plan", "retrieve", "summarize"],
        "search_queries": ["default query"],
        "tool_calls": [{"name": "search_knowledge_base", "args": {"query": "default query", "top_k": 6}}],
        "answer_format": "markdown report",
        "memory_summary": [],
    }
    raw_plan = {
        "objective": "competitive brief",
        "steps": ["review evidence", "summarize findings"],
        "search_queries": ["query one", "query two", "query three"],
        "tool_calls": [
            {"name": "search_knowledge_base", "args": {"query": "same query", "top_k": 6}},
            {"name": "search_knowledge_base", "args": {"query": "same query", "top_k": 6}},
            {"name": "search_knowledge_base", "args": {"query": "same query", "top_k": 6}},
        ],
    }

    normalized = workflow._normalize_plan_payload(raw_plan, default_plan)

    assert normalized["tool_calls"] == [
        {"name": "search_knowledge_base", "args": {"query": "query one", "top_k": 6}},
        {"name": "search_knowledge_base", "args": {"query": "query two", "top_k": 6}},
        {"name": "search_knowledge_base", "args": {"query": "query three", "top_k": 6}},
    ]


def test_normalize_plan_payload_simplifies_long_answer_format():
    workflow = LangGraphResearchWorkflow.__new__(LangGraphResearchWorkflow)
    workflow.config = AgentConfig()
    default_plan = {
        "objective": "competitive brief",
        "steps": ["plan"],
        "search_queries": ["query one"],
        "tool_calls": [],
        "answer_format": "markdown report",
        "memory_summary": [],
    }
    raw_plan = {
        "objective": "competitive brief",
        "steps": ["plan"],
        "search_queries": ["query one"],
        "tool_calls": [],
        "answer_format": (
            "{'title': 'Competitive Research Brief', 'content': 'The latest advancements in low earth orbit "
            "enterprise governance and navigation...'}"
        ),
    }

    normalized = workflow._normalize_plan_payload(raw_plan, default_plan)

    assert normalized["answer_format"] == "structured research brief"
