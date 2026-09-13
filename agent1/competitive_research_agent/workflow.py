from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

try:  # pragma: no cover - optional dependency
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception:  # pragma: no cover - optional dependency
    SqliteSaver = None

from .config import AgentConfig
from .llm import GLMClient
from .logging_utils import get_logger
from .memory import LongTermMemoryStore, ThreadHistoryStore
from .models import WorkflowExecutionError, WorkflowResult
from .observability import (
    PerformanceCollector,
    current_collector,
    measure_current,
    reset_current_collector,
    set_current_collector,
)
from .schemas import EvidenceItemSchema, PlanSchema, ReflectionSchema


WorkflowEventSink = Callable[[dict[str, Any]], None]
current_event_sink: ContextVar[WorkflowEventSink | None] = ContextVar(
    "workflow_event_sink",
    default=None,
)


@contextmanager
def workflow_event_sink(sink: WorkflowEventSink) -> Iterator[None]:
    token = current_event_sink.set(sink)
    try:
        yield
    finally:
        current_event_sink.reset(token)


def emit_workflow_event(event: dict[str, Any]) -> None:
    sink = current_event_sink.get()
    if sink is not None:
        sink(event)


class ResearchState(MessagesState):
    user_question: str
    user_id: str
    thread_id: str
    requested_mode: str
    plan: dict[str, Any]
    rewritten_queries: list[str]
    retrieval_results: list[dict[str, Any]]
    analysis: dict[str, Any]
    final_answer: str
    reflection: dict[str, Any]
    should_retry: bool
    retry_count: int
    tool_history: list[dict[str, Any]]
    memory_hits: list[dict[str, Any]]
    report_path: str | None
    session_context: str


class LangGraphResearchWorkflow:
    def __init__(
        self,
        config: AgentConfig,
        rag,
        toolbox,
        thread_history: ThreadHistoryStore,
        long_term_memory: LongTermMemoryStore,
    ):
        self.config = config
        self.rag = rag
        self.toolbox = toolbox
        self.thread_history = thread_history
        self.long_term_memory = long_term_memory
        self.logger = get_logger(self.__class__.__name__)
        self.llm = GLMClient(api_key=config.api_key, model=config.rag.llm_model)
        self._checkpoint_conn: sqlite3.Connection | None = None
        self._active_threads: set[str] = set()
        self.graph = self._build_graph()

    def _collector(self) -> PerformanceCollector | None:
        return current_collector()

    def _record_json_metrics(
        self,
        node_name: str,
        *,
        meta: dict[str, Any],
        schema_success: bool | None = None,
        used_default: bool | None = None,
    ) -> None:
        collector = self._collector()
        if collector is None:
            return
        collector.increment(f"{node_name}_json_attempts")
        if meta.get("parse_success"):
            collector.increment(f"{node_name}_json_parse_successes")
        if used_default is None:
            used_default = bool(meta.get("used_default"))
        if used_default:
            collector.increment(f"{node_name}_default_fallbacks")
        if schema_success is not None:
            collector.increment(f"{node_name}_schema_attempts")
            if schema_success:
                collector.increment(f"{node_name}_schema_successes")

    def _finalize_collector_metrics(self, collector: PerformanceCollector) -> None:
        counters = collector.snapshot().get("counters", {})
        workflow_runs = max(1, int(counters.get("workflow_run_count", 0)))
        retry_triggers = int(counters.get("retry_trigger_count", 0))
        retry_rescues = int(counters.get("retry_rescue_success_count", 0))
        parse_attempts = sum(
            int(counters.get(name, 0))
            for name in (
                "planner_json_attempts",
                "analysis_json_attempts",
                "reflection_json_attempts",
                "query_rewrite_json_attempts",
            )
        )
        parse_successes = sum(
            int(counters.get(name, 0))
            for name in (
                "planner_json_parse_successes",
                "analysis_json_parse_successes",
                "reflection_json_parse_successes",
                "query_rewrite_json_parse_successes",
            )
        )
        schema_attempts = sum(
            int(counters.get(name, 0))
            for name in ("planner_schema_attempts", "reflection_schema_attempts")
        )
        schema_successes = sum(
            int(counters.get(name, 0))
            for name in ("planner_schema_successes", "reflection_schema_successes")
        )
        collector.record_value(
            "structured_parse_success_rate",
            parse_successes / parse_attempts if parse_attempts else 0.0,
        )
        collector.record_value(
            "schema_validation_success_rate",
            schema_successes / schema_attempts if schema_attempts else 0.0,
        )
        collector.record_value(
            "workflow_error_rate",
            int(counters.get("workflow_error_count", 0)) / workflow_runs,
        )
        collector.record_value(
            "retry_rescue_success_rate",
            retry_rescues / retry_triggers if retry_triggers else 0.0,
        )

    def _build_checkpointer(self):
        if self.config.graph.use_sqlite_checkpointer and SqliteSaver is not None:
            self._checkpoint_conn = sqlite3.connect(
                self.config.paths.checkpoint_path,
                check_same_thread=False,
            )
            return SqliteSaver(self._checkpoint_conn)
        return InMemorySaver()

    def _build_graph(self):
        builder = StateGraph(ResearchState)
        builder.add_node("planner", self._timed_node("planner", self.planner_node))
        builder.add_node("retrieval", self._timed_node("retrieval", self.retrieval_node))
        builder.add_node("analysis", self._timed_node("analysis", self.analysis_node))
        builder.add_node("summary", self._timed_node("summary", self.summary_node))
        builder.add_node("reflection", self._timed_node("reflection", self.reflection_node))
        builder.add_node("finalize", self._timed_node("finalize", self.finalize_node))

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "retrieval")
        builder.add_edge("retrieval", "analysis")
        builder.add_edge("analysis", "summary")
        builder.add_edge("summary", "reflection")
        builder.add_conditional_edges(
            "reflection",
            self.route_after_reflection,
            {"retry": "retrieval", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self._build_checkpointer())

    def _timed_node(self, name: str, node):
        def wrapped(state: ResearchState) -> dict[str, Any]:
            emit_workflow_event({"event": "node", "status": "start", "node": name})
            try:
                with measure_current(name, category="nodes"):
                    result = node(state)
            except Exception as exc:
                emit_workflow_event(
                    {
                        "event": "node",
                        "status": "error",
                        "node": name,
                        "error": str(exc),
                    }
                )
                raise
            emit_workflow_event({"event": "node", "status": "completed", "node": name})
            return result

        return wrapped

    def _latest_question(self, state: ResearchState) -> str:
        if state.get("user_question"):
            return state["user_question"]
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    def _heuristic_plan(self, question: str, mode: str, memory_hits: list[dict[str, Any]]) -> dict[str, Any]:
        query_terms = [question]
        if mode == "research_brief":
            query_terms.append(self.config.topic)

        tool_calls = [
            {
                "name": "search_knowledge_base",
                "args": {"query": question, "top_k": self.config.search.semantic_top_k},
            }
        ]
        if getattr(getattr(self.toolbox, "web_research", None), "available", False):
            tool_calls.append(
                {
                    "name": "search_web",
                    "args": {"query": question, "max_results": self.config.search.max_results},
                }
            )
        return {
            "objective": question,
            "steps": ["需求规划", "多角度查询生成", "混合检索", "证据分析", "反思补检", "结构化简报生成"],
            "search_queries": query_terms[: self.config.search.query_rewrite_count],
            "tool_calls": tool_calls,
            "answer_format": "structured research brief" if mode == "research_brief" else "evidence-grounded research answer",
            "memory_summary": [item["text"][:120] for item in memory_hits],
        }

    def _rewrite_queries(self, question: str, plan: dict[str, Any], reflection_focus: str = "") -> list[str]:
        seed_queries = [question] + self._normalize_query_items(plan.get("search_queries", []))
        if reflection_focus:
            seed_queries.append(reflection_focus)
        deduped = self._normalize_query_items(seed_queries)

        if not self.config.graph.enable_query_rewrite or not self.llm.is_available:
            return deduped[: self.config.search.query_rewrite_count]

        default_payload = deduped[: self.config.search.query_rewrite_count]
        prompt = f"""
Question: {question}
Plan: {json.dumps(plan, ensure_ascii=False)}
Retry focus: {reflection_focus}

Generate up to {self.config.search.query_rewrite_count} retrieval queries.
They should cover market size, customer demand, competitor features, pricing/business model, technical route, product limitations, and concise keyword forms.
"""
        payload, meta = self.llm.complete_json_with_meta(
            system_prompt="You are the Retrieval Agent. Rewrite the user question into strong industry and competitor research queries.",
            user_prompt=prompt,
            default=default_payload,
        )
        self._record_json_metrics("query_rewrite", meta=meta)

        if isinstance(payload, dict):
            payload = payload.get("queries", default_payload)
        if not isinstance(payload, list):
            return default_payload
        rewritten = self._normalize_query_items(payload)
        return list(dict.fromkeys(rewritten))[: self.config.search.query_rewrite_count]

    def _normalize_query_items(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, dict):
            items = value.get("queries") or value.get("search_queries") or value.values()
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [value]

        normalized: list[str] = []
        for item in items:
            if isinstance(item, (list, tuple, set)):
                normalized.extend(self._normalize_query_items(list(item)))
                continue
            if isinstance(item, dict):
                normalized.extend(self._normalize_query_items(item))
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return list(dict.fromkeys(normalized))

    def _normalize_plan_payload(self, plan: dict[str, Any], default_plan: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(plan or {})

        steps = normalized.get("steps", default_plan.get("steps", []))
        if not isinstance(steps, list):
            steps = [steps]
        normalized_steps: list[str] = []
        for item in steps:
            if isinstance(item, dict):
                text = str(
                    item.get("title")
                    or item.get("name")
                    or item.get("description")
                    or item.get("step")
                    or ""
                ).strip()
            else:
                text = str(item).strip()
            if text:
                normalized_steps.append(text)
        normalized["steps"] = normalized_steps or list(default_plan.get("steps", []))

        normalized["search_queries"] = self._normalize_query_items(
            normalized.get("search_queries", default_plan.get("search_queries", []))
        )
        normalized["memory_summary"] = self._normalize_query_items(
            normalized.get("memory_summary", default_plan.get("memory_summary", []))
        )
        search_queries = list(normalized["search_queries"])

        tool_calls = normalized.get("tool_calls", default_plan.get("tool_calls", []))
        if not isinstance(tool_calls, list):
            tool_calls = []
        normalized_tool_calls: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", "")).strip()
            args = call.get("args", {})
            if not name:
                continue
            if not isinstance(args, dict):
                args = {}
            signature = json.dumps({"name": name, "args": args}, sort_keys=True, ensure_ascii=False)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            normalized_tool_calls.append({"name": name, "args": args})

        semantic_calls = [call for call in normalized_tool_calls if call["name"] == "search_knowledge_base"]
        non_semantic_calls = [call for call in normalized_tool_calls if call["name"] != "search_knowledge_base"]
        if search_queries:
            semantic_queries = self._normalize_query_items(
                [call.get("args", {}).get("query", "") for call in semantic_calls]
            )
            if not semantic_queries or len(semantic_queries) <= 1:
                semantic_calls = [
                    {
                        "name": "search_knowledge_base",
                        "args": {
                            "query": query,
                            "top_k": self.config.search.semantic_top_k,
                        },
                    }
                    for query in search_queries
                ]
            else:
                semantic_calls = [
                    {
                        "name": "search_knowledge_base",
                        "args": {
                            "query": query,
                            "top_k": self.config.search.semantic_top_k,
                        },
                    }
                    for query in semantic_queries[: self.config.search.query_rewrite_count]
                ]
        normalized_tool_calls = non_semantic_calls + semantic_calls
        normalized["tool_calls"] = normalized_tool_calls

        objective = str(normalized.get("objective", "")).strip()
        normalized["objective"] = objective or str(default_plan.get("objective", "")).strip()
        normalized["answer_format"] = self._normalize_answer_format(
            normalized.get("answer_format", ""),
            str(default_plan.get("answer_format", "evidence-grounded research answer")),
        )
        return normalized

    def _normalize_reflection_payload(
        self,
        reflection: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(reflection or {})

        for key in ("passed", "should_retry"):
            if key in normalized:
                normalized[key] = bool(normalized[key])
            else:
                normalized[key] = bool(fallback.get(key, False))

        try:
            score = float(normalized.get("score", fallback.get("score", 0.0)) or 0.0)
        except (TypeError, ValueError):
            score = float(fallback.get("score", 0.0) or 0.0)
        normalized["score"] = max(0.0, min(1.0, score))

        issues = normalized.get("issues", fallback.get("issues", []))
        normalized["issues"] = self._normalize_query_items(issues)

        retry_focus = normalized.get("retry_focus", fallback.get("retry_focus", ""))
        if isinstance(retry_focus, dict):
            retry_focus = (
                retry_focus.get("focus")
                or retry_focus.get("query")
                or retry_focus.get("description")
                or retry_focus.get("text")
                or ""
            )
        if isinstance(retry_focus, (list, tuple, set)):
            retry_focus_items = self._normalize_query_items(list(retry_focus))
            retry_focus = " ".join(retry_focus_items)
        normalized["retry_focus"] = str(retry_focus or "").strip()
        return normalized

    def _deterministic_reflection_score(self, state: ResearchState) -> tuple[float, dict[str, Any]]:
        evidence = state.get("retrieval_results", [])
        answer = re.sub(r"\s+", " ", state.get("final_answer", "")).casefold()
        graph_config = self.config.graph

        evidence_count = len(evidence)
        min_evidence = max(1, graph_config.reflection_min_evidence_items)
        evidence_count_score = min(1.0, evidence_count / min_evidence)

        source_keys = {
            str(item.get("source") or item.get("title") or "").strip().casefold()
            for item in evidence
            if str(item.get("source") or item.get("title") or "").strip()
        }
        source_count = len(source_keys)
        min_sources = max(1, graph_config.reflection_min_source_count)
        source_count_score = min(1.0, source_count / min_sources)

        cited_sources: set[str] = set()
        for item in evidence:
            title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip().casefold()
            source_key = str(item.get("source") or item.get("title") or "").strip().casefold()
            if title and title != "unknown" and f"[{title}]" in answer and source_key:
                cited_sources.add(source_key)
        required_citations = min(source_count, min_sources)
        citation_coverage = (
            min(1.0, len(cited_sources) / required_citations)
            if required_citations
            else 0.0
        )

        retrieval_scores: list[float] = []
        for item in evidence[:min_evidence]:
            try:
                score = float(item.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            retrieval_scores.append(max(0.0, min(1.0, score)))
        retrieval_score = (
            sum(retrieval_scores) / len(retrieval_scores)
            if retrieval_scores
            else 0.0
        )

        metric_weights = {
            "evidence_count": max(0.0, graph_config.reflection_evidence_count_weight),
            "source_count": max(0.0, graph_config.reflection_source_count_weight),
            "citation_coverage": max(0.0, graph_config.reflection_citation_coverage_weight),
            "retrieval_score": max(0.0, graph_config.reflection_retrieval_score_weight),
        }
        weight_total = sum(metric_weights.values()) or 1.0
        deterministic_score = (
            evidence_count_score * metric_weights["evidence_count"]
            + source_count_score * metric_weights["source_count"]
            + citation_coverage * metric_weights["citation_coverage"]
            + retrieval_score * metric_weights["retrieval_score"]
        ) / weight_total
        breakdown = {
            "evidence_count": evidence_count,
            "required_evidence_count": min_evidence,
            "evidence_count_score": round(evidence_count_score, 4),
            "source_count": source_count,
            "required_source_count": min_sources,
            "source_count_score": round(source_count_score, 4),
            "cited_source_count": len(cited_sources),
            "required_citation_count": required_citations,
            "citation_coverage": round(citation_coverage, 4),
            "retrieval_score": round(retrieval_score, 4),
        }
        return round(deterministic_score, 4), breakdown

    def _finalize_reflection_decision(
        self,
        state: ResearchState,
        reflection: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        deterministic_score, breakdown = self._deterministic_reflection_score(state)
        llm_score = max(0.0, min(1.0, float(reflection.get("score", 0.0) or 0.0)))
        llm_available = bool(getattr(getattr(self, "llm", None), "is_available", False))
        rule_weight = (
            max(0.0, min(1.0, self.config.graph.reflection_rule_score_weight))
            if llm_available
            else 1.0
        )
        combined_score = deterministic_score * rule_weight + llm_score * (1.0 - rule_weight)
        threshold = max(0.0, min(1.0, self.config.graph.reflection_retry_score_threshold))
        minimum_citation_coverage = max(
            0.0,
            min(1.0, self.config.graph.reflection_min_citation_coverage),
        )
        minimum_retrieval_score = max(
            0.0,
            min(1.0, self.config.graph.reflection_min_retrieval_score),
        )
        quality_gate_failures: list[str] = []
        if breakdown["evidence_count"] < breakdown["required_evidence_count"]:
            quality_gate_failures.append("evidence_count")
        if breakdown["source_count"] < breakdown["required_source_count"]:
            quality_gate_failures.append("source_count")
        if breakdown["citation_coverage"] < minimum_citation_coverage:
            quality_gate_failures.append("citation_coverage")
        if breakdown["retrieval_score"] < minimum_retrieval_score:
            quality_gate_failures.append("retrieval_score")
        quality_gates_failed = (
            self.config.graph.reflection_enforce_quality_gates
            and bool(quality_gate_failures)
        )
        retry_recommended = combined_score < threshold or quality_gates_failed
        retry_available = state.get("retry_count", 0) < self.config.graph.max_reflection_rounds
        should_retry = retry_recommended and retry_available
        llm_passed = bool(reflection.get("passed", True))
        llm_retry_recommended = bool(reflection.get("should_retry", False))

        deterministic_issues: list[str] = []
        retry_focus_parts: list[str] = []
        if breakdown["evidence_count"] < breakdown["required_evidence_count"]:
            deterministic_issues.append(
                f"Evidence count is {breakdown['evidence_count']}; "
                f"at least {breakdown['required_evidence_count']} items are required."
            )
            retry_focus_parts.append("retrieve more relevant evidence")
        if breakdown["source_count"] < breakdown["required_source_count"]:
            deterministic_issues.append(
                f"Source count is {breakdown['source_count']}; "
                f"at least {breakdown['required_source_count']} independent sources are required."
            )
            retry_focus_parts.append("add evidence from independent sources")
        if breakdown["citation_coverage"] < minimum_citation_coverage:
            deterministic_issues.append(
                f"Citation coverage is {breakdown['citation_coverage']:.2f}; "
                "the draft does not cite enough retrieved sources by title."
            )
            retry_focus_parts.append("support key claims with source-title citations")
        if breakdown["retrieval_score"] < minimum_retrieval_score:
            deterministic_issues.append(
                f"Average retrieval score is {breakdown['retrieval_score']:.2f}; "
                f"the minimum is {minimum_retrieval_score:.2f}."
            )
            retry_focus_parts.append("use more specific queries to retrieve stronger evidence")

        issues = list(dict.fromkeys([*reflection.get("issues", []), *deterministic_issues]))
        retry_focus = str(reflection.get("retry_focus", "")).strip()
        if retry_recommended and not retry_focus:
            retry_focus = "; ".join(dict.fromkeys(retry_focus_parts))

        reflection.update(
            {
                "passed": not retry_recommended,
                "score": round(combined_score, 4),
                "llm_score": round(llm_score, 4),
                "deterministic_score": deterministic_score,
                "score_breakdown": {
                    **breakdown,
                    "rule_score_weight": round(rule_weight, 4),
                    "llm_score_weight": round(1.0 - rule_weight, 4),
                    "retry_threshold": round(threshold, 4),
                    "minimum_citation_coverage": round(minimum_citation_coverage, 4),
                    "minimum_retrieval_score": round(minimum_retrieval_score, 4),
                    "quality_gate_failures": quality_gate_failures,
                },
                "retry_recommended": retry_recommended,
                "should_retry": should_retry,
                "retry_blocked_by_limit": retry_recommended and not retry_available,
                "llm_passed": llm_passed,
                "llm_retry_recommended": llm_retry_recommended,
                "issues": issues,
                "retry_focus": retry_focus,
            }
        )
        return reflection, should_retry

    def _normalize_answer_format(self, value: Any, default_value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return default_value
        lowered = text.casefold()
        if (
            "competitive research brief" in lowered
            or "markdown brief" in lowered
            or "structured research brief" in lowered
        ):
            return "structured research brief"
        if "json" in lowered and ("title" in lowered or "content" in lowered):
            return "structured report"
        if len(text) > 120:
            return default_value
        return text

    def _extract_file_path(self, question: str) -> str | None:
        if ".pdf" not in question.lower():
            return None
        pieces = question.replace('"', " ").replace("'", " ").split()
        for piece in pieces:
            if piece.lower().endswith(".pdf"):
                candidate = Path(piece)
                if candidate.exists():
                    return str(candidate.resolve())
        return None

    def _tool_calls_from_plan(self, state: ResearchState) -> list[dict[str, Any]]:
        plan = state.get("plan", {})
        tool_calls = list(plan.get("tool_calls", []))
        rewritten_queries = state.get("rewritten_queries", [])
        reflection_focus = state.get("reflection", {}).get("retry_focus", "")
        question = self._latest_question(state)

        if not tool_calls:
            tool_calls = [
                {
                    "name": "search_knowledge_base",
                    "args": {"query": question, "top_k": self.config.search.semantic_top_k},
                }
            ]

        for query in rewritten_queries:
            tool_calls.append(
                {
                    "name": "search_knowledge_base",
                    "args": {"query": query, "top_k": self.config.search.semantic_top_k},
                }
            )

        file_path = self._extract_file_path(question)
        if file_path:
            tool_calls.insert(0, {"name": "parse_pdf", "args": {"file_path": file_path}})

        deduped: list[dict[str, Any]] = []
        seen = set()
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", "")).strip()
            args = call.get("args", {})
            if not isinstance(args, dict):
                args = {}
            if name == "search_knowledge_base":
                args.setdefault("query", question)
                args.setdefault("top_k", self.config.search.semantic_top_k)
            elif name in {"search_web", "collect_research_pdfs"}:
                args.setdefault("query", question)
                args.setdefault("max_results", self.config.search.max_results)
            call = {"name": name, "args": args}
            signature = json.dumps(call, sort_keys=True, ensure_ascii=False)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(call)
        return deduped

    def _normalize_tool_results(self, tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for entry in tool_results:
            tool_name = entry["tool"]
            result = entry["result"]

            if tool_name == "search_knowledge_base":
                for item in result:
                    normalized.append(item)
            elif tool_name == "search_web":
                for item in result:
                    normalized.append(
                        {
                            "content": item.get("content", ""),
                            "source": item.get("url", ""),
                            "title": item.get("title", "web result"),
                            "score": item.get("score", 0.0),
                            "section": "web_search",
                            "origin": "web_search",
                            "metadata": item,
                        }
                    )
            elif tool_name == "collect_research_pdfs":
                for item in result:
                    normalized.append(
                        {
                            "content": item.get("summary", ""),
                            "source": item.get("source_url") or item.get("file_path", ""),
                            "title": item.get("title", "collected PDF"),
                            "score": item.get("relevance_score", 1.0),
                            "section": "downloaded_pdf",
                            "origin": "web_pdf_ingest",
                            "metadata": item,
                        }
                    )
            elif tool_name == "parse_pdf":
                normalized.append(
                    {
                        "content": json.dumps(result, ensure_ascii=False),
                        "source": result.get("file_path", ""),
                        "title": result.get("title", "pdf parse"),
                        "score": 1.0,
                        "section": "parsed_pdf",
                        "origin": "pdf_parser",
                        "metadata": result,
                    }
                )
        validated = []
        for item in normalized:
            validated.append(EvidenceItemSchema.model_validate(item).model_dump())
        validated.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return validated[:12]

    def _heuristic_analysis(self, state: ResearchState) -> dict[str, Any]:
        results = state.get("retrieval_results", [])
        findings = [item.get("title", "unknown") for item in results[:4]]
        sources = [item.get("source", "") for item in results[:4]]
        return {
            "key_findings": findings,
            "evidence_map": [{"claim": title, "sources": [source]} for title, source in zip(findings, sources)],
            "gaps": [] if findings else ["No strong evidence found in the current knowledge base."],
            "confidence": 0.55 if findings else 0.2,
            "recommended_sections": ["Answer", "Evidence", "Next steps"],
        }

    def _fallback_answer(self, state: ResearchState) -> str:
        question = self._latest_question(state)
        results = state.get("retrieval_results", [])
        if not results:
            return f"No relevant evidence was found for: {question}"

        lines = [f"Question: {question}", "", "Evidence summary:"]
        for index, item in enumerate(results[:5], start=1):
            lines.append(f"{index}. {item.get('title', 'unknown')} | score={item.get('score', 0)}")
            lines.append(f"   {item.get('content', '')[:220].replace(chr(10), ' ')}")
        return "\n".join(lines)

    def _build_sources(self, retrieval_results: list[dict[str, Any]]) -> list[str]:
        seen: list[str] = []
        for item in retrieval_results:
            title = item.get("title", "unknown")
            source = item.get("source", "")
            label = f"{title} | {source}"
            if label not in seen:
                seen.append(label)
        return seen[:8]

    def planner_node(self, state: ResearchState) -> dict[str, Any]:
        question = self._latest_question(state)
        user_id = state.get("user_id", self.config.graph.default_user_id)
        memory_hits = self.long_term_memory.search(
            user_id=user_id,
            query=question,
            limit=self.config.graph.max_memory_items,
        )

        default_plan = self._heuristic_plan(question, state.get("requested_mode", "qa"), memory_hits)
        if not self.llm.is_available:
            return {"plan": PlanSchema.model_validate(default_plan).model_dump(), "memory_hits": memory_hits}

        planner_prompt = f"""
Question: {question}
Mode: {state.get('requested_mode', 'qa')}
Research topic: {self.config.topic}
Relevant memories: {json.dumps(memory_hits, ensure_ascii=False)}
Available tools: {json.dumps(self.toolbox.describe_tools(), ensure_ascii=False)}

Return JSON with:
- objective
- steps
- search_queries
- tool_calls
- answer_format
"""
        plan, meta = self.llm.complete_json_with_meta(
            system_prompt="You are the Planner Agent in an Agentic RAG workflow for industry and competitor research.",
            user_prompt=planner_prompt,
            default=default_plan,
        )
        if not isinstance(plan, dict):
            plan = default_plan
        plan = self._normalize_plan_payload(plan, default_plan)
        validated_plan = PlanSchema.model_validate(plan).model_dump()
        self._record_json_metrics("planner", meta=meta, schema_success=True)
        return {"plan": validated_plan, "memory_hits": memory_hits}

    def retrieval_node(self, state: ResearchState) -> dict[str, Any]:
        question = self._latest_question(state)
        reflection_focus = state.get("reflection", {}).get("retry_focus", "")
        rewritten_queries = self._rewrite_queries(question, state.get("plan", {}), reflection_focus)
        tool_calls = self._tool_calls_from_plan({**state, "rewritten_queries": rewritten_queries})
        tool_results, history = self.toolbox.execute_calls(tool_calls)
        normalized = self._normalize_tool_results(tool_results)
        merged_history = list(state.get("tool_history", [])) + history
        return {
            "rewritten_queries": rewritten_queries,
            "retrieval_results": normalized,
            "tool_history": merged_history,
        }

    def analysis_node(self, state: ResearchState) -> dict[str, Any]:
        if not self.llm.is_available:
            return {"analysis": self._heuristic_analysis(state)}

        evidence = json.dumps(state.get("retrieval_results", [])[:8], ensure_ascii=False)
        prompt = f"""
Question: {self._latest_question(state)}
Plan: {json.dumps(state.get('plan', {}), ensure_ascii=False)}
Evidence: {evidence}
Memories: {json.dumps(state.get('memory_hits', []), ensure_ascii=False)}

Return JSON with:
- key_findings
- evidence_map
- gaps
- confidence
- recommended_sections
"""
        analysis, meta = self.llm.complete_json_with_meta(
            system_prompt="You are the Evidence Analysis Agent. Extract claims, supporting evidence, source documents, page hints, conflicts, and missing dimensions before writing conclusions.",
            user_prompt=prompt,
            default=self._heuristic_analysis(state),
        )
        if not isinstance(analysis, dict):
            analysis = self._heuristic_analysis(state)
        self._record_json_metrics("analysis", meta=meta)
        return {"analysis": analysis}

    def summary_node(self, state: ResearchState) -> dict[str, Any]:
        question = self._latest_question(state)
        analysis = state.get("analysis", {})
        evidence = state.get("retrieval_results", [])

        if not self.llm.is_available:
            answer = self._fallback_answer(state)
            emit_workflow_event({"event": "token", "content": answer})
        else:
            prompt = f"""
Question: {question}
Mode: {state.get('requested_mode', 'qa')}
Plan: {json.dumps(state.get('plan', {}), ensure_ascii=False)}
Analysis: {json.dumps(analysis, ensure_ascii=False)}
Evidence: {json.dumps(evidence[:8], ensure_ascii=False)}

Write the final answer in Chinese.
Requirements:
1. Be grounded in the evidence.
2. Cite source titles in brackets when making key claims.
3. If evidence is insufficient, say what is still missing.
4. If mode is research_brief, output a structured markdown research brief with sections for market trend, competitor comparison, product differentiation, technology route, risks, and cited evidence.
"""
            answer = self.llm.complete(
                system_prompt="You are the Summary Agent in an industry and competitor research Agentic RAG workflow. Follow an evidence-first, conclusion-second style.",
                user_prompt=prompt,
                temperature=0.2,
                on_token=lambda token: emit_workflow_event({"event": "token", "content": token}),
            )

        report_path = None
        if state.get("requested_mode") == "research_brief":
            report_name = f"Industry_Research_Brief_{state.get('thread_id', 'default')}.md"
            report_path = str(Path(self.config.paths.report_dir, report_name).resolve())
            Path(report_path).write_text(answer, encoding="utf-8")

        return {"final_answer": answer, "report_path": report_path}

    def reflection_node(self, state: ResearchState) -> dict[str, Any]:
        current_retry = state.get("retry_count", 0)
        if not self.config.graph.enable_reflection:
            return {
                "reflection": {"passed": True, "score": 1.0, "issues": []},
                "should_retry": False,
            }

        fallback = {
            "passed": True,
            "score": 0.5,
            "issues": [] if state.get("retrieval_results") else ["Not enough evidence retrieved."],
            "should_retry": False,
            "retry_focus": "",
        }
        if not self.llm.is_available:
            fallback, should_retry = self._finalize_reflection_decision(state, fallback)
            collector = self._collector()
            if collector is not None and should_retry:
                collector.increment("retry_trigger_count")
            return {
                "reflection": ReflectionSchema.model_validate(fallback).model_dump(),
                "should_retry": should_retry,
                "retry_count": current_retry + (1 if should_retry else 0),
            }

        prompt = f"""
Question: {self._latest_question(state)}
Plan: {json.dumps(state.get('plan', {}), ensure_ascii=False)}
Evidence: {json.dumps(state.get('retrieval_results', [])[:8], ensure_ascii=False)}
Draft answer: {state.get('final_answer', '')}

Evaluate the answer.
Return JSON with:
- passed
- score (a calibrated number from 0.0 to 1.0)
- issues
- should_retry
- retry_focus
Only request retry if evidence is insufficient, source coverage is too narrow, cited evidence cannot support key claims, cross-document conflicts are unresolved, or important dimensions such as pricing, feature limits, market data, or technical architecture are missing.
Score calibration: 0.0-0.39 means unsupported, 0.4-0.59 means materially incomplete, 0.6-0.79 means adequate with minor gaps, and 0.8-1.0 means strongly supported.
"""
        reflection, meta = self.llm.complete_json_with_meta(
            system_prompt="You are the Reflection Agent. Check evidence sufficiency, source diversity, citation support, cross-document conflicts, and missing research dimensions before deciding whether to retry.",
            user_prompt=prompt,
            default=fallback,
        )
        if not isinstance(reflection, dict):
            reflection = fallback

        reflection = self._normalize_reflection_payload(reflection, fallback)
        reflection, should_retry = self._finalize_reflection_decision(state, reflection)
        reflection = ReflectionSchema.model_validate(reflection).model_dump()
        self._record_json_metrics("reflection", meta=meta, schema_success=True)
        collector = self._collector()
        if collector is not None and should_retry:
            collector.increment("retry_trigger_count")
        return {
            "reflection": reflection,
            "should_retry": should_retry,
            "retry_count": current_retry + (1 if should_retry else 0),
        }

    def finalize_node(self, state: ResearchState) -> dict[str, Any]:
        final_answer = state.get("final_answer", "")
        collector = self._collector()
        if collector is not None:
            if final_answer.strip():
                collector.increment("workflow_success_count")
            else:
                collector.increment("empty_answer_count")
            if state.get("retry_count", 0) > 0:
                collector.increment("retry_completed_count")
                if state.get("retrieval_results"):
                    collector.increment("retry_rescue_success_count")
        messages = list(state.get("messages", []))
        if not messages or not isinstance(messages[-1], AIMessage) or str(messages[-1].content) != final_answer:
            messages.append(AIMessage(content=final_answer))

        thread_id = state.get("thread_id", self.config.graph.default_thread_id)
        user_id = state.get("user_id", self.config.graph.default_user_id)
        self.thread_history.save_messages(
            thread_id=thread_id,
            messages=messages,
            limit=self.config.graph.max_history_messages,
        )
        self.long_term_memory.remember_interaction(
            user_id=user_id,
            question=self._latest_question(state),
            answer=final_answer,
            topic=self.config.topic,
        )
        return {"messages": [AIMessage(content=final_answer)]}

    def route_after_reflection(self, state: ResearchState) -> str:
        return "retry" if state.get("should_retry") else "finalize"

    def invoke(
        self,
        question: str,
        *,
        thread_id: str,
        user_id: str,
        mode: str = "qa",
    ) -> WorkflowResult:
        collector = PerformanceCollector()
        collector_token = set_current_collector(collector)
        messages = [HumanMessage(content=question)]
        try:
            collector.increment("workflow_run_count")
            if thread_id not in self._active_threads:
                with measure_current("session_history_load"):
                    history = self.thread_history.load_messages(
                        thread_id=thread_id,
                        limit=self.config.graph.max_history_messages,
                    )
                messages = history + messages

            state: dict[str, Any] = {
                "messages": messages,
                "user_question": question,
                "user_id": user_id,
                "thread_id": thread_id,
                "requested_mode": mode,
                "plan": {},
                "rewritten_queries": [],
                "retrieval_results": [],
                "analysis": {},
                "final_answer": "",
                "reflection": {},
                "should_retry": False,
                "retry_count": 0,
                "tool_history": [],
                "memory_hits": [],
                "report_path": None,
                "session_context": "",
            }

            try:
                result = self.graph.invoke(
                    state,
                    {"configurable": {"thread_id": thread_id}},
                )
            except Exception as exc:
                collector.increment("workflow_error_count")
                self._finalize_collector_metrics(collector)
                raise WorkflowExecutionError(
                    question=question,
                    performance=collector.snapshot(),
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                ) from exc
            self._active_threads.add(thread_id)
        finally:
            reset_current_collector(collector_token)

        self._finalize_collector_metrics(collector)

        return WorkflowResult(
            question=question,
            answer=result.get("final_answer", ""),
            sources=self._build_sources(result.get("retrieval_results", [])),
            plan=result.get("plan", {}),
            reflection=result.get("reflection", {}),
            tool_history=result.get("tool_history", []),
            evidence=result.get("retrieval_results", []),
            report_path=result.get("report_path"),
            performance=collector.snapshot(),
        )

    def draw_mermaid(self) -> str:
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as exc:  # pragma: no cover - defensive path
            return f"Unable to render mermaid graph: {exc}"
