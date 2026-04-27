from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

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
from .models import WorkflowResult


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
        builder.add_node("planner", self.planner_node)
        builder.add_node("retrieval", self.retrieval_node)
        builder.add_node("analysis", self.analysis_node)
        builder.add_node("summary", self.summary_node)
        builder.add_node("reflection", self.reflection_node)
        builder.add_node("finalize", self.finalize_node)

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

    def _latest_question(self, state: ResearchState) -> str:
        if state.get("user_question"):
            return state["user_question"]
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    def _heuristic_plan(self, question: str, mode: str, memory_hits: list[dict[str, Any]]) -> dict[str, Any]:
        query_terms = [question]
        if mode == "daily_report":
            query_terms.append(self.config.topic)

        tool_calls = [
            {
                "name": "semantic_search",
                "args": {"query": question, "top_k": self.config.search.semantic_top_k},
            }
        ]
        lower_question = question.lower()
        if any(keyword in lower_question for keyword in ("latest", "today", "new", "recent", "arxiv")):
            tool_calls.insert(
                0,
                {
                    "name": "search_arxiv",
                    "args": {"query": question, "max_results": self.config.search.max_results},
                },
            )
        if mode == "daily_report":
            tool_calls.insert(
                0,
                {
                    "name": "download_and_index_arxiv",
                    "args": {
                        "query": self.config.topic,
                        "max_results": min(3, self.config.search.max_results),
                        "relevance_threshold": max(0.2, self.config.search.relevance_threshold - 0.05),
                    },
                },
            )

        return {
            "objective": question,
            "steps": ["plan", "retrieve", "analyze", "summarize", "reflect"],
            "search_queries": query_terms[: self.config.search.query_rewrite_count],
            "tool_calls": tool_calls,
            "answer_format": "markdown report" if mode == "daily_report" else "grounded answer",
            "memory_summary": [item["text"][:120] for item in memory_hits],
        }

    def _rewrite_queries(self, question: str, plan: dict[str, Any], reflection_focus: str = "") -> list[str]:
        seed_queries = [question] + plan.get("search_queries", [])
        if reflection_focus:
            seed_queries.append(reflection_focus)
        deduped = list(dict.fromkeys(query.strip() for query in seed_queries if query.strip()))

        if not self.config.graph.enable_query_rewrite or not self.llm.is_available:
            return deduped[: self.config.search.query_rewrite_count]

        default_payload = deduped[: self.config.search.query_rewrite_count]
        prompt = f"""
Question: {question}
Plan: {json.dumps(plan, ensure_ascii=False)}
Retry focus: {reflection_focus}

Generate up to {self.config.search.query_rewrite_count} retrieval queries.
They should cover terminology variants, academic phrasing, and concise keyword form.
"""
        payload = self.llm.complete_json(
            system_prompt="You are the Retrieval Agent. Rewrite the user question into strong research queries.",
            user_prompt=prompt,
            default=default_payload,
        )

        if isinstance(payload, dict):
            payload = payload.get("queries", default_payload)
        if not isinstance(payload, list):
            return default_payload
        rewritten = [str(item).strip() for item in payload if str(item).strip()]
        return list(dict.fromkeys(rewritten))[: self.config.search.query_rewrite_count]

    def _extract_pdf_path(self, question: str) -> str | None:
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
                    "name": "semantic_search",
                    "args": {"query": question, "top_k": self.config.search.semantic_top_k},
                }
            ]

        for query in rewritten_queries:
            tool_calls.append(
                {
                    "name": "semantic_search",
                    "args": {"query": query, "top_k": self.config.search.semantic_top_k},
                }
            )

        pdf_path = self._extract_pdf_path(question)
        if pdf_path:
            tool_calls.insert(0, {"name": "parse_pdf", "args": {"file_path": pdf_path}})

        if reflection_focus and state.get("retry_count", 0) > 0:
            tool_calls.insert(
                0,
                {
                    "name": "download_and_index_arxiv",
                    "args": {
                        "query": reflection_focus,
                        "max_results": min(3, self.config.search.max_results),
                        "relevance_threshold": max(0.2, self.config.search.relevance_threshold - 0.05),
                    },
                },
            )

        deduped: list[dict[str, Any]] = []
        seen = set()
        for call in tool_calls:
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

            if tool_name == "semantic_search":
                for item in result:
                    normalized.append(item)
            elif tool_name == "search_arxiv":
                for item in result:
                    normalized.append(
                        {
                            "content": item.get("summary", ""),
                            "source": item.get("source_url", "arxiv"),
                            "title": item.get("title", "arXiv result"),
                            "score": item.get("relevance_score", 0.0),
                            "section": "metadata",
                            "origin": "arxiv_search",
                            "metadata": item,
                        }
                    )
            elif tool_name == "download_and_index_arxiv":
                for item in result:
                    normalized.append(
                        {
                            "content": item.get("summary", ""),
                            "source": item.get("pdf_path", ""),
                            "title": item.get("title", "indexed paper"),
                            "score": item.get("relevance_score", 0.0),
                            "section": "indexed",
                            "origin": "arxiv_download",
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
        normalized.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return normalized[:12]

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
            return {"plan": default_plan, "memory_hits": memory_hits}

        planner_prompt = f"""
Question: {question}
Mode: {state.get('requested_mode', 'qa')}
Topic: {self.config.topic}
Relevant memories: {json.dumps(memory_hits, ensure_ascii=False)}
Available tools: {json.dumps(self.toolbox.describe_tools(), ensure_ascii=False)}

Return JSON with:
- objective
- steps
- search_queries
- tool_calls
- answer_format
"""
        plan = self.llm.complete_json(
            system_prompt="You are the Planner Agent in a multi-agent research workflow.",
            user_prompt=planner_prompt,
            default=default_plan,
        )
        if not isinstance(plan, dict):
            plan = default_plan
        return {"plan": plan, "memory_hits": memory_hits}

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
        analysis = self.llm.complete_json(
            system_prompt="You are the Analysis Agent. Synthesize evidence before writing the final answer.",
            user_prompt=prompt,
            default=self._heuristic_analysis(state),
        )
        if not isinstance(analysis, dict):
            analysis = self._heuristic_analysis(state)
        return {"analysis": analysis}

    def summary_node(self, state: ResearchState) -> dict[str, Any]:
        question = self._latest_question(state)
        analysis = state.get("analysis", {})
        evidence = state.get("retrieval_results", [])

        if not self.llm.is_available:
            answer = self._fallback_answer(state)
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
4. If mode is daily_report, output a structured markdown brief.
"""
            answer = self.llm.complete(
                system_prompt="You are the Summary Agent in a paper-research multi-agent workflow.",
                user_prompt=prompt,
                temperature=0.2,
            )

        report_path = None
        if state.get("requested_mode") == "daily_report":
            report_name = f"LangGraph_Daily_Report_{state.get('thread_id', 'default')}.md"
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
            "score": 0.6 if state.get("retrieval_results") else 0.2,
            "issues": [] if state.get("retrieval_results") else ["Not enough evidence retrieved."],
            "should_retry": False,
            "retry_focus": "",
        }
        if not self.llm.is_available:
            if not state.get("retrieval_results") and current_retry < self.config.graph.max_reflection_rounds:
                fallback["should_retry"] = True
                fallback["retry_focus"] = self._latest_question(state)
            return {
                "reflection": fallback,
                "should_retry": fallback["should_retry"],
                "retry_count": current_retry + (1 if fallback["should_retry"] else 0),
            }

        prompt = f"""
Question: {self._latest_question(state)}
Plan: {json.dumps(state.get('plan', {}), ensure_ascii=False)}
Evidence: {json.dumps(state.get('retrieval_results', [])[:8], ensure_ascii=False)}
Draft answer: {state.get('final_answer', '')}

Evaluate the answer.
Return JSON with:
- passed
- score
- issues
- should_retry
- retry_focus
Only request retry if the answer is weakly grounded or obviously incomplete.
"""
        reflection = self.llm.complete_json(
            system_prompt="You are the Reflection Agent. Critique the answer and decide whether to retry.",
            user_prompt=prompt,
            default=fallback,
        )
        if not isinstance(reflection, dict):
            reflection = fallback

        should_retry = bool(reflection.get("should_retry")) and current_retry < self.config.graph.max_reflection_rounds
        return {
            "reflection": reflection,
            "should_retry": should_retry,
            "retry_count": current_retry + (1 if should_retry else 0),
        }

    def finalize_node(self, state: ResearchState) -> dict[str, Any]:
        final_answer = state.get("final_answer", "")
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
        messages = [HumanMessage(content=question)]
        if thread_id not in self._active_threads:
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

        result = self.graph.invoke(
            state,
            {"configurable": {"thread_id": thread_id}},
        )
        self._active_threads.add(thread_id)

        return WorkflowResult(
            question=question,
            answer=result.get("final_answer", ""),
            sources=self._build_sources(result.get("retrieval_results", [])),
            plan=result.get("plan", {}),
            reflection=result.get("reflection", {}),
            tool_history=result.get("tool_history", []),
            report_path=result.get("report_path"),
        )

    def draw_mermaid(self) -> str:
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as exc:  # pragma: no cover - defensive path
            return f"Unable to render mermaid graph: {exc}"
