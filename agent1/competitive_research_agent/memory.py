from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .config import MemoryConfig


def _message_to_dict(message: BaseMessage) -> dict[str, str]:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, SystemMessage):
        role = "system"
    else:
        role = "assistant"
    return {"role": role, "content": str(message.content)}


def _dict_to_message(item: dict[str, str]) -> BaseMessage:
    if item.get("role") == "user":
        return HumanMessage(content=item.get("content", ""))
    if item.get("role") == "system":
        return SystemMessage(content=item.get("content", ""))
    return AIMessage(content=item.get("content", ""))


def _escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _now_ms() -> int:
    return int(time.time() * 1000)


class ThreadHistoryStore:
    def __init__(self, config: MemoryConfig | None = None, llm: Any = None, client: Any = None):
        self.config = config or MemoryConfig()
        self.llm = llm
        self.client = client or self._build_client()
        self._summary_offsets: dict[str, int] = {}

    def _build_client(self):
        try:
            from redis import Redis
        except ImportError as exc:
            raise RuntimeError(
                "Redis short-term memory requires redis-py. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from exc

        try:
            client = Redis.from_url(
                self.config.redis_url,
                decode_responses=True,
                socket_timeout=self.config.redis_socket_timeout_seconds,
                socket_connect_timeout=self.config.redis_socket_timeout_seconds,
            )
            client.ping()
            return client
        except Exception as exc:
            raise RuntimeError(
                f"Unable to connect to Redis at {self.config.redis_url}. "
                "Start Redis or update memory.redis_url."
            ) from exc

    def _keys(self, thread_id: str) -> tuple[str, str]:
        prefix = f"{self.config.redis_key_prefix}:{thread_id}"
        return f"{prefix}:messages", f"{prefix}:summary"

    def load_messages(self, thread_id: str, limit: int = 6) -> list[BaseMessage]:
        messages_key, summary_key = self._keys(thread_id)
        self._summary_offsets[thread_id] = 0
        serialized = self.client.lrange(messages_key, -limit, -1)
        messages = [_dict_to_message(json.loads(item)) for item in serialized]
        summary = self.client.get(summary_key) or ""
        if summary:
            messages.insert(0, SystemMessage(content=f"Earlier conversation summary: {summary}"))
        self._refresh_ttl(messages_key, summary_key)
        return messages

    def save_messages(self, thread_id: str, messages: list[BaseMessage], limit: int = 10) -> None:
        messages_key, summary_key = self._keys(thread_id)
        prior_summary = self.client.get(summary_key) or ""
        conversational = [
            message
            for message in messages
            if not isinstance(message, SystemMessage)
            or not str(message.content).startswith("Earlier conversation summary:")
        ]
        overflow_count = max(0, len(conversational) - limit)
        summary_offset = min(self._summary_offsets.get(thread_id, 0), overflow_count)
        overflow = conversational[summary_offset:overflow_count]
        recent = conversational[-limit:]
        summary = self._compress_history(prior_summary, overflow) if overflow else prior_summary
        self._summary_offsets[thread_id] = overflow_count
        serialized = [
            json.dumps(_message_to_dict(message), ensure_ascii=False)
            for message in recent
        ]
        pipeline = self.client.pipeline(transaction=True)
        pipeline.delete(messages_key)
        if serialized:
            pipeline.rpush(messages_key, *serialized)
        if summary:
            pipeline.set(summary_key, summary)
        else:
            pipeline.delete(summary_key)
        if self.config.redis_ttl_seconds > 0:
            pipeline.expire(messages_key, self.config.redis_ttl_seconds)
            pipeline.expire(summary_key, self.config.redis_ttl_seconds)
        pipeline.execute()

    def delete_thread(self, thread_id: str) -> None:
        messages_key, summary_key = self._keys(thread_id)
        self.client.delete(messages_key, summary_key)
        self._summary_offsets.pop(thread_id, None)

    def _refresh_ttl(self, messages_key: str, summary_key: str) -> None:
        if self.config.redis_ttl_seconds <= 0:
            return
        pipeline = self.client.pipeline(transaction=False)
        pipeline.expire(messages_key, self.config.redis_ttl_seconds)
        pipeline.expire(summary_key, self.config.redis_ttl_seconds)
        pipeline.execute()

    def _compress_history(self, prior_summary: str, messages: list[BaseMessage]) -> str:
        transcript = "\n".join(
            f"{_message_to_dict(message)['role']}: {str(message.content)}"
            for message in messages
        )
        fallback = re.sub(r"\s+", " ", f"{prior_summary} {transcript}").strip()
        fallback = fallback[-self.config.thread_summary_max_chars :]
        if (
            not self.config.enable_thread_compression
            or not self.llm
            or not self.llm.is_available
        ):
            return fallback

        payload = self.llm.complete_json(
            system_prompt=(
                "Compress older conversation history into durable context. "
                "Preserve user preferences, decisions, unresolved questions, and important facts. "
                "Remove repetition and transient wording. "
                'Return an object shaped as {"summary": "..."}.'
            ),
            user_prompt=f"Previous summary: {prior_summary}\nOlder messages:\n{transcript}",
            default={},
        )
        if not isinstance(payload, dict) or not str(payload.get("summary", "")).strip():
            return fallback
        return str(payload["summary"])[: self.config.thread_summary_max_chars]


@dataclass
class MemoryCandidate:
    text: str
    kind: str
    importance: float = 0.5
    memory_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LongTermMemoryStore:
    KIND_WEIGHTS = {
        "preference": 1.0,
        "profile": 0.95,
        "research_interest": 0.9,
        "reflection": 0.85,
        "interaction_summary": 0.75,
    }

    def __init__(
        self,
        config: MemoryConfig,
        embeddings: Any,
        llm: Any = None,
        client: Any = None,
    ):
        self.config = config
        self.embeddings = embeddings
        self.llm = llm
        self.client = client or self._build_client()
        self.dimension = len(self.embeddings.embed_query("memory dimension probe"))
        self._ensure_collection()

    def _build_client(self):
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError(
                "Milvus long-term memory requires pymilvus. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from exc

        kwargs: dict[str, Any] = {
            "uri": self.config.milvus_uri,
            "timeout": self.config.milvus_timeout_seconds,
        }
        if self.config.milvus_token:
            kwargs["token"] = self.config.milvus_token
        try:
            return MilvusClient(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to connect to Milvus at {self.config.milvus_uri}. "
                "Start Milvus or update memory.milvus_uri."
            ) from exc

    def _ensure_collection(self) -> None:
        if self.client.has_collection(collection_name=self.config.collection_name):
            self._validate_collection()
            return

        try:
            from pymilvus import DataType
        except ImportError as exc:
            raise RuntimeError("pymilvus is required to create the memory collection.") from exc

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("memory_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("user_id", DataType.VARCHAR, max_length=256)
        schema.add_field("text", DataType.VARCHAR, max_length=4096)
        schema.add_field("kind", DataType.VARCHAR, max_length=64)
        schema.add_field("memory_key", DataType.VARCHAR, max_length=128)
        schema.add_field("metadata_json", DataType.VARCHAR, max_length=8192)
        schema.add_field("importance", DataType.FLOAT)
        schema.add_field("created_at", DataType.INT64)
        schema.add_field("updated_at", DataType.INT64)
        schema.add_field("last_accessed_at", DataType.INT64)
        schema.add_field("access_count", DataType.INT64)
        schema.add_field("status", DataType.VARCHAR, max_length=32)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dimension)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=self.config.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _validate_collection(self) -> None:
        description = self.client.describe_collection(collection_name=self.config.collection_name)
        fields = {str(field.get("name", "")): field for field in description.get("fields", [])}
        required = {"memory_id", "user_id", "text", "embedding"}
        if not required <= set(fields) or "document_id" in fields:
            raise RuntimeError(
                f"Milvus collection {self.config.collection_name} is not a user-memory collection."
            )
        dimension = int(fields["embedding"].get("params", {}).get("dim", self.dimension))
        if dimension != self.dimension:
            raise RuntimeError(
                f"Memory collection dimension mismatch: stored={dimension}, embedding_model={self.dimension}."
            )

    def _filter(self, user_id: str, kind: str | None = None) -> str:
        clauses = [
            f'user_id == "{_escape_filter_value(user_id)}"',
            'status == "active"',
        ]
        if kind:
            clauses.append(f'kind == "{_escape_filter_value(kind)}"')
        return " and ".join(clauses)

    def _search_raw(
        self,
        *,
        user_id: str,
        embedding: list[float],
        limit: int,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        results = self.client.search(
            collection_name=self.config.collection_name,
            data=[embedding],
            filter=self._filter(user_id, kind),
            limit=limit,
            output_fields=[
                "memory_id",
                "text",
                "kind",
                "memory_key",
                "metadata_json",
                "importance",
                "created_at",
                "updated_at",
                "last_accessed_at",
                "access_count",
                "status",
            ],
            search_params={"metric_type": "COSINE", "params": {}},
        )
        return results[0] if results else []

    def add(
        self,
        user_id: str,
        text: str,
        kind: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        memory_key: str = "",
    ) -> str | None:
        payload = re.sub(r"\s+", " ", text).strip()
        if not payload:
            return None

        embedding = self.embeddings.embed_query(payload)
        similar_memories = self._search_raw(
            user_id=user_id,
            embedding=embedding,
            limit=1,
            kind=kind,
        )
        now = _now_ms()
        if (
            similar_memories
            and float(similar_memories[0].get("distance", 0.0)) >= self.config.duplicate_similarity
        ):
            hit = similar_memories[0]
            entity = hit.get("entity", {})
            memory_id = str(hit.get("id") or entity.get("memory_id", ""))
            self.client.upsert(
                collection_name=self.config.collection_name,
                data=[
                    {
                        "memory_id": memory_id,
                        "user_id": user_id,
                        "text": payload[:4096],
                        "kind": kind,
                        "memory_key": (memory_key or str(entity.get("memory_key", "")))[:128],
                        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False)[:8192],
                        "importance": max(0.0, min(1.0, float(importance))),
                        "created_at": int(entity.get("created_at") or now),
                        "updated_at": now,
                        "last_accessed_at": int(entity.get("last_accessed_at") or now),
                        "access_count": int(entity.get("access_count") or 0),
                        "status": "active",
                        "embedding": embedding,
                    }
                ],
            )
            return memory_id

        if memory_key:
            self.client.delete(
                collection_name=self.config.collection_name,
                filter=(
                    f'{self._filter(user_id, kind)} and '
                    f'memory_key == "{_escape_filter_value(memory_key)}"'
                ),
            )

        memory_id = str(uuid.uuid4())
        self.client.insert(
            collection_name=self.config.collection_name,
            data=[
                {
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "text": payload[:4096],
                    "kind": kind,
                    "memory_key": memory_key[:128],
                    "metadata_json": json.dumps(metadata or {}, ensure_ascii=False)[:8192],
                    "importance": max(0.0, min(1.0, float(importance))),
                    "created_at": now,
                    "updated_at": now,
                    "last_accessed_at": now,
                    "access_count": 0,
                    "status": "active",
                    "embedding": embedding,
                }
            ],
        )
        return memory_id

    def search(self, user_id: str, query: str, limit: int = 4) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        query_embedding = self.embeddings.embed_query(query)
        hits = self._search_raw(
            user_id=user_id,
            embedding=query_embedding,
            limit=max(limit, self.config.candidate_limit),
        )
        now = _now_ms()
        ranked: list[dict[str, Any]] = []
        for hit in hits:
            entity = hit.get("entity", {})
            semantic_score = float(hit.get("distance", 0.0))
            if semantic_score < self.config.min_similarity:
                continue

            created_at = int(entity.get("created_at") or now)
            age_days = max(0.0, (now - created_at) / 86_400_000)
            recency_score = math.exp(-age_days / 90)
            importance = float(entity.get("importance") or 0.5)
            kind = str(entity.get("kind") or "interaction_summary")
            kind_weight = self.KIND_WEIGHTS.get(kind, 0.7)
            final_score = (
                semantic_score * 0.65
                + importance * 0.15
                + recency_score * 0.10
                + kind_weight * 0.10
            )
            ranked.append(
                {
                    "memory_id": str(hit.get("id") or entity.get("memory_id", "")),
                    "text": str(entity.get("text", "")),
                    "kind": kind,
                    "memory_key": str(entity.get("memory_key", "")),
                    "score": round(final_score, 4),
                    "semantic_score": round(semantic_score, 4),
                    "importance": round(importance, 4),
                    "metadata": json.loads(entity.get("metadata_json") or "{}"),
                    "created_at": created_at,
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]

    def delete(self, memory_id: str) -> None:
        self.client.delete(
            collection_name=self.config.collection_name,
            ids=[memory_id],
        )

    def delete_user_memories(self, user_id: str, kind: str | None = None) -> None:
        self.client.delete(
            collection_name=self.config.collection_name,
            filter=self._filter(user_id, kind),
        )

    def migrate_legacy_json(self, path: str) -> dict[str, int]:
        source = Path(path)
        raw = json.loads(source.read_text(encoding="utf-8"))
        processed = 0
        failed = 0
        for item in raw:
            try:
                kind = str(item.get("kind") or "interaction_summary")
                self.add(
                    user_id=str(item["user_id"]),
                    text=str(item["text"]),
                    kind=kind,
                    metadata={
                        **dict(item.get("metadata") or {}),
                        "migrated_from": str(source),
                    },
                    importance=0.8 if kind in {"preference", "profile"} else 0.5,
                )
                processed += 1
            except Exception:
                failed += 1
        return {"processed": processed, "failed": failed}

    def _extract_explicit_memories(self, question: str) -> list[MemoryCandidate]:
        normalized = re.sub(r"\s+", " ", question).strip()
        lower = normalized.casefold()
        candidates: list[MemoryCandidate] = []

        preference_markers = (
            "我喜欢",
            "我偏好",
            "我希望",
            "请用",
            "以后用",
            "remember",
            "preference",
            "i prefer",
        )
        profile_markers = ("我叫", "我的名字", "我是", "my name is", "i am ")
        interest_markers = ("我关注", "我研究", "研究方向", "感兴趣", "i study", "interested in")

        if any(marker in lower for marker in preference_markers):
            memory_key = ""
            if any(marker in lower for marker in ("中文", "英文", "chinese", "english", "language")):
                memory_key = "output_language"
            elif any(marker in lower for marker in ("简洁", "详细", "concise", "detailed")):
                memory_key = "response_detail"
            elif any(marker in lower for marker in ("markdown", "格式", "format")):
                memory_key = "output_format"
            candidates.append(MemoryCandidate(normalized, "preference", 0.95, memory_key=memory_key))
        elif any(marker in lower for marker in profile_markers):
            candidates.append(MemoryCandidate(normalized, "profile", 0.95))
        elif any(marker in lower for marker in interest_markers):
            candidates.append(MemoryCandidate(normalized, "research_interest", 0.85))
        return candidates

    def _compress_interaction(self, question: str, answer: str, topic: str) -> MemoryCandidate:
        fallback = MemoryCandidate(
            text=f"Topic: {topic}; User asked: {question[:240]}; Answered: {answer[:320]}",
            kind="interaction_summary",
            importance=0.45,
            metadata={"topic": topic, "compression": "deterministic"},
        )
        if not self.config.enable_llm_compression or not self.llm or not self.llm.is_available:
            return fallback

        payload = self.llm.complete_json(
            system_prompt=(
                "Compress the interaction into one durable user-memory sentence. "
                "Keep reusable preferences, facts, research interests, decisions, and unresolved needs. "
                "Do not preserve transient wording or unsupported claims. "
                'Return an object shaped as {"summary": "...", "kind": "interaction_summary", '
                '"importance": 0.0}.'
            ),
            user_prompt=f"Topic: {topic}\nQuestion: {question}\nAnswer: {answer}",
            default={},
        )
        if not isinstance(payload, dict) or not str(payload.get("summary", "")).strip():
            return fallback
        return MemoryCandidate(
            text=str(payload["summary"])[: self.config.interaction_summary_max_chars],
            kind=str(payload.get("kind") or "interaction_summary"),
            importance=max(0.0, min(1.0, float(payload.get("importance", 0.6)))),
            metadata={"topic": topic, "compression": "llm"},
        )

    def remember_interaction(self, user_id: str, question: str, answer: str, topic: str) -> None:
        candidates = self._extract_explicit_memories(question)
        candidates.append(self._compress_interaction(question, answer, topic))
        for candidate in candidates:
            self.add(
                user_id=user_id,
                text=candidate.text,
                kind=candidate.kind,
                metadata={**candidate.metadata, "topic": topic},
                importance=candidate.importance,
                memory_key=candidate.memory_key,
            )
