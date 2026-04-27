from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _message_to_dict(message: BaseMessage) -> dict[str, str]:
    role = "assistant"
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    return {"role": role, "content": str(message.content)}


def _dict_to_message(item: dict[str, str]) -> BaseMessage:
    if item.get("role") == "user":
        return HumanMessage(content=item.get("content", ""))
    return AIMessage(content=item.get("content", ""))


class ThreadHistoryStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._threads: dict[str, list[dict[str, str]]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            self._threads = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self._threads, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_messages(self, thread_id: str, limit: int = 6) -> list[BaseMessage]:
        items = self._threads.get(thread_id, [])[-limit:]
        return [_dict_to_message(item) for item in items]

    def save_messages(self, thread_id: str, messages: list[BaseMessage], limit: int = 10) -> None:
        serialized = [_message_to_dict(message) for message in messages[-limit:]]
        self._threads[thread_id] = serialized
        self._save()


@dataclass
class LongTermMemoryItem:
    memory_id: str
    user_id: str
    text: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LongTermMemoryStore:
    def __init__(self, path: str, embeddings: Any):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self._items: list[LongTermMemoryItem] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._items = [LongTermMemoryItem(**item) for item in raw]

    def _save(self) -> None:
        self.path.write_text(
            json.dumps([item.to_dict() for item in self._items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, user_id: str, text: str, kind: str, metadata: dict[str, Any] | None = None) -> None:
        payload = text.strip()
        if not payload:
            return
        embedding = self.embeddings.embed_query(payload) if self.embeddings else []
        self._items.append(
            LongTermMemoryItem(
                memory_id=str(uuid.uuid4()),
                user_id=user_id,
                text=payload[:1200],
                kind=kind,
                metadata=metadata or {},
                embedding=embedding,
            )
        )
        self._save()

    def search(self, user_id: str, query: str, limit: int = 4) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        query_embedding = self.embeddings.embed_query(query) if self.embeddings else []
        candidates: list[tuple[float, LongTermMemoryItem]] = []
        for item in self._items:
            if item.user_id != user_id:
                continue
            score = _cosine_similarity(query_embedding, item.embedding) if query_embedding and item.embedding else 0.0
            keyword_bonus = 0.1 if any(token in item.text.lower() for token in query.lower().split()) else 0.0
            candidates.append((score + keyword_bonus, item))

        ranked = sorted(candidates, key=lambda entry: entry[0], reverse=True)[:limit]
        return [
            {
                "memory_id": item.memory_id,
                "text": item.text,
                "kind": item.kind,
                "score": round(score, 4),
                "metadata": item.metadata,
            }
            for score, item in ranked
            if score > 0
        ]

    def remember_interaction(self, user_id: str, question: str, answer: str, topic: str) -> None:
        snippets = []
        lower_question = question.lower()
        if any(flag in lower_question for flag in ("remember", "preference", "偏好", "记住", "我叫")):
            snippets.append(question)
        snippets.append(f"Topic: {topic}; Question: {question[:240]}; Answer summary: {answer[:480]}")
        for text in snippets:
            self.add(user_id=user_id, text=text, kind="interaction", metadata={"topic": topic})
