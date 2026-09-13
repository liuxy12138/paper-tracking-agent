from __future__ import annotations

import json

from competitive_research_agent.config import MemoryConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from competitive_research_agent.memory import LongTermMemoryStore, ThreadHistoryStore


class FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [
            1.0 if "中文" in lowered or "chinese" in lowered else 0.0,
            1.0 if "竞品" in lowered or "coding assistant" in lowered else 0.0,
            1.0 if "memory dimension probe" in lowered else 0.5,
        ]


class FakeMilvusClient:
    def __init__(self):
        self.rows: list[dict] = []

    def has_collection(self, collection_name: str) -> bool:
        return True

    def describe_collection(self, collection_name: str):
        return {
            "fields": [
                {"name": "memory_id"},
                {"name": "user_id"},
                {"name": "text"},
                {"name": "embedding", "params": {"dim": 3}},
            ]
        }

    def search(self, **kwargs):
        filter_text = kwargs.get("filter", "")
        user_id = filter_text.split('user_id == "', 1)[1].split('"', 1)[0]
        kind = None
        if 'kind == "' in filter_text:
            kind = filter_text.split('kind == "', 1)[1].split('"', 1)[0]

        hits = []
        for row in self.rows:
            if row["user_id"] != user_id or row["status"] != "active":
                continue
            if kind and row["kind"] != kind:
                continue
            query = kwargs["data"][0]
            dot = sum(a * b for a, b in zip(query, row["embedding"]))
            hits.append(
                {
                    "id": row["memory_id"],
                    "distance": dot,
                    "entity": {
                        key: value
                        for key, value in row.items()
                        if key != "embedding"
                    },
                }
            )
        hits.sort(key=lambda item: item["distance"], reverse=True)
        return [hits[: kwargs["limit"]]]

    def insert(self, collection_name: str, data: list[dict]):
        self.rows.extend(data)

    def upsert(self, collection_name: str, data: list[dict]):
        ids = {row["memory_id"] for row in data}
        self.rows = [row for row in self.rows if row["memory_id"] not in ids]
        self.rows.extend(data)

    def delete(self, collection_name: str, ids=None, filter=None):
        if ids:
            self.rows = [row for row in self.rows if row["memory_id"] not in ids]
        elif filter and 'memory_key == "' in filter:
            user_id = filter.split('user_id == "', 1)[1].split('"', 1)[0]
            kind = filter.split('kind == "', 1)[1].split('"', 1)[0]
            memory_key = filter.split('memory_key == "', 1)[1].split('"', 1)[0]
            self.rows = [
                row
                for row in self.rows
                if not (
                    row["user_id"] == user_id
                    and row["kind"] == kind
                    and row["memory_key"] == memory_key
                )
            ]


class FakeRedisPipeline:
    def __init__(self, client):
        self.client = client
        self.calls = []

    def __getattr__(self, name):
        def queue(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self

        return queue

    def execute(self):
        for name, args, kwargs in self.calls:
            getattr(self.client, name)(*args, **kwargs)
        return [True] * len(self.calls)


class FakeRedisClient:
    def __init__(self):
        self.strings = {}
        self.lists = {}
        self.expirations = {}

    def get(self, key):
        return self.strings.get(key)

    def set(self, key, value):
        self.strings[key] = value
        return True

    def lrange(self, key, start, end):
        values = self.lists.get(key, [])
        normalized_end = len(values) if end == -1 else end + 1
        return values[start:normalized_end]

    def rpush(self, key, *values):
        self.lists.setdefault(key, []).extend(values)
        return len(self.lists[key])

    def delete(self, *keys):
        for key in keys:
            self.strings.pop(key, None)
            self.lists.pop(key, None)
            self.expirations.pop(key, None)
        return len(keys)

    def expire(self, key, seconds):
        self.expirations[key] = seconds
        return True

    def pipeline(self, transaction=True):
        return FakeRedisPipeline(self)


def build_store() -> LongTermMemoryStore:
    return LongTermMemoryStore(
        config=MemoryConfig(
            min_similarity=0.2,
            duplicate_similarity=0.95,
            candidate_limit=10,
            enable_llm_compression=False,
        ),
        embeddings=FakeEmbeddings(),
        client=FakeMilvusClient(),
    )


def test_add_search_and_deduplicate_memory():
    store = build_store()

    first_id = store.add(
        "user-1",
        "我喜欢中文回答",
        "preference",
        metadata={"source": "first"},
        importance=0.7,
    )
    duplicate_id = store.add(
        "user-1",
        "请继续使用中文回答",
        "preference",
        metadata={"source": "latest"},
        importance=0.95,
    )
    store.add("user-2", "我喜欢中文回答", "preference", importance=0.95)

    results = store.search("user-1", "请继续使用中文", limit=4)

    assert first_id == duplicate_id
    assert len(store.client.rows) == 2
    assert len(results) == 1
    assert results[0]["kind"] == "preference"
    assert results[0]["text"] == "请继续使用中文回答"
    assert results[0]["metadata"] == {"source": "latest"}
    assert results[0]["importance"] == 0.95


def test_add_inserts_new_memory_when_similarity_is_low():
    store = build_store()

    first_id = store.add("user-1", "我喜欢中文回答", "preference")
    second_id = store.add("user-1", "我研究竞品导航", "preference")

    assert first_id != second_id
    assert len(store.client.rows) == 2


def test_remember_interaction_classifies_preference_and_summary():
    store = build_store()

    store.remember_interaction(
        user_id="user-1",
        question="请记住，我喜欢中文回答",
        answer="好的，后续会使用中文。",
        topic="competitive research",
    )

    rows = store.client.rows
    assert {row["kind"] for row in rows} == {"preference", "interaction_summary"}
    summary = next(row for row in rows if row["kind"] == "interaction_summary")
    assert json.loads(summary["metadata_json"])["compression"] == "deterministic"


def test_delete_memory():
    store = build_store()
    memory_id = store.add("user-1", "我研究低轨竞品定位", "research_interest")

    store.delete(memory_id)

    assert store.search("user-1", "低轨竞品定位") == []


def test_new_structured_preference_replaces_old_preference():
    store = build_store()
    first = store._extract_explicit_memories("我喜欢中文回答")[0]
    second = store._extract_explicit_memories("我喜欢英文回答")[0]

    store.add("user-1", first.text, first.kind, memory_key=first.memory_key)
    store.add("user-1", second.text, second.kind, memory_key=second.memory_key)

    preferences = [row for row in store.client.rows if row["kind"] == "preference"]
    assert len(preferences) == 1
    assert preferences[0]["text"] == "我喜欢英文回答"


def test_migrate_legacy_json(tmp_path):
    store = build_store()
    legacy_path = tmp_path / "long_term_memory.json"
    legacy_path.write_text(
        json.dumps(
            [
                {
                    "memory_id": "legacy-id",
                    "user_id": "user-1",
                    "text": "我研究低轨竞品定位",
                    "kind": "research_interest",
                    "metadata": {"topic": "AI coding"},
                    "embedding": [0.0, 0.0, 0.0],
                    "created_at": "2026-01-01T00:00:00",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = store.migrate_legacy_json(str(legacy_path))

    assert result == {"processed": 1, "failed": 0}
    assert store.client.rows[0]["metadata_json"].find("migrated_from") > 0


def test_thread_history_compresses_messages_outside_recent_window():
    client = FakeRedisClient()
    store = ThreadHistoryStore(
        config=MemoryConfig(enable_thread_compression=False, thread_summary_max_chars=500),
        client=client,
    )
    messages = [
        HumanMessage(content="第一轮问题"),
        AIMessage(content="第一轮回答"),
        HumanMessage(content="第二轮问题"),
        AIMessage(content="第二轮回答"),
    ]

    store.save_messages("thread-1", messages, limit=2)
    loaded = store.load_messages("thread-1", limit=2)

    assert isinstance(loaded[0], SystemMessage)
    assert "第一轮问题" in str(loaded[0].content)
    assert [str(message.content) for message in loaded[1:]] == ["第二轮问题", "第二轮回答"]
    assert client.expirations["industry_research_agent:thread:thread-1:messages"] == 604800


def test_thread_history_delete_removes_redis_keys():
    client = FakeRedisClient()
    store = ThreadHistoryStore(config=MemoryConfig(), client=client)
    store.save_messages("thread-1", [HumanMessage(content="问题")], limit=2)

    store.delete_thread("thread-1")

    assert client.lists == {}
    assert client.strings == {}
