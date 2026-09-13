from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .config import DatabaseConfig
from .models import ResearchDocumentRecord


class JsonResearchDocumentStore:
    def __init__(self, metadata_path: str):
        self.metadata_path = Path(metadata_path)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, ResearchDocumentRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.metadata_path.exists():
            return

        raw_data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        for item in raw_data:
            record = ResearchDocumentRecord(**item)
            self._records[record.document_id] = record

    def save(self) -> None:
        data = [asdict(record) for record in self._records.values()]
        data.sort(key=lambda item: item.get("published", ""), reverse=True)
        self.metadata_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, document_id: str) -> ResearchDocumentRecord | None:
        return self._records.get(document_id)

    def upsert(self, record: ResearchDocumentRecord) -> None:
        self._records[record.document_id] = record
        self.save()

    def all_records(self) -> list[ResearchDocumentRecord]:
        return list(self._records.values())


class MySQLConnectionFactory:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._connect_kwargs = self._parse_url(config.url)

    def _parse_url(self, url: str) -> dict[str, Any]:
        parsed = urlparse(url)
        if parsed.scheme not in {"mysql", "mysql+pymysql"}:
            raise ValueError("database.url must use mysql:// or mysql+pymysql://")

        query = parse_qs(parsed.query)
        return {
            "host": parsed.hostname or "127.0.0.1",
            "port": parsed.port or 3306,
            "user": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
            "database": parsed.path.lstrip("/"),
            "charset": query.get("charset", ["utf8mb4"])[0],
            "autocommit": True,
        }

    def connect(self):
        try:
            import pymysql
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "MySQL persistence is enabled, but PyMySQL is not installed. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from exc

        return pymysql.connect(**self._connect_kwargs)


class MySQLSchema:
    def __init__(self, connection_factory: MySQLConnectionFactory):
        self.connection_factory = connection_factory

    def ensure(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS research_documents (
                document_id VARCHAR(255) PRIMARY KEY,
                title TEXT NOT NULL,
                file_path TEXT NOT NULL,
                relevance_score DOUBLE NOT NULL DEFAULT 0,
                summary LONGTEXT,
                published VARCHAR(64),
                source_url TEXT,
                indexed BOOLEAN NOT NULL DEFAULT FALSE,
                added_at VARCHAR(64),
                notes_json LONGTEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS qa_runs (
                run_id VARCHAR(64) PRIMARY KEY,
                thread_id VARCHAR(255),
                user_id VARCHAR(255),
                mode VARCHAR(64),
                question LONGTEXT NOT NULL,
                answer LONGTEXT,
                sources_json LONGTEXT,
                plan_json LONGTEXT,
                reflection_json LONGTEXT,
                report_path TEXT,
                created_at VARCHAR(64) NOT NULL
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS tool_calls (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                run_id VARCHAR(64) NOT NULL,
                tool_name VARCHAR(255),
                args_json LONGTEXT,
                status VARCHAR(64),
                result_preview LONGTEXT,
                elapsed_ms INT,
                created_at VARCHAR(64) NOT NULL,
                INDEX idx_tool_calls_run_id (run_id)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS retrieval_evidence (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                run_id VARCHAR(64) NOT NULL,
                title TEXT,
                source TEXT,
                section VARCHAR(255),
                origin VARCHAR(255),
                score DOUBLE,
                content_preview LONGTEXT,
                metadata_json LONGTEXT,
                created_at VARCHAR(64) NOT NULL,
                INDEX idx_retrieval_evidence_run_id (run_id)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                eval_run_id VARCHAR(64) PRIMARY KEY,
                document_count INT,
                question_count INT,
                ingest_documents BOOLEAN,
                averages_json LONGTEXT,
                error_count INT,
                created_at VARCHAR(64) NOT NULL
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS eval_items (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                eval_run_id VARCHAR(64) NOT NULL,
                question_id VARCHAR(255),
                question LONGTEXT,
                category VARCHAR(255),
                expected_documents_json LONGTEXT,
                scores_json LONGTEXT,
                result_json LONGTEXT,
                error LONGTEXT,
                created_at VARCHAR(64) NOT NULL,
                INDEX idx_eval_items_run_id (eval_run_id)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
        ]
        with self.connection_factory.connect() as conn:
            with conn.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


class MySQLResearchDocumentStore:
    def __init__(self, connection_factory: MySQLConnectionFactory):
        self.connection_factory = connection_factory

    def get(self, document_id: str) -> ResearchDocumentRecord | None:
        with self.connection_factory.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT document_id, title, file_path, relevance_score, summary, published,
                           source_url, indexed, added_at, notes_json
                    FROM research_documents WHERE document_id = %s
                    """,
                    (document_id,),
                )
                row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def upsert(self, record: ResearchDocumentRecord) -> None:
        with self.connection_factory.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO research_documents (
                        document_id, title, file_path, relevance_score, summary, published,
                        source_url, indexed, added_at, notes_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        title = VALUES(title),
                        file_path = VALUES(file_path),
                        relevance_score = VALUES(relevance_score),
                        summary = VALUES(summary),
                        published = VALUES(published),
                        source_url = VALUES(source_url),
                        indexed = VALUES(indexed),
                        added_at = VALUES(added_at),
                        notes_json = VALUES(notes_json)
                    """,
                    (
                        record.document_id,
                        record.title,
                        record.file_path,
                        record.relevance_score,
                        record.summary,
                        record.published,
                        record.source_url,
                        record.indexed,
                        record.added_at,
                        _json_dumps(record.notes),
                    ),
                )

    def all_records(self) -> list[ResearchDocumentRecord]:
        with self.connection_factory.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT document_id, title, file_path, relevance_score, summary, published,
                           source_url, indexed, added_at, notes_json
                    FROM research_documents ORDER BY published DESC
                    """
                )
                rows = cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: tuple[Any, ...]) -> ResearchDocumentRecord:
        notes = {}
        if row[9]:
            notes = json.loads(row[9])
        return ResearchDocumentRecord(
            document_id=row[0],
            title=row[1],
            file_path=row[2],
            relevance_score=float(row[3] or 0),
            summary=row[4] or "",
            published=row[5] or "",
            source_url=row[6] or "",
            indexed=bool(row[7]),
            added_at=row[8] or "",
            notes=notes,
        )


class MySQLTraceStore:
    def __init__(self, connection_factory: MySQLConnectionFactory):
        self.connection_factory = connection_factory

    def save_workflow_result(
        self,
        *,
        run_id: str,
        thread_id: str,
        user_id: str,
        mode: str,
        result: dict[str, Any],
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connection_factory.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO qa_runs (
                        run_id, thread_id, user_id, mode, question, answer, sources_json,
                        plan_json, reflection_json, report_path, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        thread_id,
                        user_id,
                        mode,
                        result.get("question", ""),
                        result.get("answer", ""),
                        _json_dumps(result.get("sources", [])),
                        _json_dumps(result.get("plan", {})),
                        _json_dumps(result.get("reflection", {})),
                        result.get("report_path"),
                        now,
                    ),
                )
                for item in result.get("tool_history", []):
                    cursor.execute(
                        """
                        INSERT INTO tool_calls (
                            run_id, tool_name, args_json, status, result_preview,
                            elapsed_ms, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            run_id,
                            item.get("name", ""),
                            _json_dumps(item.get("args", {})),
                            item.get("status", ""),
                            item.get("result_preview", ""),
                            int(item.get("elapsed_ms") or 0),
                            now,
                        ),
                    )
                for item in result.get("evidence", []):
                    cursor.execute(
                        """
                        INSERT INTO retrieval_evidence (
                            run_id, title, source, section, origin, score,
                            content_preview, metadata_json, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            run_id,
                            item.get("title", ""),
                            item.get("source", ""),
                            item.get("section", ""),
                            item.get("origin", ""),
                            float(item.get("score") or 0),
                            str(item.get("content", ""))[:1200],
                            _json_dumps(item.get("metadata", {})),
                            now,
                        ),
                    )

    def save_eval_run(self, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        eval_run_id = summary["run_id"]
        with self.connection_factory.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO eval_runs (
                        eval_run_id, document_count, question_count, ingest_documents,
                        averages_json, error_count, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        document_count = VALUES(document_count),
                        question_count = VALUES(question_count),
                        ingest_documents = VALUES(ingest_documents),
                        averages_json = VALUES(averages_json),
                        error_count = VALUES(error_count)
                    """,
                    (
                        eval_run_id,
                        int(summary.get("document_count") or 0),
                        int(summary.get("question_count") or 0),
                        bool(summary.get("ingest_documents")),
                        _json_dumps(summary.get("averages", {})),
                        int(summary.get("error_count") or 0),
                        now,
                    ),
                )
                for row in rows:
                    cursor.execute(
                        """
                        INSERT INTO eval_items (
                            eval_run_id, question_id, question, category,
                            expected_documents_json, scores_json, result_json, error, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            eval_run_id,
                            row.get("id", ""),
                            row.get("question", ""),
                            row.get("category", ""),
                            _json_dumps(row.get("expected_documents", [])),
                            _json_dumps(row.get("scores", {})),
                            _json_dumps(row.get("result", {})),
                            row.get("error", ""),
                            now,
                        ),
                    )

    def list_workflow_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connection_factory.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        q.run_id, q.thread_id, q.user_id, q.mode, q.question,
                        q.answer, q.created_at,
                        COUNT(DISTINCT t.id) AS tool_count,
                        COUNT(DISTINCT e.id) AS evidence_count
                    FROM qa_runs q
                    LEFT JOIN tool_calls t ON t.run_id = q.run_id
                    LEFT JOIN retrieval_evidence e ON e.run_id = q.run_id
                    GROUP BY q.run_id, q.thread_id, q.user_id, q.mode, q.question, q.answer, q.created_at
                    ORDER BY q.created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
        return [
            {
                "run_id": row[0],
                "thread_id": row[1],
                "user_id": row[2],
                "mode": row[3],
                "question": row[4],
                "answer_preview": str(row[5] or "")[:280],
                "created_at": row[6],
                "tool_count": int(row[7] or 0),
                "evidence_count": int(row[8] or 0),
            }
            for row in rows
        ]

    def get_workflow_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connection_factory.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT run_id, thread_id, user_id, mode, question, answer, sources_json,
                           plan_json, reflection_json, report_path, created_at
                    FROM qa_runs WHERE run_id = %s
                    """,
                    (run_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                cursor.execute(
                    """
                    SELECT tool_name, args_json, status, result_preview, elapsed_ms, created_at
                    FROM tool_calls WHERE run_id = %s ORDER BY id ASC
                    """,
                    (run_id,),
                )
                tool_rows = cursor.fetchall()

                cursor.execute(
                    """
                    SELECT title, source, section, origin, score, content_preview, metadata_json, created_at
                    FROM retrieval_evidence WHERE run_id = %s ORDER BY score DESC, id ASC
                    """,
                    (run_id,),
                )
                evidence_rows = cursor.fetchall()

        return {
            "run_id": row[0],
            "thread_id": row[1],
            "user_id": row[2],
            "mode": row[3],
            "question": row[4],
            "answer": row[5],
            "sources": json.loads(row[6] or "[]"),
            "plan": json.loads(row[7] or "{}"),
            "reflection": json.loads(row[8] or "{}"),
            "report_path": row[9],
            "created_at": row[10],
            "tool_history": [
                {
                    "name": item[0],
                    "args": json.loads(item[1] or "{}"),
                    "status": item[2],
                    "result_preview": item[3],
                    "elapsed_ms": int(item[4] or 0),
                    "created_at": item[5],
                }
                for item in tool_rows
            ],
            "evidence": [
                {
                    "title": item[0],
                    "source": item[1],
                    "section": item[2],
                    "origin": item[3],
                    "score": float(item[4] or 0),
                    "content": item[5],
                    "metadata": json.loads(item[6] or "{}"),
                    "created_at": item[7],
                }
                for item in evidence_rows
            ],
        }

    def list_eval_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connection_factory.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        r.eval_run_id, r.document_count, r.question_count, r.ingest_documents,
                        r.averages_json, r.error_count, r.created_at,
                        COUNT(i.id) AS item_count
                    FROM eval_runs r
                    LEFT JOIN eval_items i ON i.eval_run_id = r.eval_run_id
                    GROUP BY r.eval_run_id, r.document_count, r.question_count, r.ingest_documents,
                             r.averages_json, r.error_count, r.created_at
                    ORDER BY r.created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
        return [
            {
                "eval_run_id": row[0],
                "document_count": int(row[1] or 0),
                "question_count": int(row[2] or 0),
                "ingest_documents": bool(row[3]),
                "averages": json.loads(row[4] or "{}"),
                "error_count": int(row[5] or 0),
                "created_at": row[6],
                "item_count": int(row[7] or 0),
            }
            for row in rows
        ]


def build_mysql_stores(config: DatabaseConfig) -> tuple[MySQLResearchDocumentStore, MySQLTraceStore]:
    connection_factory = MySQLConnectionFactory(config)
    if config.init_schema:
        MySQLSchema(connection_factory).ensure()
    return MySQLResearchDocumentStore(connection_factory), MySQLTraceStore(connection_factory)


def create_document_store(database_config: DatabaseConfig, metadata_path: str):
    if database_config.enabled:
        document_store, _ = build_mysql_stores(database_config)
        return document_store
    return JsonResearchDocumentStore(metadata_path)


def create_trace_store(database_config: DatabaseConfig):
    if not database_config.enabled:
        return None
    _, trace_store = build_mysql_stores(database_config)
    return trace_store


ResearchDocumentStore = JsonResearchDocumentStore
