from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import PaperRecord


class PaperIndexStore:
    def __init__(self, metadata_path: str):
        self.metadata_path = Path(metadata_path)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, PaperRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.metadata_path.exists():
            return

        raw_data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        for item in raw_data:
            record = PaperRecord(**item)
            self._records[record.paper_id] = record

    def save(self) -> None:
        data = [asdict(record) for record in self._records.values()]
        data.sort(key=lambda item: item.get("published", ""), reverse=True)
        self.metadata_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, paper_id: str) -> PaperRecord | None:
        return self._records.get(paper_id)

    def upsert(self, record: PaperRecord) -> None:
        self._records[record.paper_id] = record
        self.save()

    def all_records(self) -> list[PaperRecord]:
        return list(self._records.values())
