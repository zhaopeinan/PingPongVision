"""JSON-backed registry for manually selectable TrackNet checkpoints."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TrackNetModelRecord:
    model_id: str
    display_name: str
    checkpoint_path: str
    status: str = "available"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "checkpoint_path": self.checkpoint_path,
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TrackNetModelRecord":
        return cls(
            model_id=str(value["model_id"]),
            display_name=str(value.get("display_name", value["model_id"])),
            checkpoint_path=str(value["checkpoint_path"]),
            status=str(value.get("status", "available")),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass
class TrackNetRun:
    run_id: str
    output_dir: Path
    metadata: dict = field(default_factory=dict)


class TrackNetModelRegistry:
    """Keep base and accepted fine-tuned models independently selectable."""

    def __init__(self, registry_dir: str | Path, base_path: str | Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_root = self.registry_dir.parent.resolve()
        self.base_path = Path(base_path).expanduser().resolve()
        self.index_path = self.registry_dir / "index.json"

    def _read_index(self) -> list[TrackNetModelRecord]:
        if not self.index_path.exists():
            return []
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            return [TrackNetModelRecord.from_dict(item) for item in payload.get("models", [])]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"TrackNet 模型注册表损坏: {self.index_path}") from exc

    def _write_index(self, records: list[TrackNetModelRecord]) -> None:
        payload = {"models": [record.to_dict() for record in records]}
        temp_path = self.index_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(self.index_path)

    def _base_record(self) -> TrackNetModelRecord:
        return TrackNetModelRecord(
            model_id="base",
            display_name="TrackNet 基础模型",
            checkpoint_path=str(self.base_path),
            metadata={"immutable": True},
        )

    def list_models(self) -> list[TrackNetModelRecord]:
        records = [self._base_record()]
        records.extend(record for record in self._read_index() if record.model_id != "base")
        return records

    def resolve(self, model_id: str) -> Path:
        record = next((item for item in self.list_models() if item.model_id == model_id), None)
        if record is None:
            raise KeyError(f"未知 TrackNet 模型: {model_id}")
        if record.status != "available":
            raise PermissionError(f"TrackNet 模型不可用: {model_id}")
        path = Path(record.checkpoint_path).expanduser().resolve()
        if path != self.base_path and self.models_root not in path.parents:
            raise ValueError("TrackNet 模型路径不在模型目录内")
        if not path.is_file():
            raise FileNotFoundError(f"TrackNet 权重不存在: {path}")
        return path

    def create_temp_run(self, metadata: dict) -> TrackNetRun:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        output_dir = self.registry_dir / f".tmp_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=False)
        return TrackNetRun(run_id=run_id, output_dir=output_dir, metadata=dict(metadata))

    def save_run(
        self,
        run_id: str,
        checkpoint_path: str | Path,
        metadata: dict | None = None,
    ) -> TrackNetModelRecord:
        source = Path(checkpoint_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"临时 TrackNet 权重不存在: {source}")
        if not run_id.startswith("run_"):
            raise ValueError("非法 TrackNet run_id")

        temp_dir = self.registry_dir / f".tmp_{run_id}"
        if temp_dir.exists() and source.parent != temp_dir.resolve():
            raise ValueError("临时权重路径与 run_id 不匹配")
        destination_dir = self.registry_dir / run_id
        if destination_dir.exists():
            raise FileExistsError(f"TrackNet 模型版本已存在: {run_id}")
        destination_dir.mkdir(parents=True, exist_ok=False)
        destination = destination_dir / "TrackNet_best.pt"
        shutil.copy2(source, destination)
        model_metadata = dict(metadata or {})
        model_metadata.setdefault("checkpoint_sha256", self._sha256(destination))
        model_metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        (destination_dir / "metadata.json").write_text(
            json.dumps(model_metadata, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        record = TrackNetModelRecord(
            model_id=run_id,
            display_name=str(model_metadata.get("display_name", f"微调模型 {run_id}")),
            checkpoint_path=str(destination.resolve()),
            metadata=model_metadata,
        )
        records = [item for item in self._read_index() if item.model_id != run_id]
        records.append(record)
        self._write_index(records)
        if temp_dir.exists() and temp_dir != destination_dir:
            shutil.rmtree(temp_dir)
        return record

    def discard_run(self, run_id: str) -> None:
        if not run_id.startswith("run_"):
            return
        temp_dir = self.registry_dir / f".tmp_{run_id}"
        saved_dir = self.registry_dir / run_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if saved_dir.exists():
            records = [item for item in self._read_index() if item.model_id != run_id]
            self._write_index(records)
            shutil.rmtree(saved_dir)

    def disable(self, model_id: str) -> None:
        if model_id == "base":
            raise ValueError("基础 TrackNet 模型不能禁用")
        records = self._read_index()
        for record in records:
            if record.model_id == model_id:
                record.status = "disabled"
                self._write_index(records)
                return
        raise KeyError(f"未知 TrackNet 模型: {model_id}")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
