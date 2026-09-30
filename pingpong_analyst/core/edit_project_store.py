"""Persistent storage for highlight-editing projects.

The editor state is intentionally stored as plain JSON so it can be recovered
after a service restart without coupling the API to an ORM or database.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProjectNotFoundError(FileNotFoundError):
    """Raised when a requested edit project does not exist."""


class ProjectValidationError(ValueError):
    """Raised when a project or an edit update is malformed."""


class EditProjectStore:
    """Read and write editor projects below one controlled directory."""

    _PROJECT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
    _PROJECT_REQUIRED_FIELDS = {
        "project_id",
        "video_id",
        "source_filename",
        "source_storage_name",
        "source_duration",
        "status",
        "created_at",
        "updated_at",
        "segments",
        "render",
    }
    _SEGMENT_REQUIRED_FIELDS = {
        "segment_id",
        "analysis_index",
        "clip_filename",
        "detected_start_time",
        "detected_end_time",
        "clip_start_time",
        "clip_end_time",
        "edit_start_time",
        "edit_end_time",
        "board_count",
        "selected",
        "order",
    }
    _RENDER_FIELDS = {
        "status",
        "progress",
        "message",
        "task_id",
        "output_filename",
        "output_size_bytes",
        "duration",
        "error",
    }
    _RENDER_STATUSES = {"idle", "rendering", "completed", "failed"}
    _PROJECT_STATUSES = {"draft", "rendering", "completed", "failed"}
    _SEGMENT_UPDATE_FIELDS = {
        "segment_id",
        "selected",
        "order",
        "edit_start_time",
        "edit_end_time",
    }

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()

    def create_from_analysis(
        self,
        *,
        project_id: str,
        video_id: str,
        source_filename: str,
        source_storage_name: str,
        source_duration: float,
        rallies: list[dict[str, Any]],
        clips: list[Any],
        buffer_before: float,
        buffer_after: float,
    ) -> dict[str, Any]:
        """Create a draft project from one completed rally analysis.

        Creation is idempotent for an existing project ID. The existing file is
        returned unchanged, which lets callers safely retry after a timeout.
        """

        project_id = self._validate_project_id(project_id)
        video_id = self._validate_identifier(video_id, "video_id")
        source_filename = self._validate_filename(source_filename, "source_filename")
        source_storage_name = self._validate_filename(
            source_storage_name, "source_storage_name"
        )
        source_duration = self._number(
            source_duration, "source_duration", minimum=0.0
        )
        buffer_before = self._number(buffer_before, "buffer_before", minimum=0.0)
        buffer_after = self._number(buffer_after, "buffer_after", minimum=0.0)

        if not isinstance(rallies, list):
            raise ProjectValidationError("rallies must be a list")
        if not isinstance(clips, list):
            raise ProjectValidationError("clips must be a list")
        if len(rallies) != len(clips):
            raise ProjectValidationError("rallies and clips must have the same length")

        segments: list[dict[str, Any]] = []
        for position, (rally, clip) in enumerate(zip(rallies, clips), start=1):
            if not isinstance(rally, dict):
                raise ProjectValidationError(f"rally {position} must be an object")

            analysis_index = rally.get("analysis_index", rally.get("index"))
            if analysis_index is None:
                raise ProjectValidationError(
                    f"rally {position} is missing index or analysis_index"
                )
            analysis_index = self._integer(
                analysis_index, f"rally {position}.analysis_index", minimum=1
            )
            detected_start = self._number(
                rally.get("start_time"),
                f"rally {position}.start_time",
                minimum=0.0,
                maximum=source_duration,
            )
            detected_end = self._number(
                rally.get("end_time"),
                f"rally {position}.end_time",
                minimum=0.0,
                maximum=source_duration,
            )
            if detected_start >= detected_end:
                raise ProjectValidationError(
                    f"rally {position} start_time must be before end_time"
                )
            board_count = self._integer(
                rally.get("board_count"),
                f"rally {position}.board_count",
                minimum=0,
            )
            clip_filename = self._clip_filename(clip, position)

            clip_start = max(0.0, detected_start - buffer_before)
            clip_end = min(source_duration, detected_end + buffer_after)

            # 消除相邻片段重叠：如果当前片段的 clip_start 小于上一个片段的 clip_end，
            # 取中点作为边界
            if segments:
                prev_clip_end = segments[-1]["clip_end_time"]
                if clip_start < prev_clip_end:
                    boundary = (prev_clip_end + clip_start) / 2.0
                    # 确保边界在两个 detected 范围之间
                    boundary = max(segments[-1]["detected_end_time"], min(boundary, detected_start))
                    segments[-1]["clip_end_time"] = boundary
                    segments[-1]["edit_end_time"] = boundary
                    clip_start = boundary

            if clip_start >= clip_end:
                raise ProjectValidationError(
                    f"rally {position} has an empty buffered clip range"
                )

            segments.append(
                {
                    "segment_id": f"segment-{position:03d}",
                    "analysis_index": analysis_index,
                    "clip_filename": clip_filename,
                    "detected_start_time": detected_start,
                    "detected_end_time": detected_end,
                    "clip_start_time": clip_start,
                    "clip_end_time": clip_end,
                    "edit_start_time": clip_start,
                    "edit_end_time": clip_end,
                    "board_count": board_count,
                    "selected": True,
                    "order": position,
                }
            )

        project = {
            "project_id": project_id,
            "video_id": video_id,
            "source_filename": source_filename,
            "source_storage_name": source_storage_name,
            "source_duration": source_duration,
            "status": "draft",
            "created_at": self._now(),
            "updated_at": self._now(),
            "segments": segments,
            "render": self._default_render(),
        }
        self._validate_project(project, expected_project_id=project_id)

        project_path = self._project_path(project_id)
        if project_path.exists():
            return self._read(project_id)

        self._atomic_write(project)
        return copy.deepcopy(project)

    def get(self, project_id: str) -> dict[str, Any]:
        """Load one project from disk."""

        return self._read(project_id)

    def list(
        self,
        video_id: str | None = None,
        source_storage_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return valid projects, newest updated project first."""

        if video_id is not None:
            video_id = self._validate_identifier(video_id, "video_id")
        if source_storage_name is not None:
            source_storage_name = self._validate_filename(
                source_storage_name, "source_storage_name"
            )

        projects = []
        for project_path in self.root.glob("*.json"):
            if project_path.is_symlink() or not project_path.is_file():
                raise ProjectValidationError(
                    f"project path is not a regular file: {project_path.name}"
                )
            project = self._read(project_path.stem)
            if video_id is not None and project["video_id"] != video_id:
                continue
            if (
                source_storage_name is not None
                and project["source_storage_name"] != source_storage_name
            ):
                continue
            projects.append(project)

        return sorted(
            projects,
            key=lambda project: project["updated_at"],
            reverse=True,
        )

    def update_segments(
        self, project_id: str, segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Update editable fields and atomically persist a new segment order.

        The input can contain all segments or a subset identified by
        ``segment_id``. Existing segments not included in the update are kept.
        """

        project = self._read(project_id)
        if project["render"]["status"] == "rendering":
            raise ProjectValidationError("cannot edit a project while rendering")
        if not isinstance(segments, list) or not segments:
            raise ProjectValidationError("segments must be a non-empty list")

        existing = {
            segment["segment_id"]: copy.deepcopy(segment)
            for segment in project["segments"]
        }
        seen_ids: set[str] = set()
        for update in segments:
            if not isinstance(update, dict):
                raise ProjectValidationError("each segment update must be an object")
            unknown_fields = set(update) - self._SEGMENT_UPDATE_FIELDS
            if unknown_fields:
                fields = ", ".join(sorted(unknown_fields))
                raise ProjectValidationError(f"unsupported segment fields: {fields}")
            segment_id = update.get("segment_id")
            if not isinstance(segment_id, str) or not segment_id:
                raise ProjectValidationError("segment_id is required")
            if segment_id in seen_ids:
                raise ProjectValidationError(f"duplicate segment_id: {segment_id}")
            seen_ids.add(segment_id)
            if segment_id not in existing:
                raise ProjectValidationError(f"unknown segment_id: {segment_id}")

            target = existing[segment_id]
            for field in self._SEGMENT_UPDATE_FIELDS - {"segment_id"}:
                if field in update:
                    target[field] = update[field]

        candidate = copy.deepcopy(project)
        candidate["segments"] = list(existing.values())
        candidate["updated_at"] = self._now()
        candidate["render"] = self._default_render()
        self._validate_project(
            candidate, expected_project_id=self._validate_project_id(project_id)
        )
        candidate["segments"].sort(key=lambda segment: segment["order"])
        self._atomic_write(candidate)
        return copy.deepcopy(candidate)

    def update_render(self, project_id: str, **fields: Any) -> dict[str, Any]:
        """Update render metadata after validating the complete project."""

        project = self._read(project_id)
        unknown_fields = set(fields) - self._RENDER_FIELDS
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ProjectValidationError(f"unsupported render fields: {names}")

        candidate = copy.deepcopy(project)
        candidate["render"].update(fields)
        candidate["updated_at"] = self._now()
        self._validate_project(
            candidate, expected_project_id=self._validate_project_id(project_id)
        )
        self._atomic_write(candidate)
        return copy.deepcopy(candidate)

    def _read(self, project_id: str) -> dict[str, Any]:
        project_id = self._validate_project_id(project_id)
        project_path = self._project_path(project_id)
        if not project_path.is_file() or project_path.is_symlink():
            raise ProjectNotFoundError(f"edit project not found: {project_id}")
        try:
            with project_path.open("r", encoding="utf-8") as handle:
                project = json.load(handle)
        except FileNotFoundError as exc:
            raise ProjectNotFoundError(
                f"edit project not found: {project_id}"
            ) from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProjectValidationError(
                f"invalid project JSON: {project_id}"
            ) from exc

        self._validate_project(project, expected_project_id=project_id)
        return project

    def _atomic_write(self, project: dict[str, Any]) -> None:
        project_id = self._validate_project_id(project["project_id"])
        self._validate_project(project, expected_project_id=project_id)
        destination = self._project_path(project_id)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=f".{project_id}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(project, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
        except OSError as exc:
            raise ProjectValidationError(
                f"could not persist project: {project_id}"
            ) from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _project_path(self, project_id: str) -> Path:
        project_id = self._validate_project_id(project_id)
        destination = self.root / f"{project_id}.json"
        try:
            destination.resolve(strict=False).relative_to(self.root)
        except ValueError as exc:
            raise ProjectValidationError("project path escapes the project root") from exc
        return destination

    @classmethod
    def _validate_project_id(cls, value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise ProjectValidationError("project_id must be a non-empty string")
        if (
            os.path.isabs(value)
            or "/" in value
            or "\\" in value
            or ".." in value
            or value in {".", ".."}
            or not cls._PROJECT_ID_RE.fullmatch(value)
        ):
            raise ProjectValidationError("project_id must be a safe file name")
        return value

    @staticmethod
    def _validate_identifier(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise ProjectValidationError(f"{field} must be a non-empty string")
        if (
            os.path.isabs(value)
            or "/" in value
            or "\\" in value
            or ".." in value
        ):
            raise ProjectValidationError(f"{field} must be a safe identifier")
        return value

    @staticmethod
    def _validate_filename(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise ProjectValidationError(f"{field} must be a non-empty file name")
        if (
            os.path.isabs(value)
            or "/" in value
            or "\\" in value
            or ".." in value
            or Path(value).name != value
        ):
            raise ProjectValidationError(f"{field} must be a safe file name")
        return value

    @staticmethod
    def _clip_filename(value: Any, position: int) -> str:
        if isinstance(value, dict):
            value = value.get("clip_filename", value.get("filename"))
        return EditProjectStore._validate_filename(value, f"clip {position}")

    @staticmethod
    def _number(
        value: Any,
        field: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProjectValidationError(f"{field} must be a number")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ProjectValidationError(f"{field} must be a number") from exc
        if not math.isfinite(number):
            raise ProjectValidationError(f"{field} must be finite")
        if minimum is not None and number < minimum:
            raise ProjectValidationError(f"{field} must be >= {minimum}")
        if maximum is not None and number > maximum:
            raise ProjectValidationError(f"{field} must be <= {maximum}")
        return number

    @classmethod
    def _integer(
        cls, value: Any, field: str, *, minimum: int | None = None
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProjectValidationError(f"{field} must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ProjectValidationError(f"{field} must be an integer")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ProjectValidationError(f"{field} must be an integer") from exc
        if minimum is not None and integer < minimum:
            raise ProjectValidationError(f"{field} must be >= {minimum}")
        return integer

    @classmethod
    def _validate_project(
        cls, project: Any, *, expected_project_id: str | None = None
    ) -> None:
        if not isinstance(project, dict):
            raise ProjectValidationError("project JSON must contain an object")
        missing = cls._PROJECT_REQUIRED_FIELDS - set(project)
        if missing:
            raise ProjectValidationError(
                f"project is missing required fields: {', '.join(sorted(missing))}"
            )

        project_id = cls._validate_project_id(project["project_id"])
        if expected_project_id is not None and project_id != expected_project_id:
            raise ProjectValidationError("project_id does not match its file name")
        cls._validate_identifier(project["video_id"], "video_id")
        cls._validate_filename(project["source_filename"], "source_filename")
        cls._validate_filename(project["source_storage_name"], "source_storage_name")
        source_duration = cls._number(
            project["source_duration"], "source_duration", minimum=0.0
        )
        if (
            not isinstance(project["status"], str)
            or project["status"] not in cls._PROJECT_STATUSES
        ):
            raise ProjectValidationError("invalid project status")
        if not isinstance(project["created_at"], str) or not isinstance(
            project["updated_at"], str
        ):
            raise ProjectValidationError("project timestamps must be strings")

        segments = project["segments"]
        if not isinstance(segments, list):
            raise ProjectValidationError("segments must be a list")
        orders: set[int] = set()
        segment_ids: set[str] = set()
        for position, segment in enumerate(segments, start=1):
            cls._validate_segment(
                segment,
                source_duration=source_duration,
                position=position,
                orders=orders,
                segment_ids=segment_ids,
            )

        render = project["render"]
        if not isinstance(render, dict):
            raise ProjectValidationError("render must be an object")
        missing_render = cls._RENDER_FIELDS - set(render)
        if missing_render:
            raise ProjectValidationError(
                "render is missing required fields: "
                + ", ".join(sorted(missing_render))
            )
        if (
            not isinstance(render["status"], str)
            or render["status"] not in cls._RENDER_STATUSES
        ):
            raise ProjectValidationError("invalid render status")
        cls._number(render["progress"], "render.progress", minimum=0.0, maximum=100.0)
        if render["task_id"] is not None and not isinstance(render["task_id"], str):
            raise ProjectValidationError("render.task_id must be a string or null")
        if render["message"] is not None and not isinstance(render["message"], str):
            raise ProjectValidationError("render.message must be a string or null")
        if render["output_filename"] is not None:
            cls._validate_filename(render["output_filename"], "render.output_filename")
        if render["output_size_bytes"] is not None:
            cls._integer(
                render["output_size_bytes"],
                "render.output_size_bytes",
                minimum=0,
            )
        if render["duration"] is not None:
            cls._number(render["duration"], "render.duration", minimum=0.0)
        if render["error"] is not None and not isinstance(render["error"], str):
            raise ProjectValidationError("render.error must be a string or null")

    @classmethod
    def _validate_segment(
        cls,
        segment: Any,
        *,
        source_duration: float,
        position: int,
        orders: set[int],
        segment_ids: set[str],
    ) -> None:
        if not isinstance(segment, dict):
            raise ProjectValidationError(f"segment {position} must be an object")
        missing = cls._SEGMENT_REQUIRED_FIELDS - set(segment)
        if missing:
            raise ProjectValidationError(
                f"segment {position} is missing required fields: "
                + ", ".join(sorted(missing))
            )
        segment_id = segment["segment_id"]
        if not isinstance(segment_id, str) or not segment_id:
            raise ProjectValidationError(f"segment {position}.segment_id is invalid")
        if segment_id in segment_ids:
            raise ProjectValidationError(f"duplicate segment_id: {segment_id}")
        segment_ids.add(segment_id)

        order = cls._integer(segment["order"], f"segment {position}.order", minimum=1)
        if order in orders:
            raise ProjectValidationError(f"duplicate segment order: {order}")
        orders.add(order)
        if not isinstance(segment["selected"], bool):
            raise ProjectValidationError(f"segment {position}.selected must be boolean")
        cls._integer(
            segment["analysis_index"], f"segment {position}.analysis_index", minimum=1
        )
        cls._validate_filename(segment["clip_filename"], f"segment {position}.clip_filename")

        detected_start = cls._number(
            segment["detected_start_time"],
            f"segment {position}.detected_start_time",
            minimum=0.0,
            maximum=source_duration,
        )
        detected_end = cls._number(
            segment["detected_end_time"],
            f"segment {position}.detected_end_time",
            minimum=0.0,
            maximum=source_duration,
        )
        if detected_start >= detected_end:
            raise ProjectValidationError(
                f"segment {position} detected range is invalid"
            )

        clip_start = cls._number(
            segment["clip_start_time"],
            f"segment {position}.clip_start_time",
            minimum=0.0,
            maximum=source_duration,
        )
        clip_end = cls._number(
            segment["clip_end_time"],
            f"segment {position}.clip_end_time",
            minimum=0.0,
            maximum=source_duration,
        )
        if clip_start >= clip_end:
            raise ProjectValidationError(f"segment {position} clip range is invalid")

        edit_start = cls._number(
            segment["edit_start_time"],
            f"segment {position}.edit_start_time",
            minimum=clip_start,
            maximum=clip_end,
        )
        edit_end = cls._number(
            segment["edit_end_time"],
            f"segment {position}.edit_end_time",
            minimum=clip_start,
            maximum=clip_end,
        )
        if edit_start >= edit_end:
            raise ProjectValidationError(f"segment {position} edit range is invalid")

        cls._integer(segment["board_count"], f"segment {position}.board_count", minimum=0)

    @staticmethod
    def _default_render() -> dict[str, Any]:
        return {
            "status": "idle",
            "progress": 0,
            "message": "",
            "task_id": None,
            "output_filename": None,
            "output_size_bytes": None,
            "duration": None,
            "error": None,
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
