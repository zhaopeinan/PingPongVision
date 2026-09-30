"""Focused tests for persistent highlight-editing project state."""

import json

import pytest

from pingpong_analyst.core.edit_project_store import (
    EditProjectStore,
    ProjectNotFoundError,
    ProjectValidationError,
)


def sample_analysis_args():
    return {
        "project_id": "project-001",
        "video_id": "video-001",
        "source_filename": "match.mp4",
        "source_storage_name": "match.mp4",
        "source_duration": 20.0,
        "rallies": [
            {"index": 1, "start_time": 3.0, "end_time": 8.0, "board_count": 12},
            {"index": 2, "start_time": 12.0, "end_time": 19.8, "board_count": 9},
        ],
        "clips": ["rally_001.mp4", "rally_002.mp4"],
        "buffer_before": 0.5,
        "buffer_after": 0.5,
    }


def test_create_project_derives_buffered_edit_ranges(tmp_path):
    store = EditProjectStore(tmp_path / "projects")
    project = store.create_from_analysis(
        project_id="project-001",
        video_id="video-001",
        source_filename="match.mp4",
        source_storage_name="match.mp4",
        source_duration=20.0,
        rallies=[
            {"index": 1, "start_time": 3.0, "end_time": 8.0, "board_count": 12}
        ],
        clips=["rally_001.mp4"],
        buffer_before=0.5,
        buffer_after=0.5,
    )

    assert project["status"] == "draft"
    assert project["segments"][0]["edit_start_time"] == 2.5
    assert project["segments"][0]["edit_end_time"] == 8.5
    assert store.get("project-001")["source_storage_name"] == "match.mp4"
    assert (tmp_path / "projects" / "project-001.json").is_file()


def test_create_clamps_buffered_ranges_to_source_duration(tmp_path):
    store = EditProjectStore(tmp_path / "projects")
    project = store.create_from_analysis(
        project_id="project-001",
        video_id="video-001",
        source_filename="match.mp4",
        source_storage_name="match.mp4",
        source_duration=10.0,
        rallies=[
            {"index": 1, "start_time": 0.2, "end_time": 9.9, "board_count": 8}
        ],
        clips=["rally_001.mp4"],
        buffer_before=1.0,
        buffer_after=2.0,
    )

    segment = project["segments"][0]
    assert segment["clip_start_time"] == 0.0
    assert segment["clip_end_time"] == 10.0
    assert segment["edit_start_time"] == 0.0
    assert segment["edit_end_time"] == 10.0


def test_project_update_is_reloaded_from_disk(tmp_path):
    store = EditProjectStore(tmp_path / "projects")
    store.create_from_analysis(**sample_analysis_args())
    updated = store.update_segments(
        "project-001",
        [
            {
                "segment_id": "segment-001",
                "selected": False,
                "order": 2,
                "edit_start_time": 2.8,
                "edit_end_time": 7.9,
            },
            {"segment_id": "segment-002", "order": 1},
        ],
    )

    assert updated["segments"][0]["segment_id"] == "segment-002"
    assert updated["segments"][1]["selected"] is False
    assert updated["segments"][1]["edit_start_time"] == 2.8
    reloaded = EditProjectStore(tmp_path / "projects").get("project-001")
    assert reloaded["segments"][1]["edit_start_time"] == 2.8


def test_update_render_persists_only_render_fields(tmp_path):
    store = EditProjectStore(tmp_path / "projects")
    store.create_from_analysis(**sample_analysis_args())

    updated = store.update_render(
        "project-001",
        status="completed",
        progress=100,
        task_id="render-001",
        output_filename="project-001.mp4",
        duration=12.5,
    )

    assert updated["status"] == "draft"
    assert updated["render"] == {
        "status": "completed",
        "progress": 100,
        "message": "",
        "task_id": "render-001",
        "output_filename": "project-001.mp4",
        "output_size_bytes": None,
        "duration": 12.5,
        "error": None,
    }


def test_list_filters_by_video_and_source_and_sorts_newest_first(tmp_path):
    store = EditProjectStore(tmp_path / "projects")
    store.create_from_analysis(**sample_analysis_args())
    second = sample_analysis_args()
    second.update(
        project_id="project-002",
        video_id="video-002",
        source_filename="other.mp4",
        source_storage_name="other.mp4",
    )
    store.create_from_analysis(**second)

    assert [p["project_id"] for p in store.list(video_id="video-001")] == [
        "project-001"
    ]
    assert [p["project_id"] for p in store.list(source_storage_name="other.mp4")] == [
        "project-002"
    ]
    assert {p["project_id"] for p in store.list()} == {
        "project-001",
        "project-002",
    }


def test_invalid_segment_update_leaves_original_file_unchanged(tmp_path):
    store = EditProjectStore(tmp_path / "projects")
    store.create_from_analysis(**sample_analysis_args())
    project_path = tmp_path / "projects" / "project-001.json"
    original = project_path.read_text(encoding="utf-8")

    with pytest.raises(ProjectValidationError, match="edit range"):
        store.update_segments(
            "project-001",
            [
                {
                    "segment_id": "segment-001",
                    "selected": False,
                    "order": 1,
                    "edit_start_time": 7.0,
                    "edit_end_time": 7.0,
                },
                {"segment_id": "segment-002", "order": 2},
            ],
        )

    assert project_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "update, message",
    [
        ([{"segment_id": "segment-001", "selected": False, "order": 1}], None),
        (
            [
                {"segment_id": "segment-001", "order": 1},
                {"segment_id": "segment-002", "order": 1},
            ],
            "duplicate segment order",
        ),
        ([{"segment_id": "segment-001", "order": -1}], "must be >= 1"),
    ],
)
def test_segment_update_validation(tmp_path, update, message):
    store = EditProjectStore(tmp_path / "projects")
    store.create_from_analysis(**sample_analysis_args())

    if message is None:
        updated = store.update_segments("project-001", update)
        assert all(not segment["selected"] for segment in updated["segments"] if segment["segment_id"] == "segment-001")
    else:
        with pytest.raises(ProjectValidationError, match=message):
            store.update_segments("project-001", update)


def test_empty_update_is_rejected_without_changing_file(tmp_path):
    store = EditProjectStore(tmp_path / "projects")
    store.create_from_analysis(**sample_analysis_args())
    project_path = tmp_path / "projects" / "project-001.json"
    original = project_path.read_text(encoding="utf-8")

    with pytest.raises(ProjectValidationError, match="non-empty"):
        store.update_segments("project-001", [])

    assert project_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("project_id", ["../escape", "/absolute", "foo/bar", "foo\\bar", "foo..bar"])
def test_project_id_must_be_safe(tmp_path, project_id):
    store = EditProjectStore(tmp_path / "projects")
    args = sample_analysis_args()
    args["project_id"] = project_id

    with pytest.raises(ProjectValidationError, match="safe"):
        store.create_from_analysis(**args)

    assert list((tmp_path / "projects").iterdir()) == []


def test_missing_and_corrupt_projects_raise_typed_errors(tmp_path):
    root = tmp_path / "projects"
    store = EditProjectStore(root)
    with pytest.raises(ProjectNotFoundError):
        store.get("missing-001")

    (root / "broken-001.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ProjectValidationError, match="invalid project JSON"):
        store.get("broken-001")


def test_idempotent_create_keeps_existing_project(tmp_path):
    store = EditProjectStore(tmp_path / "projects")
    first = store.create_from_analysis(**sample_analysis_args())
    second_args = sample_analysis_args()
    second_args["source_filename"] = "changed.mp4"

    second = store.create_from_analysis(**second_args)

    assert second["created_at"] == first["created_at"]
    assert second["source_filename"] == "match.mp4"
    assert json.loads(
        (tmp_path / "projects" / "project-001.json").read_text(encoding="utf-8")
    )["source_filename"] == "match.mp4"
