from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from run_pipeline import build_stage_plan


def _args(**overrides) -> argparse.Namespace:
    payload = {
        "stop_after": None,
        "skip_download": False,
        "skip_prepare": False,
        "skip_process": False,
        "skip_upload": False,
        "trim_method": "ffmpeg-vad-asr",
        "trim_debug": False,
        "trim_force": False,
        "trim_dry_run": False,
        "trim_config": Path("/repo/config/trim_audio.toml"),
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_default_stage_plan_runs_all_stages() -> None:
    plan = build_stage_plan(_args(), base_dir=Path("/repo"))

    assert [stage.name for stage in plan] == ["download", "prepare", "process", "upload"]
    assert "ffmpeg-vad-asr" in plan[1].command
    assert plan[1].command[2:5] == ["--config", "/repo/config/trim_audio.toml", "run"]
    assert "/repo/audio-input" in plan[1].command
    assert "/repo/audio-prepared" in plan[1].command
    assert "--skip-prepare" in plan[2].command


def test_stage_plan_respects_stop_after_and_skips() -> None:
    plan = build_stage_plan(
        _args(stop_after="process", skip_download=True, trim_force=True, trim_debug=True),
        base_dir=Path("/repo"),
    )

    assert [stage.name for stage in plan] == ["prepare", "process"]
    assert "--force" in plan[0].command
    assert "--debug" in plan[0].command
    assert "--skip-existing" not in plan[0].command


def test_trim_dry_run_requires_pipeline_to_stop_after_prepare() -> None:
    with pytest.raises(ValueError, match="--trim-dry-run"):
        build_stage_plan(_args(trim_dry_run=True), base_dir=Path("/repo"))
