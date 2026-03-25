#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from trim_raw_audio.pipeline import METHODS


STAGES = ("download", "prepare", "process", "upload")


@dataclass(frozen=True, slots=True)
class StageCommand:
    name: str
    command: list[str]
    description: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Sacred Space audio pipeline end-to-end with one command.",
    )
    parser.add_argument(
        "--stop-after",
        choices=STAGES,
        help="Run through the named stage and then stop.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip ClickUp download.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip raw-audio prepare.")
    parser.add_argument("--skip-process", action="store_true", help="Skip denoise/export.")
    parser.add_argument("--skip-upload", action="store_true", help="Skip ClickUp upload.")
    parser.add_argument("--clean", action="store_true", help="Remove audio working files after successful upload.")
    parser.add_argument(
        "--trim-method",
        choices=METHODS,
        default="ffmpeg-vad-asr",
        help="Trim method to use during the prepare stage.",
    )
    parser.add_argument("--trim-debug", action="store_true", help="Emit trim debug artifacts.")
    parser.add_argument("--trim-force", action="store_true", help="Recompute trim artifacts even if cached.")
    parser.add_argument(
        "--trim-dry-run",
        action="store_true",
        help="Analyze trim only; valid only when the pipeline stops after prepare.",
    )
    parser.add_argument(
        "--trim-config",
        type=Path,
        default=Path(__file__).parent / "config" / "trim_audio.toml",
        help="Path to the trim config file.",
    )
    return parser


def _stage_enabled(stage: str, args: argparse.Namespace) -> bool:
    return not getattr(args, f"skip_{stage.replace('-', '_')}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.trim_dry_run and not args.skip_prepare:
        stop_after = args.stop_after or STAGES[-1]
        if STAGES.index(stop_after) > STAGES.index("prepare"):
            raise ValueError("--trim-dry-run requires --stop-after prepare or skipping downstream stages.")


def build_stage_plan(args: argparse.Namespace, *, base_dir: Path | None = None) -> list[StageCommand]:
    _validate_args(args)
    base_dir = base_dir or Path(__file__).parent
    python_bin = sys.executable
    stop_index = STAGES.index(args.stop_after) if args.stop_after else len(STAGES) - 1

    stage_commands = {
        "download": StageCommand(
            name="download",
            command=[python_bin, str(base_dir / "workflow1_download.py")],
            description="Download raw recordings from ClickUp.",
        ),
        "prepare": StageCommand(
            name="prepare",
            command=[
                python_bin,
                str(base_dir / "prepare_raw_audio.py"),
                "--config",
                str(args.trim_config),
                "run",
                "--input-dir",
                str(base_dir / "audio-input"),
                "--output-dir",
                str(base_dir / "audio-prepared"),
                "--artifact-dir",
                str(base_dir / "trim-artifacts"),
                "--method",
                args.trim_method,
            ]
            + (["--skip-existing"] if not args.trim_force else [])
            + (["--force"] if args.trim_force else [])
            + (["--debug"] if args.trim_debug else [])
            + (["--dry-run"] if args.trim_dry_run else []),
            description="Prepare raw audio with silence/VAD/ASR trimming.",
        ),
        "process": StageCommand(
            name="process",
            command=["bash", str(base_dir / "process_audio.sh"), "--skip-prepare"],
            description="Denoise, normalize, and export MP3s.",
        ),
        "upload": StageCommand(
            name="upload",
            command=[python_bin, str(base_dir / "workflow2_upload.py")],
            description="Upload finished MP3s back to ClickUp.",
        ),
    }

    plan: list[StageCommand] = []
    for index, stage in enumerate(STAGES):
        if index > stop_index:
            break
        if _stage_enabled(stage, args):
            plan.append(stage_commands[stage])
    return plan


CLEAN_DIRS = ("audio-input", "audio-prepared", "audio-output", "trim-artifacts")


def clean_working_dirs(base_dir: Path) -> None:
    print("\n== CLEAN ==")
    print("Removing audio working files...")
    for name in CLEAN_DIRS:
        d = base_dir / name
        if not d.exists():
            continue
        # Preserve .metadata subdirectories
        for item in d.iterdir():
            if item.name == ".metadata":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print(f"  - {name}/  cleaned")
    print("Ready for next session.")


def run_stage(stage: StageCommand, *, cwd: Path) -> float:
    started = time.monotonic()
    print(f"\n== {stage.name.upper()} ==")
    print(stage.description)
    subprocess.run(stage.command, cwd=cwd, check=True)
    return time.monotonic() - started


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    base_dir = Path(__file__).parent

    try:
        plan = build_stage_plan(args, base_dir=base_dir)
    except ValueError as exc:
        parser.error(str(exc))

    if not plan:
        parser.error("No stages selected. Remove one of the skip flags or choose a later --stop-after stage.")

    print("Sacred Space pipeline")
    print(f"Working directory: {base_dir}")
    print("Plan:")
    for stage in plan:
        print(f"  - {stage.name}: {' '.join(stage.command)}")

    timings: list[tuple[str, float]] = []
    for stage in plan:
        elapsed = run_stage(stage, cwd=base_dir)
        timings.append((stage.name, elapsed))

    print("\nPipeline complete.")
    for name, elapsed in timings:
        print(f"  - {name}: {elapsed:.1f}s")

    if args.clean and any(s.name == "upload" for s in plan):
        clean_working_dirs(base_dir)


if __name__ == "__main__":
    main()
