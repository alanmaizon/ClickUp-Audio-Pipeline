#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trim_raw_audio import RawAudioPreparer, load_trim_config
from trim_raw_audio.pipeline import METHODS, BackendNotReadyError, _backend_ready_reason
from trim_raw_audio.experiment import run_experiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare raw audio by trimming silence and spoken metadata before downstream processing.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "trim_audio.toml",
        help="Path to TOML config file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    run_parser = subparsers.add_parser("run", help="Process raw audio into prepared WAVs.")
    run_parser.add_argument("--input-dir", type=Path, help="Raw input directory.")
    run_parser.add_argument("--output-dir", type=Path, help="Prepared WAV output directory.")
    run_parser.add_argument("--artifact-dir", type=Path, help="Artifact root directory.")
    run_parser.add_argument("--method", choices=METHODS, default="ffmpeg-vad-asr")
    run_parser.add_argument("--files", nargs="*", help="Optional file names inside the input directory.")
    run_parser.add_argument("--dry-run", action="store_true", help="Analyze only; do not emit trimmed WAVs.")
    run_parser.add_argument("--debug", action="store_true", help="Emit waveform/transcript debug artifacts.")
    run_parser.add_argument("--force", action="store_true", help="Recompute even if cached artifacts exist.")
    run_parser.add_argument("--skip-existing", action="store_true", help="Reuse cached sidecars when still fresh.")
    run_parser.add_argument("--min-silence-duration", type=float, help="Override the ffmpeg silence minimum duration.")
    run_parser.add_argument("--silence-threshold-db", type=float, help="Override the ffmpeg silence threshold in dB.")
    run_parser.add_argument("--vad-confidence-threshold", type=float, help="Override the minimum usable VAD confidence.")
    run_parser.add_argument("--intro-analysis-window", type=float, help="Override the ASR intro-analysis window in seconds.")
    run_parser.add_argument("--start-pad", type=float, help="Override both silence/VAD opening pads.")
    run_parser.add_argument("--end-pad", type=float, help="Override both silence/VAD trailing pads.")
    run_parser.add_argument("--max-auto-trim-start", type=float, help="Override the maximum automatic opening trim.")
    run_parser.add_argument("--unsafe-mode", action="store_true", help="Disable safe-mode guardrails.")

    experiment_parser = subparsers.add_parser("experiment", help="Compare multiple trim methods on a sample batch.")
    experiment_parser.add_argument("--input-dir", type=Path, help="Raw input directory.")
    experiment_parser.add_argument("--artifact-dir", type=Path, help="Experiment artifact root directory.")
    experiment_parser.add_argument("--files", nargs="*", help="Optional file names inside the input directory.")
    experiment_parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=METHODS,
        help="Methods to compare.",
    )
    experiment_parser.add_argument("--sample-size", type=int, help="Limit the experiment batch size.")
    experiment_parser.add_argument("--dry-run", action="store_true", help="Analyze only; do not emit trimmed WAVs.")
    experiment_parser.add_argument("--force", action="store_true", help="Recompute experiment outputs even if cached.")
    experiment_parser.add_argument("--no-debug", action="store_true", help="Disable waveform/transcript debug artifacts.")

    subparsers.add_parser("doctor", help="Show Python/backend readiness for advanced trimming.")

    return parser


def _apply_overrides(config, args) -> None:
    if getattr(args, "min_silence_duration", None) is not None:
        config.silence.min_silence_duration_sec = args.min_silence_duration
    if getattr(args, "silence_threshold_db", None) is not None:
        config.silence.noise_threshold_db = args.silence_threshold_db
    if getattr(args, "vad_confidence_threshold", None) is not None:
        config.vad.confidence_threshold = args.vad_confidence_threshold
    if getattr(args, "intro_analysis_window", None) is not None:
        config.asr.analysis_window_sec = args.intro_analysis_window
    if getattr(args, "start_pad", None) is not None:
        config.silence.start_pad_sec = args.start_pad
        config.vad.start_pad_sec = args.start_pad
    if getattr(args, "end_pad", None) is not None:
        config.silence.end_pad_sec = args.end_pad
        config.vad.end_pad_sec = args.end_pad
    if getattr(args, "max_auto_trim_start", None) is not None:
        config.intro_classifier.max_auto_trim_start_sec = args.max_auto_trim_start
        config.decision.max_start_trim_sec = args.max_auto_trim_start
    if getattr(args, "unsafe_mode", False):
        config.decision.safe_mode = False


def _doctor(preparer: RawAudioPreparer, config) -> dict[str, object]:
    vad_reason = _backend_ready_reason(preparer.vad_backend)
    asr_reason = _backend_ready_reason(preparer.asr_backend, config=config.asr)
    return {
        "python_executable": sys.executable,
        "requested_default_method": "ffmpeg-vad-asr",
        "vad": {
            "ready": vad_reason is None,
            "reason": vad_reason or "",
        },
        "asr": {
            "ready": asr_reason is None,
            "reason": asr_reason or "",
            "model_size": config.asr.model_size,
        },
        "recommendation": (
            "Advanced trimming is ready."
            if vad_reason is None and asr_reason is None
            else "Run this script with the Python above, and ensure the Whisper model is available locally."
        ),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command is None:
        args.command = "run"

    config = load_trim_config(args.config)
    _apply_overrides(config, args)
    base_dir = Path(__file__).parent
    input_dir = getattr(args, "input_dir", None) or (base_dir / config.paths.input_dir)
    preparer = RawAudioPreparer(config)

    try:
        if args.command == "doctor":
            print(json.dumps(_doctor(preparer, config), indent=2))
            return

        if args.command == "experiment":
            artifact_dir = getattr(args, "artifact_dir", None) or (base_dir / config.paths.experiment_dir)
            summary = run_experiment(
                preparer,
                config=config,
                input_dir=input_dir,
                artifact_root=artifact_dir,
                methods=args.methods or config.experiment.default_methods,
                file_names=args.files,
                sample_size=args.sample_size or config.experiment.sample_size,
                dry_run=args.dry_run,
                debug=not args.no_debug,
                force=args.force,
            )
            print(json.dumps(summary, indent=2))
            return

        output_dir = getattr(args, "output_dir", None) or (base_dir / config.paths.output_dir)
        artifact_dir = getattr(args, "artifact_dir", None) or (base_dir / config.paths.artifact_dir)
        result = preparer.prepare_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            artifact_root=artifact_dir,
            requested_method=args.method,
            file_names=args.files,
            dry_run=args.dry_run,
            debug=args.debug,
            force=args.force,
            skip_existing=args.skip_existing,
            run_label="production",
        )
        print(json.dumps(result["summary"], indent=2))
    except BackendNotReadyError as exc:
        doctor_cmd = f"{sys.executable} {Path(__file__).resolve()} doctor"
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Run this to verify the active interpreter and backend readiness: {doctor_cmd}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
