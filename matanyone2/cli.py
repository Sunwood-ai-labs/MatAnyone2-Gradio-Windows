from __future__ import annotations

import argparse
import os

import torch

from matanyone2.api import run_pipeline
from matanyone2.demo_core import (
    MODEL_DISPLAY_TO_FILE,
    PROFILE_CHOICES,
    SAM_MODEL_CHOICES,
    resolve_sam_model_type,
)


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Run the shared MatAnyone demo core from the CLI.")
    parser.add_argument("--input", required=True, help="Path to an input image or video.")
    parser.add_argument("--device", default=default_device, choices=["cpu", "cuda"], help="Inference device.")
    parser.add_argument("--model", default="MatAnyone 2", choices=list(MODEL_DISPLAY_TO_FILE.keys()))
    parser.add_argument("--performance_profile", default="auto", choices=PROFILE_CHOICES)
    parser.add_argument("--sam_model_type", default="auto", choices=SAM_MODEL_CHOICES)
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--frame_limit", type=int, default=None, help="Optional cap for loaded frames.")
    parser.add_argument("--video_target_fps", type=float, default=None, help="Optional override for video sampling FPS. Use 0 to keep all frames.")
    parser.add_argument(
        "--output_fps",
        type=float,
        default=None,
        help="Optional FPS override for video outputs. Defaults to the processing FPS of the loaded media.",
    )
    parser.add_argument("--select_frame", type=int, default=0, help="Frame index used to place the click prompt.")
    parser.add_argument("--end_frame", type=int, default=None, help="Optional exclusive end frame for processing.")
    parser.add_argument("--output_dir", default=os.path.join(REPO_DIR, "results", "cli"))
    parser.add_argument("--positive_point", action="append", default=[], help='Positive point in "x,y" format.')
    parser.add_argument("--negative_point", action="append", default=[], help='Negative point in "x,y" format.')
    parser.add_argument("--erode_kernel_size", type=int, default=0)
    parser.add_argument("--dilate_kernel_size", type=int, default=0)
    parser.add_argument("--refine_iter", type=int, default=None, help="Warmup/refinement iterations for image mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    resolved_sam_model_type = resolve_sam_model_type(args.sam_model_type, args.device)
    result = run_pipeline(
        input_path=args.input,
        device=args.device,
        model=args.model,
        performance_profile=args.performance_profile,
        sam_model_type=resolved_sam_model_type,
        cpu_threads=args.cpu_threads,
        frame_limit=args.frame_limit,
        video_target_fps=args.video_target_fps,
        output_fps=args.output_fps,
        select_frame=args.select_frame,
        end_frame=args.end_frame,
        output_dir=args.output_dir,
        positive_points=list(args.positive_point),
        negative_points=list(args.negative_point),
        erode_kernel_size=args.erode_kernel_size,
        dilate_kernel_size=args.dilate_kernel_size,
        refine_iter=args.refine_iter,
    )
    print(f"Completed shared MatAnyone pipeline. Outputs saved to {result['run_output_dir']}")
    print(f"Debug artifacts saved to {result['debug_dir']}")


if __name__ == "__main__":
    main()
