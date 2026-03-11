from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image

from matanyone2.demo_core import (
    MODEL_DISPLAY_TO_FILE,
    PROFILE_CHOICES,
    SAM_MODEL_CHOICES,
    RuntimeModelManager,
    SamMaskGenerator,
    apply_sam_points,
    configure_runtime,
    create_run_output_dir,
    export_debug_artifacts,
    load_image_state,
    load_video_state,
    parse_point_spec,
    prepare_sam_frame,
    resolve_sam_model_type,
    save_cli_outputs,
    run_matting,
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


def is_image_input(input_path: str):
    return input_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def maybe_truncate_media_state(media_state: dict, frame_limit: int | None):
    if frame_limit is None or frame_limit <= 0:
        return media_state
    current_frames = media_state["origin_images"]
    if len(current_frames) <= frame_limit:
        return media_state
    media_state["origin_images"] = current_frames[:frame_limit]
    media_state["painted_images"] = media_state["painted_images"][:frame_limit]
    media_state["masks"] = media_state["masks"][:frame_limit]
    media_state["logits"] = media_state["logits"][:frame_limit]
    return media_state


def load_media_state(
    input_path: str,
    device_name: str,
    performance_profile: str,
    frame_limit: int | None,
    video_target_fps: float | None,
):
    if is_image_input(input_path):
        image = np.array(Image.open(input_path).convert("RGB"))
        media_state, _media_info, _runtime_profile = load_image_state(image, device_name, performance_profile)
        return media_state, False

    target_fps_override = video_target_fps
    media_state, _media_info, _runtime_profile = load_video_state(
        input_path,
        device_name,
        performance_profile,
        video_target_fps_override=target_fps_override,
    )
    return maybe_truncate_media_state(media_state, frame_limit), True


def build_click_prompt(media_state: dict, positive_points: list[str], negative_points: list[str]):
    if not positive_points:
        positive_points = ["center"]

    point_list = [parse_point_spec(spec, media_state["origin_images"][0].shape) for spec in positive_points]
    label_list = [1] * len(point_list)
    for spec in negative_points:
        point_list.append(parse_point_spec(spec, media_state["origin_images"][0].shape))
        label_list.append(0)
    return point_list, label_list


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    configure_runtime(args.device, args.cpu_threads)
    sam_model_type = resolve_sam_model_type(args.sam_model_type, args.device)
    checkpoint_folder = os.path.join(REPO_DIR, "pretrained_models")
    runtime_models = RuntimeModelManager(args.device, checkpoint_folder)
    sam_generator = SamMaskGenerator(runtime_models.get_sam_checkpoint(sam_model_type), sam_model_type, args.device)

    media_state, is_video = load_media_state(
        args.input,
        args.device,
        args.performance_profile,
        args.frame_limit,
        args.video_target_fps,
    )
    if args.select_frame < 0 or args.select_frame >= len(media_state["origin_images"]):
        raise ValueError(f"--select_frame must be between 0 and {len(media_state['origin_images']) - 1}")
    if args.end_frame is not None and (args.end_frame <= args.select_frame or args.end_frame > len(media_state["origin_images"])):
        raise ValueError(f"--end_frame must be greater than --select_frame and at most {len(media_state['origin_images'])}")
    media_state["select_frame_number"] = args.select_frame

    prepare_sam_frame(sam_generator, media_state, args.select_frame, force=True)
    points, labels = build_click_prompt(media_state, args.positive_point, args.negative_point)
    mask, _logit, painted = apply_sam_points(
        sam_generator,
        media_state,
        points,
        labels,
        frame_index=args.select_frame,
    )
    media_state["masks"][args.select_frame] = mask
    media_state["painted_images"][args.select_frame] = painted

    selected_model = runtime_models.load_model(args.model)
    foreground, alpha, _runtime_profile = run_matting(
        selected_model,
        media_state,
        mask,
        args.performance_profile,
        args.device,
        erode_kernel_size=args.erode_kernel_size,
        dilate_kernel_size=args.dilate_kernel_size,
        refine_iter=args.refine_iter,
        start_frame=args.select_frame,
        end_frame=args.end_frame,
    )

    fps = args.output_fps if args.output_fps and args.output_fps > 0 else media_state.get("fps", 12)
    run_output_dir = create_run_output_dir(args.output_dir, media_state)
    save_cli_outputs(
        run_output_dir,
        args.input,
        media_state.get("source_size"),
        mask.astype(np.uint8),
        painted,
        foreground,
        alpha,
        is_video,
        fps=fps,
        audio_path=media_state.get("audio", ""),
    )
    debug_dir = export_debug_artifacts(
        run_output_dir,
        media_state,
        mask,
        foreground,
        alpha,
        device_name=args.device,
        performance_profile=args.performance_profile,
        model_name=args.model,
    )
    print(f"Completed shared MatAnyone pipeline. Outputs saved to {run_output_dir}")
    print(f"Debug artifacts saved to {debug_dir}")


if __name__ == "__main__":
    main()
