from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from matanyone2.demo_core import (
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


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_DIR / "results" / "cli"


@dataclass(frozen=True)
class MatAnyoneRunResult:
    run_output_dir: str
    debug_dir: str
    foreground_path: str
    alpha_path: str
    mask_path: str
    sam_preview_path: str
    is_video: bool
    fps: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_image_input(input_path: str) -> bool:
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
        media_state, _media_info, _runtime_profile = load_image_state(
            image,
            device_name,
            performance_profile,
        )
        return media_state, False

    media_state, _media_info, _runtime_profile = load_video_state(
        input_path,
        device_name,
        performance_profile,
        video_target_fps_override=video_target_fps,
    )
    return maybe_truncate_media_state(media_state, frame_limit), True


def build_click_prompt(
    media_state: dict,
    positive_points: list[str] | None,
    negative_points: list[str] | None,
):
    resolved_positive = list(positive_points or ["center"])
    resolved_negative = list(negative_points or [])

    point_list = [
        parse_point_spec(spec, media_state["origin_images"][0].shape)
        for spec in resolved_positive
    ]
    label_list = [1] * len(point_list)
    for spec in resolved_negative:
        point_list.append(parse_point_spec(spec, media_state["origin_images"][0].shape))
        label_list.append(0)
    return point_list, label_list


def run_pipeline(
    *,
    input_path: str,
    device: str,
    model: str = "MatAnyone 2",
    performance_profile: str = "auto",
    sam_model_type: str = "auto",
    cpu_threads: int | None = None,
    frame_limit: int | None = None,
    video_target_fps: float | None = None,
    output_fps: float | None = None,
    select_frame: int = 0,
    end_frame: int | None = None,
    output_dir: str | None = None,
    positive_points: list[str] | None = None,
    negative_points: list[str] | None = None,
    erode_kernel_size: int = 0,
    dilate_kernel_size: int = 0,
    refine_iter: int | None = None,
) -> dict[str, Any]:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    configure_runtime(device, cpu_threads)
    resolved_sam_model_type = resolve_sam_model_type(sam_model_type, device)
    checkpoint_folder = str(REPO_DIR / "pretrained_models")
    runtime_models = RuntimeModelManager(device, checkpoint_folder)
    sam_generator = SamMaskGenerator(
        runtime_models.get_sam_checkpoint(resolved_sam_model_type),
        resolved_sam_model_type,
        device,
    )

    media_state, is_video = load_media_state(
        str(input_file),
        device,
        performance_profile,
        frame_limit,
        video_target_fps,
    )
    if select_frame < 0 or select_frame >= len(media_state["origin_images"]):
        raise ValueError(
            f"select_frame must be between 0 and {len(media_state['origin_images']) - 1}"
        )
    if end_frame is not None and (
        end_frame <= select_frame or end_frame > len(media_state["origin_images"])
    ):
        raise ValueError(
            "end_frame must be greater than select_frame and at most "
            f"{len(media_state['origin_images'])}"
        )
    media_state["select_frame_number"] = select_frame

    prepare_sam_frame(sam_generator, media_state, select_frame, force=True)
    points, labels = build_click_prompt(media_state, positive_points, negative_points)
    mask, _logit, painted = apply_sam_points(
        sam_generator,
        media_state,
        points,
        labels,
        frame_index=select_frame,
    )
    media_state["masks"][select_frame] = mask
    media_state["painted_images"][select_frame] = painted

    selected_model = runtime_models.load_model(model)
    foreground, alpha, _runtime_profile = run_matting(
        selected_model,
        media_state,
        mask,
        performance_profile,
        device,
        erode_kernel_size=erode_kernel_size,
        dilate_kernel_size=dilate_kernel_size,
        refine_iter=refine_iter,
        start_frame=select_frame,
        end_frame=end_frame,
    )

    resolved_output_fps = (
        output_fps if output_fps and output_fps > 0 else media_state.get("fps", 12)
    )
    run_output_dir = create_run_output_dir(
        output_dir or str(DEFAULT_OUTPUT_DIR),
        media_state,
    )
    save_cli_outputs(
        run_output_dir,
        str(input_file),
        media_state.get("source_size"),
        mask.astype(np.uint8),
        painted,
        foreground,
        alpha,
        is_video,
        fps=resolved_output_fps,
        audio_path=media_state.get("audio", ""),
    )
    debug_dir = export_debug_artifacts(
        run_output_dir,
        media_state,
        mask,
        foreground,
        alpha,
        device_name=device,
        performance_profile=performance_profile,
        model_name=model,
    )

    stem = input_file.stem
    if is_video:
        foreground_path = Path(run_output_dir) / f"{stem}_foreground.mp4"
        alpha_path = Path(run_output_dir) / f"{stem}_alpha.mp4"
    else:
        foreground_path = Path(run_output_dir) / f"{stem}_foreground.png"
        alpha_path = Path(run_output_dir) / f"{stem}_alpha.png"

    result = MatAnyoneRunResult(
        run_output_dir=str(run_output_dir),
        debug_dir=str(debug_dir),
        foreground_path=str(foreground_path),
        alpha_path=str(alpha_path),
        mask_path=str(Path(run_output_dir) / f"{stem}_mask.png"),
        sam_preview_path=str(Path(run_output_dir) / f"{stem}_sam_preview.png"),
        is_video=is_video,
        fps=resolved_output_fps,
    )
    return result.to_dict()
