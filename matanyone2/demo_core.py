from __future__ import annotations

import gc
import glob
import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Sequence

import cv2
import ffmpeg
import imageio
import imageio_ffmpeg
import numpy as np
import psutil
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict
from PIL import Image

from hugging_face.matanyone2_wrapper import matanyone2
from hugging_face.tools.download_util import load_file_from_url
from hugging_face.tools.interact_tools import SamControler
from hugging_face.tools.painter import mask_painter
from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.device import set_default_device
from matanyone2.utils.get_default_model import get_matanyone2_model


PERFORMANCE_PROFILES = {
    "quality": {
        "video_target_fps": None,
        "video_max_short_side": 1080,
        "image_max_short_side": 1536,
        "max_internal_size": -1,
        "n_warmup": 10,
    },
    "balanced": {
        "video_target_fps": 12,
        "video_max_short_side": 720,
        "image_max_short_side": 1024,
        "max_internal_size": 768,
        "n_warmup": 4,
    },
    "fast": {
        "video_target_fps": 8,
        "video_max_short_side": 640,
        "image_max_short_side": 832,
        "max_internal_size": 640,
        "n_warmup": 2,
    },
}

PROFILE_CHOICES = ["auto", "balanced", "fast", "quality"]
SAM_MODEL_CHOICES = ["auto", "vit_h", "vit_l", "vit_b"]
MODEL_DISPLAY_TO_FILE = {
    "MatAnyone 2": "matanyone2.pth",
    "MatAnyone": "matanyone.pth",
}
MODEL_URLS = {
    "matanyone.pth": "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth",
    "matanyone2.pth": "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth",
}
SAM_MODEL_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    sam_model_type: str = "auto"
    performance_profile: str = "auto"
    cpu_threads: int | None = None
    mask_save: bool = False


def configure_ffmpeg_binary():
    candidates = []

    env_ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if env_ffmpeg:
        candidates.append(env_ffmpeg)

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        candidates.append(path_ffmpeg)

    try:
        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled_ffmpeg:
            candidates.append(bundled_ffmpeg)
    except Exception:
        pass

    winget_root = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    if os.path.isdir(winget_root):
        candidates.extend(glob.glob(os.path.join(winget_root, "**", "ffmpeg.exe"), recursive=True))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            ffmpeg_dir = os.path.dirname(candidate)
            os.environ["IMAGEIO_FFMPEG_EXE"] = candidate
            current_path = os.environ.get("PATH", "")
            path_entries = current_path.split(os.pathsep) if current_path else []
            if ffmpeg_dir not in path_entries:
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path if current_path else ffmpeg_dir
            return candidate

    return None


def configure_runtime(device_name: str, cpu_threads: int | None = None):
    set_default_device(device_name)
    device = torch.device(device_name)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True

    if device.type != "cpu":
        return

    cpu_count = os.cpu_count() or 4
    threads = cpu_threads if cpu_threads and cpu_threads > 0 else max(1, cpu_count - 1 if cpu_count > 4 else cpu_count)
    torch.set_num_threads(threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(4, threads)))
        except RuntimeError:
            pass
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    cv2.setNumThreads(max(1, min(4, threads)))


def resolve_performance_profile(profile_name: str | None, device_name: str):
    normalized = (profile_name or "auto").strip().lower()
    device_type = torch.device(device_name).type
    if normalized == "auto":
        normalized = "fast" if device_type == "cpu" else "quality"
    profile = dict(PERFORMANCE_PROFILES[normalized])
    profile["name"] = normalized
    profile["device_type"] = device_type
    return profile


def resolve_sam_model_type(requested_model_type: str | None, device_name: str):
    normalized = (requested_model_type or "auto").strip().lower()
    if normalized != "auto":
        return normalized
    return "vit_b" if torch.device(device_name).type == "cpu" else "vit_h"


def maybe_resize_frame(frame: np.ndarray, max_short_side: int | None):
    if max_short_side is None or max_short_side <= 0:
        return frame

    height, width = frame.shape[:2]
    short_side = min(height, width)
    if short_side <= max_short_side:
        return frame

    scale = max_short_side / float(short_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def resize_output_frame(frame: np.ndarray, target_size, interpolation=cv2.INTER_LINEAR):
    if target_size is None:
        return frame

    target_h, target_w = target_size
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        return frame
    resized = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)
    if frame.ndim == 3 and frame.shape[2] == 1 and resized.ndim == 2:
        resized = resized[:, :, None]
    return resized


def sam_image_key(state: dict, frame_index: int):
    return f"{state.get('user_name', 'session')}:{frame_index}"


def create_empty_media_state(performance_profile: str = "auto", mask_save: bool = False):
    media_state = {
        "user_name": "",
        "video_name": "",
        "image_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30,
        "source_fps": 30,
        "frame_stride": 1,
        "source_size": None,
        "working_size": None,
        "performance_profile": performance_profile,
        "audio": "",
    }
    interactive_state = {
        "inference_times": 0,
        "negative_click_times": 0,
        "positive_click_times": 0,
        "mask_save": mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": [],
        },
        "track_end_number": None,
    }
    return media_state, interactive_state


def load_image_state(image_input: np.ndarray, device_name: str, performance_profile: str):
    runtime_profile = resolve_performance_profile(performance_profile, device_name)
    user_name = time.time()
    source_size = (image_input.shape[0], image_input.shape[1])
    working_image = maybe_resize_frame(image_input, runtime_profile["image_max_short_side"])
    frames = [working_image.copy(), working_image.copy()]
    image_size = (frames[0].shape[0], frames[0].shape[1])
    image_state = {
        "user_name": user_name,
        "image_name": "output.png",
        "origin_images": frames,
        "painted_images": [frame.copy() for frame in frames],
        "masks": [np.zeros((frames[0].shape[0], frames[0].shape[1]), np.uint8) for _ in frames],
        "logits": [None] * len(frames),
        "select_frame_number": 0,
        "fps": None,
        "source_size": source_size,
        "working_size": image_size,
        "performance_profile": runtime_profile["name"],
    }
    image_info = (
        "Image Name: N/A,\n"
        f"FPS: N/A,\n"
        f"Total Frames: {len(frames)},\n"
        f"Source Size:{source_size},\n"
        f"Working Size:{image_size},\n"
        f"Profile:{runtime_profile['name']}"
    )
    return image_state, image_info, runtime_profile


def load_video_state(
    video_path: str,
    device_name: str,
    performance_profile: str,
    video_target_fps_override: float | None = None,
):
    runtime_profile = resolve_performance_profile(performance_profile, device_name)
    frames = []
    user_name = time.time()
    source_size = None
    audio_path = ""

    ffmpeg_binary = configure_ffmpeg_binary()
    if ffmpeg_binary:
        try:
            video_root, _ = os.path.splitext(video_path)
            audio_path = f"{video_root}_audio.wav"
            (
                ffmpeg.input(video_path)
                .output(audio_path, format="wav", acodec="pcm_s16le", ac=2, ar="44100")
                .run(overwrite_output=True, quiet=True)
            )
        except Exception:
            audio_path = ""

    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 30.0
    frame_stride = 1
    target_fps = video_target_fps_override if video_target_fps_override is not None else runtime_profile["video_target_fps"]
    if target_fps:
        frame_stride = max(1, int(round(source_fps / target_fps)))
    fps = source_fps / frame_stride
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if source_size is None:
            source_size = (frame.shape[0], frame.shape[1])
        if frame_index % frame_stride != 0:
            frame_index += 1
            continue
        current_memory_usage = psutil.virtual_memory().percent
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(maybe_resize_frame(rgb_frame, runtime_profile["video_max_short_side"]))
        if current_memory_usage > 90:
            break
        frame_index += 1
    cap.release()

    if not frames:
        raise ValueError("No frames could be extracted from the selected video.")

    image_size = (frames[0].shape[0], frames[0].shape[1])
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0], frames[0].shape[1]), np.uint8)] * len(frames),
        "logits": [None] * len(frames),
        "select_frame_number": 0,
        "fps": fps,
        "audio": audio_path,
        "source_fps": source_fps,
        "frame_stride": frame_stride,
        "source_size": source_size or image_size,
        "working_size": image_size,
        "performance_profile": runtime_profile["name"],
    }
    video_info = (
        f"Video Name: {video_state['video_name']},\n"
        f"Source FPS: {round(video_state['source_fps'], 1)},\n"
        f"Processing FPS: {round(video_state['fps'], 1)},\n"
        f"Total Frames: {len(frames)},\n"
        f"Source Size:{video_state['source_size']},\n"
        f"Working Size:{image_size},\n"
        f"Profile:{runtime_profile['name']}"
    )
    return video_state, video_info, runtime_profile


def prepare_sam_frame(mask_generator: "SamMaskGenerator", media_state: dict, frame_index: int | None = None, force: bool = False):
    selected_frame = media_state["select_frame_number"] if frame_index is None else frame_index
    mask_generator.prepare_image(
        media_state["origin_images"][selected_frame],
        sam_image_key(media_state, selected_frame),
        force=force,
    )


def apply_sam_points(
    mask_generator: "SamMaskGenerator",
    media_state: dict,
    points: Sequence[Sequence[int]],
    labels: Sequence[int],
    frame_index: int | None = None,
    multimask=True,
):
    selected_frame = media_state["select_frame_number"] if frame_index is None else frame_index
    prepare_sam_frame(mask_generator, media_state, selected_frame)
    return mask_generator.first_frame_click(
        image=media_state["origin_images"][selected_frame],
        points=np.asarray(points, dtype=np.int32),
        labels=np.asarray(labels, dtype=np.int32),
        multimask=multimask,
    )


def compose_selected_mask(base_mask: np.ndarray, multi_masks: Sequence[np.ndarray], mask_dropdown: Sequence[str] | None):
    if not multi_masks:
        return base_mask

    selected_names = sorted(mask_dropdown or ["mask_001"])
    template_mask = multi_masks[int(selected_names[0].split("_")[1]) - 1] * int(selected_names[0].split("_")[1])
    for name in selected_names[1:]:
        mask_number = int(name.split("_")[1]) - 1
        template_mask = np.clip(template_mask + multi_masks[mask_number] * (mask_number + 1), 0, mask_number + 1)
    return template_mask


def ensure_non_empty_mask(mask: np.ndarray):
    safe_mask = np.array(mask, copy=True)
    if len(np.unique(safe_mask)) == 1:
        safe_mask[0][0] = 1
    return safe_mask


def build_inference_core(selected_model, performance_profile: str, device_name: str):
    runtime_profile = resolve_performance_profile(performance_profile, device_name)
    runtime_cfg = OmegaConf.create(OmegaConf.to_container(selected_model.cfg, resolve=True))
    with open_dict(runtime_cfg):
        runtime_cfg.max_internal_size = runtime_profile["max_internal_size"]
    return InferenceCore(selected_model, cfg=runtime_cfg, device=device_name), runtime_profile


def run_matting(
    selected_model,
    media_state: dict,
    template_mask: np.ndarray,
    performance_profile: str,
    device_name: str,
    erode_kernel_size: int = 0,
    dilate_kernel_size: int = 0,
    refine_iter: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
):
    processor, runtime_profile = build_inference_core(selected_model, performance_profile, device_name)
    actual_start = media_state["select_frame_number"] if start_frame is None else start_frame
    actual_end = media_state.get("track_end_number") if end_frame is None else end_frame
    following_frames = media_state["origin_images"][actual_start:actual_end] if actual_end else media_state["origin_images"][actual_start:]
    safe_mask = ensure_non_empty_mask(template_mask)
    if refine_iter is None:
        n_warmup = runtime_profile["n_warmup"]
    else:
        refine_value = int(refine_iter)
        n_warmup = min(refine_value, runtime_profile["n_warmup"]) if runtime_profile["device_type"] == "cpu" else refine_value

    foreground, alpha = matanyone2(
        processor,
        following_frames,
        safe_mask * 255,
        r_erode=erode_kernel_size,
        r_dilate=dilate_kernel_size,
        n_warmup=n_warmup,
    )
    return foreground, alpha, runtime_profile


def add_audio_to_video(video_path: str, audio_path: str, output_path: str):
    try:
        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(audio_path)
        (
            ffmpeg.output(video_input, audio_input, output_path, vcodec="copy", acodec="aac")
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        return output_path
    except ffmpeg.Error:
        return None


def generate_video_from_frames(
    frames: Sequence[np.ndarray],
    output_path: str,
    fps: float = 30,
    gray2rgb: bool = False,
    audio_path: str = "",
    target_size=None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    video_temp_path = output_path.replace(".mp4", "_temp.mp4")
    writer = imageio.get_writer(
        video_temp_path,
        fps=fps,
        quality=7,
        codec="libx264",
        macro_block_size=1,
    )

    try:
        for frame in frames:
            frame_np = np.asarray(frame)
            if frame_np.ndim == 2:
                frame_np = frame_np[:, :, None]
            if gray2rgb and frame_np.shape[2] == 1:
                frame_np = np.repeat(frame_np, 3, axis=2)
            frame_np = resize_output_frame(frame_np, target_size, interpolation=cv2.INTER_LINEAR)

            height, width = frame_np.shape[:2]
            even_height = height // 2 * 2
            even_width = width // 2 * 2
            if height != even_height or width != even_width:
                frame_np = cv2.resize(frame_np, (even_width, even_height), interpolation=cv2.INTER_LINEAR)

            writer.append_data(frame_np)
    finally:
        writer.close()

    if audio_path and os.path.exists(audio_path):
        merged_path = add_audio_to_video(video_temp_path, audio_path, output_path)
        if merged_path:
            os.remove(video_temp_path)
            return merged_path
    return video_temp_path


def save_cli_outputs(
    output_dir: str,
    input_path: str,
    source_size,
    mask: np.ndarray,
    sam_preview: Image.Image,
    foreground: Sequence[np.ndarray],
    alpha: Sequence[np.ndarray],
    is_video: bool,
    fps: float = 12,
    audio_path: str = "",
):
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]

    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(output_dir, f"{stem}_mask.png"))
    sam_preview.save(os.path.join(output_dir, f"{stem}_sam_preview.png"))

    if not is_video:
        foreground_frame = resize_output_frame(foreground[-1], source_size)
        alpha_frame = resize_output_frame(alpha[-1], source_size)
        Image.fromarray(foreground_frame).save(os.path.join(output_dir, f"{stem}_foreground.png"))
        Image.fromarray(alpha_frame[:, :, 0]).save(os.path.join(output_dir, f"{stem}_alpha.png"))
        return

    generate_video_from_frames(
        foreground,
        output_path=os.path.join(output_dir, f"{stem}_foreground.mp4"),
        fps=fps,
        audio_path=audio_path,
        target_size=source_size,
    )
    generate_video_from_frames(
        alpha,
        output_path=os.path.join(output_dir, f"{stem}_alpha.mp4"),
        fps=fps,
        gray2rgb=True,
        audio_path=audio_path,
        target_size=source_size,
    )


def sanitize_debug_name(name: str):
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return cleaned.strip("_") or "session"


def create_run_output_dir(base_output_dir: str, media_state: dict):
    source_name = media_state.get("video_name") or media_state.get("image_name") or "session"
    session_tag = str(media_state.get("user_name") or int(time.time()))
    safe_name = sanitize_debug_name(os.path.splitext(source_name)[0])
    safe_session = sanitize_debug_name(session_tag.replace(".", "_"))
    run_dir = os.path.join(base_output_dir, f"{safe_name}_{safe_session}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _save_rgb_frame(frame: np.ndarray, path: str):
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)


def _save_mask(mask: np.ndarray, path: str):
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).save(path)


def export_debug_artifacts(
    output_dir: str,
    media_state: dict,
    template_mask: np.ndarray,
    foreground: Sequence[np.ndarray],
    alpha: Sequence[np.ndarray],
    *,
    device_name: str,
    performance_profile: str,
    model_name: str,
):
    selected_index = media_state.get("select_frame_number", 0)
    selected_frame = media_state["origin_images"][selected_index]
    selected_painted = media_state.get("painted_images", [selected_frame])[selected_index]
    overlay = mask_painter(selected_frame, np.asarray(template_mask).astype("uint8"), mask_color=3)

    _save_rgb_frame(media_state["origin_images"][0], os.path.join(output_dir, "input_first_frame.png"))
    _save_rgb_frame(selected_frame, os.path.join(output_dir, "input_selected_frame.png"))
    _save_rgb_frame(np.asarray(selected_painted), os.path.join(output_dir, "sam_selected_preview.png"))
    _save_rgb_frame(overlay, os.path.join(output_dir, "sam_selected_overlay.png"))
    _save_mask(template_mask, os.path.join(output_dir, "sam_selected_mask.png"))
    _save_rgb_frame(foreground[0], os.path.join(output_dir, "matting_output_first_foreground.png"))
    _save_rgb_frame(alpha[0].squeeze(-1), os.path.join(output_dir, "matting_output_first_alpha.png"))
    _save_rgb_frame(foreground[-1], os.path.join(output_dir, "matting_output_last_foreground.png"))
    _save_rgb_frame(alpha[-1].squeeze(-1), os.path.join(output_dir, "matting_output_last_alpha.png"))

    metadata = {
        "device": device_name,
        "model": model_name,
        "performance_profile": performance_profile,
        "source_name": media_state.get("video_name") or media_state.get("image_name"),
        "selected_frame_number": selected_index,
        "track_end_number": media_state.get("track_end_number"),
        "num_input_frames": len(media_state.get("origin_images") or []),
        "num_output_frames": len(foreground),
        "fps": media_state.get("fps"),
        "source_fps": media_state.get("source_fps"),
        "frame_stride": media_state.get("frame_stride"),
        "source_size": media_state.get("source_size"),
        "working_size": media_state.get("working_size"),
        "audio_path": media_state.get("audio"),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    return output_dir


def parse_point_spec(spec: str, image_shape):
    height, width = image_shape[:2]
    if spec == "center":
        return [width // 2, height // 2]
    x_str, y_str = spec.split(",", 1)
    return [int(x_str), int(y_str)]


class SamMaskGenerator:
    def __init__(self, sam_checkpoint: str, sam_model_type: str, device_name: str):
        self.device = device_name
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.samcontroler = None

    def _ensure_loaded(self):
        if self.samcontroler is None:
            self.samcontroler = SamControler(self.sam_checkpoint, self.sam_model_type, self.device)
        return self.samcontroler

    def prepare_image(self, image: np.ndarray, image_key=None, force: bool = False):
        self._ensure_loaded().prepare_image(image, image_key=image_key, force=force)

    def first_frame_click(self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=True):
        return self._ensure_loaded().first_frame_click(image, points, labels, multimask)

    def release(self):
        if self.samcontroler is None:
            return
        self.samcontroler.release()
        self.samcontroler = None
        gc.collect()
        if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()


class RuntimeModelManager:
    def __init__(self, device_name: str, checkpoint_folder: str):
        self.device = device_name
        self.checkpoint_folder = checkpoint_folder
        self.loaded_models = {}
        self.model_paths = {}
        self.sam_paths = {}

    def get_sam_checkpoint(self, sam_model_type: str):
        if sam_model_type not in self.sam_paths:
            self.sam_paths[sam_model_type] = load_file_from_url(SAM_MODEL_URLS[sam_model_type], self.checkpoint_folder)
        return self.sam_paths[sam_model_type]

    def get_model_path(self, model_file: str):
        if model_file not in self.model_paths:
            self.model_paths[model_file] = load_file_from_url(MODEL_URLS[model_file], self.checkpoint_folder)
        return self.model_paths[model_file]

    def prefetch_available_models(self):
        available_models = []
        for display_name, model_file in MODEL_DISPLAY_TO_FILE.items():
            model_path = self.get_model_path(model_file)
            if os.path.exists(model_path):
                available_models.append(display_name)
        return available_models

    def load_model(self, display_name: str):
        if display_name in MODEL_DISPLAY_TO_FILE:
            model_file = MODEL_DISPLAY_TO_FILE[display_name]
        elif display_name in MODEL_URLS:
            model_file = display_name
        else:
            raise ValueError(f"Unknown model: {display_name}")

        if model_file in self.loaded_models:
            return self.loaded_models[model_file]

        if torch.device(self.device).type == "cpu" and self.loaded_models:
            self.loaded_models.clear()
            gc.collect()

        ckpt_path = self.get_model_path(model_file)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file not found: {ckpt_path}")

        try:
            GlobalHydra.instance().clear()
        except Exception:
            pass

        model = get_matanyone2_model(ckpt_path, self.device)
        model = model.to(self.device).eval()
        self.loaded_models[model_file] = model
        return model
