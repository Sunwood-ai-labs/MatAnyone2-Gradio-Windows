import argparse
import os
import sys
from typing import Iterable

import cv2
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUGGING_FACE_DIR = os.path.join(REPO_DIR, "hugging_face")

for path in (REPO_DIR, HUGGING_FACE_DIR):
    if path not in sys.path:
        sys.path.append(path)

from hugging_face.matanyone2_wrapper import matanyone2
from hugging_face.tools.download_util import load_file_from_url
from hugging_face.tools.interact_tools import SamControler
from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.device import set_default_device
from matanyone2.utils.get_default_model import get_matanyone2_model


PERFORMANCE_PROFILES = {
    "quality": {
        "video_max_short_side": 1080,
        "image_max_short_side": 1536,
        "max_internal_size": -1,
        "n_warmup": 10,
    },
    "balanced": {
        "video_max_short_side": 720,
        "image_max_short_side": 1024,
        "max_internal_size": 768,
        "n_warmup": 4,
    },
    "fast": {
        "video_max_short_side": 640,
        "image_max_short_side": 832,
        "max_internal_size": 512,
        "n_warmup": 2,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SAM -> MatAnyone pipeline without Gradio.")
    parser.add_argument("--input", required=True, help="Path to an input image or video.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device.")
    parser.add_argument("--performance_profile", default="fast", choices=list(PERFORMANCE_PROFILES.keys()))
    parser.add_argument("--sam_model_type", default="auto", choices=["auto", "vit_h", "vit_l", "vit_b"])
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--frame_limit", type=int, default=8, help="Max number of video frames to process.")
    parser.add_argument("--output_dir", default=os.path.join(REPO_DIR, "results", "pipeline-check"))
    parser.add_argument("--positive_point", default="center", help='Either "center" or "x,y".')
    parser.add_argument("--negative_point", action="append", default=[], help='Optional "x,y" points.')
    parser.add_argument("--model_checkpoint", default=os.path.join(REPO_DIR, "pretrained_models", "matanyone2.pth"))
    parser.add_argument("--sam_checkpoint_dir", default=os.path.join(REPO_DIR, "pretrained_models"))
    return parser.parse_args()


def resolve_sam_model_type(requested, device):
    if requested != "auto":
        return requested
    return "vit_b" if device == "cpu" else "vit_h"


def configure_runtime(device, cpu_threads):
    set_default_device(device)
    if device != "cpu":
        return
    torch.set_num_threads(max(1, cpu_threads))
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
        except RuntimeError:
            pass
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True
    cv2.setNumThreads(max(1, min(4, cpu_threads)))


def maybe_resize_frame(frame, max_short_side):
    if max_short_side <= 0:
        return frame
    height, width = frame.shape[:2]
    short_side = min(height, width)
    if short_side <= max_short_side:
        return frame
    scale = max_short_side / float(short_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def resize_output_frame(frame, target_size):
    target_h, target_w = target_size
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        return frame
    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if frame.ndim == 3 and frame.shape[2] == 1 and resized.ndim == 2:
        resized = resized[:, :, None]
    return resized


def parse_point(spec, image_shape):
    height, width = image_shape[:2]
    if spec == "center":
        return [width // 2, height // 2]
    x_str, y_str = spec.split(",", 1)
    return [int(x_str), int(y_str)]


def load_input_frames(input_path, profile, frame_limit):
    if input_path.lower().endswith((".jpg", ".jpeg", ".png")):
        image = np.array(Image.open(input_path).convert("RGB"))
        image = maybe_resize_frame(image, profile["image_max_short_side"])
        return [image, image.copy()], (image.shape[0], image.shape[1]), False

    cap = cv2.VideoCapture(input_path)
    frames = []
    source_size = None
    while cap.isOpened() and len(frames) < frame_limit:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if source_size is None:
            source_size = (rgb.shape[0], rgb.shape[1])
        frames.append(maybe_resize_frame(rgb, profile["video_max_short_side"]))
    cap.release()
    if not frames:
        raise ValueError(f"No frames could be loaded from {input_path}")
    return frames, source_size, True


def build_mask(frame, sam_model_type, sam_checkpoint_dir, device, positive_point, negative_points):
    sam_model_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    checkpoint_path = load_file_from_url(sam_model_urls[sam_model_type], sam_checkpoint_dir)

    controller = SamControler(checkpoint_path, sam_model_type, device)
    controller.prepare_image(frame, image_key="pipeline-check", force=True)
    point_list = [parse_point(positive_point, frame.shape)]
    label_list = [1]
    for item in negative_points:
        point_list.append(parse_point(item, frame.shape))
        label_list.append(0)

    mask, _logit, painted = controller.first_frame_click(
        frame,
        np.array(point_list, dtype=np.int32),
        np.array(label_list, dtype=np.int32),
        multimask=True,
    )
    controller.release()
    return mask.astype(np.uint8), painted


def build_processor(model_checkpoint, device, performance_profile):
    model = get_matanyone2_model(model_checkpoint, device).eval()
    cfg = OmegaConf.create(OmegaConf.to_container(model.cfg, resolve=True))
    with open_dict(cfg):
        cfg.max_internal_size = performance_profile["max_internal_size"]
    return InferenceCore(model, cfg=cfg, device=device), performance_profile["n_warmup"]


def save_outputs(output_dir, input_path, source_size, mask, painted, foreground, alpha, is_video):
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]

    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(os.path.join(output_dir, f"{stem}_mask.png"))
    painted.save(os.path.join(output_dir, f"{stem}_sam_preview.png"))

    if not is_video:
        foreground_frame = resize_output_frame(foreground[-1], source_size)
        alpha_frame = resize_output_frame(alpha[-1], source_size)
        Image.fromarray(foreground_frame).save(os.path.join(output_dir, f"{stem}_foreground.png"))
        Image.fromarray(alpha_frame[:, :, 0]).save(os.path.join(output_dir, f"{stem}_alpha.png"))
        return

    foreground_path = os.path.join(output_dir, f"{stem}_foreground.mp4")
    alpha_path = os.path.join(output_dir, f"{stem}_alpha.mp4")

    with imageio.get_writer(foreground_path, fps=12, codec="libx264", quality=7, macro_block_size=1) as writer:
        for frame in foreground:
            writer.append_data(resize_output_frame(frame, source_size))

    with imageio.get_writer(alpha_path, fps=12, codec="libx264", quality=7, macro_block_size=1) as writer:
        for frame in alpha:
            alpha_rgb = np.repeat(resize_output_frame(frame, source_size), 3, axis=2)
            writer.append_data(alpha_rgb)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    configure_runtime(args.device, args.cpu_threads)
    profile = PERFORMANCE_PROFILES[args.performance_profile]
    sam_model_type = resolve_sam_model_type(args.sam_model_type, args.device)
    frames, source_size, is_video = load_input_frames(args.input, profile, args.frame_limit)
    mask, painted = build_mask(
        frames[0],
        sam_model_type,
        args.sam_checkpoint_dir,
        args.device,
        args.positive_point,
        args.negative_point,
    )
    processor, n_warmup = build_processor(args.model_checkpoint, args.device, profile)
    foreground, alpha = matanyone2(processor, frames, mask * 255, n_warmup=n_warmup)
    save_outputs(args.output_dir, args.input, source_size, mask, painted, foreground, alpha, is_video)
    print(f"Pipeline check completed. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
