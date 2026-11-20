"""ComfyUI nodes for Apple Depth Pro 2 metric monocular depth."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar

from .depth_pro_2 import PRESETS, DepthPro2
from .depth_pro_2.model import to_normalized_tensor
from .fp8 import QuantizedLinear, quantize_linears

LOGGER = logging.getLogger(__name__)

MODELS_DIR = Path(folder_paths.models_dir) / "depthpro2"
folder_paths.add_model_folder_path("depthpro2", str(MODELS_DIR))

HF_REPOS = {
    "depthpro2-s": "apple/DepthPro2-S",
    "depthpro2-b": "apple/DepthPro2-B",
    "depthpro2-l": "apple/DepthPro2-L",
    "depthpro2-h": "apple/DepthPro2-H",
}
FOCAL_REPO = "apple/DepthPro2-focal"

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# (embed_dim, num_blocks) -> preset, for local weight files without a config.json.
ARCH_BY_SHAPE = {
    (384, 12): "depthpro2-s",
    (768, 12): "depthpro2-b",
    (1024, 24): "depthpro2-l",
    (1280, 32): "depthpro2-h",
}


def _download_repo(repo_id: str, name: str) -> Path:
    from huggingface_hub import snapshot_download

    local = MODELS_DIR / name
    if not (local / "model.safetensors").exists():
        LOGGER.info("Downloading %s to %s", repo_id, local)
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["config.json", "model.safetensors", "LICENSE"],
            local_dir=str(local),
        )
    return local


def _local_weight_files() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    files = []
    for path in sorted(MODELS_DIR.rglob("*")):
        if path.suffix in (".safetensors", ".pt") and path.is_file():
            files.append(str(path.relative_to(MODELS_DIR)))
    return files


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True)


def _detect_preset(state_dict: dict[str, torch.Tensor]) -> str:
    embed_dim = state_dict["encoder.cls_token"].shape[-1]
    num_blocks = (
        max(
            int(key.split(".")[2])
            for key in state_dict
            if key.startswith("encoder.blocks.")
        )
        + 1
    )
    preset = ARCH_BY_SHAPE.get((embed_dim, num_blocks))
    if preset is None:
        raise ValueError(
            f"Cannot detect DepthPro2 size from weights (embed_dim={embed_dim}, "
            f"blocks={num_blocks}). Place a config.json next to the weight file."
        )
    return preset


def _resolve_weights(name: str) -> tuple[dict, dict[str, torch.Tensor], Path | None]:
    """Return (config kwargs, state dict, focal weights path or None) for a model choice."""
    if name in HF_REPOS:
        local = _download_repo(HF_REPOS[name], name)
        config = json.loads((local / "config.json").read_text())
        state_dict = _load_state_dict(local / "model.safetensors")
    else:
        path = MODELS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Weight file not found: {path}")
        state_dict = _load_state_dict(path)
        config_path = path.with_name("config.json")
        if config_path.exists():
            config = json.loads(config_path.read_text())
        else:
            config = vars(PRESETS[_detect_preset(state_dict)]).copy()
            config = {
                key: config[key]
                for key in (
                    "backbone_preset",
                    "reassembled_blocks",
                    "cross_frame_blocks",
                    "dims_reassemble",
                    "dim_decoder",
                )
            }
    has_fov = any(key.startswith("fov.") for key in state_dict)
    focal_path = None if has_fov else _download_repo(FOCAL_REPO, "depthpro2-focal") / "model.safetensors"
    return config, state_dict, focal_path


def _cast_storage(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Cast floating params/buffers to ``dtype``, leaving fp8-quantized modules untouched."""
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            continue
        for name, param in module.named_parameters(recurse=False):
            param.data = param.data.to(dtype)
        for name, buf in module.named_buffers(recurse=False):
            if buf is not None and buf.is_floating_point():
                module._buffers[name] = buf.to(dtype)


def load_depthpro2(name: str, precision: str, device: str) -> dict:
    if device == "auto":
        torch_device = mm.get_torch_device()
    else:
        torch_device = torch.device(device)
    if precision == "auto":
        precision = "fp32" if torch_device.type == "cpu" else "bf16"

    config, state_dict, focal_path = _resolve_weights(name)
    compute_dtype = DTYPES["bf16" if precision == "fp8" else precision]
    if precision == "fp8" and torch_device.type == "cpu":
        compute_dtype = torch.float32

    model = DepthPro2(
        **{key: value for key, value in config.items() if key != "fov"},
        fov=focal_path is not None or any(k.startswith("fov.") for k in state_dict),
        precision=compute_dtype,
    )
    if focal_path is not None:
        state_dict.update({f"fov.{k}": v for k, v in _load_state_dict(focal_path).items()})
    model.load_state_dict(state_dict, strict=True)
    del state_dict

    if precision == "fp8":
        count = quantize_linears(model.encoder, compute_dtype)
        LOGGER.info("DepthPro2: quantized %d linear layers to fp8.", count)
    if compute_dtype != torch.float32:
        # Store the autocast-covered parts in low precision; the heads run in float32.
        _cast_storage(model.encoder, compute_dtype)
        _cast_storage(model.decoder, compute_dtype)
        if model.fov is not None:
            _cast_storage(model.fov.trunk, compute_dtype)

    model.to(torch_device).eval()
    return {"model": model, "device": torch_device, "name": name, "precision": precision}


def _prepare_frames(
    images: torch.Tensor, device: torch.device, max_size: int, f_px: float
) -> tuple[list[torch.Tensor], float | None, float]:
    """ComfyUI IMAGE batch (T, H, W, 3) float 0..1 -> normalized frame tensors and focal."""
    height, width = images.shape[1:3]
    scale = 1.0
    if max_size > 0 and max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_h = max(32, round(height * scale))
        new_w = max(32, round(width * scale))
        images = F.interpolate(
            images.permute(0, 3, 1, 2), size=(new_h, new_w), mode="bilinear", antialias=True
        ).permute(0, 2, 3, 1)
    frames = [
        to_normalized_tensor((frame.clamp(0, 1) * 255).round().to(torch.uint8).numpy(), device)
        for frame in images
    ]
    focal = None if f_px <= 0 else f_px * scale
    return frames, focal, scale


def _normalize_inverse_depth(inverse_depth: torch.Tensor) -> torch.Tensor:
    """Per-frame min-max to 0..1, near white; sky (inverse depth 0) stays black."""
    finite = torch.isfinite(inverse_depth)
    out = torch.zeros_like(inverse_depth)
    for i in range(inverse_depth.shape[0]):
        values = inverse_depth[i][finite[i]]
        if values.numel() == 0:
            continue
        lo, hi = values.min(), values.max()
        if hi > lo:
            out[i] = (inverse_depth[i] - lo) / (hi - lo)
    return out.clamp(0, 1)


def _depth_outputs(
    predictions: list[dict], original_size: tuple[int, int], scale: float
) -> tuple[torch.Tensor, torch.Tensor, float]:
    inverse = torch.stack([p["inverse_depth"].float().cpu() for p in predictions])
    depth = torch.stack(
        [
            p["depth"].float().cpu()
            if p["depth"] is not None
            else torch.full_like(p["inverse_depth"], float("nan"))
            for p in predictions
        ]
    )
    if inverse.shape[-2:] != original_size:
        inverse = F.interpolate(inverse[:, None], size=original_size, mode="bilinear")[:, 0]
        depth = F.interpolate(depth[:, None], size=original_size, mode="bilinear")[:, 0]
    image = _normalize_inverse_depth(inverse).unsqueeze(-1).repeat(1, 1, 1, 3)
    focal = 0.0
    if predictions[0]["focallength_px"] is not None:
        # Report in pixels of the original input, undoing any max_size downscale.
        focal = float(predictions[0]["focallength_px"]) / scale
    return image, depth, focal


class DepthPro2Loader:
    @classmethod
    def INPUT_TYPES(cls):
        choices = list(HF_REPOS) + _local_weight_files()
        return {
            "required": {
                "model": (choices, {"default": "depthpro2-s"}),
                "precision": (["auto", "bf16", "fp16", "fp32", "fp8"], {"default": "auto"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("DEPTHPRO2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "depth/DepthPro2"
    DESCRIPTION = (
        "Load Apple Depth Pro 2. Preset names download weights from Hugging Face into "
        "models/depthpro2 on first use; local .safetensors/.pt files in that folder are also "
        "listed. fp8 stores encoder weights as float8 with per-channel scales (about half the "
        "memory of bf16); the focal-length estimator stays in high precision."
    )

    def load(self, model: str, precision: str, device: str):
        return (load_depthpro2(model, precision, device),)


class DepthPro2Estimate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DEPTHPRO2_MODEL",),
                "image": ("IMAGE",),
                "f_px": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
                "max_size": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32}),
            }
        }

    RETURN_TYPES = ("IMAGE", "DEPTHPRO2_DEPTH", "FLOAT")
    RETURN_NAMES = ("depth_image", "depth_meters", "focal_length_px")
    FUNCTION = "estimate"
    CATEGORY = "depth/DepthPro2"
    DESCRIPTION = (
        "Estimate metric depth for each image of the batch independently. depth_image is "
        "normalized inverse depth (near white, far black) for preview/ControlNet; depth_meters "
        "is raw metric depth (T, H, W) with inf at sky. f_px is the focal length in pixels of "
        "the input image; 0 estimates it with the focal-length model. max_size caps the longest "
        "side processed (0 = native resolution, sharpest edges)."
    )

    def estimate(self, model: dict, image: torch.Tensor, f_px: float, max_size: int):
        net, device = model["model"], model["device"]
        frames, focal, scale = _prepare_frames(image, device, max_size, f_px)
        progress = ProgressBar(len(frames))
        predictions = []
        for frame in frames:
            predictions.append(net.infer(frame, f_px=focal))
            progress.update(1)
        mm.soft_empty_cache()
        return _depth_outputs(predictions, tuple(image.shape[1:3]), scale)


class DepthPro2Video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DEPTHPRO2_MODEL",),
                "images": ("IMAGE",),
                "window": ("INT", {"default": 16, "min": 1, "max": 64}),
                "overlap": ("INT", {"default": 4, "min": 0, "max": 63}),
                "f_px": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
                "max_size": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32}),
            }
        }

    RETURN_TYPES = ("IMAGE", "DEPTHPRO2_DEPTH", "FLOAT")
    RETURN_NAMES = ("depth_image", "depth_meters", "focal_length_px")
    FUNCTION = "estimate"
    CATEGORY = "depth/DepthPro2"
    DESCRIPTION = (
        "Temporally consistent depth for a video batch: frames are processed jointly in "
        "overlapping windows and share one focal length. Lower window/overlap if you run out "
        "of memory. depth_image is normalized over the whole batch so brightness is steady "
        "across frames."
    )

    def estimate(
        self,
        model: dict,
        images: torch.Tensor,
        window: int,
        overlap: int,
        f_px: float,
        max_size: int,
    ):
        if overlap >= window:
            raise ValueError("overlap must be smaller than window.")
        net, device = model["model"], model["device"]
        frames, focal, scale = _prepare_frames(images, device, max_size, f_px)
        progress = ProgressBar(len(frames))
        predictions = []
        for prediction in net.infer_video(iter(frames), window=window, overlap=overlap, f_px=focal):
            predictions.append(prediction)
            progress.update(1)
        mm.soft_empty_cache()
        image, depth, focal_out = _depth_outputs(predictions, tuple(images.shape[1:3]), scale)
        # One shared range keeps the visualization temporally steady.
        finite = torch.isfinite(depth) & (depth > 0)
        if finite.any():
            inverse = 1.0 / depth.clamp(min=1e-6)
            inverse[~finite] = 0.0
            lo = inverse[finite].min()
            hi = inverse[finite].max()
            if hi > lo:
                image = ((inverse - lo) / (hi - lo)).clamp(0, 1).unsqueeze(-1).repeat(1, 1, 1, 3)
        return image, depth, focal_out


NODE_CLASS_MAPPINGS = {
    "DepthPro2Loader": DepthPro2Loader,
    "DepthPro2Estimate": DepthPro2Estimate,
    "DepthPro2Video": DepthPro2Video,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthPro2Loader": "DepthPro2 Loader",
    "DepthPro2Estimate": "DepthPro2 Estimate Depth",
    "DepthPro2Video": "DepthPro2 Video Depth",
}