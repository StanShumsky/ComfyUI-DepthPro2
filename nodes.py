"""ComfyUI nodes for Apple Depth Pro 2 metric monocular depth."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors import SafetensorError, safe_open

import folder_paths
import comfy.model_management as mm
import comfy.model_patcher
from comfy.utils import ProgressBar
from comfy_api.latest import io, ui
from tqdm.auto import tqdm

from .depth_pro_2 import PRESETS, DepthPro2
from .depth_pro_2.model import to_normalized_tensor
from .depth_pro_2.network.fov import FOVEstimator
from .colormaps import COLORMAPS, color_scale, colorize, inverse_metric
from .fp8 import QuantizedLinear, quantize_linears

LOGGER = logging.getLogger(__name__)

MODELS_DIR = Path(folder_paths.models_dir) / "depthpro2"
folder_paths.add_model_folder_path("depthpro2", str(MODELS_DIR))

# Native placement: the loader sits with the model loaders, the rest with the
# geometry estimation models (the same split the built-in ComfyUI depth nodes use).
LOADER_CATEGORY = "model/loaders"
CATEGORY = "image/geometry estimation"
DEPTHPRO2_MODEL = io.Custom("DEPTHPRO2_MODEL")
DEPTHPRO2_DEPTH = io.Custom("DEPTHPRO2_DEPTH")

HF_REPOS = {
    "depthpro2-s": "apple/DepthPro2-S",
    "depthpro2-b": "apple/DepthPro2-B",
    "depthpro2-l": "apple/DepthPro2-L",
    "depthpro2-h": "apple/DepthPro2-H",
}
FOCAL_REPO = "apple/DepthPro2-focal"
HF_REVISIONS = {
    "apple/DepthPro2-S": "903a1c1ade918831d138adcc4aedbcc8f4da0078",
    "apple/DepthPro2-B": "85e67fdfb7f5b36a6f6c4495e6789d331b1d8a04",
    "apple/DepthPro2-L": "849dc5461267e0bc006ec63ab46920ea1ed98702",
    "apple/DepthPro2-H": "68426c789f75a7c760273e857222625b4fe69576",
    "apple/DepthPro2-focal": "622659d44a2002cf9ae4333f2569afa938290d9a",
}

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# (embed_dim, num_blocks) -> preset, for local weight files without a config.json.
ARCH_BY_SHAPE = {
    (384, 12): "depthpro2-s",
    (768, 12): "depthpro2-b",
    (1024, 24): "depthpro2-l",
    (1280, 32): "depthpro2-h",
}


def _model_dirs() -> list[Path]:
    """Folders searched for weights, in priority order; the first one is where downloads go."""
    dirs: list[Path] = [Path(p) for p in folder_paths.get_folder_paths("depthpro2")]
    if MODELS_DIR not in dirs:
        dirs.append(MODELS_DIR)
    unique: list[Path] = []
    for d in dirs:
        if d not in unique:
            unique.append(d)
    return unique


def _find_model_dir(name: str) -> Path | None:
    """The folder of an already downloaded model, searching every model dir."""
    for base in _model_dirs():
        if (base / name / "model.safetensors").exists():
            return base / name
    return None


def _quarantine_weight(path: Path) -> None:
    """Preserve an unreadable weight beside the fresh download instead of deleting user data."""
    if not path.exists():
        return
    backup = path.with_name(f"{path.name}.corrupt-{time.time_ns()}")
    path.replace(backup)
    LOGGER.warning("DepthPro2: preserved unreadable weights as %s", backup)


def _download_repo(repo_id: str, name: str, force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    local = _model_dirs()[0] / name
    if force:
        _quarantine_weight(local / "model.safetensors")
    else:
        found = _find_model_dir(name)
        if found is not None:
            return found
    local.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s to %s", repo_id, local)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=["config.json", "model.safetensors", "LICENSE"],
        local_dir=str(local),
        force_download=force,
        revision=HF_REVISIONS[repo_id],
    )
    if not (local / "model.safetensors").exists():
        raise FileNotFoundError(f"Download of {repo_id} did not produce {local / 'model.safetensors'}.")
    return local


def _local_weight_files() -> list[str]:
    files: list[str] = []
    for base in _model_dirs():
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in (".safetensors", ".pt") or not path.is_file():
                continue
            rel = path.relative_to(base)
            # Downloads of the presets (depthpro2-s/model.safetensors, ...) and the focal
            # estimator are already offered by name or loaded automatically.
            if ".cache" in rel.parts or rel.parts[0] in (*HF_REPOS, "depthpro2-focal"):
                continue
            if str(rel) not in files:
                files.append(str(rel))
    return files


def _find_local_file(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Weight file path must stay inside the model directories: {name!r}.")
    for base in _model_dirs():
        root = base.resolve()
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise ValueError(f"Weight file path resolves outside the model directories: {name!r}.") from None
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Weight file {name!r} not found in any of: {[str(d) for d in _model_dirs()]}"
    )


def _corrupt_safetensors(message: str, path: str, incomplete: bool = True) -> ValueError:
    """The native corrupt-file diagnostic, with the file path, so the UI shows the same
    wording for this pack's weights as for the native models' (e.g. a truncated download
    is reported exactly as ComfyUI reports one for the built-in loaders). The wording is
    sourced from comfy.utils so it can never drift from what ComfyUI itself reports."""
    try:
        from comfy.utils import _incomplete_safetensors_error, _invalid_safetensors_error
    except ImportError:  # outside ComfyUI (unit tests); wording kept in sync with comfy.utils
        if incomplete:
            hint = (
                "The safetensors file is corrupt/incomplete. Check the file size and make sure "
                "you have copied/downloaded it correctly."
            )
        else:
            hint = (
                "The safetensors file is corrupt or invalid. Make sure this is actually a "
                "safetensors file and not a ckpt or pt or other filetype."
            )
        return ValueError(f"{message}\n\nFile path: {path}\n\n{hint}")
    return (_incomplete_safetensors_error if incomplete else _invalid_safetensors_error)(message, path)


def _check_weights(state_dict: dict[str, torch.Tensor], name: str, path: str) -> None:
    """Raise if the weights are visibly broken, e.g. a truncated or zero-filled download.

    Such a file still loads with strict=True (all keys and shapes are there) but predicts
    garbage, typically a black depth image, so it is caught here instead, with the same
    diagnostics ComfyUI's native loaders report.
    """
    nonfinite, zeros = [], []
    for key, value in state_dict.items():
        if not value.is_floating_point() or value.numel() <= 100:
            continue
        if not torch.isfinite(value).all():
            nonfinite.append(key)
        elif not value.any():
            zeros.append(key)
    if nonfinite or len(zeros) >= 3:
        bad = (nonfinite + zeros)[:3]
        raise _corrupt_safetensors(
            f"{name}: weights look corrupt ({len(nonfinite)} tensors with NaN/inf, "
            f"{len(zeros)} all-zero tensors, e.g. {bad}).",
            path,
            incomplete=False,
        )


def _load_repo_verified(
    repo_id: str, name: str, dtype: torch.dtype
) -> tuple[Path, dict[str, torch.Tensor]]:
    """Load a downloaded model in ``dtype``; if it is unreadable or corrupt, re-download once."""
    last_error: Exception | None = None
    path = ""
    for attempt in range(2):
        local = _download_repo(repo_id, name, force=attempt == 1)
        path = str(local / "model.safetensors")
        try:
            state_dict = _load_state_dict(local / "model.safetensors", dtype)
            _check_weights(state_dict, name, path)
            return local, state_dict
        except (SafetensorError, ValueError) as error:
            last_error = error
            if attempt == 0:
                LOGGER.warning("DepthPro2: %s; re-downloading %s.", error, name)
    raise _corrupt_safetensors(
        f"{name} is still unusable after a fresh download from {repo_id}: {last_error}. "
        f"Check free disk space in {_model_dirs()[0]} and your connection.",
        path,
    ) from last_error


def _load_state_dict(path: Path, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Load the weights straight into ``dtype`` so no fp32 copy ever lives in RAM.

    The native ComfyUI model loaders build the model in the compute dtype on the
    offload device; the fp32 file contents are read from the memory map and converted
    tensor by tensor instead of being resident a second time.
    """
    if path.suffix == ".safetensors":
        state: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                value = f.get_tensor(key)
                state[key] = value.to(dtype) if value.is_floating_point() else value
        return state
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key: value.to(dtype) if value.is_floating_point() else value for key, value in state.items()}


def _annotate_source(state_dict: dict[str, torch.Tensor], path: str) -> None:
    """Tag the weights with their file path, for ComfyUI's fast-storage detection."""
    try:
        import comfy.storage

        comfy.storage.annotate_state_dict(state_dict, path)
    except Exception:
        pass


def _fast_disk(state_dict: dict[str, torch.Tensor]) -> bool:
    """Whether the weights live on fast storage, as the native model loaders report it."""
    try:
        import comfy.storage

        return bool(comfy.storage.state_dict_fast_disk(state_dict))
    except Exception:
        return False


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


def _resolve_weights(
    name: str, dtype: torch.dtype
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Return config kwargs and the main-model state dict in ``dtype``."""
    if name in HF_REPOS:
        local, state_dict = _load_repo_verified(HF_REPOS[name], name, dtype)
        _annotate_source(state_dict, str(local / "model.safetensors"))
        config = json.loads((local / "config.json").read_text())
    else:
        path = _find_local_file(name)
        try:
            state_dict = _load_state_dict(path, dtype)
        except FileNotFoundError:
            raise
        except Exception as error:  # truncated/corrupt file: native-style diagnostic
            raise _corrupt_safetensors(f"Could not load the weights: {error}", str(path)) from error
        _check_weights(state_dict, name, str(path))
        _annotate_source(state_dict, str(path))
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
    return config, state_dict


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
        # The native ComfyUI device policy for auxiliary models: honors --gpu-only,
        # aimdo and the session VRAM state (a NO_VRAM session runs the model on the CPU).
        load_device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()
        # Text encoders stay on CPU in SHARED sessions; this model should use the accelerator.
        if load_device.type == "cpu" and mm.vram_state == mm.VRAMState.SHARED:
            accelerator = mm.get_torch_device()
            if accelerator.type != "cpu":
                load_device = accelerator
    else:
        load_device = torch.device(device)
        offload_device = torch.device("cpu")
    if precision == "auto":
        # ComfyUI's session dtype for auxiliary models (honors --fp16-text-enc &
        # friends; fp8 text-enc flags fall back to fp16). CPU stays fp32: half-precision
        # autocast is much slower there, as in Apple's reference loading.
        if load_device.type == "cpu":
            precision = "fp32"
        else:
            auto = mm.text_encoder_dtype(load_device)
            precision = "bf16" if auto == torch.bfloat16 else "fp16"

    compute_dtype = DTYPES["bf16" if precision == "fp8" else precision]
    if precision == "fp8" and load_device.type == "cpu":
        compute_dtype = torch.float32
    config, state_dict = _resolve_weights(name, compute_dtype)
    preset = _detect_preset(state_dict)
    has_fov = any(key.startswith("fov.") for key in state_dict)

    # Build in system RAM and hand the weights to ComfyUI's model management (the
    # native pattern for auxiliary models): the model rides in the offload device until
    # a run loads it onto the GPU, and is evicted again when other nodes need the VRAM.
    # The storage
    # is cast to the compute dtype before the load, so the low-precision copy is the
    # only one in RAM (the native pattern: the model is built in the compute dtype on
    # the offload device, never in fp32).
    model = DepthPro2(
        **{key: value for key, value in config.items() if key != "fov"},
        fov=has_fov,
        precision=compute_dtype,
    )
    if compute_dtype != torch.float32:
        # Store the autocast-covered parts in low precision; the heads run in float32.
        _cast_storage(model.encoder, compute_dtype)
        _cast_storage(model.decoder, compute_dtype)
        if model.fov is not None:
            _cast_storage(model.fov.trunk, compute_dtype)
    model.load_state_dict(state_dict, strict=True)
    fast_disk = _fast_disk(state_dict)
    del state_dict

    if precision == "fp8":
        count = quantize_linears(model.encoder, compute_dtype)
        LOGGER.info("DepthPro2: quantized %d linear layers to fp8.", count)

    model.eval()
    patcher = comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        fast_disk=fast_disk,
    )
    wrapper = {
        "model": model,
        "patcher": patcher,
        "device": load_device,
        "name": preset,
        "precision": precision,
        "focal_pending": not has_fov,
        "_focal_lock": threading.Lock(),
    }
    model._focal_loader = lambda: _ensure_focal_model(wrapper)
    return wrapper


def _ensure_focal_model(model: dict) -> None:
    """Load the separate focal estimator once, only when inference needs it."""
    net = model["model"]
    if net.fov is not None or not model.get("focal_pending"):
        return
    with model["_focal_lock"]:
        if net.fov is not None:
            return
        patcher = model["patcher"]
        if patcher in mm.loaded_models():
            patcher.detach()
        precision = model["precision"]
        compute_dtype = DTYPES["bf16" if precision == "fp8" else precision]
        if precision == "fp8" and model["device"].type == "cpu":
            compute_dtype = torch.float32
        focal_local, focal_state = _load_repo_verified(
            FOCAL_REPO, "depthpro2-focal", compute_dtype
        )
        _annotate_source(focal_state, str(focal_local / "model.safetensors"))
        fov = FOVEstimator(compute_dtype)
        if compute_dtype != torch.float32:
            _cast_storage(fov.trunk, compute_dtype)
        fov.load_state_dict(focal_state, strict=True)
        net.fov = fov.to(net.weight_device).eval()
        patcher.size = 0
        patcher.fast_disk = getattr(patcher, "fast_disk", False) and _fast_disk(focal_state)
        model["focal_pending"] = False
        del focal_state


# Boundary MP per preset: the input size at which one bf16 image forward consumes the
# whole free VRAM of the reference card (32 GiB, ~29.8 GiB free). Derived analytically
# from the architecture (stage-by-stage live-tensor accounting: ViT token sequence, the
# 4-level reassembly pyramid, the DPT decoder fusions, the fp32 inverse-depth head) in
# scripts/derive_boundaries.py, whose single correction factor (1.74x, for cudnn
# workspaces, autocast accumulation and allocator variance) is fitted to the one
# reliable peak measurement on file (L: 24 GiB torch peak at 16 MP). The derived
# boundaries cross-check against the other observations: S predicts 17.0 GiB at 32 MP
# vs ~19.4 GiB reported (-12%, hence the floor below the 51.4 MP derivation), and B
# derives 32.3 MP vs its observed OOM window of 24-32 MP (hence 28). H shares L's
# decoder/head and takes the extra headroom for its 1.63 GiB of weights. These anchor
# both the "auto" max_size policy, which scales the cap with the VRAM actually free at
# run time, and the activation estimate reserved through ComfyUI's model management
# before the run. Local weight files of unknown preset use the conservative boundary.
AUTO_BOUNDARY_MPS = {
    "depthpro2-s": 48.0,
    "depthpro2-b": 28.0,
    "depthpro2-l": 18.0,
    "depthpro2-h": 16.0,
}
AUTO_BOUNDARY_UNKNOWN = 16.0
# Extra driver memory per video window frame per input MP (bf16): buffered and
# unbuffered windows both keep `window` frames' tokens in one cross-frame attention.
VIDEO_WINDOW_COST = {
    "depthpro2-s": 0.037,
    "depthpro2-b": 0.074,
    "depthpro2-l": 0.099,
    "depthpro2-h": 0.123,
}
VIDEO_WINDOW_COST_UNKNOWN = 0.074


def _window_boundary(name: str, window: int) -> float:
    """The boundary MP with the per-window-frame cross-frame attention cost accounted for."""
    boundary = AUTO_BOUNDARY_MPS.get(name, AUTO_BOUNDARY_UNKNOWN)
    if window > 1:
        cost = VIDEO_WINDOW_COST.get(name, VIDEO_WINDOW_COST_UNKNOWN)
        boundary = boundary / (1.0 + cost * (window - 1) * boundary / AUTO_REF_FREE_GIB)
    return boundary


# Free VRAM (GiB) during the boundary runs: 32 GiB card minus the ~2 GiB baseline.
AUTO_REF_FREE_GIB = 29.8
# Headroom kept out of the free VRAM for allocator variance and session overhead.
AUTO_MARGIN_GIB = 2.5
MIN_AUTO_MPS = 2.0


def _mps_available_memory() -> int | None:
    """Metal working-set headroom, excluding reusable MPS allocator caches."""
    recommended = getattr(torch.mps, "recommended_max_memory", None)
    active = getattr(torch.mps, "current_allocated_memory", None)
    if not callable(recommended) or not callable(active):
        return None
    return max(0, recommended() - active())


def _auto_max_size(model: dict, height: int, width: int, window: int = 1) -> int:
    """The "auto" max_size policy: largest longest side that fits available device memory.

    Returns 0 when the native resolution fits or no device budget is available.
    """
    device = model["device"]
    if device.type not in ("cuda", "mps"):
        return 0
    boundary = _window_boundary(model["name"], window)
    if device.type == "mps":
        mps_free = _mps_available_memory()
        if mps_free is None:
            return 0
        free_bytes = mps_free
    else:
        free_bytes = mm.get_free_memory(device)
    free_gib = free_bytes / 2**30
    if model["patcher"] not in mm.loaded_models():
        # The weights are not on the GPU yet: account for them so the policy reads the
        # same whether measured before or after the model load.
        free_gib -= model["patcher"].model_size() / 2**30
    budget = free_gib - AUTO_MARGIN_GIB
    if budget <= 0.0:
        raise RuntimeError(
            f"DepthPro2: only {free_gib:.1f} GiB of device memory is free; the model needs at "
            f"least ~{AUTO_MARGIN_GIB + 1:.0f} GiB headroom. Free other models, choose a "
            "smaller preset, or use the fp8 precision."
        )
    native_mps = height * width / 1e6
    allowed_mps = boundary * budget / AUTO_REF_FREE_GIB
    if allowed_mps >= native_mps:
        return 0
    longest, shortest = max(height, width), min(height, width)
    side = (allowed_mps * 1e6 * longest / shortest) ** 0.5
    min_side = (MIN_AUTO_MPS * 1e6 * longest / shortest) ** 0.5
    return max(int(side // 32 * 32), int(min_side // 32 * 32))


def _resolve_max_size(model: dict, image: torch.Tensor, max_size: int, window: int = 1) -> int:
    """Widget value -> processing longest-side cap (0 = budget-aware auto, negative = native)."""
    if max_size != 0:
        return 0 if max_size < 0 else max_size
    capped = _auto_max_size(model, *image.shape[1:3], window=window)
    if capped > 0:
        LOGGER.info(
            "DepthPro2: auto resolution capped %dx%d to longest side %d to fit available device memory.",
            image.shape[2], image.shape[1], capped,
        )
    return capped


def _activation_bytes(model: dict, height: int, width: int, side: int, window: int = 1) -> int:
    """Estimated activation VRAM (bytes) of one forward at the resolved processing size.

    Anchored on the boundary runs (see AUTO_BOUNDARY_MPS): at the boundary MP the session
    consumed the reference card's whole free VRAM, so the non-weight cost per MP is
    (ref free - weights) / boundary. Activations scale linearly with the megapixels
    (flash attention is linear in the token count), and the estimate is scaled by the
    compute dtype versus the bf16 calibration.
    """
    device = model["device"]
    if device.type != "cuda":
        return 0
    boundary = _window_boundary(model["name"], window)
    longest = max(height, width)
    if side > 0 and longest > side:
        scale = side / longest
        mps = height * width * scale * scale / 1e6
    else:
        mps = height * width / 1e6
    weights_gib = model["patcher"].model_size() / 2**30
    per_mp_gib = max(0.0, AUTO_REF_FREE_GIB - weights_gib) / boundary
    itemsize = 4 if model["precision"] == "fp32" else 2
    return int(per_mp_gib * mps * (itemsize / 2.0) * 2**30)


def _focal_mm_to_px(focal_mm: float, height: int, width: int) -> float:
    """35mm-equivalent focal length -> pixels of the input image (35mm frame: 36x24mm)."""
    if not math.isfinite(focal_mm) or focal_mm < 0:
        raise ValueError("focal_length_mm must be a finite non-negative number.")
    if focal_mm == 0:
        return 0.0
    return focal_mm * math.hypot(width, height) / math.hypot(36.0, 24.0)


def _resolve_focal_px(
    focal_mm: float, height: int, width: int, focal_px: float | None
) -> float:
    if focal_px is None:
        return _focal_mm_to_px(focal_mm, height, width)
    if not math.isfinite(focal_px) or focal_px < 0:
        raise ValueError("focal_length_px must be a finite non-negative number.")
    return float(focal_px)


# Sensor width is what matters for focal conversion; heights are listed for recognition.
SENSOR_PRESETS = {
    "Full frame 36x24 (Sony Venice, Canon R5)": (36.0, 24.0),
    "ARRI ALEXA 35 (27.99x19.22)": (27.99, 19.22),
    "ARRI ALEXA Mini LF (36.70x25.54)": (36.70, 25.54),
    "ARRI ALEXA Mini / AMIRA (28.25x18.17)": (28.25, 18.17),
    "Blackmagic URSA 12K (27.03x14.25)": (27.03, 14.25),
    "Blackmagic Pocket 6K (23.10x12.99)": (23.10, 12.99),
    "Super 35 3-perf (24.89x18.66)": (24.89, 18.66),
    "APS-C Sony/Fujifilm (23.5x15.6)": (23.5, 15.6),
    "APS-C Canon (22.3x14.9)": (22.3, 14.9),
    "Micro Four Thirds (17.3x13)": (17.3, 13.0),
    "1-inch (13.2x8.8)": (13.2, 8.8),
    "custom": None,
    "35mm equivalent (EXIF / unknown sensor)": None,
}


def _focal_length_calculation(
    focal_length_mm: float,
    sensor: str,
    sensor_width_mm: float,
    image_side_px: float,
    image: torch.Tensor | None = None,
) -> tuple[float, dict]:
    if not math.isfinite(focal_length_mm) or focal_length_mm < 0:
        raise ValueError("focal_length_mm must be a finite non-negative number.")
    if sensor not in SENSOR_PRESETS:
        raise ValueError(f"Unknown sensor mode {sensor!r}.")
    if image is not None:
        _validate_image_batch(image, "DepthPro2 Focal Length image")

    height = int(image.shape[1]) if image is not None else None
    width = int(image.shape[2]) if image is not None else None
    if focal_length_mm == 0:
        side = max(height, width) if image is not None else (
            image_side_px if math.isfinite(image_side_px) and image_side_px > 0 else None
        )
        return 0.0, {"height": height, "width": width, "side": side, "sensor": sensor}

    if sensor == "35mm equivalent (EXIF / unknown sensor)":
        if image is None:
            raise ValueError(
                "35mm-equivalent mode needs an IMAGE input so its diagonal can be measured; "
                "connect the image used for depth, or choose a physical sensor mode."
            )
        return _focal_mm_to_px(focal_length_mm, height, width), {
            "height": height,
            "width": width,
            "side": max(height, width),
            "sensor": sensor,
        }

    preset = SENSOR_PRESETS[sensor]
    sensor_w = preset[0] if preset is not None else sensor_width_mm
    if sensor_w is None or not math.isfinite(sensor_w) or sensor_w <= 0:
        raise ValueError(
            f"sensor '{sensor}' has no known size: set a positive sensor_width_mm "
            "(the 'custom' sensor needs its active readout width in mm)."
        )
    if image is not None:
        # The sensor-width direction follows the image's long edge, including portrait images.
        side = max(height, width)
    else:
        side = image_side_px
        if not math.isfinite(side) or side <= 0:
            raise ValueError(
                "image_side_px must be a positive pixel count when no IMAGE is connected."
            )
    return focal_length_mm * side / sensor_w, {
        "height": height,
        "width": width,
        "side": side,
        "sensor": sensor,
        "sensor_width_mm": sensor_w,
    }


def _focal_length_readout(focal_length_px: float, info: dict) -> str:
    lines = [f"Focal length: {focal_length_px:.2f} px"]
    height, width = info["height"], info["width"]
    if height is not None:
        orientation = "square" if height == width else ("landscape" if width > height else "portrait")
        lines.append(f"Image: {width} × {height} ({orientation})")
    elif info["side"] is not None:
        lines.append(f"Image side: {info['side']:.0f} px (manual)")
    if focal_length_px == 0:
        lines.append("0 requests automatic focal estimation in DepthPro2 Estimate/Video.")
        return "\n".join(lines)

    equivalent_mode = info["sensor"] == "35mm equivalent (EXIF / unknown sensor)"
    lines.append("Mode: 35mm equivalent / EXIF" if equivalent_mode else "Mode: physical lens + sensor")
    sensor_w = info.get("sensor_width_mm")
    if sensor_w is not None:
        lines.append(f"Active sensor width: {sensor_w:.2f} mm")
    if width is not None:
        diagonal = math.hypot(width, height)
        equivalent_mm = focal_length_px * math.hypot(36.0, 24.0) / diagonal
        horizontal_fov = math.degrees(2.0 * math.atan(width / (2.0 * focal_length_px)))
        vertical_fov = math.degrees(2.0 * math.atan(height / (2.0 * focal_length_px)))
        lines.append(f"35mm equivalent: {equivalent_mm:.2f} mm")
        lines.append(f"FOV: {horizontal_fov:.2f}° × {vertical_fov:.2f}° (H × V)")
    elif info["side"] is not None:
        fov = math.degrees(2.0 * math.atan(info["side"] / (2.0 * focal_length_px)))
        lines.append(f"FOV along sensor width: {fov:.2f}°")
    return "\n".join(lines)


class DepthPro2FocalLength(io.ComfyNode):
    """Convert physical or 35mm-equivalent lens focal length to image pixels."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthPro2FocalLength",
            display_name="DepthPro2 Focal Length",
            category=CATEGORY,
            is_output_node=True,
            search_aliases=["focal length", "focal", "lens", "sensor", "35mm equivalent", "EXIF"],
            description=(
                "Convert a lens focal length (mm) to pixels and link focal_length_px to "
                "DepthPro2 Estimate/Video. Basic: connect the exact IMAGE, choose a physical sensor "
                "or '35mm equivalent (EXIF / unknown sensor)', then enter the focal length. Physical "
                "mode reads the image's long side automatically, so portrait images need no manual "
                "math. Professional: choose 'custom' and enter the active sensor/readout width for "
                "cropped or anamorphic cameras; disconnect IMAGE to use image_side_px manually. "
                "0 requests automatic estimation downstream. The node reports pixels, 35mm-equivalent "
                "focal length and field of view after execution."
            ),
            inputs=[
                io.Float.Input("focal_length_mm", display_name="Focal length (mm)",
                               default=35.0, min=0.0, step=0.5,
                               tooltip="Physical mm engraved on the lens, or the 35mm-equivalent EXIF value "
                                       "when that sensor mode is selected; 0 estimates downstream."),
                io.Combo.Input("sensor", display_name="Mode / sensor", options=list(SENSOR_PRESETS),
                               default=next(iter(SENSOR_PRESETS)),
                               tooltip="Physical camera sensor, 'custom' active readout width, or 35mm-equivalent "
                                       "EXIF / unknown sensor (requires IMAGE)."),
                io.Float.Input("sensor_width_mm", display_name="Active sensor width (mm)",
                               default=36.0, min=0.1, max=100.0, step=0.01, advanced=True,
                               tooltip="Used only by 'custom': width in mm of the sensor area actually read out, "
                                       "not the full sensor when cropped."),
                io.Float.Input("image_side_px", display_name="Manual image side (px)",
                               default=1920.0, min=1.0, max=16384.0, step=1.0, advanced=True,
                               tooltip="Fallback when IMAGE is not connected: pixels along the sensor-width "
                                       "direction (width for landscape, height for portrait). Ignored with IMAGE."),
                io.Image.Input("image", display_name="Image (auto size)", optional=True,
                               tooltip="Connect the exact image used for depth. Its dimensions and orientation "
                                       "are read automatically; image_side_px is ignored."),
            ],
            outputs=[
                io.Float.Output(id="focal_length_px", display_name="focal_length_px",
                                tooltip="Focal length in pixels of the image; 0 requests automatic estimation."),
            ],
        )

    @classmethod
    def validate_inputs(cls, focal_length_mm: float, sensor: str, sensor_width_mm: float,
                        image_side_px: float, image: torch.Tensor | None = None) -> bool | str:
        if not math.isfinite(focal_length_mm) or focal_length_mm < 0:
            return "focal_length_mm must be a finite non-negative number."
        if sensor not in SENSOR_PRESETS:
            return f"Unknown sensor mode {sensor!r}."
        # Frontend validation runs before linked IMAGE tensors are supplied; execute checks the link.
        if sensor == "35mm equivalent (EXIF / unknown sensor)" and image is None:
            return True
        try:
            _focal_length_calculation(focal_length_mm, sensor, sensor_width_mm, image_side_px, image)
        except ValueError as error:
            return str(error)
        return True

    @classmethod
    def execute(cls, focal_length_mm: float, sensor: str, sensor_width_mm: float,
                image_side_px: float, image: torch.Tensor | None = None) -> io.NodeOutput:
        focal_px, info = _focal_length_calculation(
            focal_length_mm, sensor, sensor_width_mm, image_side_px, image
        )
        return io.NodeOutput(focal_px, ui=ui.PreviewText(_focal_length_readout(focal_px, info)))


def _validate_image_batch(images: torch.Tensor, label: str) -> None:
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] != 3:
        shape = tuple(images.shape) if isinstance(images, torch.Tensor) else type(images).__name__
        raise ValueError(f"{label} must be a ComfyUI IMAGE tensor shaped (B, H, W, 3); got {shape}.")
    if images.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one image.")
    if images.shape[1] == 0 or images.shape[2] == 0:
        raise ValueError(f"{label} must have non-empty spatial dimensions.")


def _validate_depth(depth: torch.Tensor, label: str) -> None:
    if not isinstance(depth, torch.Tensor) or depth.ndim != 3:
        shape = tuple(depth.shape) if isinstance(depth, torch.Tensor) else type(depth).__name__
        raise ValueError(f"{label} must be a depth tensor shaped (T, H, W); got {shape}.")
    if depth.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one depth map.")
    if depth.shape[1] == 0 or depth.shape[2] == 0:
        raise ValueError(f"{label} must have non-empty spatial dimensions.")


def _prepare_frames(
    images: torch.Tensor, max_size: int, f_px: float
) -> tuple[list[torch.Tensor], float | None, float]:
    """ComfyUI IMAGE batch (T, H, W, 3) float 0..1 -> CPU uint8 staging frames and focal.

    The frames stay on the CPU; each one moves to and normalizes on the GPU as it is
    consumed (the native per-chunk pattern: the input batch stays on the CPU and only
    the active chunk moves to the GPU), so a long batch never sits in VRAM.
    """
    height, width = images.shape[1:3]
    scale = 1.0
    if max_size > 0 and max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_h = max(32, round(height * scale))
        new_w = max(32, round(width * scale))
        images = F.interpolate(
            images.permute(0, 3, 1, 2), size=(new_h, new_w), mode="bilinear", antialias=True
        ).permute(0, 2, 3, 1)
    frames = [(frame.clamp(0, 1) * 255).round().to(torch.uint8) for frame in images]
    focal = None if f_px <= 0 else f_px * scale
    return frames, focal, scale


def _to_cpu(prediction: dict) -> None:
    """Move a prediction off the GPU the moment it is final (the native per-chunk offload)."""
    for key in ("inverse_depth", "depth"):
        if prediction[key] is not None:
            prediction[key] = prediction[key].detach().float().cpu()
    if prediction["focallength_px"] is not None:
        prediction["focallength_px"] = prediction["focallength_px"].detach().float().cpu()


def _run_inference(
    model: dict,
    image: torch.Tensor,
    f_px: float,
    max_size: int,
    desc: str,
    window: int = 1,
    mode: str | None = None,
    overlap: int = 0,
) -> tuple[list[dict], float]:
    """Run inference over the batch at the resolved processing size.

    Follows the native ComfyUI model-management pattern: the estimated activation VRAM is
    reserved through ``load_models_gpu`` before the weights load (the frame-interpolation,
    Trellis and TripoSplat nodes all do this), so other resident models are evicted in
    time instead of mid-run out-of-memory errors; the input frames stay on the CPU and
    move to the GPU one at a time; and every prediction is moved back to the CPU the
    moment it is final. If a frame still fails with an accelerator out-of-memory, the
    processing size halves and retries, so a busy session degrades in detail instead of
    crashing.
    """
    net, device = model["model"], model["device"]
    if f_px <= 0 and model.get("focal_pending"):
        _ensure_focal_model(model)
    height, width = image.shape[1:3]
    side = _resolve_max_size(model, image, max_size, window=window)
    longest = max(height, width)
    attempts = 0
    while True:
        mm.throw_exception_if_processing_interrupted()
        frames, focal, scale = _prepare_frames(image, side, f_px)
        progress = ProgressBar(len(frames))
        bar = tqdm(total=len(frames), desc=desc)
        try:
            if device.type != "cpu":
                mm.load_models_gpu(
                    [model["patcher"]],
                    memory_required=_activation_bytes(model, height, width, side, window=window),
                )
            predictions = []
            if mode is None:
                for frame in frames:
                    mm.throw_exception_if_processing_interrupted()
                    prediction = net.infer(to_normalized_tensor(frame, device), f_px=focal)
                    _to_cpu(prediction)
                    predictions.append(prediction)
                    progress.update(1)
                    bar.update(1)
            else:
                iterator = iter(net.infer_video(
                    (to_normalized_tensor(frame, device) for frame in frames),
                    window=window, overlap=overlap, f_px=focal, mode=mode,
                ))
                while True:
                    mm.throw_exception_if_processing_interrupted()
                    try:
                        prediction = next(iterator)
                    except StopIteration:
                        break
                    _to_cpu(prediction)
                    predictions.append(prediction)
                    progress.update(1)
                    bar.update(1)
            return predictions, scale
        except Exception as error:
            if not mm.is_oom(error):
                raise
            del frames
            mm.soft_empty_cache()
            attempts += 1
            current = min(longest, side) if side > 0 else longest
            if attempts >= 3 or current <= 32:
                raise RuntimeError(
                    "DepthPro2: out of memory even at the smallest processing size. "
                    "Free memory used by other models, choose a smaller preset, use "
                    "the fp8 precision, or set a smaller max_size."
                ) from None
            side = max(32, current // (2**attempts))
            LOGGER.warning(
                "DepthPro2: out of memory at longest side %d; retrying at %d.",
                current, side,
            )
        finally:
            bar.close()


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
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    if not predictions:
        raise ValueError("DepthPro2: inference produced no predictions.")
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
        valid = torch.isfinite(depth)
        weights = F.interpolate(valid.float()[:, None], size=original_size, mode="bilinear")[:, 0]
        depth = F.interpolate(
            torch.where(valid, depth, 0.0)[:, None], size=original_size, mode="bilinear"
        )[:, 0]
        depth = torch.where(
            weights > 0, depth / weights.clamp_min(1e-12), torch.full_like(depth, float("inf"))
        )
    image = _normalize_inverse_depth(inverse).unsqueeze(-1).repeat(1, 1, 1, 3)
    focal = 0.0
    if predictions[0]["focallength_px"] is not None:
        # Report in pixels of the original input, undoing any max_size downscale.
        focal = float(predictions[0]["focallength_px"]) / scale
    sky = (~torch.isfinite(depth)).to(torch.float32)
    return image, depth, focal, sky


class DepthPro2Loader(io.ComfyNode):
    """Load Apple Depth Pro 2 as the DEPTHPRO2_MODEL payload the estimate nodes consume."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthPro2Loader",
            display_name="DepthPro2 Loader",
            category=LOADER_CATEGORY,
            description=(
                "Load Apple Depth Pro 2. Preset names download weights from Hugging Face into "
                "models/depthpro2 on first use; local .safetensors/.pt files in that folder are also "
                "listed. auto uses ComfyUI's session dtype for auxiliary models (usually fp16 on an "
                "accelerator, fp32 on CPU; honors --fp16-text-enc/--bf16-text-enc); fp16 is the native "
                "default and bf16 Apple's training precision; fp8 stores encoder weights as float8 "
                "with per-channel scales and dequantizes them on every forward (about half the bf16 "
                "weight storage, not an fp8 compute kernel). The model is managed by ComfyUI's model "
                "management: it rides in RAM and is evicted from the GPU when other nodes need it."
            ),
            search_aliases=["depthpro2", "depth pro 2", "apple depth", "metric depth", "depth estimation"],
            inputs=[
                io.Combo.Input("model", options=list(HF_REPOS) + _local_weight_files(),
                               default="depthpro2-s",
                               tooltip="A preset (auto-downloaded into models/depthpro2 on first use) or "
                                       "a local .safetensors/.pt file in that folder."),
                io.Combo.Input("precision", options=["auto", "bf16", "fp16", "fp32", "fp8"], default="auto",
                               tooltip="auto uses ComfyUI's session dtype; fp8 stores encoder weights as "
                                       "float8 and dequantizes them on every forward."),
                io.Combo.Input(
                    "device", options=["auto", "cuda", "mps", "cpu"], default="auto",
                    tooltip="auto uses ComfyUI's device policy for auxiliary models (honors --gpu-only; "
                            "keeps DepthPro2 on MPS in Apple Silicon shared-memory sessions; falls back "
                            "to CPU without a usable accelerator); an explicit device is used as-is.",
                ),
            ],
            outputs=[DEPTHPRO2_MODEL.Output(id="model", display_name="model")],
        )

    @classmethod
    def execute(cls, model: str, precision: str, device: str) -> io.NodeOutput:
        return io.NodeOutput(load_depthpro2(model, precision, device))


class DepthPro2Estimate(io.ComfyNode):
    """Estimate metric depth for each image of the batch independently."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthPro2Estimate",
            display_name="DepthPro2 Estimate Depth",
            category=CATEGORY,
            description=(
                "Estimate metric depth for each image of the batch independently. depth_image is "
                "normalized inverse depth (near white, far black) for preview/ControlNet; depth_meters "
                "is raw metric depth (T, H, W) with inf at sky; sky_mask is a MASK (1 at sky) for comp "
                "holdouts. focal_length_mm is the lens focal length "
                "as a 35mm-equivalent (e.g. 24, 35, 50 — the EXIF FocalLengthIn35mmFilm value); 0 "
                "estimates it with the focal-length model, as long as the 'Estimate focal length' "
                "toggle (on by default) is on — with it off, an explicit focal length is required "
                "(focal_length_mm or a linked focal_length_px) and a ValueError is raised without one. "
                "Depth in meters scales linearly with focal "
                "length. max_size: 0 (default) chooses a budget-aware processing resolution, -1 runs "
                "at native resolution (sharpest edges, may OOM on very large images), and any positive "
                "value caps the longest side processed; the output is always upsampled back to the "
                "input size. CUDA reserves estimated processing memory through ComfyUI's model "
                "management before the run; Apple Silicon budgets against Metal working-set headroom. "
                "If a frame still runs out of memory, the node retries at half the processing size, "
                "down to a 32 px longest side. For physical mm with a known "
                "camera sensor, link the Focal Length node's focal_length_px output to the optional "
                "focal_length_px input; it overrides focal_length_mm."
            ),
            search_aliases=["depth", "metric depth", "depth estimation", "depthpro2"],
            inputs=[
                DEPTHPRO2_MODEL.Input("model"),
                io.Image.Input("image", tooltip="An image, or a batch of images (each estimated independently)."),
                io.Boolean.Input("estimate_focal_length", default=True,
                                 tooltip="Let the model estimate the focal length when none is given "
                                         "(focal_length_mm is 0 and focal_length_px is not linked). Off: an "
                                         "explicit focal length is required (focal_length_mm or a linked "
                                         "focal_length_px) and a ValueError is raised without one."),
                io.Float.Input("focal_length_mm", default=0.0, min=0.0, step=0.5, optional=True,
                               tooltip="Lens focal length as a 35mm-equivalent (e.g. 24, 35, 50); 0 estimates "
                                       "it with the focal-length model (while 'Estimate focal length' is on)."),
                io.Float.Input("focal_length_px", default=0.0, min=0.0, step=0.1, optional=True,
                               tooltip="Focal length in pixels of the image, e.g. from the Focal Length node; "
                                       "overrides focal_length_mm."),
                io.Int.Input("max_size", default=0, min=-1, max=16384, step=32, optional=True,
                             tooltip="0 (default) chooses a budget-aware processing resolution; -1 runs "
                                     "at native resolution (may OOM on very large images); any positive "
                                     "value caps the longest side processed. The output is always upsampled "
                                     "back to the input size."),
            ],
            outputs=[
                io.Image.Output(id="depth_image", display_name="depth_image",
                                tooltip="Normalized inverse depth (near white, far black) for preview/ControlNet."),
                DEPTHPRO2_DEPTH.Output(id="depth_meters", display_name="depth_meters",
                                       tooltip="Raw metric depth (T, H, W) with inf at sky."),
                io.Float.Output(id="focal_length_px", display_name="focal_length_px",
                                tooltip="Focal length in pixels used for the run."),
                io.Mask.Output(id="sky_mask", display_name="sky_mask",
                               tooltip="1 at sky, 0 elsewhere; for comp holdouts."),
            ],
        )

    @classmethod
    def execute(cls, model: dict, image: torch.Tensor, estimate_focal_length: bool,
                focal_length_mm: float, focal_length_px: float | None = None,
                max_size: int = 0) -> io.NodeOutput:
        _validate_image_batch(image, "DepthPro2 estimate image")
        f_px = _resolve_focal_px(focal_length_mm, *image.shape[1:3], focal_length_px)
        if f_px <= 0 and not estimate_focal_length:
            raise ValueError(
                "DepthPro2 estimate: no focal length is set (focal_length_mm is 0 and "
                "focal_length_px is not linked) and 'Estimate focal length' is off. Set "
                "focal_length_mm or link focal_length_px, or re-enable the toggle."
            )
        predictions, scale = _run_inference(model, image, f_px, max_size, "DepthPro2 estimate")
        mm.soft_empty_cache()
        return io.NodeOutput(*_depth_outputs(predictions, tuple(image.shape[1:3]), scale))


class DepthPro2Video(io.ComfyNode):
    """Temporally consistent depth for a video batch, in overlapping windows."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthPro2Video",
            display_name="DepthPro2 Video Depth",
            category=CATEGORY,
            description=(
                "Temporally consistent depth for a video batch: frames are processed jointly in "
                "overlapping windows and share one focal length. focal_length_mm is the 35mm-equivalent "
                "focal length; 0 estimates it from the first window, as long as the 'Estimate focal "
                "length' toggle (on by default) is on — with it off, an explicit focal length is "
                "required (focal_length_mm or a linked focal_length_px) and a ValueError is raised "
                "without one. Lower window/overlap if you run out of "
                "memory. depth_image is normalized over the whole batch so brightness is steady "
                "across frames. buffered lets each frame also see later frames (overlap applies); "
                "unbuffered predicts each frame from itself and the frames before it, at lower latency "
                "and memory (overlap is ignored). max_size: 0 (default) chooses a budget-aware "
                "processing resolution accounting for the window, -1 runs at native resolution, and "
                "any positive value caps the longest side processed; the output is always upsampled "
                "back to the input size. Frames are staged on the CPU and moved to the GPU one window "
                "at a time. CUDA reserves estimated processing memory through ComfyUI's model "
                "management before the run; Apple Silicon budgets against Metal working-set headroom. "
                "Link the Focal Length node's focal_length_px output "
                "to the optional focal_length_px input to override focal_length_mm with a sensor-aware "
                "physical focal length."
            ),
            search_aliases=["depth", "metric depth", "video depth", "depth estimation", "depthpro2"],
            inputs=[
                DEPTHPRO2_MODEL.Input("model"),
                io.Image.Input("images", tooltip="The video frames (an IMAGE batch)."),
                io.Int.Input("window", default=16, min=1, max=64,
                             tooltip="Frames attended together; lower it if you run out of memory."),
                io.Int.Input("overlap", default=4, min=0, max=63, advanced=True,
                             tooltip="Frames shared by consecutive clips (buffered mode only)."),
                io.Boolean.Input("estimate_focal_length", default=True,
                                 tooltip="Let the model estimate the focal length from the first window "
                                         "when none is given (focal_length_mm is 0 and focal_length_px is "
                                         "not linked). Off: an explicit focal length is required "
                                         "(focal_length_mm or a linked focal_length_px) and a ValueError "
                                         "is raised without one."),
                io.Float.Input("focal_length_mm", default=0.0, min=0.0, step=0.5, optional=True,
                               tooltip="35mm-equivalent focal length; 0 estimates it from the first window "
                                       "(while 'Estimate focal length' is on)."),
                io.Float.Input("focal_length_px", default=0.0, min=0.0, step=0.1, optional=True,
                               tooltip="Focal length in pixels, e.g. from the Focal Length node; overrides "
                                       "focal_length_mm."),
                io.Int.Input("max_size", default=0, min=-1, max=16384, step=32, optional=True,
                             tooltip="0 (default) chooses a budget-aware processing resolution accounting "
                                     "for the window; -1 native; positive caps the longest side. Output is "
                                     "upsampled back to the input size."),
                io.Combo.Input("mode", options=["buffered", "unbuffered"], default="buffered", advanced=True,
                               tooltip="buffered lets each frame also see later frames; unbuffered predicts "
                                       "from the past only, at lower latency and memory."),
            ],
            outputs=[
                io.Image.Output(id="depth_image", display_name="depth_image",
                                tooltip="Normalized inverse depth, steady across frames."),
                DEPTHPRO2_DEPTH.Output(id="depth_meters", display_name="depth_meters",
                                       tooltip="Raw metric depth (T, H, W) with inf at sky."),
                io.Float.Output(id="focal_length_px", display_name="focal_length_px",
                                tooltip="Focal length in pixels used for the run."),
                io.Mask.Output(id="sky_mask", display_name="sky_mask",
                               tooltip="1 at sky, 0 elsewhere; for comp holdouts."),
            ],
        )

    @classmethod
    def execute(cls, model: dict, images: torch.Tensor, window: int, overlap: int,
                estimate_focal_length: bool, focal_length_mm: float,
                focal_length_px: float | None = None, max_size: int = 0,
                mode: str = "buffered") -> io.NodeOutput:
        _validate_image_batch(images, "DepthPro2 video images")
        if mode == "buffered" and overlap >= window:
            raise ValueError("overlap must be smaller than window.")
        f_px = _resolve_focal_px(focal_length_mm, *images.shape[1:3], focal_length_px)
        if f_px <= 0 and not estimate_focal_length:
            raise ValueError(
                "DepthPro2 video: no focal length is set (focal_length_mm is 0 and "
                "focal_length_px is not linked) and 'Estimate focal length' is off. Set "
                "focal_length_mm or link focal_length_px, or re-enable the toggle."
            )
        predictions, scale = _run_inference(
            model, images, f_px, max_size, "DepthPro2 video",
            window=window, mode=mode, overlap=overlap,
        )
        mm.soft_empty_cache()
        image, depth, focal_out, sky = _depth_outputs(predictions, tuple(images.shape[1:3]), scale)
        # One shared range keeps the visualization temporally steady.
        finite = torch.isfinite(depth) & (depth > 0)
        if finite.any():
            inverse = 1.0 / depth.clamp(min=1e-6)
            inverse[~finite] = 0.0
            lo = inverse[finite].min()
            hi = inverse[finite].max()
            if hi > lo:
                image = ((inverse - lo) / (hi - lo)).clamp(0, 1).unsqueeze(-1).repeat(1, 1, 1, 3)
        return io.NodeOutput(image, depth, focal_out, sky)


class DepthPro2Colorize(io.ComfyNode):
    """Color preview of metric depth, as in Apple's CLI."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthPro2Colorize",
            display_name="DepthPro2 Colorize Depth",
            category=CATEGORY,
            search_aliases=["colorize", "colormap", "depth color", "turbo", "depth preview"],
            description=(
                "Color preview of metric depth, as in Apple's CLI: turbo maps dark blue at infinitely "
                "far to red/white near. The scale is fixed by the first frame's nearest point plus 20% "
                "headroom, so a video stays steady and points coming nearer later saturate. gray uses "
                "the same scale in grayscale (white near, black infinitely far)."
            ),
            inputs=[
                DEPTHPRO2_DEPTH.Input("depth"),
                io.Combo.Input("colormap", options=list(COLORMAPS), default="turbo",
                               tooltip="turbo is Apple's preview map (dark blue = infinitely far, "
                                       "red/white = near); the others are standard names; gray is the "
                                       "same scale in grayscale (white near, black far)."),
            ],
            outputs=[io.Image.Output(id="depth_color", display_name="depth_color")],
        )

    @classmethod
    def execute(cls, depth: torch.Tensor, colormap: str) -> io.NodeOutput:
        _validate_depth(depth, "DepthPro2 colorize depth")
        inverse = inverse_metric(depth)
        return io.NodeOutput(colorize(inverse, colormap, color_scale(inverse)))


class DepthPro2PositionPass(io.ComfyNode):
    """Camera-space position pass in meters, back-projected from metric depth."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthPro2PositionPass",
            display_name="DepthPro2 Position Pass",
            category=CATEGORY,
            search_aliases=["position pass", "point cloud", "unproject", "backproject", "3d points"],
            description=(
                "Camera-space position pass in meters: X right, Y down, Z along the viewing direction, "
                "back-projected from metric depth with focal_length_px (link it from the estimate node) "
                "and the image center as principal point, shifted by cx_offset/cy_offset for calibrated "
                "cameras. Sky and pixels beyond max_depth (0 = no clip) are zero with valid = 0; clip "
                "before making point clouds, as sky depth is unreliable."
            ),
            inputs=[
                DEPTHPRO2_DEPTH.Input("depth"),
                io.Float.Input("focal_length_px", default=0.0, min=0.0, step=0.1,
                               tooltip="Focal length in pixels; link it from the estimate node."),
                io.Float.Input("max_depth", default=0.0, min=0.0, step=0.5, advanced=True,
                               tooltip="Clip beyond this many meters (0 = no clip); sky depth is unreliable."),
                io.Float.Input("cx_offset", default=0.0, min=-4096.0, max=4096.0, step=0.5, advanced=True,
                               tooltip="Principal point offset from the image center, in pixels (x)."),
                io.Float.Input("cy_offset", default=0.0, min=-4096.0, max=4096.0, step=0.5, advanced=True,
                               tooltip="Principal point offset from the image center, in pixels (y)."),
            ],
            outputs=[
                io.Image.Output(id="position_xyz", display_name="position_xyz",
                                tooltip="Camera-space position in meters (X right, Y down, Z forward)."),
                io.Mask.Output(id="valid", display_name="valid",
                               tooltip="1 where a valid point was written, 0 at sky/clipped pixels."),
            ],
        )

    @classmethod
    def execute(cls, depth: torch.Tensor, focal_length_px: float, max_depth: float,
                cx_offset: float, cy_offset: float) -> io.NodeOutput:
        _validate_depth(depth, "DepthPro2 position depth")
        if focal_length_px <= 0:
            raise ValueError("focal_length_px must be positive; link it from the estimate node.")
        height, width = depth.shape[-2:]
        valid = torch.isfinite(depth) & (depth > 0)
        if max_depth > 0:
            valid = valid & (depth <= max_depth)
        z = torch.where(valid, depth, torch.zeros_like(depth))
        u = torch.arange(width, dtype=torch.float32) + 0.5 - (width / 2 + cx_offset)
        v = torch.arange(height, dtype=torch.float32) + 0.5 - (height / 2 + cy_offset)
        x = z * u.view(1, 1, width) / focal_length_px
        y = z * v.view(1, height, 1) / focal_length_px
        return io.NodeOutput(torch.stack([x, y, z], dim=-1), valid.to(torch.float32))


def _reserve_output_paths(
    folder: str, filename: str, extension: str, counter: int, count: int
) -> list[Path]:
    """Reserve unique output names, including across concurrent prompts."""
    directory = Path(folder)
    flags = re.IGNORECASE if os.name == "nt" else 0
    pattern = re.compile(rf"{re.escape(filename)}_(\d+){re.escape(extension)}", flags)
    indices = [
        int(match.group(1))
        for path in directory.iterdir()
        if (match := pattern.fullmatch(path.name))
    ]
    counter = max(counter, max(indices, default=0) + 1)
    paths = []
    while len(paths) < count:
        path = directory / f"{filename}_{counter:05d}{extension}"
        try:
            with path.open("xb"):
                pass
        except FileExistsError:
            counter += 1
            continue
        paths.append(path)
        counter += 1
    return paths


class DepthPro2Save16(io.ComfyNode):
    """Save inverse metric depth as 16-bit grayscale PNGs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthPro2Save16",
            display_name="DepthPro2 Save 16-bit Depth",
            category=CATEGORY,
            is_output_node=True,
            search_aliases=["save depth", "16-bit", "depth png", "depthpro2"],
            description=(
                "Save inverse metric depth as 16-bit grayscale PNGs (white near, black infinitely far) "
                "on the color preview's scale, for compositing tools: read as raw data (Raw in Nuke, "
                "Non-Color in Blender). Depth in meters = k / value; k is logged and stored in each "
                "PNG's depthpro2_k text chunk. 0 is infinitely far. Shows an 8-bit preview on the node."
            ),
            inputs=[
                DEPTHPRO2_DEPTH.Input("depth"),
                io.String.Input("filename_prefix", default="depthpro2/depth16"),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, depth: torch.Tensor, filename_prefix: str) -> io.NodeOutput:
        _validate_depth(depth, "DepthPro2 16-bit depth")
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        inverse = inverse_metric(depth)
        scale = color_scale(inverse)
        k = 65535.0 / scale
        height, width = depth.shape[-2:]
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height
        )
        normalized_frames = []
        paths = _reserve_output_paths(folder, filename, ".png", counter, len(inverse))
        for frame, path in zip(inverse, paths):
            normalized = (frame / scale).clamp(0, 1)
            values = normalized.mul(65535).round().to(torch.uint16).numpy()
            meta = PngInfo()
            meta.add_text("depthpro2_k", f"{k:.6g}")
            Image.fromarray(values).save(path, pnginfo=meta, compress_level=1)
            normalized_frames.append(normalized)
        LOGGER.info("DepthPro2 16-bit: depth in meters = %.2f / value, 0 is infinitely far.", k)
        preview = torch.stack(normalized_frames).unsqueeze(-1).repeat(1, 1, 1, 3)
        return io.NodeOutput(ui=ui.PreviewImage(preview, cls=cls))


class DepthPro2SaveFloat(io.ComfyNode):
    """Save metric depth in meters as 32-bit float EXR/TIFF."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthPro2SaveFloat",
            display_name="DepthPro2 Save Float Depth",
            category=CATEGORY,
            is_output_node=True,
            search_aliases=["save depth", "float depth", "exr", "tiff", "z depth"],
            description=(
                "Save metric depth in meters as 32-bit float for compositing (Nuke, Fusion, Blender): "
                "single-channel EXR (channel Z, needs the optional OpenEXR package: pip install OpenEXR) "
                "or float32 TIFF with no extra dependencies. Unlike the 16-bit node this is lossless and "
                "stores true meters, not an inverse scale. sky_value replaces sky/invalid pixels with a "
                "far distance (0 keeps infinity). Link focal_length_px from the estimate node to embed "
                "it, with units and sky convention, in the file metadata. Shows an 8-bit preview."
            ),
            inputs=[
                DEPTHPRO2_DEPTH.Input("depth"),
                io.String.Input("filename_prefix", default="depthpro2/depth_float"),
                io.Combo.Input("format", options=["exr", "tiff"], default="exr",
                               tooltip="Both are lossless 32-bit float in meters. exr needs the optional "
                                       "OpenEXR package (pip install OpenEXR); tiff has no extra "
                                       "dependencies."),
                io.Float.Input("sky_value", default=0.0, min=0.0, step=100.0, advanced=True,
                               tooltip="Replace sky/invalid pixels with this far distance (0 keeps infinity)."),
                io.Float.Input("focal_length_px", default=0.0, min=0.0, step=0.1, optional=True,
                               tooltip="Link it from the estimate node to embed it in the file metadata."),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        depth: torch.Tensor,
        filename_prefix: str,
        format: str,
        sky_value: float,
        focal_length_px: float | None = None,
    ) -> io.NodeOutput:
        _validate_depth(depth, "DepthPro2 float depth")
        from PIL import Image

        meters = depth.float().cpu().clone()
        invalid = ~torch.isfinite(meters)
        if sky_value > 0:
            meters[invalid] = sky_value
        else:
            meters[torch.isnan(meters)] = float("inf")
        sky_desc = f"{sky_value:g} m" if sky_value > 0 else "inf"
        focal = float(focal_length_px) if focal_length_px else 0.0
        height, width = meters.shape[-2:]
        ext = ".exr" if format == "exr" else ".tif"
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height
        )
        if format == "exr":
            try:
                import Imath
                import OpenEXR
            except ImportError as e:
                raise ImportError(
                    "EXR export needs the OpenEXR package (pip install OpenEXR), "
                    "or choose the tiff format."
                ) from e
        inverse = inverse_metric(depth)
        scale = color_scale(inverse)
        preview_frames = []
        paths = _reserve_output_paths(folder, filename, ext, counter, len(meters))
        for i, (frame, path) in enumerate(zip(meters, paths)):
            if format == "exr":
                header = OpenEXR.Header(width, height)
                header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
                header["depthpro2_units"] = b"meters"
                header["depthpro2_sky"] = sky_desc.encode()
                if focal > 0:
                    header["depthpro2_focal_px"] = f"{focal:.6g}".encode()
                out = OpenEXR.OutputFile(str(path), header)
                out.writePixels({"Z": frame.numpy().tobytes()})
                out.close()
            else:
                from PIL.TiffImagePlugin import ImageFileDirectory_v2

                info = ImageFileDirectory_v2()
                info[65000] = "meters"
                info[65001] = sky_desc
                if focal > 0:
                    info[65002] = f"{focal:.6g}"
                Image.fromarray(frame.numpy()).save(path, tiffinfo=info)
            preview_frames.append((inverse[i] / scale).clamp(0, 1))
        LOGGER.info("DepthPro2 float depth: %s, meters, sky = %s.", path, sky_desc)
        preview = torch.stack(preview_frames).unsqueeze(-1).repeat(1, 1, 1, 3)
        return io.NodeOutput(ui=ui.PreviewImage(preview, cls=cls))


class DepthPro2SavePLY(io.ComfyNode):
    """Write a metric point cloud (binary PLY, in meters) from the Position Pass outputs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthPro2SavePLY",
            display_name="DepthPro2 Save PLY",
            category=CATEGORY,
            is_output_node=True,
            search_aliases=["save ply", "point cloud", "pointcloud", "points", "depthpro2"],
            description=(
                "Write a metric point cloud (binary PLY, in meters) from the Position Pass outputs: "
                "link position_xyz and valid; only valid points are written, sky and clipped pixels "
                "are dropped. Link the original image to color the points. Coordinates are camera-space "
                "(X right, Y down, Z forward) — flip Y in DCCs that expect Y up. Opens in MeshLab, "
                "Blender and CloudCompare."
            ),
            inputs=[
                io.Image.Input("position", tooltip="The Position Pass position_xyz output (T, H, W, 3)."),
                io.Mask.Input("valid", tooltip="The Position Pass valid output."),
                io.String.Input("filename_prefix", default="depthpro2/points"),
                io.Image.Input("image", optional=True,
                               tooltip="The original image, to color the points."),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        position: torch.Tensor,
        valid: torch.Tensor,
        filename_prefix: str,
        image: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        import numpy as np

        if not isinstance(position, torch.Tensor) or position.ndim != 4 or position.shape[-1] != 3:
            raise ValueError("position must be the Position Pass position_xyz output (T, H, W, 3).")
        if any(size == 0 for size in position.shape):
            raise ValueError("position must contain at least one non-empty point map.")
        if not isinstance(valid, torch.Tensor) or valid.shape != position.shape[:3]:
            raise ValueError("valid must be a MASK shaped (T, H, W) matching position.")
        if image is not None and (
            not isinstance(image, torch.Tensor)
            or image.ndim != 4
            or image.shape[-1] != 3
            or image.shape != position.shape
        ):
            raise ValueError("image must match position in batch, spatial dimensions and RGB channels.")
        height, width = position.shape[1:3]
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height
        )
        paths = _reserve_output_paths(folder, filename, ".ply", counter, position.shape[0])
        colored = image is not None
        props = "property float x\nproperty float y\nproperty float z\n"
        fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if colored:
            props += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        for i, path in enumerate(paths):
            mask = valid[i].bool()
            points = position[i][mask].numpy().astype("<f4")
            record = np.empty(len(points), dtype=np.dtype(fields))
            record["x"], record["y"], record["z"] = points.T
            if colored:
                rgb = (image[i][mask].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
                record["red"], record["green"], record["blue"] = rgb.T
            header = (
                f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
                f"{props}end_header\n"
            )
            path.write_bytes(header.encode("ascii") + record.tobytes())
            LOGGER.info("DepthPro2 PLY: %s (%d points, meters).", path, len(points))
        return io.NodeOutput()