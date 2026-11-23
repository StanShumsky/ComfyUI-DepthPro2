"""Unit tests: no weights, no ComfyUI install, no network."""

import asyncio
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file


def synthetic_depth(t=2, h=16, w=24):
    depth = torch.linspace(1.0, 10.0, h * w).reshape(1, h, w).repeat(t, 1, 1)
    depth[0, :4, :4] = float("inf")
    return depth


def _entry_node_list():
    """The node classes exactly as the ComfyExtension entrypoint exposes them to ComfyUI."""
    extension = sys.modules["dp2node"].DepthPro2Pack()
    return asyncio.run(extension.get_node_list())


@pytest.fixture(scope="session")
def node_classes():
    return {cls.GET_SCHEMA().node_id: cls for cls in _entry_node_list()}


def test_node_schemas(node_classes):
    assert set(node_classes) == {
        "DepthPro2Loader", "DepthPro2Estimate", "DepthPro2Video", "DepthPro2FocalLength",
        "DepthPro2Colorize", "DepthPro2PositionPass", "DepthPro2Save16",
        "DepthPro2SaveFloat", "DepthPro2SavePLY",
    }
    for name, cls in node_classes.items():
        # GET_SCHEMA finalizes and validates the definition the way ComfyUI does on load.
        schema = cls.GET_SCHEMA()
        assert schema.node_id == name
        assert schema.display_name
        # The native placement: the loader with the model loaders, the rest with the
        # geometry estimation models (the same split the built-in ComfyUI depth nodes use).
        assert schema.category == ("model/loaders" if name == "DepthPro2Loader" else "image/geometry estimation")
        assert schema.description
        assert callable(cls.execute)
        # Every schema input name must be an execute() parameter (and vice versa, for the
        # non-optional ones), as ComfyUI's executor passes inputs through by name.
        import inspect

        params = inspect.signature(cls.execute).parameters
        for inp in schema.inputs:
            assert inp.id in params, (name, inp.id)
        for param in params:
            if param != "cls":
                assert param in {i.id for i in schema.inputs} or params[param].default is not inspect.Parameter.empty, (name, param)


def test_focal_mm_to_px(nodes):
    assert nodes._focal_mm_to_px(0.0, 256, 384) == 0.0
    expected = 50.0 * math.hypot(384, 256) / math.hypot(36.0, 24.0)
    assert abs(nodes._focal_mm_to_px(50.0, 256, 384) - expected) < 1e-9
    # On a full-frame 3:2 sensor, 35mm-equivalent mm == physical mm.
    assert abs(nodes._focal_mm_to_px(50.0, 2400, 3600) - 5000.0) < 1e-6


def test_focal_length_node(nodes):
    execute = nodes.DepthPro2FocalLength.execute
    full_frame = next(iter(nodes.SENSOR_PRESETS))
    (fpx,) = execute(50.0, full_frame, 36.0, 3600.0).result
    assert abs(fpx - 5000.0) < 1e-6
    (fpx,) = execute(35.0, "custom", 27.03, 4096.0).result
    assert abs(fpx - 35.0 * 4096 / 27.03) < 1e-6
    # Portrait: the image height spans the sensor width, same math.
    (fpx,) = execute(50.0, full_frame, 36.0, 2400.0).result
    assert abs(fpx - 50.0 * 2400 / 36.0) < 1e-6
    (fpx,) = execute(0.0, "custom", 36.0, 1920.0).result
    assert fpx == 0.0


def test_focal_length_validation(nodes):
    execute = nodes.DepthPro2FocalLength.execute
    full_frame = next(iter(nodes.SENSOR_PRESETS))
    with pytest.raises(ValueError, match="sensor_width_mm"):
        execute(35.0, "custom", 0.0, 4096.0)
    with pytest.raises(ValueError, match="image_side_px"):
        execute(35.0, full_frame, 36.0, 0.0)


class _Sentinel(Exception):
    """Stops the faked _run_inference so these tests need no real model."""


def test_estimate_focal_toggle(nodes, make_batch, monkeypatch):
    """Toggle off requires an explicit focal length; on (the default), 0 lets the model estimate it."""
    batch = make_batch(t=1, h=32, w=32)
    seen = []

    def fake_run(model, image, f_px, max_size, desc, **kwargs):
        seen.append(f_px)
        raise _Sentinel()

    monkeypatch.setattr(nodes, "_run_inference", fake_run)

    # Off, nothing set -> ValueError before inference starts.
    with pytest.raises(ValueError, match="Estimate focal length"):
        nodes.DepthPro2Estimate.execute(None, batch, False, 0.0, 0)

    # Off, 35mm-equivalent -> converted pixels reach inference.
    with pytest.raises(_Sentinel):
        nodes.DepthPro2Estimate.execute(None, batch, False, 35.0, 0)
    assert abs(seen[-1] - 35.0 * math.hypot(32, 32) / math.hypot(36.0, 24.0)) < 1e-6

    # Off, linked pixels -> used as-is.
    with pytest.raises(_Sentinel):
        nodes.DepthPro2Estimate.execute(None, batch, False, 0.0, 500.0, 0)
    assert seen[-1] == 500.0

    # On, nothing set -> 0, the model estimates.
    with pytest.raises(_Sentinel):
        nodes.DepthPro2Estimate.execute(None, batch, True, 0.0, 0)
    assert seen[-1] == 0.0


def test_prepare_frames_resize_and_focal(nodes, make_batch):
    batch = make_batch(t=2, h=64, w=96)
    frames, focal, scale = nodes._prepare_frames(batch, 48, 500.0)
    assert scale == 0.5 and focal == 250.0
    # CPU uint8 staging: the frames stay on the CPU until consumed (the native pattern).
    assert len(frames) == 2 and frames[0].shape == (32, 48, 3)
    assert frames[0].dtype == torch.uint8 and frames[0].device.type == "cpu"
    frames, focal, scale = nodes._prepare_frames(batch, 0, 0.0)
    assert scale == 1.0 and focal is None and frames[0].shape == (64, 96, 3)


def test_normalize_uint8_tensor(nodes):
    u8 = (torch.rand(8, 12, 3) * 255.0).round().to(torch.uint8)
    x = nodes.to_normalized_tensor(u8, torch.device("cpu"))
    assert x.shape == (3, 8, 12) and x.dtype == torch.float32 and x.device.type == "cpu"
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    expected = (u8.permute(2, 0, 1).float() / 255.0 - mean) / std
    assert torch.allclose(x, expected, atol=1e-6)
    with pytest.raises(TypeError):
        nodes.to_normalized_tensor(torch.rand(3, 8, 12), torch.device("cpu"))
    with pytest.raises(TypeError):
        nodes.to_normalized_tensor(torch.rand(8, 12, 3), torch.device("cpu"))


def test_normalize_inverse_depth(nodes):
    inverse = torch.tensor([[[0.0, 1.0], [2.0, 4.0]]])
    out = nodes._normalize_inverse_depth(inverse)
    assert out[0, 0, 0] == 0.0 and out[0, 1, 1] == 1.0 and out[0, 1, 0] == 0.5


def test_depth_outputs_upscale_and_focal(nodes):
    inverse = torch.rand(2, 8, 8) + 0.1
    preds = [
        {"inverse_depth": inverse[i], "depth": 1.0 / inverse[i], "focallength_px": torch.tensor(250.0)}
        for i in range(2)
    ]
    image, depth, focal, sky = nodes._depth_outputs(preds, (16, 16), 0.5)
    assert image.shape == (2, 16, 16, 3) and depth.shape == (2, 16, 16)
    assert focal == 500.0 and sky.max() == 0.0
    preds[1]["depth"] = None
    preds[1]["focallength_px"] = None
    _, depth, focal, sky = nodes._depth_outputs(preds, (8, 8), 1.0)
    assert torch.isnan(depth[1]).all() and focal == 250.0
    assert sky[1].min() == 1.0 and sky[0].max() == 0.0


def test_detect_preset(nodes):
    sd = {"encoder.cls_token": torch.zeros(1, 1, 384), "encoder.blocks.11.mlp.fc1.weight": torch.zeros(1)}
    assert nodes._detect_preset(sd) == "depthpro2-s"
    sd["encoder.cls_token"] = torch.zeros(1, 1, 512)
    with pytest.raises(ValueError):
        nodes._detect_preset(sd)


def test_fp8_quantized_linear(nodes):
    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 64)
    quant = nodes.QuantizedLinear(linear, torch.bfloat16)
    weight = quant.dequantized_weight()
    rel = (weight - linear.weight).norm() / linear.weight.norm()
    assert rel < 0.05
    assert torch.equal(quant.weight, weight)
    x = torch.randn(4, 64)
    reference = x @ weight.float().T + linear.bias
    assert torch.allclose(quant(x).float(), reference, atol=5e-2)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(64, 64))
    assert nodes.quantize_linears(model, torch.bfloat16) == 1
    assert isinstance(model[1], nodes.QuantizedLinear) and not isinstance(model[0], nodes.QuantizedLinear)


def test_position_pass(nodes):
    depth = synthetic_depth()
    focal = 500.0
    execute = nodes.DepthPro2PositionPass.execute
    pos, valid = execute(depth, focal, 0.0, 0.0, 0.0).result
    finite = torch.isfinite(depth) & (depth > 0)
    assert pos.shape == (2, 16, 24, 3) and torch.equal(valid.bool(), finite)
    v, u = 10, 20
    z = depth[0, v, u]
    assert pos[0, v, u, 2] == z
    assert abs(pos[0, v, u, 0] - (u + 0.5 - 12) * z / focal) < 1e-4
    assert abs(pos[0, v, u, 1] - (v + 0.5 - 8) * z / focal) < 1e-4
    assert torch.all(pos[~finite] == 0)
    pos_c, valid_c = execute(depth, focal, 5.0, 0.0, 0.0).result
    assert torch.equal(valid_c.bool(), finite & (depth <= 5.0))
    assert torch.all(pos_c[~valid_c.bool()] == 0)
    pos_o, _ = execute(depth, focal, 0.0, 10.0, -5.0).result
    assert torch.allclose(pos_o[..., 0], pos[..., 0] - 10.0 * pos[..., 2] / focal, atol=1e-4)
    assert torch.allclose(pos_o[..., 1], pos[..., 1] + 5.0 * pos[..., 2] / focal, atol=1e-4)
    with pytest.raises(ValueError):
        execute(depth, 0.0, 0.0, 0.0, 0.0)


def test_colorize_scale_and_sky(nodes):
    depth = synthetic_depth()
    inverse = nodes.inverse_metric(depth)
    assert inverse[0, 0, 0] == 0.0
    scale = nodes.color_scale(inverse)
    assert abs(scale - float(inverse[0].max()) * 1.2) < 1e-6
    execute = nodes.DepthPro2Colorize.execute
    (color,) = execute(depth, "turbo").result
    lut = sys.modules["dp2node.colormaps"]._LUTS
    lut0 = torch.from_numpy(lut[0][0] / 255.0).float()
    assert torch.allclose(color[0, :4, :4], lut0)
    assert color.min() >= 0 and color.max() <= 1
    (gray,) = execute(depth, "gray").result
    assert torch.allclose(gray[..., 0], (inverse / scale).clamp(0, 1))
    assert torch.equal(gray[..., 0], gray[..., 2])


def test_save16(nodes, output_dir):
    depth = synthetic_depth()
    result = nodes.DepthPro2Save16.execute(depth, "unit/depth16")
    files = sorted((output_dir / "unit").glob("depth16_*.png"))
    assert len(files) == 2
    previews = result.ui.values
    assert len(previews) == 2 and all(p["type"] == "temp" for p in previews)
    preview = Image.open(output_dir / "temp" / previews[0]["filename"])
    arr = np.array(preview)
    assert preview.size == (24, 16)
    if arr.ndim == 3:  # saved as gray-on-RGB; a true preview means equal channels
        assert (arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all()
    assert arr[:4, :4].max() == 0
    img = Image.open(files[0])
    assert img.mode == "I;16"
    scale = nodes.color_scale(nodes.inverse_metric(depth))
    k = float(img.text["depthpro2_k"])
    assert abs(k - 65535.0 / scale) / k < 1e-5
    values = np.array(img).astype(np.float64)
    assert values[:4, :4].max() == 0
    near = values > 1000
    rel = np.abs(k / values[near] - depth[0].numpy()[near]) / depth[0].numpy()[near]
    assert np.median(rel) < 1e-3


def test_save_float_tiff(nodes, output_dir):
    depth = synthetic_depth()
    result = nodes.DepthPro2SaveFloat.execute(depth, "unit/float", "tiff", 0.0, focal_length_px=500.0)
    files = sorted((output_dir / "unit").glob("float_*.tif"))
    assert len(files) == 2
    img = Image.open(files[0])
    assert img.mode == "F"
    values = torch.from_numpy(np.array(img))
    finite = torch.isfinite(depth[0])
    assert torch.equal(values[finite], depth[0][finite])
    assert torch.isinf(values[~finite]).all()
    assert img.tag_v2[65000] == "meters" and img.tag_v2[65002] == "500"
    assert len(result.ui.values) == 2
    nodes.DepthPro2SaveFloat.execute(depth, "unit/clipped", "tiff", 5000.0)
    clipped = torch.from_numpy(np.array(Image.open(sorted((output_dir / "unit").glob("clipped_*.tif"))[0])))
    assert clipped[0, 0] == 5000.0


def test_save_ply(nodes, output_dir):
    depth = synthetic_depth()
    pos, valid = nodes.DepthPro2PositionPass.execute(depth, 500.0, 0.0, 0.0, 0.0).result
    nodes.DepthPro2SavePLY.execute(pos, valid, "unit/points")
    files = sorted((output_dir / "unit").glob("points_*.ply"))
    assert len(files) == 2
    header, _, body = files[0].read_bytes().partition(b"end_header\n")
    n = int(header.split(b"element vertex ")[1].split(b"\n")[0])
    assert n == int(valid[0].sum()) and b"property uchar red" not in header
    rec = np.frombuffer(body, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]))
    assert len(rec) == n and np.isfinite(rec["z"]).all()
    z = float(depth[0, 0, 4])
    assert abs(rec[0]["x"] - (4 + 0.5 - 12) * z / 500.0) < 1e-4
    assert abs(rec[0]["z"] - z) < 1e-4
    nodes.DepthPro2SavePLY.execute(pos, valid, "unit/color", image=torch.rand(2, 16, 24, 3))
    cheader, _, cbody = sorted((output_dir / "unit").glob("color_*.ply"))[0].read_bytes().partition(b"end_header\n")
    assert b"property uchar red" in cheader and len(cbody) == n * 15
    with pytest.raises(ValueError):
        nodes.DepthPro2SavePLY.execute(pos, valid, "unit/bad", image=torch.rand(2, 8, 8, 3))


def test_video_overlap_validation(nodes, make_batch):
    batch = make_batch(t=2, h=32, w=32)
    with pytest.raises(ValueError):
        nodes.DepthPro2Video.execute(None, batch, 2, 2, True, 0.0, 0)


def test_video_focal_toggle(nodes, make_batch, monkeypatch):
    """Same contract as the Estimate node: off requires an explicit focal length; on, 0 estimates from the first window."""
    batch = make_batch(t=2, h=32, w=32)
    seen = []

    def fake_run(model, image, f_px, max_size, desc, **kwargs):
        seen.append(f_px)
        raise _Sentinel()

    monkeypatch.setattr(nodes, "_run_inference", fake_run)

    # Off, nothing set -> ValueError before inference starts.
    with pytest.raises(ValueError, match="Estimate focal length"):
        nodes.DepthPro2Video.execute(None, batch, 2, 1, False, 0.0, 0)

    # Off, 35mm-equivalent -> converted pixels reach inference.
    with pytest.raises(_Sentinel):
        nodes.DepthPro2Video.execute(None, batch, 2, 1, False, 35.0, 0)
    assert abs(seen[-1] - 35.0 * math.hypot(32, 32) / math.hypot(36.0, 24.0)) < 1e-6

    # Off, linked pixels -> used as-is.
    with pytest.raises(_Sentinel):
        nodes.DepthPro2Video.execute(None, batch, 2, 1, False, 0.0, 500.0, 0)
    assert seen[-1] == 500.0

    # On, nothing set -> 0, the model estimates from the first window.
    with pytest.raises(_Sentinel):
        nodes.DepthPro2Video.execute(None, batch, 2, 1, True, 0.0, 0)
    assert seen[-1] == 0.0


TINY_NAME = "unit-tiny.safetensors"


@pytest.fixture(scope="session")
def tmp_model_dir():
    """A throwaway models dir registered after the real one, so test weight files
    never touch the user's models folder."""
    import folder_paths

    extra = Path(tempfile.gettempdir()) / "dp2-unit-models"
    extra.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path("depthpro2", str(extra))
    yield extra
    for stale in extra.glob("unit-*"):
        stale.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def tiny_model(nodes, tmp_model_dir):
    """A random-weight DepthPro2-S in the test models dir, for loader tests.

    Built with pretrained=False, so no weights download and no network.
    """
    config = nodes.PRESETS["depthpro2-s"]
    model = nodes.DepthPro2(
        backbone_preset=config.backbone_preset,
        reassembled_blocks=config.reassembled_blocks,
        cross_frame_blocks=config.cross_frame_blocks,
        dims_reassemble=config.dims_reassemble,
        dim_decoder=config.dim_decoder,
        fov=True,
        precision=torch.float16,
    )
    path = tmp_model_dir / TINY_NAME
    if not path.exists():
        state = {
            key: value.half() if value.is_floating_point() else value
            for key, value in model.state_dict().items()
        }
        # The architecture zero-initializes a few parameters (block biases, the FOV
        # query/prefix embeddings). Nudge the large ones so the corrupt-weights guard
        # (>=3 all-zero tensors) does not fire on a perfectly healthy fixture.
        for key, value in state.items():
            if value.is_floating_point() and value.numel() > 100 and not value.any():
                state[key] = torch.randn(value.shape, dtype=torch.float32).mul(1e-3).half()
        save_file(state, str(path))
        del state
    del model
    return TINY_NAME


def test_loader_auto_device_follows_session_policy(nodes, tiny_model, monkeypatch):
    # "auto" resolves through ComfyUI's device policy for auxiliary models (the
    # native pattern): a NO_VRAM session lands on the CPU, with
    # the model riding on the CPU as well.
    cpu = torch.device("cpu")
    monkeypatch.setattr(nodes.mm, "text_encoder_device", lambda: cpu)
    monkeypatch.setattr(nodes.mm, "text_encoder_offload_device", lambda: cpu)
    wrapper = nodes.load_depthpro2(tiny_model, "auto", "auto")
    assert wrapper["patcher"].load_device == cpu
    assert wrapper["patcher"].offload_device == cpu
    assert wrapper["precision"] == "fp32"  # CPU sessions compute in fp32


def test_loader_explicit_device_offloads_to_ram(nodes, tiny_model):
    wrapper = nodes.load_depthpro2(tiny_model, "fp16", "cuda")
    assert wrapper["patcher"].load_device == torch.device("cuda")
    assert wrapper["patcher"].offload_device == torch.device("cpu")


def test_corrupt_safetensors_message(nodes):
    # The same wording and layout as comfy.utils' native diagnostics.
    error = nodes._corrupt_safetensors("boom", "/w/weights.safetensors")
    assert str(error) == (
        "boom\n\nFile path: /w/weights.safetensors\n\n"
        "The safetensors file is corrupt/incomplete. Check the file size and make sure "
        "you have copied/downloaded it correctly."
    )
    error = nodes._corrupt_safetensors("boom", "/w/weights.safetensors", incomplete=False)
    message = str(error)
    assert "corrupt or invalid. Make sure this is actually a safetensors file" in message
    assert message.endswith("not a ckpt or pt or other filetype.")


def test_corrupt_local_weights_native_diagnostic(nodes, tmp_model_dir):
    path = tmp_model_dir / "unit-corrupt.safetensors"
    path.write_bytes(b"this is not a safetensors file, just some plain text bytes........")
    try:
        with pytest.raises(ValueError, match="corrupt/incomplete. Check the file size") as excinfo:
            nodes.load_depthpro2("unit-corrupt.safetensors", "bf16", "auto")
        assert str(path) in str(excinfo.value)
    finally:
        path.unlink(missing_ok=True)


def test_zero_filled_weights_native_diagnostic(nodes, tiny_model, tmp_model_dir):
    # A file that loads fine but is all zeros is reported with the native wording too.
    path = tmp_model_dir / tiny_model
    state = nodes._load_state_dict(path, torch.float16)
    for value in state.values():
        if value.is_floating_point():
            value.zero_()
    with pytest.raises(ValueError, match="corrupt or invalid") as excinfo:
        nodes._check_weights(state, tiny_model, str(path))
    assert str(path) in str(excinfo.value)