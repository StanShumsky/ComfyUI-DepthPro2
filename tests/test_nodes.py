"""Unit tests: no weights, no ComfyUI install, no network."""

import asyncio
import math
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
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


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_focal_mm_rejects_invalid_values(nodes, value):
    with pytest.raises(ValueError, match="finite non-negative"):
        nodes._focal_mm_to_px(value, 256, 384)


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
    with pytest.raises(ValueError, match="finite non-negative"):
        execute(float("nan"), full_frame, 36.0, 4096.0)
    with pytest.raises(ValueError, match="image_side_px"):
        execute(35.0, full_frame, 36.0, float("nan"))
    with pytest.raises(ValueError, match="sensor_width_mm"):
        execute(35.0, "custom", float("nan"), 4096.0)


def test_focal_length_schema_compatibility(nodes):
    schema = nodes.DepthPro2FocalLength.GET_SCHEMA()
    assert [inp.id for inp in schema.inputs] == [
        "focal_length_mm", "sensor", "sensor_width_mm", "image_side_px", "image",
    ]
    assert [inp.default for inp in schema.inputs[:4]] == [35.0, next(iter(nodes.SENSOR_PRESETS)), 36.0, 1920.0]
    assert schema.inputs[4].optional
    assert schema.is_output_node
    assert [out.id for out in schema.outputs] == ["focal_length_px"]


def test_focal_length_frontend_readout_asset():
    package = sys.modules["dp2node"]
    root = Path(__file__).resolve().parents[1]
    assert package.WEB_DIRECTORY == "./web"
    source = (root / package.WEB_DIRECTORY / "focal_length.js").read_text(encoding="utf-8")
    assert "DepthPro2FocalLength" in source
    assert "onNodeOutputsUpdated" in source
    assert "node.onExecuted" in source


@pytest.mark.parametrize(
    ("height", "width", "orientation"),
    [(2400, 3600, "landscape"), (3600, 2400, "portrait"), (3000, 3000, "square")],
)
def test_focal_length_image_dimensions(nodes, height, width, orientation):
    image = torch.zeros(1, height, width, 3)
    output = nodes.DepthPro2FocalLength.execute(
        50.0, next(iter(nodes.SENSOR_PRESETS)), 36.0, 1920.0, image
    )
    (focal_px,) = output.result
    assert abs(focal_px - 50.0 * max(height, width) / 36.0) < 1e-6
    assert f"Image: {width} × {height} ({orientation})" in output.ui.value
    assert "Active sensor width: 36.00 mm" in output.ui.value
    assert "FOV:" in output.ui.value


def test_focal_length_exif_mode(nodes):
    image = torch.zeros(1, 256, 384, 3)
    mode = "35mm equivalent (EXIF / unknown sensor)"
    output = nodes.DepthPro2FocalLength.execute(35.0, mode, 0.0, 0.0, image)
    (focal_px,) = output.result
    assert abs(focal_px - nodes._focal_mm_to_px(35.0, 256, 384)) < 1e-6
    assert "Mode: 35mm equivalent / EXIF" in output.ui.value
    assert "35mm equivalent: 35.00 mm" in output.ui.value


def test_focal_length_custom_readout_with_image(nodes):
    image = torch.zeros(1, 2160, 3840, 3)
    (focal_px,) = nodes.DepthPro2FocalLength.execute(
        35.0, "custom", 27.03, 1920.0, image
    ).result
    assert abs(focal_px - 35.0 * 3840 / 27.03) < 1e-6


def test_focal_length_image_validation(nodes):
    execute = nodes.DepthPro2FocalLength.execute
    full_frame = next(iter(nodes.SENSOR_PRESETS))
    with pytest.raises(ValueError, match="IMAGE"):
        execute(35.0, "35mm equivalent (EXIF / unknown sensor)", 36.0, 1920.0)
    with pytest.raises(ValueError, match=r"\(B, H, W, 3\)"):
        execute(35.0, full_frame, 36.0, 1920.0, torch.zeros(3, 256, 384))
    with pytest.raises(ValueError, match="at least one image"):
        execute(35.0, full_frame, 36.0, 1920.0, torch.zeros(0, 256, 384, 3))


def test_focal_length_zero_and_frontend_validation(nodes):
    node = nodes.DepthPro2FocalLength
    full_frame = next(iter(nodes.SENSOR_PRESETS))
    output = node.execute(0.0, full_frame, 36.0, 0.0)
    assert output.result == (0.0,)
    assert "automatic focal estimation" in output.ui.value
    assert node.validate_inputs(35.0, full_frame, 36.0, 1920.0) is True
    assert node.validate_inputs(
        35.0, "35mm equivalent (EXIF / unknown sensor)", 36.0, 1920.0
    ) is True
    error = node.validate_inputs(
        -1.0, "35mm equivalent (EXIF / unknown sensor)", 36.0, 1920.0
    )
    assert isinstance(error, str) and "finite non-negative" in error


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
        nodes.DepthPro2Estimate.execute(None, batch, False, 0.0, None)

    # Off, 35mm-equivalent -> converted pixels reach inference.
    with pytest.raises(_Sentinel):
        nodes.DepthPro2Estimate.execute(None, batch, False, 35.0, None)
    assert abs(seen[-1] - 35.0 * math.hypot(32, 32) / math.hypot(36.0, 24.0)) < 1e-6

    # Off, linked pixels -> used as-is.
    with pytest.raises(_Sentinel):
        nodes.DepthPro2Estimate.execute(None, batch, False, 0.0, 500.0, 0)
    assert seen[-1] == 500.0

    # On, nothing set -> 0, the model estimates.
    with pytest.raises(_Sentinel):
        nodes.DepthPro2Estimate.execute(None, batch, True, 0.0, None)
    assert seen[-1] == 0.0


@pytest.mark.parametrize("node", ["estimate", "video"])
def test_linked_zero_focal_overrides_mm(nodes, make_batch, monkeypatch, node):
    batch = make_batch(t=1, h=32, w=32)
    seen = []

    def fake_run(model, image, f_px, max_size, desc, **kwargs):
        seen.append(f_px)
        raise _Sentinel()

    monkeypatch.setattr(nodes, "_run_inference", fake_run)
    with pytest.raises(_Sentinel):
        if node == "estimate":
            nodes.DepthPro2Estimate.execute(
                None, batch, True, 35.0, focal_length_px=0.0, max_size=0
            )
        else:
            nodes.DepthPro2Video.execute(
                None, batch, 2, 1, True, 35.0,
                focal_length_px=0.0, max_size=0, mode="buffered",
            )
    assert seen == [0.0]


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_linked_focal_rejects_invalid_values(nodes, make_batch, monkeypatch, value):
    batch = make_batch(t=1, h=32, w=32)

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid focal length must fail before inference")

    monkeypatch.setattr(nodes, "_run_inference", unexpected)
    with pytest.raises(ValueError, match="finite non-negative"):
        nodes.DepthPro2Estimate.execute(
            None, batch, True, 0.0, focal_length_px=value, max_size=0
        )


def test_prepare_frames_resize_and_focal(nodes, make_batch):
    batch = make_batch(t=2, h=64, w=96)
    frames, focal, scale = nodes._prepare_frames(batch, 48, 500.0)
    assert scale == 0.5 and focal == 250.0
    # CPU uint8 staging: the frames stay on the CPU until consumed (the native pattern).
    assert len(frames) == 2 and frames[0].shape == (32, 48, 3)
    assert frames[0].dtype == torch.uint8 and frames[0].device.type == "cpu"
    frames, focal, scale = nodes._prepare_frames(batch, 0, 0.0)
    assert scale == 1.0 and focal is None and frames[0].shape == (64, 96, 3)


def _mps_memory_model(nodes, monkeypatch, recommended, active, loaded=False, system_free=64 << 30):
    class Patcher:
        def model_size(self):
            return 1 << 30

    patcher = Patcher()
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: recommended)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: active)
    monkeypatch.setattr(nodes.mm, "get_free_memory", lambda device: system_free)
    monkeypatch.setattr(nodes.mm, "loaded_models", lambda: [patcher] if loaded else [])
    return {
        "device": torch.device("mps"),
        "name": "depthpro2-s",
        "precision": "fp16",
        "patcher": patcher,
    }


def test_auto_max_size_keeps_native_mps_resolution_when_it_fits(nodes, monkeypatch):
    model = _mps_memory_model(nodes, monkeypatch, 32 << 30, 1 << 29)
    assert nodes._auto_max_size(model, 2160, 4096) == 0


def test_auto_max_size_caps_mps_to_available_working_set(nodes, monkeypatch):
    model = _mps_memory_model(nodes, monkeypatch, 8 << 30, 1 << 30)
    assert nodes._auto_max_size(model, 2160, 4096) == 3264


def test_auto_max_size_mps_accounts_for_active_models(nodes, monkeypatch):
    model = _mps_memory_model(nodes, monkeypatch, 8 << 30, 4 << 30, loaded=True)
    assert nodes._auto_max_size(model, 2160, 4096) == 2112


def test_auto_max_size_mps_ignores_comfyui_system_ram_budget(nodes, monkeypatch):
    model = _mps_memory_model(
        nodes, monkeypatch, 32 << 30, 1 << 29, system_free=1 << 30
    )
    assert nodes._auto_max_size(model, 2160, 4096) == 0


def test_auto_max_size_mps_falls_back_when_budget_api_is_missing(nodes, monkeypatch):
    model = _mps_memory_model(nodes, monkeypatch, 8 << 30, 0)
    monkeypatch.delattr(torch.mps, "recommended_max_memory")
    assert nodes._auto_max_size(model, 2160, 4096) == 0


def test_auto_max_size_cpu_does_not_query_mps_budget(nodes, monkeypatch):
    def unexpected():
        raise AssertionError("CPU auto sizing must not query MPS")

    monkeypatch.setattr(torch.mps, "recommended_max_memory", unexpected)
    assert nodes._auto_max_size({"device": torch.device("cpu")}, 2160, 4096) == 0


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


def test_depth_outputs_upscale_does_not_spread_sky(nodes):
    inverse = torch.ones(1, 4, 4)
    inverse[:, :2] = 0.0
    depth = 1.0 / inverse
    prediction = {
        "inverse_depth": inverse[0],
        "depth": depth[0],
        "focallength_px": torch.tensor(250.0),
    }

    _, upsampled, _, sky = nodes._depth_outputs([prediction], (8, 8), 0.5)

    assert not torch.isnan(upsampled).any()
    assert torch.isinf(upsampled[0, 2]).all()
    assert torch.equal(upsampled[0, 3], torch.ones(8))
    assert torch.equal(sky.bool(), ~torch.isfinite(upsampled))


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


def test_save_float_uses_literal_prefix_and_max_index(nodes, output_dir):
    folder = output_dir / "unit"
    folder.mkdir()
    (folder / "literal[1]_00003.tif").write_bytes(b"keep-three")
    (folder / "literal[1]_00009.tif").write_bytes(b"keep-nine")
    (folder / "literal1_00020.tif").write_bytes(b"unrelated")

    nodes.DepthPro2SaveFloat.execute(synthetic_depth(t=1), "unit/literal[1]", "tiff", 0.0)

    assert (folder / "literal[1]_00010.tif").exists()
    assert (folder / "literal[1]_00003.tif").read_bytes() == b"keep-three"
    assert (folder / "literal[1]_00009.tif").read_bytes() == b"keep-nine"
    assert (folder / "literal1_00020.tif").read_bytes() == b"unrelated"


def test_save_ply_concurrent_prefixes_do_not_overwrite(nodes, output_dir):
    position = torch.ones(1, 2, 3, 3)
    valid = torch.ones(1, 2, 3)

    def save(_):
        nodes.DepthPro2SavePLY.execute(position, valid, "unit/concurrent")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(save, range(4)))

    files = sorted((output_dir / "unit").glob("concurrent_*.ply"))
    assert len(files) == 4
    assert all(b"element vertex 6" in path.read_bytes() for path in files)


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


def test_depth_consumers_reject_empty_or_malformed_payloads(nodes):
    invalid_depths = [
        torch.empty(0, 16, 24),
        torch.empty(1, 0, 24),
        torch.empty(1, 16, 24, 1),
    ]
    for depth in invalid_depths:
        calls = [
            lambda: nodes.DepthPro2Colorize.execute(depth, "turbo"),
            lambda: nodes.DepthPro2PositionPass.execute(depth, 500.0, 0.0, 0.0, 0.0),
            lambda: nodes.DepthPro2Save16.execute(depth, "unit/invalid"),
            lambda: nodes.DepthPro2SaveFloat.execute(depth, "unit/invalid", "tiff", 0.0),
        ]
        for call in calls:
            with pytest.raises(ValueError):
                call()


def test_save_ply_rejects_mismatched_or_empty_inputs(nodes):
    position = torch.zeros(1, 16, 24, 3)
    valid = torch.zeros(1, 16, 24)
    with pytest.raises(ValueError, match="at least one"):
        nodes.DepthPro2SavePLY.execute(torch.empty(0, 16, 24, 3), valid, "unit/invalid")
    with pytest.raises(ValueError, match="valid must be a MASK"):
        nodes.DepthPro2SavePLY.execute(position, torch.zeros(1, 8, 12), "unit/invalid")
    with pytest.raises(ValueError, match="image must match position"):
        nodes.DepthPro2SavePLY.execute(
            position, valid, "unit/invalid", image=torch.zeros(2, 16, 24, 3)
        )


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
        nodes.DepthPro2Video.execute(None, batch, 2, 1, False, 0.0, None)

    # Off, 35mm-equivalent -> converted pixels reach inference.
    with pytest.raises(_Sentinel):
        nodes.DepthPro2Video.execute(None, batch, 2, 1, False, 35.0, None)
    assert abs(seen[-1] - 35.0 * math.hypot(32, 32) / math.hypot(36.0, 24.0)) < 1e-6

    # Off, linked pixels -> used as-is.
    with pytest.raises(_Sentinel):
        nodes.DepthPro2Video.execute(None, batch, 2, 1, False, 0.0, 500.0, 0)
    assert seen[-1] == 500.0

    # On, nothing set -> 0, the model estimates from the first window.
    with pytest.raises(_Sentinel):
        nodes.DepthPro2Video.execute(None, batch, 2, 1, True, 0.0, None)
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


def test_local_weights_use_detected_preset_for_auto_sizing(nodes, tiny_model):
    wrapper = nodes.load_depthpro2(tiny_model, "fp32", "cpu")

    assert wrapper["name"] == "depthpro2-s"
    assert nodes._window_boundary(wrapper["name"], 4) == nodes._window_boundary("depthpro2-s", 4)
    assert nodes.AUTO_BOUNDARY_MPS[wrapper["name"]] == 48.0


@pytest.fixture
def main_only_model(nodes, tiny_model, tmp_model_dir, monkeypatch):
    source = nodes._load_state_dict(tmp_model_dir / tiny_model, torch.float16)
    main = {key: value for key, value in source.items() if not key.startswith("fov.")}
    focal = {key[4:]: value for key, value in source.items() if key.startswith("fov.")}
    path = tmp_model_dir / "unit-main-only.safetensors"
    save_file(main, str(path))
    calls = []

    def load_focal(repo_id, name, dtype):
        calls.append((repo_id, name, dtype))
        return tmp_model_dir / "depthpro2-focal", focal

    monkeypatch.setattr(nodes, "_load_repo_verified", load_focal)
    return nodes.load_depthpro2(path.name, "fp16", "cpu"), calls


def test_focal_model_loads_lazily_once(nodes, main_only_model):
    wrapper, calls = main_only_model

    assert calls == []
    assert wrapper["model"].fov is None
    assert wrapper["focal_pending"]

    nodes._ensure_focal_model(wrapper)
    nodes._ensure_focal_model(wrapper)

    assert calls == [("apple/DepthPro2-focal", "depthpro2-focal", torch.float16)]
    assert wrapper["model"].fov is not None
    assert not wrapper["focal_pending"]


def test_direct_python_auto_focal_initializes_once(nodes, main_only_model, monkeypatch):
    wrapper, calls = main_only_model
    net = wrapper["model"]
    monkeypatch.setattr(net, "predict_inverse_depth", lambda x, cache=None: torch.ones(x.shape[0], 32, 32))

    prediction = net.infer(torch.zeros(3, 32, 32))
    assert prediction["focallength_px"] is not None
    assert net.fov is not None

    net.stream(window=1, overlap=0)
    assert list(net.infer_video(iter(()), window=1, overlap=0)) == []
    assert len(calls) == 1


def test_direct_python_auto_focal_after_explicit_loaded_run(nodes, main_only_model, monkeypatch):
    wrapper, calls = main_only_model
    net = wrapper["model"]
    patcher = wrapper["patcher"]
    detached = []
    monkeypatch.setattr(net, "predict_inverse_depth", lambda x, cache=None: torch.ones(x.shape[0], 32, 32))
    monkeypatch.setattr(nodes.mm, "loaded_models", lambda only_currently_used=False: [patcher])
    monkeypatch.setattr(patcher, "detach", lambda: detached.append(True), raising=False)

    explicit = net.infer(torch.zeros(3, 32, 32), f_px=500.0)
    assert explicit["focallength_px"].item() == 500.0
    assert net.fov is None
    assert calls == []

    automatic = net.infer(torch.zeros(3, 32, 32))
    assert automatic["focallength_px"] is not None
    assert detached == [True]
    assert len(calls) == 1


def test_concurrent_focal_initialization_loads_once(nodes, main_only_model):
    wrapper, calls = main_only_model
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(nodes._ensure_focal_model, [wrapper] * 8))
    assert len(calls) == 1
    assert wrapper["model"].fov is not None


def test_explicit_focal_skips_pending_focal_model(nodes, make_batch, monkeypatch):
    class Net:
        def infer(self, frame, f_px=None):
            return _fake_prediction()

    def unexpected(_):
        raise AssertionError("explicit focal length must not load the focal model")

    monkeypatch.setattr(nodes, "_ensure_focal_model", unexpected)
    monkeypatch.setattr(
        nodes, "to_normalized_tensor", lambda frame, device: frame.permute(2, 0, 1).float()
    )
    model = {
        "model": Net(), "device": torch.device("cpu"), "focal_pending": True,
    }

    predictions, _ = nodes._run_inference(
        model, make_batch(t=1, h=32, w=32), 500.0, -1, "explicit focal test"
    )

    assert len(predictions) == 1


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


def test_loader_auto_uses_accelerator_in_shared_memory_session(nodes, tiny_model, monkeypatch):
    cpu = torch.device("cpu")
    mps = torch.device("mps")
    monkeypatch.setattr(nodes.mm, "vram_state", nodes.mm.VRAMState.SHARED)
    monkeypatch.setattr(nodes.mm, "get_torch_device", lambda: mps)
    monkeypatch.setattr(nodes.mm, "text_encoder_device", lambda: cpu)
    monkeypatch.setattr(nodes.mm, "text_encoder_offload_device", lambda: cpu)
    wrapper = nodes.load_depthpro2(tiny_model, "auto", "auto")
    assert wrapper["patcher"].load_device == mps
    assert wrapper["patcher"].offload_device == cpu
    assert wrapper["device"] == mps
    assert wrapper["precision"] == "fp16"


def test_loader_auto_keeps_cpu_when_shared_accelerator_is_unavailable(nodes, tiny_model, monkeypatch):
    cpu = torch.device("cpu")
    monkeypatch.setattr(nodes.mm, "vram_state", nodes.mm.VRAMState.SHARED)
    monkeypatch.setattr(nodes.mm, "get_torch_device", lambda: cpu)
    monkeypatch.setattr(nodes.mm, "text_encoder_device", lambda: cpu)
    monkeypatch.setattr(nodes.mm, "text_encoder_offload_device", lambda: cpu)
    wrapper = nodes.load_depthpro2(tiny_model, "auto", "auto")
    assert wrapper["device"] == cpu and wrapper["precision"] == "fp32"


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


def test_repo_download_uses_pinned_revision(nodes, tmp_path, monkeypatch):
    captured = {}

    def fake_download(repo_id, **kwargs):
        captured.update(kwargs)
        (Path(kwargs["local_dir"]) / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(nodes, "_model_dirs", lambda: [tmp_path])
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)

    nodes._download_repo("apple/DepthPro2-S", "depthpro2-s")

    assert captured["revision"] == "903a1c1ade918831d138adcc4aedbcc8f4da0078"


def test_forced_redownload_quarantines_only_target_weight(nodes, tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    local = first / "depthpro2-s"
    other = second / "depthpro2-s"
    local.mkdir(parents=True)
    other.mkdir(parents=True)
    (local / "model.safetensors").write_bytes(b"corrupt")
    (local / "config.json").write_text("{}")
    (local / "notes.txt").write_text("keep")
    (other / "model.safetensors").write_bytes(b"other copy")
    monkeypatch.setattr(nodes, "_model_dirs", lambda: [first, second])

    def fake_download(repo_id, allow_patterns, local_dir, force_download, revision):
        assert repo_id == "apple/DepthPro2-S" and force_download
        assert revision == "903a1c1ade918831d138adcc4aedbcc8f4da0078"
        (Path(local_dir) / "model.safetensors").write_bytes(b"fresh")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    result = nodes._download_repo("apple/DepthPro2-S", "depthpro2-s", force=True)
    assert result == local
    assert (local / "model.safetensors").read_bytes() == b"fresh"
    assert (local / "config.json").read_text() == "{}"
    assert (local / "notes.txt").read_text() == "keep"
    assert (other / "model.safetensors").read_bytes() == b"other copy"
    backups = list(local.glob("model.safetensors.corrupt-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == b"corrupt"


def test_repo_verification_retries_corruption(nodes, tmp_path, monkeypatch):
    local = tmp_path / "depthpro2-s"
    local.mkdir()
    forces = []

    def fake_download(repo_id, name, force=False):
        forces.append(force)
        return local

    attempts = 0

    def fake_load(path, dtype):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise nodes.SafetensorError("truncated header")
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(nodes, "_download_repo", fake_download)
    monkeypatch.setattr(nodes, "_load_state_dict", fake_load)
    monkeypatch.setattr(nodes, "_check_weights", lambda *args: None)
    result, state = nodes._load_repo_verified("apple/DepthPro2-S", "depthpro2-s", torch.float16)
    assert result == local and state["weight"].item() == 1
    assert forces == [False, True]


def test_repo_verification_does_not_redownload_runtime_errors(nodes, tmp_path, monkeypatch):
    local = tmp_path / "depthpro2-s"
    local.mkdir()
    forces = []

    def fake_download(repo_id, name, force=False):
        forces.append(force)
        return local

    def fake_load(path, dtype):
        raise RuntimeError("out of memory while reading weights")

    monkeypatch.setattr(nodes, "_download_repo", fake_download)
    monkeypatch.setattr(nodes, "_load_state_dict", fake_load)
    with pytest.raises(RuntimeError, match="out of memory"):
        nodes._load_repo_verified("apple/DepthPro2-S", "depthpro2-s", torch.float16)
    assert forces == [False]


def test_local_weight_paths_stay_inside_model_roots(nodes, tmp_path, monkeypatch):
    root = tmp_path / "models"
    nested = root / "nested"
    nested.mkdir(parents=True)
    inside = nested / "weights.safetensors"
    inside.write_bytes(b"inside")
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    monkeypatch.setattr(nodes, "_model_dirs", lambda: [root])

    assert nodes._find_local_file("nested/weights.safetensors") == inside.resolve()
    for name in ("../outside.safetensors", str(outside)):
        with pytest.raises(ValueError, match="outside"):
            nodes._find_local_file(name)
    with pytest.raises(FileNotFoundError):
        nodes._find_local_file("missing.safetensors")


def test_local_weight_paths_reject_escaping_symlinks(nodes, tmp_path, monkeypatch):
    root = tmp_path / "models"
    nested = root / "nested"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    try:
        (nested / "escape.safetensors").symlink_to(outside)
    except OSError as error:
        if getattr(error, "winerror", None) != 1314:
            raise
        pytest.skip("Creating symlinks on Windows requires elevated privileges")
    monkeypatch.setattr(nodes, "_model_dirs", lambda: [root])

    with pytest.raises(ValueError, match="outside"):
        nodes._find_local_file("nested/escape.safetensors")


def _fake_prediction():
    return {
        "inverse_depth": torch.ones(4, 4),
        "depth": torch.ones(4, 4),
        "focallength_px": torch.tensor(100.0),
    }


@pytest.mark.parametrize(
    "height,width,max_size,expected",
    [
        (64, 128, -1, [(64, 128), (32, 64)]),
        (64, 64, 256, [(64, 64), (32, 32)]),
    ],
)
def test_inference_retries_accelerator_oom_without_upscaling(
    nodes, make_batch, monkeypatch, height, width, max_size, expected
):
    sizes = []

    class Net:
        def infer(self, frame, f_px=None):
            sizes.append(tuple(frame.shape[-2:]))
            if len(sizes) == 1:
                raise torch.OutOfMemoryError("MPS backend out of memory")
            return _fake_prediction()

    monkeypatch.setattr(
        nodes, "to_normalized_tensor", lambda frame, device: frame.permute(2, 0, 1).float()
    )
    model = {"model": Net(), "device": torch.device("mps"), "patcher": object()}
    predictions, scale = nodes._run_inference(
        model, make_batch(t=1, h=height, w=width), 0.0, max_size, "OOM retry test"
    )
    assert sizes == expected
    assert len(predictions) == 1
    assert scale == 0.5


def test_inference_retries_oom_during_model_load(nodes, make_batch, monkeypatch):
    sizes = []
    reservations = []

    class Net:
        def infer(self, frame, f_px=None):
            sizes.append(tuple(frame.shape[-2:]))
            return _fake_prediction()

    def load(models, memory_required=0, **kwargs):
        reservations.append(memory_required)
        if len(reservations) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory while loading weights")

    monkeypatch.setattr(nodes.mm, "load_models_gpu", load)
    monkeypatch.setattr(
        nodes, "to_normalized_tensor", lambda frame, device: frame.permute(2, 0, 1).float()
    )

    class Patcher:
        def model_size(self):
            return 0

    model = {
        "model": Net(), "device": torch.device("cuda"), "patcher": Patcher(),
        "name": "depthpro2-s", "precision": "fp32",
    }

    predictions, scale = nodes._run_inference(
        model, make_batch(t=1, h=64, w=128), 0.0, -1, "load OOM retry test"
    )

    assert sizes == [(32, 64)]
    assert reservations[0] > reservations[1] > 0
    assert len(predictions) == 1 and scale == 0.5


def test_inference_stops_retrying_at_patch_floor(nodes, make_batch, monkeypatch):
    sizes = []

    class Net:
        def infer(self, frame, f_px=None):
            sizes.append(tuple(frame.shape[-2:]))
            raise torch.OutOfMemoryError("MPS backend out of memory")

    monkeypatch.setattr(
        nodes, "to_normalized_tensor", lambda frame, device: frame.permute(2, 0, 1).float()
    )
    model = {"model": Net(), "device": torch.device("mps"), "patcher": object()}
    with pytest.raises(RuntimeError, match="out of memory"):
        nodes._run_inference(model, make_batch(t=1, h=16, w=16), 0.0, -1, "OOM floor test")
    assert sizes == [(16, 16)]


def test_inference_retries_video_accelerator_oom(nodes, make_batch, monkeypatch):
    sizes = []

    class Net:
        def infer_video(self, frames, **kwargs):
            def generate():
                for frame in frames:
                    sizes.append(tuple(frame.shape[-2:]))
                    if len(sizes) == 1:
                        raise torch.OutOfMemoryError("MPS backend out of memory")
                    yield _fake_prediction()

            return generate()

    monkeypatch.setattr(
        nodes, "to_normalized_tensor", lambda frame, device: frame.permute(2, 0, 1).float()
    )
    model = {"model": Net(), "device": torch.device("mps"), "patcher": object()}
    predictions, scale = nodes._run_inference(
        model, make_batch(t=2, h=64, w=128), 0.0, -1, "video OOM retry test",
        window=1, mode="unbuffered",
    )
    assert sizes == [(64, 128), (32, 64), (32, 64)]
    assert len(predictions) == 2 and scale == 0.5


def test_inference_reraises_non_oom_accelerator_errors(nodes, make_batch, monkeypatch):
    error = RuntimeError("MPS backend received an invalid tensor")
    calls = []

    class Net:
        def infer(self, frame, f_px=None):
            calls.append(tuple(frame.shape[-2:]))
            raise error

    monkeypatch.setattr(
        nodes, "to_normalized_tensor", lambda frame, device: frame.permute(2, 0, 1).float()
    )
    model = {"model": Net(), "device": torch.device("mps"), "patcher": object()}
    with pytest.raises(RuntimeError) as caught:
        nodes._run_inference(model, make_batch(t=1, h=64, w=64), 0.0, -1, "invalid tensor test")
    assert caught.value is error
    assert calls == [(64, 64)]


def test_inference_checks_interrupt_between_images(nodes, make_batch, monkeypatch):
    calls = []

    class Net:
        def infer(self, frame, f_px=None):
            calls.append("infer")
            return _fake_prediction()

    checks = 0

    def interrupt():
        nonlocal checks
        checks += 1
        if checks == 3:
            raise _Sentinel()

    monkeypatch.setattr(nodes.mm, "throw_exception_if_processing_interrupted", interrupt)
    model = {"model": Net(), "device": torch.device("cpu")}
    with pytest.raises(_Sentinel):
        nodes._run_inference(model, make_batch(t=3, h=8, w=8), 0.0, -1, "interrupt test")
    assert calls == ["infer"]


def test_inference_checks_interrupt_before_next_video_window(nodes, make_batch, monkeypatch):
    calls = []

    class Net:
        def infer_video(self, frames, **kwargs):
            calls.append("window 1")
            yield _fake_prediction()
            calls.append("window 2")
            yield _fake_prediction()

    checks = 0

    def interrupt():
        nonlocal checks
        checks += 1
        if checks == 3:
            raise _Sentinel()

    monkeypatch.setattr(nodes.mm, "throw_exception_if_processing_interrupted", interrupt)
    model = {"model": Net(), "device": torch.device("cpu")}
    with pytest.raises(_Sentinel):
        nodes._run_inference(
            model, make_batch(t=2, h=8, w=8), 0.0, -1, "interrupt test",
            window=1, mode="unbuffered",
        )
    assert calls == ["window 1"]


def test_estimate_and_video_reject_empty_or_malformed_batches(nodes):
    empty = torch.empty(0, 32, 32, 3)
    with pytest.raises(ValueError, match="at least one image"):
        nodes.DepthPro2Estimate.execute(None, empty, True, 0.0)
    with pytest.raises(ValueError, match="at least one image"):
        nodes.DepthPro2Video.execute(None, empty, 2, 1, True, 0.0)
    malformed = torch.empty(1, 32, 32, 4)
    with pytest.raises(ValueError, match=r"\(B, H, W, 3\)"):
        nodes.DepthPro2Estimate.execute(None, malformed, True, 0.0)
    with pytest.raises(ValueError, match="non-empty spatial"):
        nodes.DepthPro2Video.execute(None, torch.empty(1, 0, 32, 3), 2, 1, True, 0.0)


def test_depth_outputs_rejects_empty_predictions(nodes):
    with pytest.raises(ValueError, match="no predictions"):
        nodes._depth_outputs([], (32, 32), 1.0)
