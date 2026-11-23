"""Unit tests: no weights, no ComfyUI install, no network."""

import math
import sys

import numpy as np
import pytest
import torch
from PIL import Image


def synthetic_depth(t=2, h=16, w=24):
    depth = torch.linspace(1.0, 10.0, h * w).reshape(1, h, w).repeat(t, 1, 1)
    depth[0, :4, :4] = float("inf")
    return depth


def test_node_schemas(nodes):
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        assert isinstance(cls.DESCRIPTION, str) and cls.DESCRIPTION
        assert cls.CATEGORY == "depth/DepthPro2"
        assert callable(getattr(cls, cls.FUNCTION))
        returns = cls.RETURN_TYPES
        assert len(returns) == len(getattr(cls, "RETURN_NAMES", returns))
        required = cls.INPUT_TYPES()["required"]
        for spec in required.values():
            assert isinstance(spec, tuple) and len(spec) in (1, 2)


def test_focal_mm_to_px(nodes):
    assert nodes._focal_mm_to_px(0.0, 256, 384) == 0.0
    expected = 50.0 * math.hypot(384, 256) / math.hypot(36.0, 24.0)
    assert abs(nodes._focal_mm_to_px(50.0, 256, 384) - expected) < 1e-9
    # On a full-frame 3:2 sensor, 35mm-equivalent mm == physical mm.
    assert abs(nodes._focal_mm_to_px(50.0, 2400, 3600) - 5000.0) < 1e-6


def test_focal_length_node(nodes, make_batch):
    convert = nodes.DepthPro2FocalLength().convert
    full_frame = next(iter(nodes.SENSOR_PRESETS))
    (fpx,) = convert(50.0, full_frame, 36.0, 3600)
    assert abs(fpx - 5000.0) < 1e-6
    (fpx,) = convert(35.0, "custom", 27.03, 4096)
    assert abs(fpx - 35.0 * 4096 / 27.03) < 1e-6
    batch = make_batch(t=1, h=256, w=384)
    (fpx,) = convert(50.0, full_frame, 36.0, 1920, image=batch)
    assert abs(fpx - 50.0 * 384 / 36.0) < 1e-6
    (fpx,) = convert(0.0, "custom", 36.0, 1920)
    assert fpx == 0.0


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
    pos, valid = nodes.DepthPro2PositionPass().position(depth, focal, 0.0, 0.0, 0.0)
    finite = torch.isfinite(depth) & (depth > 0)
    assert pos.shape == (2, 16, 24, 3) and torch.equal(valid.bool(), finite)
    v, u = 10, 20
    z = depth[0, v, u]
    assert pos[0, v, u, 2] == z
    assert abs(pos[0, v, u, 0] - (u + 0.5 - 12) * z / focal) < 1e-4
    assert abs(pos[0, v, u, 1] - (v + 0.5 - 8) * z / focal) < 1e-4
    assert torch.all(pos[~finite] == 0)
    pos_c, valid_c = nodes.DepthPro2PositionPass().position(depth, focal, 5.0, 0.0, 0.0)
    assert torch.equal(valid_c.bool(), finite & (depth <= 5.0))
    assert torch.all(pos_c[~valid_c.bool()] == 0)
    pos_o, _ = nodes.DepthPro2PositionPass().position(depth, focal, 0.0, 10.0, -5.0)
    assert torch.allclose(pos_o[..., 0], pos[..., 0] - 10.0 * pos[..., 2] / focal, atol=1e-4)
    assert torch.allclose(pos_o[..., 1], pos[..., 1] + 5.0 * pos[..., 2] / focal, atol=1e-4)
    with pytest.raises(ValueError):
        nodes.DepthPro2PositionPass().position(depth, 0.0, 0.0, 0.0, 0.0)


def test_colorize_scale_and_sky(nodes):
    depth = synthetic_depth()
    inverse = nodes.inverse_metric(depth)
    assert inverse[0, 0, 0] == 0.0
    scale = nodes.color_scale(inverse)
    assert abs(scale - float(inverse[0].max()) * 1.2) < 1e-6
    (color,) = nodes.DepthPro2Colorize().colorize(depth, "turbo")
    lut = sys.modules["dp2node.colormaps"]._LUTS
    lut0 = torch.from_numpy(lut[0][0] / 255.0).float()
    assert torch.allclose(color[0, :4, :4], lut0)
    assert color.min() >= 0 and color.max() <= 1
    (gray,) = nodes.DepthPro2Colorize().colorize(depth, "gray")
    assert torch.allclose(gray[..., 0], (inverse / scale).clamp(0, 1))
    assert torch.equal(gray[..., 0], gray[..., 2])


def test_save16(nodes, output_dir):
    depth = synthetic_depth()
    result = nodes.DepthPro2Save16().save(depth, "unit/depth16")
    files = sorted((output_dir / "unit").glob("depth16_*.png"))
    assert len(files) == 2
    previews = result["ui"]["images"]
    assert len(previews) == 2 and all(p["type"] == "temp" for p in previews)
    preview = Image.open(output_dir / "temp" / previews[0]["filename"])
    assert preview.mode == "L" and preview.size == (24, 16)
    assert np.array(preview)[:4, :4].max() == 0
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
    result = nodes.DepthPro2SaveFloat().save(depth, "unit/float", "tiff", 0.0, focal_length_px=500.0)
    files = sorted((output_dir / "unit").glob("float_*.tif"))
    assert len(files) == 2
    img = Image.open(files[0])
    assert img.mode == "F"
    values = torch.from_numpy(np.array(img))
    finite = torch.isfinite(depth[0])
    assert torch.equal(values[finite], depth[0][finite])
    assert torch.isinf(values[~finite]).all()
    assert img.tag_v2[65000] == "meters" and img.tag_v2[65002] == "500"
    assert len(result["ui"]["images"]) == 2
    nodes.DepthPro2SaveFloat().save(depth, "unit/clipped", "tiff", 5000.0)
    clipped = torch.from_numpy(np.array(Image.open(sorted((output_dir / "unit").glob("clipped_*.tif"))[0])))
    assert clipped[0, 0] == 5000.0


def test_save_ply(nodes, output_dir):
    depth = synthetic_depth()
    pos, valid = nodes.DepthPro2PositionPass().position(depth, 500.0, 0.0, 0.0, 0.0)
    nodes.DepthPro2SavePLY().save(pos, valid, "unit/points")
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
    nodes.DepthPro2SavePLY().save(pos, valid, "unit/color", image=torch.rand(2, 16, 24, 3))
    cheader, _, cbody = sorted((output_dir / "unit").glob("color_*.ply"))[0].read_bytes().partition(b"end_header\n")
    assert b"property uchar red" in cheader and len(cbody) == n * 15
    with pytest.raises(ValueError):
        nodes.DepthPro2SavePLY().save(pos, valid, "unit/bad", image=torch.rand(2, 8, 8, 3))


def test_video_overlap_validation(nodes, make_batch):
    batch = make_batch(t=2, h=32, w=32)
    with pytest.raises(ValueError):
        nodes.DepthPro2Video().estimate(None, batch, 2, 2, 0.0, 0, "buffered")