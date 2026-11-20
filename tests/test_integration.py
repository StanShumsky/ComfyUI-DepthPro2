"""Integration tests: load real weights and run the nodes end to end.

Opt in with RUN_DP2_WEIGHT_TESTS=1; weights download once into DP2_MODELS_DIR
(default: a temp dir) on first use.
"""

import math
import os

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_DP2_WEIGHT_TESTS") != "1",
        reason="set RUN_DP2_WEIGHT_TESTS=1 to download weights and run integration tests",
    ),
]


@pytest.fixture(scope="module")
def wrapper(nodes):
    (wrapper,) = nodes.DepthPro2Loader().load("depthpro2-s", "bf16", "auto")
    return wrapper


def test_estimate(nodes, wrapper, make_batch):
    batch = make_batch(t=2)
    image, depth, focal, sky = nodes.DepthPro2Estimate().estimate(wrapper, batch, 0.0, 0)
    assert image.shape == (2, 256, 384, 3) and depth.shape == (2, 256, 384)
    assert 0 <= image.min() and image.max() <= 1 and focal > 0
    assert torch.isfinite(depth).float().mean() > 0.5
    assert sky.shape == (2, 256, 384)
    assert torch.equal(sky.bool(), ~torch.isfinite(depth))


def test_estimate_focal_override_and_resize(nodes, wrapper, make_batch):
    batch = make_batch(t=1)
    expected = 50.0 * math.hypot(384, 256) / math.hypot(36.0, 24.0)
    _, _, focal, _ = nodes.DepthPro2Estimate().estimate(wrapper, batch, 50.0, 192)
    assert abs(focal - expected) / expected < 1e-5


def test_focal_px_override(nodes, wrapper, make_batch):
    batch = make_batch(t=1)
    _, _, focal, _ = nodes.DepthPro2Estimate().estimate(wrapper, batch, 0.0, 0, focal_length_px=500.0)
    assert abs(focal - 500.0) < 1e-3
    _, _, focal, _ = nodes.DepthPro2Video().estimate(
        wrapper, batch, 2, 1, 0.0, 0, "buffered", focal_length_px=500.0
    )
    assert abs(focal - 500.0) < 1e-3


def test_video_modes(nodes, wrapper, make_batch):
    batch = make_batch(t=4)
    for mode in ("buffered", "unbuffered"):
        image, depth, focal, sky = nodes.DepthPro2Video().estimate(wrapper, batch, 4, 2, 0.0, 0, mode)
        assert image.shape == (4, 256, 384, 3) and focal > 0
        assert torch.isfinite(depth).float().mean() > 0.5
        assert torch.equal(sky.bool(), ~torch.isfinite(depth))
    nodes.DepthPro2Video().estimate(wrapper, batch[:2], 2, 2, 0.0, 0, "unbuffered")


def test_colorize_matches_upstream_cli(nodes, wrapper, make_batch):
    matplotlib = pytest.importorskip("matplotlib")
    batch = make_batch(t=2)
    _, depth, _, _ = nodes.DepthPro2Estimate().estimate(wrapper, batch, 0.0, 0)
    inverse = nodes.inverse_metric(depth)
    scale = float(inverse[0].max()) * 1.2
    lut = (matplotlib.colormaps["turbo"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    expected = lut[np.clip(inverse.numpy() * (256 / scale), 0, 255).astype(np.int32)] / 255.0
    (color,) = nodes.DepthPro2Colorize().colorize(depth, "turbo")
    assert np.allclose(color.numpy(), expected, atol=1e-6)


def test_position_pass_matches_formula(nodes, wrapper, make_batch):
    batch = make_batch(t=1)
    _, depth, focal, _ = nodes.DepthPro2Estimate().estimate(wrapper, batch, 0.0, 0)
    pos, valid = nodes.DepthPro2PositionPass().position(depth, focal, 0.0, 0.0, 0.0)
    finite = torch.isfinite(depth) & (depth > 0)
    assert torch.equal(valid.bool(), finite)
    v, u = 100, 200
    z = depth[0, v, u]
    assert abs(pos[0, v, u, 0] - (u + 0.5 - 192) * z / focal) < 1e-2
    assert abs(pos[0, v, u, 1] - (v + 0.5 - 128) * z / focal) < 1e-2


def test_save16_roundtrip(nodes, wrapper, output_dir, make_batch):
    batch = make_batch(t=1)
    _, depth, _, _ = nodes.DepthPro2Estimate().estimate(wrapper, batch, 0.0, 0)
    nodes.DepthPro2Save16().save(depth, "integration/depth16")
    from PIL import Image

    img = Image.open(next((output_dir / "integration").glob("depth16_*.png")))
    values = np.array(img).astype(np.float64)
    k = float(img.text["depthpro2_k"])
    near = values > 1000
    rel = np.abs(k / values[near] - depth[0].numpy()[near]) / depth[0].numpy()[near]
    assert np.median(rel) < 1e-3


def test_save_float_exr_roundtrip(nodes, wrapper, output_dir, make_batch):
    OpenEXR = pytest.importorskip("OpenEXR")
    Imath = pytest.importorskip("Imath")
    batch = make_batch(t=1)
    _, depth, focal, _ = nodes.DepthPro2Estimate().estimate(wrapper, batch, 0.0, 0)
    nodes.DepthPro2SaveFloat().save(depth, "integration/float", "exr", 0.0, focal_length_px=focal)
    path = next((output_dir / "integration").glob("float_*.exr"))
    src = OpenEXR.InputFile(str(path))
    header = src.header()
    assert header["depthpro2_units"] == b"meters"
    assert abs(float(header["depthpro2_focal_px"]) - focal) / focal < 1e-5
    w, h = header["dataWindow"].max.x + 1, header["dataWindow"].max.y + 1
    values = np.frombuffer(src.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(h, w)
    assert np.array_equal(values, depth[0].numpy(), equal_nan=True)


def test_save_ply_roundtrip(nodes, wrapper, output_dir, make_batch):
    batch = make_batch(t=1)
    _, depth, focal, _ = nodes.DepthPro2Estimate().estimate(wrapper, batch, 0.0, 0)
    pos, valid = nodes.DepthPro2PositionPass().position(depth, focal, 100.0, 0.0, 0.0)
    nodes.DepthPro2SavePLY().save(pos, valid, "integration/points", image=batch)
    path = next((output_dir / "integration").glob("points_*.ply"))
    header, _, body = path.read_bytes().partition(b"end_header\n")
    n = int(header.split(b"element vertex ")[1].split(b"\n")[0])
    assert n == int(valid.sum()) and n > 0
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec = np.frombuffer(body, dtype=dtype)
    assert len(rec) == n and ((rec["z"] > 0) & (rec["z"] <= 100.0)).all()
    expected_z = depth[0][valid[0].bool()].numpy()
    assert np.allclose(np.sort(rec["z"]), np.sort(expected_z), atol=1e-4)


def test_fp8_stream_reads_dequantized_weight(nodes, make_batch):
    (wrapper,) = nodes.DepthPro2Loader().load("depthpro2-s", "fp8", "auto")
    net, device = wrapper["model"], wrapper["device"]
    frames, _, _ = nodes._prepare_frames(make_batch(t=4), device, 0, 0.0)
    stream = net.stream(window=4, mode="unbuffered")
    preds = []
    for frame in frames:
        preds.extend(stream.step(frame))
    preds.extend(stream.flush())
    assert len(preds) == 4 and float(preds[0]["focallength_px"]) > 0