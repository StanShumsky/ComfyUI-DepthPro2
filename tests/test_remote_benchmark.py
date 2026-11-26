import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PIL.TiffImagePlugin import ImageFileDirectory_v2


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dp2_remote_benchmark", ROOT / "benchmarks" / "benchmark_remote_api.py"
)
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)


def test_view_path_encodes_nested_output(monkeypatch):
    api = remote.ComfyAPI("http://example.invalid")
    paths = []

    def request(path):
        paths.append(path)
        return b"data"

    monkeypatch.setattr(api, "request", request)
    assert api.view_path("bench/run/depth_00001.tif") == b"data"
    assert paths == [
        "/api/view?filename=depth_00001.tif&subfolder=bench%2Frun&type=output"
    ]


def test_depth_metrics_reads_direct_export_paths(tmp_path):
    class FakeAPI:
        def view_path(self, path, asset_type="output"):
            return (tmp_path / path).read_bytes()

    prefix = "quality"
    folder = tmp_path / prefix
    folder.mkdir()
    frames = [
        np.array([[1.0, 2.0, np.inf], [0.5, 4.0, 3.0]], dtype=np.float32),
        np.array([[2.0, 4.0, np.inf], [1.0, 2.0, 6.0]], dtype=np.float32),
    ]
    for index, depth in enumerate(frames):
        info = ImageFileDirectory_v2()
        info[65002] = "123.5"
        Image.fromarray(depth).save(folder / f"depth_float_{index:05d}.tif", tiffinfo=info)
        values = np.where(np.isfinite(depth), np.rint(60.0 / depth), 0).astype(np.uint16)
        meta = PngInfo()
        meta.add_text("depthpro2_k", "60")
        Image.fromarray(values).save(folder / f"depth16_{index + 1:05d}.png", pnginfo=meta)

    metrics = remote.depth_metrics(FakeAPI(), prefix, len(frames))
    assert metrics["frames"] == 2
    assert metrics["finite_depth_fraction"] == pytest.approx(5 / 6)
    assert metrics["depth_min_m"] == 0.5
    assert metrics["depth_max_m"] == 6.0
    assert metrics["focal_length_px"] == 123.5
    assert metrics["focal_consistent"] is True
    assert metrics["png_compared_pixels"] == 10
    assert metrics["png_vs_float_max_abs_depth_m"] == 0.0
    assert metrics["temporal_inverse_depth_mae"] == pytest.approx(13 / 36)
