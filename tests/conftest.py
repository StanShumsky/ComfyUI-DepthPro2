"""Stubs for the ComfyUI modules the nodes import, so tests run without a ComfyUI install."""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
STATE = {
    "output_dir": tempfile.gettempdir(),
    "temp_dir": str(Path(tempfile.gettempdir()) / "dp2-temp"),
}


def _install_stubs() -> None:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = os.environ.get(
        "DP2_MODELS_DIR", str(Path(tempfile.gettempdir()) / "dp2-models")
    )
    folder_paths.add_model_folder_path = lambda *a, **k: None
    folder_paths.get_output_directory = lambda: STATE["output_dir"]

    def get_temp_directory():
        Path(STATE["temp_dir"]).mkdir(parents=True, exist_ok=True)
        return STATE["temp_dir"]

    folder_paths.get_temp_directory = get_temp_directory

    def get_save_image_path(prefix, output_dir, width, height):
        sub = Path(prefix).parent
        subfolder = "" if str(sub) == "." else str(sub)
        filename = Path(prefix).name
        folder = Path(output_dir) / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        counter = len(list(folder.glob(filename + "_*.png")))
        return str(folder), filename, counter, subfolder, prefix

    folder_paths.get_save_image_path = get_save_image_path
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    mm.soft_empty_cache = lambda: None
    cu = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            self.total = total

        def update(self, n):
            pass

    cu.ProgressBar = ProgressBar
    comfy.model_management = mm
    comfy.utils = cu
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    sys.modules["comfy.utils"] = cu


_install_stubs()
_spec = importlib.util.spec_from_file_location(
    "dp2node", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["dp2node"] = _pkg
_spec.loader.exec_module(_pkg)
# pytest's Package.setup imports the repo-root __init__.py as top-level "__init__"
# (the hyphenated directory name is not a valid package identifier); reuse our
# correctly-loaded package instead of letting it re-import without a parent package.
sys.modules.setdefault("__init__", _pkg)


@pytest.fixture(scope="session")
def nodes():
    return sys.modules["dp2node.nodes"]


@pytest.fixture
def output_dir(tmp_path):
    STATE["output_dir"] = str(tmp_path)
    STATE["temp_dir"] = str(tmp_path / "temp")
    return tmp_path


def _make_batch(t=3, h=256, w=384):
    """Synthetic scene: floor gradient plus a box moving across frames."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    frames = []
    for i in range(t):
        img = torch.stack([0.3 + 0.4 * yy, 0.4 + 0.3 * xx, 0.6 - 0.2 * yy], dim=-1)
        x0 = 0.1 + 0.15 * i
        box = (xx > x0) & (xx < x0 + 0.2) & (yy > 0.3) & (yy < 0.7)
        img[box] = torch.tensor([0.9, 0.2, 0.2])
        frames.append(img)
    return torch.stack(frames)


@pytest.fixture
def make_batch():
    return _make_batch