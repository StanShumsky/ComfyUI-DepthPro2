"""Stubs for the ComfyUI modules the nodes import, so tests run without a ComfyUI install."""

import importlib.util
import os
import random
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

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
    # Extra model dirs unit tests register, kept out of the real models folder.
    extra_model_dirs: list[str] = []

    def add_model_folder_path(name, path):
        if name == "depthpro2" and str(path) not in extra_model_dirs:
            extra_model_dirs.append(str(path))

    folder_paths.add_model_folder_path = add_model_folder_path

    def get_folder_paths(name):
        # Real ComfyUI returns [] for unregistered folder names; MODELS_DIR leads the
        # search order (so downloads still go there), test extras come last.
        if name == "depthpro2":
            return [os.path.join(folder_paths.models_dir, "depthpro2")] + extra_model_dirs
        return []

    folder_paths.get_folder_paths = get_folder_paths
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
    # Submodules we do not stub (e.g. comfy.memory_management, needed by
    # comfy_api.latest's video inputs) resolve against the real ComfyUI checkout
    # when one is importable; the stubbed modules below still win via sys.modules.
    try:
        _api_spec = importlib.util.find_spec("comfy_api")
        if _api_spec is not None and _api_spec.submodule_search_locations:
            _root = Path(next(iter(_api_spec.submodule_search_locations))).parent
            if (_root / "comfy").is_dir():
                comfy.__path__ = [str(_root / "comfy")]
    except Exception:
        pass
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    mm.soft_empty_cache = lambda: None
    # Generous free memory: the auto max_size policy stays at native resolution in tests.
    mm.get_free_memory = lambda dev=None: 1 << 40
    mm.load_model_gpu = lambda patcher: patcher
    mm.load_models_gpu = lambda models, memory_required=0, **kwargs: models
    mm.loaded_models = lambda only_currently_used=False: []
    mm.text_encoder_dtype = lambda device=None: torch.float16
    # As in a NORMAL_VRAM session: the model loads on the accelerator and rides in RAM.
    mm.text_encoder_device = lambda: torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    mm.text_encoder_offload_device = lambda: torch.device("cpu")
    mp = types.ModuleType("comfy.model_patcher")

    class CoreModelPatcher:
        """Stand-in for ComfyUI's CoreModelPatcher (weights stay put in tests)."""

        def __init__(self, model, load_device, offload_device, size=0, fast_disk=False):
            self.model = model
            self.load_device = load_device
            self.offload_device = offload_device

    mp.CoreModelPatcher = CoreModelPatcher
    cu = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            self.total = total

        def update(self, n):
            pass

    cu.ProgressBar = ProgressBar
    comfy.model_management = mm
    comfy.model_patcher = mp
    comfy.utils = cu
    try:
        # The real module (via __path__); stop cli_args from parsing pytest's argv.
        # importlib (not `import comfy.options`): a plain import would rebind the local
        # `comfy` to the real package the import machinery creates first ("comfy" is not
        # in sys.modules yet), and the sys.modules registration below would then keep the
        # real package instead of this stub.
        import importlib

        real_options = importlib.import_module("comfy.options")
        real_options.args_parsing = False
        real_memory_management = importlib.import_module("comfy.memory_management")
    except Exception:
        real_memory_management = types.ModuleType("comfy.memory_management")
        real_memory_management.aimdo_enabled = False
    sys.modules["comfy.memory_management"] = real_memory_management
    comfy.memory_management = real_memory_management
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    sys.modules["comfy.model_patcher"] = mp
    sys.modules["comfy.utils"] = cu

    try:
        import tqdm.auto  # noqa: F401
    except ImportError:
        tqdm_mod = types.ModuleType("tqdm")
        auto_mod = types.ModuleType("tqdm.auto")

        class _Tqdm:
            def __init__(self, *args, **kwargs):
                pass

            def update(self, n=1):
                pass

            def close(self):
                pass

        auto_mod.tqdm = _Tqdm
        tqdm_mod.auto = auto_mod
        sys.modules["tqdm"] = tqdm_mod
        sys.modules["tqdm.auto"] = auto_mod


def _install_comfy_api_stub() -> None:
    """The real comfy_api.latest drags in the whole ComfyUI runtime; the tests only need the io
    surface the nodes use, so install a minimal stand-in when the real one is not importable.
    When the real API is importable (a ComfyUI checkout on PYTHONPATH), the tests run against
    it instead, so the schemas are validated exactly the way ComfyUI validates them."""
    try:
        import comfy_api.latest  # noqa: F401
        return
    except Exception:  # noqa: BLE001
        pass

    class _Input:
        def __init__(self, id, display_name=None, optional=False, tooltip=None,
                     lazy=None, extra_dict=None, raw_link=None, advanced=None,
                     default=None, **kwargs):
            self.id = id
            self.display_name = display_name
            self.optional = optional
            self.tooltip = tooltip
            self.lazy = lazy
            self.advanced = advanced
            self.default = default

    class _Output:
        def __init__(self, id=None, display_name=None, tooltip=None, is_output_list=False):
            self.id = id
            self.display_name = display_name if display_name else id
            self.tooltip = tooltip
            self.is_output_list = is_output_list

    def _type(name, io_type, input_cls=_Input, output_cls=None, with_output=True):
        # A distinct Input class per type: the real API has one class per socket type,
        # and the subgraph tests compare both type identity (widget detection) and the
        # LiteGraph io_type string.
        in_cls = type(f"_{name}Input", (input_cls,), {"io_type": io_type})
        class T:
            Input = in_cls

        if with_output:
            out = type("Output", (output_cls or _Output,), {"io_type": io_type})
            T.Output = out
        return T

    class _ComboInput(_Input):
        def __init__(self, id, options=None, display_name=None, tooltip=None,
                     optional=False, default=None, **kwargs):
            super().__init__(id, display_name, tooltip, optional, **kwargs)
            self.options = options or []
            self.default = default

    class _TensorInput(_Input):
        """Image/Mask sockets: distinct classes, as in the real API, so widget-type
        filters (type identity against the value-type inputs) exclude them exactly as
        ComfyUI's frontend does - tensor inputs are link sockets, never widgets."""

    def _custom(io_type):
        """Mirror of comfy_api.latest.io.Custom: a type factory for pack-private payloads."""
        outer = type("Custom", (), {"io_type": io_type})
        outer.Input = type("Input", (_Input,), {"io_type": io_type})
        outer.Output = type("Output", (_Output,), {"io_type": io_type})
        return outer

    class _Schema:
        def __init__(self, node_id=None, display_name=None, category=None, description=None,
                     search_aliases=None, inputs=None, outputs=None, is_output_node=False,
                     **kwargs):
            self.node_id = node_id
            self.display_name = display_name
            self.category = category
            self.description = description
            self.search_aliases = search_aliases or []
            self.inputs = inputs or []
            self.outputs = outputs or []
            self.is_output_node = is_output_node
            self.hidden = []

        def finalize(self):
            # Mirror the real Schema.finalize: hidden for output nodes, default output ids.
            if self.is_output_node:
                self.hidden += ["prompt", "extra_pnginfo"]
            for i, output in enumerate(self.outputs):
                if output.id is None:
                    output.id = f"_{i}_{getattr(output, 'io_type', 'X')}_"

        def validate(self):
            # Mirror the real Schema.validate: ids unique within inputs and within outputs
            # (a node may reuse a name for both an input and an output).
            for label, items in (("Input", self.inputs), ("Output", self.outputs)):
                ids = [i.id for i in items]
                dupes = sorted({x for x in ids if ids.count(x) > 1})
                if dupes:
                    raise ValueError(f"{label} ids must be unique, but {dupes} are not.")

    class _ComfyNode:
        SCHEMA = None

        @classmethod
        def define_schema(cls):
            raise NotImplementedError

        @classmethod
        def FINALIZE_SCHEMA(cls):
            schema = cls.define_schema()
            schema.finalize()
            return schema

        @classmethod
        def GET_SCHEMA(cls):
            schema = cls.FINALIZE_SCHEMA()
            schema.validate()
            cls.SCHEMA = schema
            return schema

    class _NodeOutput:
        def __init__(self, *args, ui=None, expand=None, block_execution=None):
            self.args = args
            self.ui = ui
            self.expand = expand
            self.block_execution = block_execution

        @property
        def result(self):
            return self.args if len(self.args) > 0 else None

        def __getitem__(self, index):
            return self.args[index]

    class _PreviewImage:
        """Mirror of ui.PreviewImage's save path (ImageSaveHelper.save_images): one temp PNG per
        frame, .values entries shaped like the real SavedResult dicts."""

        def __init__(self, image, animated=False, cls=None, **kwargs):
            import folder_paths

            temp = folder_paths.get_temp_directory()
            prefix = "ComfyUI_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
            folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                prefix, temp, image[0].shape[1], image[0].shape[0]
            )
            values = []
            for frame in image:
                arr = np.clip(255.0 * frame.cpu().numpy(), 0, 255).astype(np.uint8)
                file = f"{filename}_{counter:05}_.png"
                Image.fromarray(arr).save(Path(folder) / file, compress_level=1)
                values.append({"filename": file, "subfolder": subfolder, "type": "temp"})
                counter += 1
            self.values = values
            self.animated = animated

    _combo = type("Combo", (), {"Input": type("_ComboInput", (_ComboInput,), {"io_type": "COMBO"})})
    io = types.SimpleNamespace(
        Boolean=_type("Boolean", "BOOLEAN"),
        Int=_type("Int", "INT"),
        Float=_type("Float", "FLOAT"),
        String=_type("String", "STRING"),
        Combo=_combo,
        Image=_type("Image", "IMAGE", input_cls=_TensorInput),
        Mask=_type("Mask", "MASK", input_cls=_TensorInput),
        Custom=_custom,
        Schema=_Schema,
        ComfyNode=_ComfyNode,
        NodeOutput=_NodeOutput,
    )
    ui = types.ModuleType("comfy_api.latest.ui")
    ui.PreviewImage = _PreviewImage
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ui = ui
    latest.ComfyExtension = type("ComfyExtension", (), {})
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest
    sys.modules["comfy_api.latest.ui"] = ui


_install_stubs()
_install_comfy_api_stub()
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