# ComfyUI-DepthPro2

ComfyUI nodes for [Apple Depth Pro 2](https://huggingface.co/apple/DepthPro2) — sharp,
temporally consistent **metric** monocular depth (in meters) at full input resolution, for
single images and video batches.

## Why a custom loader?

Depth Pro 2 is a multi-view DINOv3 transformer + DPT decoder with a separate focal-length
estimator. It is not one of the architectures ComfyUI's built-in loaders (UNET/CLIP/VAE)
know about, so converting its weights to a "ComfyUI format" would not make them loadable.
This pack instead ships a custom loader with its own memory-saving weight formats:

| Precision | Weight storage | S-model VRAM* | Depth deviation vs fp32* |
|-----------|----------------|---------------|--------------------------|
| `fp32`    | float32 (as released) | 192 MB | — |
| `bf16` / `fp16` | low-precision weights, fp32 heads | 100 MB | ~0.7% median |
| `fp8`     | encoder Linears as float8_e4m3fn + per-channel scales, dequantized on the fly | 79 MB | ~7% median |
| `auto`    | bf16 on GPU, fp32 on CPU | | |

\* Measured on Apple MPS with DepthPro2-S (24M + 24M focal params). Larger models scale
accordingly (H+ is ~872M params). fp8 needs no extra dependencies and works on CUDA, MPS
and CPU; the focal-length estimator always stays in high precision.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/StanShumsky/ComfyUI-DepthPro2
pip install -r ComfyUI-DepthPro2/requirements.txt
```

Restart ComfyUI. Weights download automatically from Hugging Face into
`ComfyUI/models/depthpro2/` on first use (S: 97 MB, B: 386 MB, L: 1.3 GB, H+: 3.5 GB, plus
the shared 94 MB focal estimator). You can also drop your own `.safetensors` / `.pt` files
into that folder — the size is auto-detected, or read from a `config.json` next to the file.

## Nodes

### DepthPro2 Loader
- `model`: `depthpro2-s/b/l/h` (auto-download) or any local weight file in `models/depthpro2`.
- `precision`: `auto`, `bf16`, `fp16`, `fp32`, `fp8`.
- `device`: `auto` (ComfyUI's device), `cuda`, `mps`, `cpu`.

### DepthPro2 Estimate Depth
Each image of the batch is processed independently (use this for unrelated photos).
- `focal_length_mm`: lens focal length as a **35mm-equivalent** in mm (e.g. 24, 35, 50 — the
  EXIF `FocalLengthIn35mmFilm` value); `0` estimates it with the focal-length model. Depth in
  meters scales linearly with focal length — provide it when you know it.
- `focal_length_px` (optional input): overrides `focal_length_mm` when linked — use the
  Focal Length node below for physical mm with a known camera sensor.
- `max_size`: cap the longest side processed (`0` = native resolution, sharpest edges).
- Outputs:
  - `depth_image` (IMAGE): normalized inverse depth, near white / far black, for preview or
    ControlNet-style conditioning.
  - `depth_meters` (DEPTHPRO2_DEPTH): raw metric depth tensor `(T, H, W)`, `inf` at sky.
  - `focal_length_px` (FLOAT): focal length used, in pixels of the original input.
  - `sky` (MASK): 1 at sky pixels — a ready-made comp holdout.

### DepthPro2 Video Depth
Frames are processed jointly in overlapping windows (`window`, `overlap`) and share one
focal length, giving temporally steady depth. The visualization is normalized over the whole
batch. Reduce `window`/`overlap` first if you run out of memory.
- `mode`: `buffered` (each frame also sees later frames; `overlap` applies) or `unbuffered`
  (each frame predicted from itself and the frames before it — lower latency and memory,
  `overlap` ignored), as in the upstream CLI.

### DepthPro2 Focal Length
Converts a **physical** lens focal length (mm, as engraved on the lens) to pixels for a known
sensor size — for cine/mirrorless cameras (ARRI, Blackmagic, Sony…). Presets cover full frame,
ALEXA 35 / Mini LF / Mini-AMIRA, URSA 12K, Pocket 6K, Super 35, APS-C, Micro Four Thirds and
1-inch; `custom` takes an exact `sensor_width_mm` (for cropped readouts, use the width of the
sensor area actually used). Link `image` to take the width from it, or set `width` manually.
Link the `focal_length_px` output to the optional `focal_length_px` input of the Estimate/Video
nodes; it overrides `focal_length_mm`. For phone photos and unknown cameras, keep using the
35mm-equivalent `focal_length_mm` widget instead.

### DepthPro2 Colorize Depth
Color preview of `depth_meters`, matching Apple's CLI: `turbo` (byte-identical LUT) maps dark
blue at infinitely far to red/white near; also `viridis`, `inferno`, `magma`, `plasma`, `gray`.
The scale is fixed by the first frame's nearest point plus 20% headroom, so videos stay steady
and points coming nearer later saturate. Outputs `depth_color` (IMAGE).

### DepthPro2 Position Pass
Camera-space position pass in meters from `depth_meters` + `focal_length_px`: X right, Y down,
Z along the viewing direction, principal point at the image center (`cx_offset`/`cy_offset`
shift it for calibrated cameras). `max_depth` clips the unreliable far range (`0` = no clip).
Outputs `position_xyz` (IMAGE, `(T, H, W, 3)`) and `valid` (MASK); sky and clipped pixels are
zero with `valid = 0`.

### DepthPro2 Save 16-bit Depth
Writes `depth_meters` as 16-bit grayscale PNGs (white near, black infinitely far) on the color
preview's scale, for compositing tools — read as raw data (Raw in Nuke, Non-Color in Blender).
Depth in meters = `k / value`; `k` is logged and stored in each PNG's `depthpro2_k` text chunk.
An 8-bit preview is shown on the node (written to ComfyUI's temp dir, not the output folder).

### DepthPro2 Save Float Depth
Lossless 32-bit float export of `depth_meters` in **true meters** for compositing (Nuke, Fusion,
Blender) — unlike the 16-bit node, no inverse scale is baked in. `format`: `exr` (single channel
`Z`, needs the optional [OpenEXR](https://pypi.org/project/OpenEXR/) package:
`pip install OpenEXR`) or `tiff` (float32, no extra dependencies). `sky_value` replaces sky with
a far distance (`0` keeps infinity). Link `focal_length_px` from the estimate node to embed it in
the file metadata together with units and the sky convention. Shows an 8-bit preview on the node.

### DepthPro2 Save PLY Points
Metric point cloud (binary PLY, in meters) from the Position Pass outputs: link `position_xyz`
and `valid`; only valid points are written, sky and clipped pixels are dropped. Link the original
`image` for per-point RGB. Coordinates are camera-space (X right, Y down, Z forward) — flip Y in
DCCs that expect Y up. Opens in MeshLab, Blender and CloudCompare.

## Example workflows

Drag into ComfyUI (or use Workflow → Open):

- [example_workflows/i2depth.json](example_workflows/i2depth.json) — image → depth:
  Loader → Estimate → Preview/Save, plus a turbo color preview branch.
- [example_workflows/v2depth.json](example_workflows/v2depth.json) — video → depth:
  LoadVideo → GetVideoComponents → Video Depth → CreateVideo → SaveVideo, with the source
  fps carried through. Uses ComfyUI's built-in video nodes (keep ComfyUI up to date).

## Notes

- Depth is the distance along the camera's viewing direction (z), not radial distance.
- Sky pixels have infinite depth (or several km with the smaller models); clip before making
  point clouds — the Position Pass node's `max_depth` does this for you.
- Depth → 3D points: the **Position Pass** node outputs camera-space XYZ with the image center
  as principal point; chain it into **Save PLY Points** for a metric point cloud.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                  # unit + example-workflow tests: no weights, no network
RUN_DP2_WEIGHT_TESTS=1 pytest -m integration   # end-to-end on real weights (S + focal, ~190 MB once)
```

Integration tests download into `$DP2_MODELS_DIR` (default: a temp dir) and run on ComfyUI's
device selection (CUDA/MPS/CPU). Tests live in the repository but are excluded from the
published registry archive via `.comfyignore`, as the registry docs recommend.

## License

- The node code in this repository is Apache-2.0 (see `LICENSE`).
- The vendored [depth_pro_2/](depth_pro_2/) code is Apple's, redistributed under its
  permissive sample-code license (see `depth_pro_2/LICENSE`); modifications here make
  `av`/`pillow_heif` optional.
- **Model weights are released under the Apple Machine Learning Research Model license
  (`LICENSE_MODEL`): non-commercial scientific research and academic use only.**

## Credits

Apple ML Research — [Depth Pro 2: Sharp and Temporally Consistent Metric Monocular
Depth](https://arxiv.org/abs/2611.26603),
[upstream code](https://github.com/apple-aiml-research/ml-depth-pro-2).

```bibtex
@article{DepthPro2:2026:arxiv,
  title   = {Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth},
  author  = {Peiyun Hu and Marko Mihajlovic and Lyndon Duong and Lahav Lipson and Wei Dong
             and Ama\"{e}l Delaunoy and Stephan R. Richter and Vladlen Koltun},
  journal = {arXiv preprint arXiv:2611.26603},
  year    = {2026},
}
```