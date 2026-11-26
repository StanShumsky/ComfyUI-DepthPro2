# ComfyUI-DepthPro2

ComfyUI nodes for [Apple Depth Pro 2](https://huggingface.co/apple/DepthPro2) — sharp,
temporally consistent **metric** monocular depth (in meters) for single images and video
batches.

## Why a custom loader?

Depth Pro 2 is a multi-view DINOv3 transformer with a DPT decoder and a separate
focal-length estimator. It is not one of the architectures ComfyUI's built-in loaders
(UNET/CLIP/VAE) know about, so this pack ships its own loader with memory-saving weight
formats instead.

| Precision | Weights | Compute |
|-----------|---------|---------|
| `fp32`    | full precision, as released | fp32 |
| `bf16` / `fp16` | reduced-precision weights, fp32 output heads | reduced precision, fp32 heads |
| `fp8`     | smallest weights (encoder layers compressed, dequantized on the fly) | reduced precision / fp32 |
| `auto`    | uses ComfyUI's session precision | session policy |

Basic guidance: `auto` is a good default; `fp8` stores less but is **weight-storage
compression**, not a faster fp8 matrix-multiply kernel — compute still runs in a lower
precision (or fp32 on CPU) and results are not guaranteed to be identical to fp32. No
absolute accuracy figures are promised across precisions; verify on your own work.

Weight files for the S preset (main + focal) occupy roughly 97 MB for fp32 and less for
low-precision formats. Larger presets scale up accordingly — H+ is multiple GB. Runtime
memory is higher than the weight sizes: activations grow with input resolution and video
window length.

## Install

Use an existing ComfyUI checkout with Python 3.9+ and ComfyUI API 0.11.0 or newer.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/StanShumsky/ComfyUI-DepthPro2
pip install -r ComfyUI-DepthPro2/requirements.txt
```

Restart ComfyUI. Weights download automatically from Hugging Face into
`ComfyUI/models/depthpro2/` on first use (S: 97 MB, B: 386 MB, L: 1.3 GB, H+: 3.5 GB, plus
the shared 94 MB focal estimator). You can also drop your own `.safetensors` / `.pt` files
into that folder. Automatic focal estimation downloads the separate focal checkpoint on the
first run that needs it; if you always set the focal length yourself, the main checkpoint
is enough.

## Quick start

1. Copy [example_workflows/example.png](example_workflows/example.png) into
   `ComfyUI/input/`, then drag [example_workflows/i2depth.json](example_workflows/i2depth.json)
   onto the canvas (or Workflow → Open). For the video example, copy
   [example_workflows/example.mp4](example_workflows/example.mp4) into the same folder.
2. Pick a preset in the Loader — S is smallest and fastest, H+ is largest and sharpest.
   Weights (and, if you estimate focal length, the focal checkpoint) download on first run.
3. Set the lens focal length on the Estimate node: the 35mm-equivalent in `focal_length_mm`
   (EXIF has it as `FocalLengthIn35mmFilm`), or leave it at 0 with no linked pixel focal
   input and let the model estimate it. For a known camera/lens, use the
   [explicit focal workflow](example_workflows/focal_length.json) and the Focal Length node.
4. Queue the prompt. `depth_image` is the preview, `depth_meters` the metric depth tensor.

## Nodes

Nine nodes in two families: **estimation** (Loader, Estimate Depth, Video Depth, Focal
Length) and **export** (Colorize, Position Pass, Save 16-bit, Save Float, Save PLY).

### DepthPro2 Loader
- `model`: `depthpro2-s/b/l/h` (auto-download) or any local weight file in `models/depthpro2`.
- `precision`: `auto`, `bf16`, `fp16`, `fp32`, `fp8`.
- `device`: `auto` picks the best available device (CUDA, MPS or CPU) and follows ComfyUI's
  device policy; you can also choose `cuda`, `mps` or `cpu` explicitly.

### DepthPro2 Estimate Depth
Each image of the batch is processed independently (use this for unrelated photos).
- `estimate_focal_length` (toggle, on by default): with it on, `0` in `focal_length_mm` and
  no linked `focal_length_px` lets the focal-length model estimate the focal length; with it
  off, an explicit positive focal length is required.
- `focal_length_mm`: lens focal length as a **35mm-equivalent** in mm (e.g. 24, 35, 50 — the
  EXIF `FocalLengthIn35mmFilm` value); `0` requests estimation while the toggle is on and no
  pixel focal input is linked. Depth in meters scales linearly with focal length — provide it
  when you know it.
- `focal_length_px` (optional input): overrides `focal_length_mm` whenever linked, including
  a linked `0.0` (which requests estimation while the toggle is on). Use the Focal Length
  node below for physical mm with a known camera sensor. Negative, NaN and infinite values
  are rejected before inference.
- `max_size`: `0` (default) picks a resolution that fits the free memory at run time,
  `-1` runs at native resolution (sharpest edges, may run out of memory on very large images),
  any positive value caps the longest side processed. The output is always restored to the
  input size.
- Outputs:
  - `depth_image` (IMAGE): normalized inverse depth, near white / far black, for preview or
    ControlNet-style conditioning.
  - `depth_meters` (DEPTHPRO2_DEPTH): raw metric depth tensor `(T, H, W)`, `inf` at sky.
  - `focal_length_px` (FLOAT): focal length used, in pixels of the original input. For an
    independently processed batch with automatic estimation, this scalar is the first
    image's focal length.
  - `sky_mask` (MASK): 1 at sky pixels — a ready-made comp holdout.

### DepthPro2 Video Depth
Frames are processed jointly in overlapping windows (`window`, `overlap`) and share one
focal length, giving temporally steady depth. The visualization is normalized over the whole
batch. Reduce `window`/`overlap` first if you run out of memory.
- `estimate_focal_length` (toggle, on by default): same as the Estimate node — with it on, `0` in
  `focal_length_mm` and no linked `focal_length_px` estimates the focal length from the first
  window; a linked `0.0` also requests estimation and overrides the mm value. With the toggle
  off, an explicit positive focal length is required.
- `mode`: `buffered` (each frame also sees later frames; `overlap` applies) or `unbuffered`
  (each frame predicted from itself and the frames before it — lower latency and memory,
  `overlap` ignored).
- `max_size`: same as the Estimate node.

### DepthPro2 Focal Length
Converts a lens focal length in mm to `focal_length_px` and shows the calculated pixels,
35mm-equivalent focal length and field of view on the node after execution.

**Basic**
1. Connect the exact IMAGE used for depth to `Image (auto size)`. Landscape, portrait and
   square dimensions are detected automatically.
2. In `Mode / sensor`, choose a physical camera preset, or `35mm equivalent (EXIF / unknown
   sensor)` for phone photos and the EXIF `FocalLengthIn35mmFilm` value.
3. Enter `Focal length (mm)` and link `focal_length_px` to the optional `focal_length_px`
   input of Estimate/Video. A linked value overrides their `focal_length_mm` widget; `0`
   requests automatic estimation downstream while **Estimate focal length** is on.

**Professional**
- Physical presets cover full frame, ALEXA 35 / Mini LF / Mini-AMIRA, URSA 12K, Pocket 6K,
  Super 35, APS-C, Micro Four Thirds and 1-inch. The image's long side supplies the
  sensor-width pixel count, so portrait images need no manual orientation math.
- For cropped, anamorphic or unusual readouts, choose `custom` and enter the **active
  sensor/readout width** in `Active sensor width (mm)`, not the full sensor width.
- Disconnect IMAGE to use `Manual image side (px)`: the width for landscape, the height for
  portrait, measured along the sensor-width direction. Existing manual workflows keep
  their original defaults and behavior.

### DepthPro2 Colorize Depth
Color preview of `depth_meters`: `turbo` (Apple's preview colormap) maps dark blue at
infinitely far to red/white near; also `viridis`, `inferno`, `magma`, `plasma`, `gray`.
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
- [example_workflows/focal_length.json](example_workflows/focal_length.json) — explicit
  camera/lens focal length: the same image feeds Focal Length and Estimate, with the
  calculated pixel focal linked into the depth node.

## Notes

- Depth is the distance along the camera's viewing direction (z), not radial distance.
- Sky pixels have infinite depth (or several km with the smaller models); clip before making
  point clouds — the Position Pass node's `max_depth` does this for you.
- Depth → 3D points: the **Position Pass** node outputs camera-space XYZ with the image center
  as principal point; chain it into **Save PLY Points** for a metric point cloud.
- On memory-limited GPUs, weights stay in RAM and only occupy VRAM while a run is in progress.
  If a run still runs out of memory, processing is automatically reduced and retried, then
  raises an error — on a tight session, use a smaller preset, a smaller positive `max_size`,
  or fp8. With no usable accelerator, `device=auto` runs on the CPU; it works, but slowly.

## Memory and troubleshooting

- Weight storage is only the fixed cost: activations grow with input resolution and video
  window length. Start with S, `max_size=0`, and a smaller video `window`; lower the longest
  side explicitly if your session shares memory with other models.
- A successful `auto` resolution is still not a guarantee: close other GPU workloads if
  inference raises an out-of-memory error.
- If a weight file cannot be decoded, the pack preserves it beside the original file with a
  `.corrupt-<timestamp>` suffix and redownloads a fresh copy. Review the preserved file before
  deleting or moving it; repeated suffixes indicate repeated failed downloads.
- For CPU-only use, choose `device=auto` or `device=cpu` and keep resolutions modest. CUDA,
  MPS and CPU are supported, but each backend has different memory limits and performance.
- EXR export needs `pip install OpenEXR`; TIFF and PLY export have no extra dependency.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

The unit and workflow tests run without downloading weights or touching the network. An
opt-in integration suite runs against the real weights (S preset plus the focal estimator):

```bash
RUN_DP2_WEIGHT_TESTS=1 pytest -m integration
```

## License

- The node code in this repository is Apache-2.0 (see `LICENSE`).
- The bundled [depth_pro_2/](depth_pro_2/) code is Apple's, redistributed under its
  permissive sample-code license (see `depth_pro_2/LICENSE`).
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
