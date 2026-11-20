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
git clone <this repo> ComfyUI-DepthPro2
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
- `f_px`: focal length in pixels of the input; `0` estimates it with the focal-length model.
  Depth in meters scales linearly with focal length — provide it when you know it.
- `max_size`: cap the longest side processed (`0` = native resolution, sharpest edges).
- Outputs:
  - `depth_image` (IMAGE): normalized inverse depth, near white / far black, for preview or
    ControlNet-style conditioning.
  - `depth_meters` (DEPTHPRO2_DEPTH): raw metric depth tensor `(T, H, W)`, `inf` at sky.
  - `focal_length_px` (FLOAT): focal length used, in pixels of the original input.

### DepthPro2 Video Depth
Frames are processed jointly in overlapping windows (`window`, `overlap`) and share one
focal length, giving temporally steady depth. The visualization is normalized over the whole
batch. Reduce `window`/`overlap` first if you run out of memory.

## Example workflows

Drag into ComfyUI (or use Workflow → Open):

- [example_workflows/i2depth.json](example_workflows/i2depth.json) — image → depth:
  Loader → Estimate → Preview/Save.
- [example_workflows/v2depth.json](example_workflows/v2depth.json) — video → depth:
  LoadVideo → GetVideoComponents → Video Depth → CreateVideo → SaveVideo, with the source
  fps carried through. Uses ComfyUI's built-in video nodes (keep ComfyUI up to date).

## Notes

- Depth is the distance along the camera's viewing direction (z), not radial distance.
- Sky pixels have infinite depth (or several km with the smaller models); clip before making
  point clouds.
- To convert depth to 3D points, use `focal_length_px` with the image center as principal
  point: `X = (u - cx) * Z / f`, `Y = (v - cy) * Z / f`.

## License

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