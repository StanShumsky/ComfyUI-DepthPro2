"""Derive the AUTO_BOUNDARY_MPS table analytically (no GPU needed).

Boundary MP = the input size at which one bf16 image forward consumes the whole free VRAM
of the reference card (32 GiB 5090, ~2 GiB baseline -> 29.8 GiB free). The per-MP cost is
modeled stage by stage from the architecture (encoder token sequence, the 4-level reassembly
pyramid, the DPT decoder fusions, the fp32 inverse-depth head); a single global correction
factor absorbs what static accounting cannot see (cudnn workspaces, autocast accumulation,
allocator variance) and is fitted to the one reliable peak measurement on file:

depthpro2-l: 24 GiB torch peak at 16 MP (healthy torch counters, verified that session).

Cross-checks the model also has to satisfy:
- depthpro2-s: ~19.4 GiB torch peak reported at 32 MP
- depthpro2-b: OOM observed between 24 and 32 MP (24 was the last success)

Re-run when Apple ships a new preset size or a new measurement replaces the anchor.
"""

import math

# (embed_dim, dims_reassemble L0..L3, dim_decoder, weights GiB bf16)
PRESETS = {
    "depthpro2-s": (384, (48, 96, 192, 384), 64, 0.05),
    "depthpro2-b": (768, (96, 192, 384, 768), 128, 0.18),
    "depthpro2-l": (1024, (256, 512, 1024, 1024), 256, 0.62),
    "depthpro2-h": (1280, (256, 512, 1024, 1024), 256, 1.63),
}

REF_FREE_GIB = 29.8   # 32 GiB card minus the ~2 GiB driver/session baseline
MARGIN_GIB = 2.5      # AUTO_MARGIN_GIB in nodes.py

# The anchor: (preset, MP, torch peak GiB incl. weights) - the only reliable measurement.
ANCHOR = ("depthpro2-l", 16.0, 24.0)

# Live-tensor bytes per 1 MP of input at each stage (bf16 = 2 B, head = fp32 = 4 B).
# P = 1e6 px; the ViT grid is P/256 tokens (patch 16).
def stage_bytes_per_mp(embed: int, dims: tuple[int, int, int, int], dd: int) -> dict[str, float]:
    P = 1e6
    g = P / 256.0                      # 1/16-resolution pixel count
    tokens = g * embed * 2             # bf16 token sequence
    d0, d1, d2, d3 = dims
    py0, py1, py2, py3 = P/16*d0*2, P/64*d1*2, P/256*d2*2, P/1024*d3*2   # pyramid levels
    # Decoder peak: last fusion (1/4 -> 1/2 res): previous 1/2 map, deconv 1/2 map, out 1/2 map,
    # conv workspace for the 1/2 deconv, plus the 1/4-res working maps (conv out, two resnet
    # outputs, their workspaces) and the still-referenced pyramid inputs.
    dec = (500_000*dd*4 + 125_000*dd*6 + py1 + py2 + py3)
    # Head peak (fp32): decoder 1/2 map (bf16) + conv 1/2 x dd/2 + deconv full x dd/2 +
    # conv full x 32 + final full-res maps.
    head = 500_000*dd + 500_000*dd + 2_000_000*dd + 128_000_000 + 8_000_000
    return {
        "encoder": tokens * 8 + 4 * tokens,          # seq + qkv/attn/o transients + 4 live taps
        "reassemble": py0 + py1 + py2 + py3 + 3 * py0 + tokens,  # pyramid + L0 working set
        "decoder": dec,
        "head": head,
    }


def main() -> None:
    # Fit the global correction factor to the anchor measurement.
    a = PRESETS[ANCHOR[0]]
    anchor_model = max(stage_bytes_per_mp(*a[:3]).values()) / 2**30
    factor = (ANCHOR[2] - a[3]) / (ANCHOR[1] * anchor_model)
    print(f"correction factor (fitted to {ANCHOR[0]} @ {ANCHOR[1]:.0f} MP -> {ANCHOR[2]:.0f} GiB): {factor:.2f}\n")

    rows = []
    for name, (embed, dims, dd, w) in PRESETS.items():
        st = stage_bytes_per_mp(embed, dims, dd)
        peak = max(st.values()) / 2**30 * factor          # GiB per MP, calibrated
        boundary = (REF_FREE_GIB - w - MARGIN_GIB) / peak
        rows.append((name, st, peak, boundary))
        stages = " ".join(f"{k}={v/2**20:.0f}MB" for k, v in st.items())
        print(f"{name}: {stages}")
        print(f"   -> {peak:.2f} GiB/MP (calibrated), boundary = {boundary:.1f} MP (weights {w} GiB)")

    # Cross-checks against the other observations.
    srow = [r for r in rows if r[0] == "depthpro2-s"][0]
    brow = [r for r in rows if r[0] == "depthpro2-b"][0]
    pred_s32 = 0.05 + 32 * srow[2]
    print(f"\ncheck s @ 32 MP: predicted {pred_s32:.1f} GiB vs ~19.4 reported")
    print(f"check b boundary: predicted {brow[3]:.1f} MP vs OOM observed in 24-32 MP")

    print("\nsuggested AUTO_BOUNDARY_MPS (conservative floors):")
    for name, _st, peak, boundary in rows:
        print(f"    {name!r}: {math.floor(boundary / 2) * 2:.0f},")


if __name__ == "__main__":
    main()
