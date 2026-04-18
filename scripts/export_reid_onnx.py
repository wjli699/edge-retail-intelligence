#!/usr/bin/env python3
"""
Export OSNet-x1.0 (ImageNet pretrained) to ONNX for DeepStream nvinfer.
Input:  (B, 3, 256, 128)  RGB normalised  [nchw]
Output: (B, 512)           L2-normalised embedding  [output node: fc_pred]

Requirements:
  pip3 install torchreid

Run from anywhere; DEST is relative to this script's repo root.

Why ImageNet backbone instead of Market-1501 fine-tuned?
  The only publicly available Market-1501 OSNet checkpoints were trained with
  softmax-only loss (cross-entropy on 751 identity classes).  Softmax training
  does not constrain embedding geometry: the BN neck collapses (running_var ≈
  0.0001) and all person crops land in a tiny angular region (mean cosine sim
  0.95+, even for random inputs).  This makes it useless for distance-based
  retrieval.  The ImageNet backbone has no ReID-specific training but its
  embedding space is well-spread across real person crops (mean sim ~0.73,
  std ~0.08), giving the cross-camera matcher real signal to work with.
  A checkpoint trained with triplet or ArcFace loss (e.g. strong-baseline,
  FastReID) would be the correct production upgrade path.
"""
import os
import sys
import importlib.util
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEST  = os.path.join(REPO_ROOT, "models", "reid", "osnet_x1_0_market1501.onnx")
BATCH = 8

# -------------------------------------------------------------------
# Locate torchreid's osnet.py and load it as a standalone module.
#
# WHY: torchreid.__init__ eagerly imports torchvision via its data
# sub-package.  On Jetson JP5 the Jetson-specific torch==2.4.1 wheel
# is incompatible with every pip-available torchvision release, so
# installing torchvision silently downgrades torch.  Loading only
# osnet.py avoids that dependency entirely — the file only imports
# torch and torch.nn.
# -------------------------------------------------------------------
def _find_osnet_py() -> str:
    candidates = sys.path + [os.path.expanduser("~/.local/lib/python3.8/site-packages")]
    for base in candidates:
        p = os.path.join(base, "torchreid", "reid", "models", "osnet.py")
        if os.path.exists(p):
            return p
    raise RuntimeError("torchreid not found.  Install with: pip3 install torchreid")


# ── Load model ──────────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("osnet", _find_osnet_py())
osnet_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osnet_mod)

# pretrained=True: downloads ImageNet weights (~5 MB, cached by torchreid).
# num_classes=1000: ImageNet head — discarded; ONNX output is the 512-d GAP vector.
# loss='softmax': in eval() mode, forward() returns feature vector, not logits.
backbone = osnet_mod.osnet_x1_0(num_classes=1000, pretrained=True, loss="softmax")
print("[export] ImageNet pretrained weights loaded.")

# ── Wrap with L2-normalisation ───────────────────────────────────────────────
# OSNet's forward() returns raw GAP features with norm ~25-35.
# Baking L2-norm into the ONNX means:
#   • nvinfer writes unit-norm vectors into NvDsObjectMeta
#   • the matcher can do a plain dot-product (no Python-side normalisation)
#   • the output node fc_pred is directly interpretable as cosine similarity
class OSNetWithL2Norm(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                     # (B, 512)  raw GAP features
        return nn.functional.normalize(feat, p=2, dim=1)  # (B, 512)  unit norm

model = OSNetWithL2Norm(backbone)
model.eval()

# ── Sanity check ─────────────────────────────────────────────────────────────
dummy = torch.zeros(BATCH, 3, 256, 128)
with torch.no_grad():
    out = model(dummy)

assert out.shape == (BATCH, 512), f"unexpected shape {out.shape}"
norms = torch.linalg.norm(out, dim=1)
assert norms.min() > 0.99, f"L2 norm not ≈ 1.0: {norms}"
print(f"[export] output shape={tuple(out.shape)}  norms min={norms.min():.4f} max={norms.max():.4f}")

# ── ONNX export ──────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(DEST), exist_ok=True)
torch.onnx.export(
    model,
    dummy,
    DEST,
    input_names=["input"],
    output_names=["fc_pred"],
    dynamic_axes={"input": {0: "batch"}, "fc_pred": {0: "batch"}},
    opset_version=11,
)
print(f"[export] ONNX saved → {DEST}")
print("[export] Delete the stale .engine file so TRT rebuilds on next pipeline start:")
engine = DEST + "_b8_gpu0_fp16.engine"
if os.path.exists(engine):
    os.remove(engine)
    print(f"[export] deleted {engine}")
else:
    print(f"[export] (no stale engine found at {engine})")
print("[export] Done.  TRT engine auto-generated on first DeepStream run (~5 min).")
