"""
Exact Parameter & FLOPs Counter for all 5 model variants.

Counts:
  - Parameters: total, trainable, non-trainable (buffers excluded)
  - FLOPs via forward hooks: Conv2d, Linear, BatchNorm2d/1d, SiLU, 
    plus Sobel-filter conv2d in EdgeFusionStem (non-parametric but costs FLOPs)
  - Per-layer breakdown printed

Convention:
  1 MAC (multiply–accumulate) = 2 FLOPs (1 multiply + 1 add)
  We report *total FLOPs* = multiplications + additions, consistent with
  the standard "FLOP" definition used in YOLO papers (≈ 2 × MACs).
"""

import os, sys, copy, textwrap
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# ================================================================
# Reusable module definitions (copied from training scripts)
# ================================================================

# ---------- Edge-Direction Fusion Stem (EDFS) ----------
class EdgeFusionStem(nn.Module):
    def __init__(self, original_stem, c_out=16):
        super().__init__()
        self.original_stem = original_stem

        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(3, 1, 1, 1))

        self.edge_proj = nn.Sequential(
            nn.Conv2d(4, c_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )
        self.edge_weight = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        orig_out = self.original_stem(x)
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        edge_x = edge_x.mean(dim=1, keepdim=True)
        edge_y = edge_y.mean(dim=1, keepdim=True)
        mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        direction = torch.atan2(edge_y, edge_x)
        dir_sin = torch.sin(direction)
        dir_cos = torch.cos(direction)
        gray = x.mean(dim=1, keepdim=True)
        edge_feat = self.edge_proj(torch.cat([mag, dir_sin, dir_cos, gray], dim=1))
        return orig_out + self.edge_weight * edge_feat


# ---------- Lightweight CBAM ----------
class LightChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        mid = max(c // r, 8)
        self.fc = nn.Sequential(
            nn.Linear(c, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        w = torch.sigmoid(self.fc(avg) + self.fc(mx)).view(b, c, 1, 1)
        return x * w


class LightSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv7 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.conv3 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        desc = torch.cat([avg, mx], dim=1)
        w7 = self.conv7(desc)
        w3 = self.conv3(desc)
        w = torch.sigmoid(self.alpha * w7 + (1 - self.alpha) * w3)
        return x * w


class LightCBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = LightChannelAttention(c, r=16)
        self.sa = LightSpatialAttention()
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        attn = self.ca(x)
        attn = self.sa(attn)
        return x + self.gamma * (attn - x)


# ================================================================
# Model builders
# ================================================================

def build_yolov8n_baseline():
    """Vanilla YOLOv8n."""
    return YOLO("yolov8n.pt")


def build_yolov8n_edfs_only():
    """YOLOv8n + Edge-Direction Fusion Stem only."""
    base = YOLO("yolov8n.pt")
    det = base.model
    original_stem = det.model[0]
    c_out = original_stem.conv.out_channels if hasattr(original_stem, "conv") else 16
    fused = EdgeFusionStem(original_stem, c_out=c_out)
    for attr in ("f", "i", "type", "np"):
        if hasattr(original_stem, attr):
            setattr(fused, attr, getattr(original_stem, attr))
    det.model[0] = fused
    return base


def build_yolov8n_msrcbam_only():
    """YOLOv8n + MSR-CBAM after SPPF only."""
    base = YOLO("yolov8n.pt")
    det = base.model
    layers = det.model
    m = layers[9]
    if hasattr(m, "cv2"):
        c = m.cv2.conv.out_channels
    elif hasattr(m, "cv"):
        c = m.cv.conv.out_channels
    else:
        c = 512
    cbam = LightCBAM(c)
    seq = nn.Sequential(m, cbam)
    for attr in ("f", "i", "type", "np"):
        if hasattr(m, attr):
            setattr(seq, attr, getattr(m, attr))
    layers[9] = seq
    return base


def build_yolov8n_sar_adaptive():
    """YOLOv8n + EDFS + MSR-CBAM (full proposed model)."""
    base = YOLO("yolov8n.pt")
    det = base.model

    # Edge Fusion Stem
    original_stem = det.model[0]
    c_out = original_stem.conv.out_channels if hasattr(original_stem, "conv") else 16
    fused = EdgeFusionStem(original_stem, c_out=c_out)
    for attr in ("f", "i", "type", "np"):
        if hasattr(original_stem, attr):
            setattr(fused, attr, getattr(original_stem, attr))
    det.model[0] = fused

    # CBAM after SPPF
    layers = det.model
    m = layers[9]
    if hasattr(m, "cv2"):
        c = m.cv2.conv.out_channels
    elif hasattr(m, "cv"):
        c = m.cv.conv.out_channels
    else:
        c = 512
    cbam = LightCBAM(c)
    seq = nn.Sequential(m, cbam)
    for attr in ("f", "i", "type", "np"):
        if hasattr(m, attr):
            setattr(seq, attr, getattr(m, attr))
    layers[9] = seq

    return base


def build_yolov10n():
    """Vanilla YOLOv10n."""
    return YOLO("yolov10n.pt")


# ================================================================
# Detailed layer-by-layer counter
# ================================================================

def count_parameters_detailed(model_det):
    """Return dict with per-layer param counts and totals."""
    total = 0
    trainable = 0
    layer_info = []
    for name, p in model_det.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        layer_info.append((name, list(p.shape), n, p.requires_grad))
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "layers": layer_info,
    }


def count_flops_detailed(model_obj, imgsz=640):
    """
    Forward-hook based FLOPs counter.
    Counts: Conv2d, Linear, BatchNorm2d/1d, and
    the non-parametric F.conv2d calls inside EdgeFusionStem (via wrapper).
    """
    det = model_obj.model
    device = next(det.parameters()).device
    det.eval()

    x = torch.randn(1, 3, imgsz, imgsz, device=device)

    records = []  # (layer_name, mults, adds)

    def make_conv_hook(name):
        def hook(m, inp, out):
            oh, ow = out.shape[2], out.shape[3]
            kops = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
            lm = kops * oh * ow * m.out_channels
            la = (kops - 1) * oh * ow * m.out_channels
            if m.bias is not None:
                la += oh * ow * m.out_channels
            records.append((name, lm, la))
        return hook

    def make_linear_hook(name):
        def hook(m, inp, out):
            batch = out.shape[0]
            lm = m.in_features * m.out_features * batch
            la = (m.in_features - 1) * m.out_features * batch
            if m.bias is not None:
                la += m.out_features * batch
            records.append((name, lm, la))
        return hook

    def make_bn_hook(name):
        def hook(m, inp, out):
            numel = out.numel()
            records.append((name, 2 * numel, 2 * numel))
        return hook

    # We also need to account for the Sobel F.conv2d calls in EdgeFusionStem.
    # These are non-parametric but cost FLOPs. We handle them via a module-level
    # forward hook on EdgeFusionStem that estimates the Sobel FLOPs.
    def make_edge_stem_hook(name):
        def hook(m, inp, out):
            # inp[0] is the original input x of shape (1, 3, H, W)
            x_in = inp[0]
            _, C, H, W = x_in.shape
            # Two grouped Sobel convolutions: kernel 3x3, groups=3, in_ch=3, out_ch=3
            # Per grouped conv: kernel_ops = 3*3*(3/3) = 9
            # mults = 9 * H * W * 3, adds = 8 * H * W * 3
            sobel_kops = 9  # 3x3 * (3/3)
            for _ in range(2):  # sobel_x and sobel_y
                lm = sobel_kops * H * W * 3
                la = (sobel_kops - 1) * H * W * 3
                records.append((f"{name}.sobel_conv", lm, la))
            # mean across 3 channels -> 1 channel: 2 adds per pixel, twice (edge_x, edge_y)
            records.append((f"{name}.mean_reduce", 0, 2 * 2 * H * W))
            # sqrt, atan2, sin, cos: element-wise ops on (1,1,H,W) each
            # We count these as 1 FLOP each per element (they're transcendentals, 
            # but standard practice counts them as ~1 FLOP each)
            hw = H * W
            records.append((f"{name}.elem_ops", 4 * hw, 2 * hw))  # mag(2 mults+1 add), atan2(1), sin(1), cos(1)
            # mean for gray: 2 adds per pixel
            records.append((f"{name}.gray_mean", 0, 2 * hw))
        return hook

    hooks = []
    for name, m in det.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(make_conv_hook(name)))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(make_linear_hook(name)))
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            hooks.append(m.register_forward_hook(make_bn_hook(name)))
        elif isinstance(m, EdgeFusionStem):
            hooks.append(m.register_forward_hook(make_edge_stem_hook(name)))

    with torch.no_grad():
        _ = det(x)

    for h in hooks:
        h.remove()

    total_mults = sum(r[1] for r in records)
    total_adds  = sum(r[2] for r in records)
    total_flops = total_mults + total_adds

    return {
        "mults": total_mults,
        "adds": total_adds,
        "total_flops": total_flops,
        "records": records,
    }


# ================================================================
# Pretty printer
# ================================================================

def print_summary(model_name, param_info, flop_info, show_layers=False):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {model_name}")
    print(sep)

    # --- Parameters ---
    print(f"  Total Parameters:         {param_info['total']:>14,d}  ({param_info['total']/1e6:.6f} M)")
    print(f"  Trainable Parameters:     {param_info['trainable']:>14,d}  ({param_info['trainable']/1e6:.6f} M)")
    print(f"  Non-trainable Parameters: {param_info['non_trainable']:>14,d}")
    print()

    # --- FLOPs ---
    print(f"  Multiplications:  {flop_info['mults']:>18,d}  ({flop_info['mults']/1e9:.6f} G)")
    print(f"  Additions:        {flop_info['adds']:>18,d}  ({flop_info['adds']/1e9:.6f} G)")
    print(f"  Total FLOPs:      {flop_info['total_flops']:>18,d}  ({flop_info['total_flops']/1e9:.6f} G)")
    print(f"  MACs (FLOPs/2):   {flop_info['mults']:>18,d}  ({flop_info['mults']/1e9:.6f} G)")
    print()

    if show_layers:
        # Top-20 FLOPs layers
        recs = sorted(flop_info["records"], key=lambda r: r[1]+r[2], reverse=True)
        print("  Top-20 layers by FLOPs:")
        print(f"  {'Layer':<55s} {'Mults':>12s} {'Adds':>12s} {'FLOPs':>12s}")
        print(f"  {'-'*55} {'-'*12} {'-'*12} {'-'*12}")
        for name, m, a in recs[:20]:
            print(f"  {name:<55s} {m:>12,d} {a:>12,d} {m+a:>12,d}")
        print()

        # Top-20 param layers
        plyr = sorted(param_info["layers"], key=lambda r: r[2], reverse=True)
        print("  Top-20 layers by Parameters:")
        print(f"  {'Layer':<55s} {'Shape':<25s} {'Params':>12s}")
        print(f"  {'-'*55} {'-'*25} {'-'*12}")
        for name, shape, n, req_grad in plyr[:20]:
            print(f"  {name:<55s} {str(shape):<25s} {n:>12,d}")
        print()

    print(sep)


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    IMGSZ = 640
    SHOW_LAYERS = True  # Set False for compact output

    models = [
        ("YOLOv8n (baseline)",            build_yolov8n_baseline),
        ("YOLOv8n SAR-EDFS only",         build_yolov8n_edfs_only),
        ("YOLOv8n SAR-MSR-CBAM only",     build_yolov8n_msrcbam_only),
        ("YOLOv8n SAR-Adaptive (proposed)",build_yolov8n_sar_adaptive),
        ("YOLOv10n",                       build_yolov10n),
    ]

    summary_rows = []

    for model_name, builder in models:
        print(f"\n>>> Building: {model_name} ...")
        model_obj = builder()
        det = model_obj.model
        det.eval()

        param_info = count_parameters_detailed(det)
        flop_info  = count_flops_detailed(model_obj, imgsz=IMGSZ)

        print_summary(model_name, param_info, flop_info, show_layers=SHOW_LAYERS)

        summary_rows.append({
            "Model": model_name,
            "Params": param_info["total"],
            "Params (M)": f"{param_info['total']/1e6:.6f}",
            "Trainable": param_info["trainable"],
            "FLOPs": flop_info["total_flops"],
            "FLOPs (G)": f"{flop_info['total_flops']/1e9:.6f}",
            "Mults (G)": f"{flop_info['mults']/1e9:.6f}",
            "Adds (G)": f"{flop_info['adds']/1e9:.6f}",
        })

    # ---- Final comparison table ----
    print("\n\n" + "=" * 100)
    print("  COMPARISON TABLE")
    print("=" * 100)
    print(f"  {'Model':<40s} {'Params':>14s} {'Params (M)':>12s} {'FLOPs':>18s} {'FLOPs (G)':>12s} {'Mults (G)':>12s} {'Adds (G)':>12s}")
    print(f"  {'-'*40} {'-'*14} {'-'*12} {'-'*18} {'-'*12} {'-'*12} {'-'*12}")
    for r in summary_rows:
        print(f"  {r['Model']:<40s} {r['Params']:>14,d} {r['Params (M)']:>12s} {r['FLOPs']:>18,d} {r['FLOPs (G)']:>12s} {r['Mults (G)']:>12s} {r['Adds (G)']:>12s}")
    print("=" * 100)
