import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from ultralytics import YOLO


# ============================================================
# Detailed FLOPs Calculator (Multiplications + Additions)
# ============================================================
def print_detailed_flops(model, imgsz=640):
    """Calculate and print detailed FLOPs showing multiplications and additions separately."""
    det = model.model
    device = next(det.parameters()).device
    det.eval()

    x = torch.randn(1, 3, imgsz, imgsz, device=device)

    mults = 0
    adds = 0
    total_params = 0

    def conv_flops(m, inp, out):
        nonlocal mults, adds
        out_h, out_w = out.shape[2], out.shape[3]
        kernel_ops = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        # Multiplications: kernel_h * kernel_w * in_channels/groups * out_h * out_w * out_channels
        layer_mults = kernel_ops * out_h * out_w * m.out_channels
        # Additions: (kernel_ops - 1) * out_h * out_w * out_channels (summing products)
        layer_adds = (kernel_ops - 1) * out_h * out_w * m.out_channels
        if m.bias is not None:
            layer_adds += out_h * out_w * m.out_channels
        mults += layer_mults
        adds += layer_adds

    def linear_flops(m, inp, out):
        nonlocal mults, adds
        batch = out.shape[0]
        # Multiplications: in_features * out_features * batch
        layer_mults = m.in_features * m.out_features * batch
        # Additions: (in_features - 1) * out_features * batch
        layer_adds = (m.in_features - 1) * m.out_features * batch
        if m.bias is not None:
            layer_adds += m.out_features * batch
        mults += layer_mults
        adds += layer_adds

    def bn_flops(m, inp, out):
        nonlocal mults, adds
        # BatchNorm: 2 mults (scale, shift) and 2 adds (mean subtraction, shift) per element
        numel = out.numel()
        mults += 2 * numel
        adds += 2 * numel

    hooks = []
    for m in det.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_flops))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_flops))
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            hooks.append(m.register_forward_hook(bn_flops))

    with torch.no_grad():
        _ = det(x)

    for h in hooks:
        h.remove()

    total_params = sum(p.numel() for p in det.parameters())
    total_flops = mults + adds

    print(f"\n{'='*50}")
    print(f"📊 DETAILED MODEL COMPLEXITY ({imgsz}x{imgsz})")
    print(f"{'='*50}")
    print(f"Parameters:      {total_params / 1e6:.3f} M")
    print(f"Multiplications: {mults / 1e9:.3f} G")
    print(f"Additions:       {adds / 1e9:.3f} G")
    print(f"Total FLOPs:     {total_flops / 1e9:.3f} G")
    print(f"{'='*50}\n")

    return mults, adds, total_params


# ============================================================
# Baseline YOLOv10n Training Script (NO modifications)
# ============================================================
if __name__ == "__main__":
    print("🔥 Training Baseline YOLOv10n on SSDD (No EdgeStem, No CBAM)")

    # Load vanilla YOLOv10n with pretrained COCO weights
    model = YOLO("yolov10n.pt")

    # Calculate detailed FLOPs
    print_detailed_flops(model, imgsz=640)

    # -------------------------
    # Force GPU
    # -------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. To run on CPU change this behavior or install CUDA.")
    device = '0'  # use CUDA device 0
    torch_device = torch.device('cuda:0')
    try:
        # move model parameters to GPU
        model.model.to(torch_device)
    except Exception:
        pass

    # -------------------------
    # TRAIN
    # -------------------------
    results = model.train(
        model="yolov10n.yaml",
        data=r"C:\Users\user\iCloudDrive\Desktop\PHD_SND\summaries_gui\ssdd\datasets\data.yaml",
        imgsz=640,
        epochs=100,
        batch=4,
        workers=0,
        optimizer="AdamW",
        lr0=0.003,
        pretrained=True,
        half=True,
        device=device,
        seed=0,
        deterministic=True,
        project="runs/ssdd/yolov10n_ssdd_baseline",
        name="exp1_ssdd",
        val=True,
        verbose=False,

        # SAR-friendly augments
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.0,
        mixup=0.0,
        multi_scale=False,
    )

    # -------------------------
    # FINAL EVAL ON VAL SET
    # -------------------------
    print("\n🔍 Validating Baseline YOLOv10n on val set...")
    metrics = model.val(device=device)

    rd = metrics.results_dict
    print("\n===== FINAL BASELINE YOLOv10n METRICS =====")
    print(f"metrics/precision(B): {rd.get('metrics/precision(B)')}")
    print(f"metrics/recall(B):    {rd.get('metrics/recall(B)')}")
    print(f"metrics/mAP50(B):     {rd.get('metrics/mAP50(B)')}")
    print(f"metrics/mAP50-95(B):  {rd.get('metrics/mAP50-95(B)')}")
    print(f"fitness:              {rd.get('fitness')}")
