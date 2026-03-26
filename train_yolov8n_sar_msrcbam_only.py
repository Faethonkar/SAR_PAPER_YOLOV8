import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

"""
============================================================
A2: YOLOv8n + MSR-CBAM (attention only)
============================================================
- Keep original stem (no edge fusion)
- Add CBAM after SPPF

Purpose: Measures the effect of high-level attention alone.
============================================================
"""


# ============================================================
# Custom Trainer with CSV logging
# ============================================================
class CustomTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_path = None
        self.csv_initialized = False

    def get_model(self, cfg=None, weights=None, verbose=True):
        if hasattr(self, "custom_model_obj"):
            return self.custom_model_obj
        return super().get_model(cfg, weights, verbose)

    def init_csv(self):
        if self.csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_path = self.save_dir / f"results_{timestamp}.csv"

        headers = [
            "epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "val/box_loss", "val/cls_loss", "val/dfl_loss",
            "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)", "lr/pg0"
        ]

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        self.csv_initialized = True
        print(f"CSV logging to: {self.csv_path}")

    def save_epoch_to_csv(self):
        if not self.csv_initialized:
            self.init_csv()

        metrics = self.metrics

        if hasattr(self, "tloss") and self.tloss is not None:
            try:
                tloss = self.tloss.cpu().numpy() if hasattr(self.tloss, "cpu") else self.tloss
                box_loss = f"{tloss[0]:.5f}" if len(tloss) > 0 else "N/A"
                cls_loss = f"{tloss[1]:.5f}" if len(tloss) > 1 else "N/A"
                dfl_loss = f"{tloss[2]:.5f}" if len(tloss) > 2 else "N/A"
            except Exception:
                box_loss = cls_loss = dfl_loss = "N/A"
        else:
            box_loss = cls_loss = dfl_loss = "N/A"

        try:
            val_box = f"{self.validator.loss[0]:.5f}" if hasattr(self, "validator") and self.validator and hasattr(self.validator, "loss") else "N/A"
            val_cls = f"{self.validator.loss[1]:.5f}" if hasattr(self, "validator") and self.validator and hasattr(self.validator, "loss") else "N/A"
            val_dfl = f"{self.validator.loss[2]:.5f}" if hasattr(self, "validator") and self.validator and hasattr(self.validator, "loss") else "N/A"
        except Exception:
            val_box = val_cls = val_dfl = "N/A"

        row = [
            self.epoch + 1,
            box_loss, cls_loss, dfl_loss,
            val_box, val_cls, val_dfl,
            f"{metrics.get('metrics/precision(B)', 0):.5f}",
            f"{metrics.get('metrics/recall(B)', 0):.5f}",
            f"{metrics.get('metrics/mAP50(B)', 0):.5f}",
            f"{metrics.get('metrics/mAP50-95(B)', 0):.5f}",
            f"{self.optimizer.param_groups[0]['lr']:.6f}" if self.optimizer else "N/A",
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def on_fit_epoch_end(self, *args, **kwargs):
        super().on_fit_epoch_end(*args, **kwargs) if hasattr(super(), "on_fit_epoch_end") else None
        try:
            self.save_epoch_to_csv()
        except Exception as e:
            print(f"CSV save error: {e}")


# ============================================================
# MSR-CBAM (lightweight)
# ============================================================
class LightChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        mid = max(c // r, 8)
        self.fc = nn.Sequential(
            nn.Linear(c, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
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


# ============================================================
# Apply modifications
# ============================================================
def apply_single_cbam(det):
    layers = det.model
    target_id = 9

    m = layers[target_id]
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

    layers[target_id] = seq
    print(f"CBAM added after SPPF (channels={c})")


def build_model():
    base = YOLO("yolov8n.pt")
    det = base.model

    apply_single_cbam(det)

    total_params = sum(p.numel() for p in det.parameters())
    print(f"Total model parameters: {total_params / 1e6:.3f} M")

    return base


# ============================================================
# Main Training
# ============================================================
if __name__ == "__main__":
    model = build_model()

    args = dict(
        model="yolov8n.yaml",
        data=r"C:\Users\user\iCloudDrive\Desktop\PHD_SND\summaries_gui\ssdd\datasets\data.yaml",
        imgsz=640,
        epochs=100,
        batch=4,
        workers=0,
        optimizer="AdamW",
        lr0=0.003,
        pretrained=False,
        half=True,
        device="0",
        seed=0,
        deterministic=True,
        project="runs/ssdd/yolov8n_cbam_edge",
        name="ablation_msrcbam_only",
        val=True,
        verbose=False,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.0,
        mixup=0.0,
        multi_scale=False,
    )

    trainer = CustomTrainer(overrides=args)
    trainer.custom_model_obj = model.model

    trainer.add_callback("on_fit_epoch_end", lambda t: t.save_epoch_to_csv())
    trainer.init_csv()

    trainer.train()

    print("Training complete. Results saved to:", trainer.save_dir)
