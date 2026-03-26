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
A1: YOLOv8n + Edge-Direction Fusion Stem (EDFS only)
============================================================
- Keep original YOLOv8n backbone and head
- Add Edge-Direction Fusion Stem
- No CBAM

Purpose: Measures the effect of structural edge cues alone.
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
# Edge-Direction Fusion Stem (EDFS)
# ============================================================
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


# ============================================================
# Apply modifications
# ============================================================
def apply_edge_fusion(det):
    original_stem = det.model[0]
    c_out = original_stem.conv.out_channels if hasattr(original_stem, "conv") else 16

    fused_stem = EdgeFusionStem(original_stem, c_out=c_out)

    for attr in ("f", "i", "type", "np"):
        if hasattr(original_stem, attr):
            setattr(fused_stem, attr, getattr(original_stem, attr))

    det.model[0] = fused_stem
    print("Edge fusion stem applied (EDFS only)")


def build_model():
    base = YOLO("yolov8n.pt")
    det = base.model

    apply_edge_fusion(det)

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
        name="ablation_edfs_only",
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
