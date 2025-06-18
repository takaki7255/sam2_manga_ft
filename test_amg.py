import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import numpy as np
from pathlib import Path
from torch.backends.cuda import sdp_kernel, SDPBackend
import torch.cuda.amp as amp



dev = "cuda" if torch.cuda.is_available() else "cpu"
flash_ok = dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8

# 1) モデルロード
sam = build_sam2("sam2_hiera_s", "checkpoints/sam2_hiera_small.pt", device="cuda")

if flash_ok:
    sam.half()

# 2) マスクジェネレータを作成（閾値等は調整可）
amg = SAM2AutomaticMaskGenerator(
        model=sam,
        points_per_side=32,       # 解像度 〜 1/32 タイル
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=256  # 小片を捨てる
     )

# 3) 推論
img = cv2.imread("test1.jpg")[..., ::-1].copy()      # BGR→RGB

ctx = sdp_kernel(SDPBackend.FLASH_ATTENTION) if flash_ok \
      else sdp_kernel(SDPBackend.MATH)
autocast = amp.autocast(dtype=torch.float16) if flash_ok else amp.autocast(enabled=False)
with ctx, autocast:
    masks = amg.generate(img)                     # list[dict] (segmentation, bbox …)

# 4) 可視化（白マスク重ね）
out = img.copy()
for m in masks:
    out[m["segmentation"]] = (255,255,255)    # 単色例
cv2.imwrite("panel_amg.png", out[..., ::-1])  # RGB→BGR
