# -*- coding: utf-8 -*-
"""
SAM-2 推論用
------------
python infer.py --ckpt checkpoints/sam2_manga_epoch40.pt \
                --imgs sample1.jpg sample2.jpg \
                --out_dir outputs \
                --size 1024 --clicks 4 --th 0.4
"""

import argparse, cv2, numpy as np, torch
from pathlib import Path
from torch.backends.cuda import sdp_kernel, SDPBackend     # ★2.7 正式 API
from contextlib import nullcontext
import torch.cuda.amp as amp

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------- 共通 util ---------- #
def load_ckpt(model, path: str):
    obj = torch.load(path, map_location="cpu")
    sd  = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(sd, strict=False)

def grid_points(h, w, n=4):
    g = np.linspace(0.2, 0.8, n)
    return np.array([[int(x*w), int(y*h)] for y in g for x in g], np.int64)

# ---------- 画像 1 枚 ---------- #
@torch.no_grad()
def infer_one(predictor, img_path: Path, size: int,
              n_clicks: int, th: float, out_dir: Path):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"! 読込失敗: {img_path}")
        return
    h0, w0 = bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)

    predictor.set_image(rgb)
    pts = grid_points(size, size, n_clicks)             # (N,2)
    lbl = np.ones(len(pts), np.int64)                   # 正例

    masks, scores, _ = predictor.predict(
        point_coords=pts[None], point_labels=lbl[None],
        multimask_output=True)

    best = masks[np.argmax(scores)]
    print(f"  best min/max = {best.min():.3f} / {best.max():.3f}"
          f"   score = {scores.max():.3f}")
    mask_bin = (best > th).astype(np.uint8)*255
    mask_out = cv2.resize(mask_bin, (w0, h0), cv2.INTER_NEAREST)

    out_path = out_dir/f"{img_path.stem}_mask.png"
    cv2.imwrite(str(out_path), mask_out)
    if scores.max() < 0.2:          # スコアが低い時だけ描画
        dbg = rgb.copy()
        for (x,y) in pts:
            cv2.circle(dbg, (x,y), 6, (255,0,0), -1)
        cv2.imwrite("debug_clicks.png", cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
    print(f"[✓] {img_path.name:16} → {out_path.name} (score {scores.max():.3f})")

# ---------- エントリ ---------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--imgs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--clicks", type=int, default=4)
    ap.add_argument("--th", type=float, default=0.5)
    opt = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else
           "mps"  if torch.backends.mps.is_available() else "cpu")
    print("device:", dev)

    model = build_sam2("sam2_hiera_s", None, device=dev).half().eval()
    load_ckpt(model, opt.ckpt)
    predictor = SAM2ImagePredictor(model)

    out_dir = Path(opt.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # -------- math-SDPA コンテキスト --------
    sdpa_ctx = (sdp_kernel(SDPBackend.MATH) if dev=="cuda"
                else nullcontext())

    with sdpa_ctx, amp.autocast(dtype=torch.float16):                                     # ここで math 強制
        for p in map(Path, opt.imgs):
            infer_one(predictor, p, opt.size,
                      opt.clicks, opt.th, out_dir)
    
if __name__ == "__main__":
    main()
