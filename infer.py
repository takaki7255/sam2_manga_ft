# -*- coding: utf-8 -*-
"""
SAM2 Manga-Balloon 推論スクリプト
--------------------------------
使い方:
  python infer.py --ckpt checkpoints/sam2_manga_epoch40.pt \
                  --imgs test1.jpg test2.jpg \
                  --out_dir outputs            \
                  --size 1024 --points 4       \
                  --th 0.4
"""

import argparse, cv2, numpy as np, torch
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.backends.cuda import sdp_kernel, SDPBackend   # 2.5 以降の正式 API

torch.backends.cuda.enable_math_sdp(True)    # math 実装を有効化
torch.backends.cuda.enable_flash_sdp(False)  # flash を無効化
torch.backends.cuda.enable_mem_efficient_sdp(False)

# ---------- ヘルパ ---------- #
def load_checkpoint(model, ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(sd, strict=False)

def sample_grid_points(h, w, n=4):
    g = np.linspace(0.2, 0.8, n)
    return np.array([[int(x*w), int(y*h)] for y in g for x in g], dtype=np.int64)

# ---------- 推論本体 ---------- #
@torch.no_grad()
def predict_image(predictor: SAM2ImagePredictor,
                  img_path: Path,
                  img_size: int,
                  n_clicks: int,
                  th: float,
                  out_dir: Path):
    # 画像読込
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"! 読み込み失敗: {img_path}")
        return
    h0, w0 = bgr.shape[:2]

    # リサイズ & RGB
    rgb = cv2.cvtColor(cv2.resize(bgr, (img_size, img_size)), cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    # クリック点 (中央寄りグリッド)
    pts = sample_grid_points(img_size, img_size, n=n_clicks)   # (N,2)
    lbl = np.ones(len(pts), dtype=np.int64)                    # すべて正例

    masks, scores, _ = predictor.predict(
        point_coords=pts[None],        # (1,N,2)
        point_labels=lbl[None],        # (1,N)
        multimask_output=True)

    # 最良スコアのマスクを選択
    best = masks[np.argmax(scores)]
    mask_bin = (best > th).astype(np.uint8) * 255
    mask_orig = cv2.resize(mask_bin, (w0, h0), interpolation=cv2.INTER_NEAREST)

    out_path = out_dir / f"{img_path.stem}_mask.png"
    cv2.imwrite(str(out_path), mask_orig)
    print(f"[✓] {img_path.name:15} → {out_path.name}, score={scores.max():.3f}")

# ---------- エントリ ---------- #
def main():
    args = argparse.ArgumentParser()
    args.add_argument("--ckpt", required=True, help="学習済み ckpt (.pt)")
    args.add_argument("--imgs", nargs="+", required=True, help="画像ファイル群")
    args.add_argument("--out_dir", default="outputs", help="保存先ディレクトリ")
    args.add_argument("--size", type=int, default=1024, help="リサイズ解像度")
    args.add_argument("--points", type=int, default=4, help="グリッドクリック数")
    args.add_argument("--th", type=float, default=0.5, help="2値化閾値")
    a = args.parse_args()

    device = "cuda" if torch.cuda.is_available() \
             else "mps" if torch.backends.mps.is_available() else "cpu"

    model = build_sam2("sam2_hiera_s", None, device=device)
    load_checkpoint(model, a.ckpt)
    predictor = SAM2ImagePredictor(model)

    out_dir = Path(a.out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    for p in map(Path, a.imgs):
        predict_image(predictor, p, a.size, a.points, a.th, out_dir)

if __name__ == "__main__":
    with sdp_kernel(SDPBackend.MATH):
        main()
