"""
SAM2 (Hiera-Small) + Manga Balloon 重みで推論
Usage:
  python infer.py --ckpt checkpoints/sam2_manga_epoch20.pt \
                  --imgs test1.jpg test2.jpg \
                  --out_dir outputs
"""

import argparse, os, cv2, numpy as np, torch
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_model(ckpt_path: str, cfg: str = "sam2_hiera_s", device="cpu"):
    """重みをロードして predictor を返す"""
    sam2 = build_sam2(cfg, None, device=device)
    state = torch.load(ckpt_path, map_location="cpu")
    sam2.load_state_dict(state)
    predictor = SAM2ImagePredictor(sam2)
    return predictor


def random_point(mask: np.ndarray):
    """マスクがあればその内部、無ければ画像中央を返す"""
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        i = np.random.randint(len(xs))
        return [[int(xs[i]), int(ys[i])]]  # [[x,y]]
    h, w = mask.shape
    return [[w // 2, h // 2]]


@torch.no_grad()
def run_inference(predictor, img_path: Path, out_dir: Path,
                  img_size=1024, device="cpu"):
    """1 枚の画像で推論し、PNG マスクを書き出す"""
    # ---------- 画像読み込み & リサイズ ----------
    img_bgr = cv2.imread(str(img_path))
    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size),
                             interpolation=cv2.INTER_LINEAR)

    # Predictor に画像をセット
    predictor.set_image(img_resized)

    # 中央クリック 1 点をプロンプト（マスク無し 0-shot）
    grid = np.linspace(0.2, 0.8, 4)        # 画面中央寄り 4×4=16 点
    point_coords = np.array([[int(x*w0), int(y*h0)] for y in grid for x in grid])
    point_labels = np.ones(len(point_coords), dtype=np.int64)
    # 推論
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    mask_out = (masks[0] > 0).astype(np.uint8) * 255  # H×W 0/255

    # ---------- 元解像度に戻して保存 ----------
    mask_orig = cv2.resize(mask_out, (w0, h0), interpolation=cv2.INTER_NEAREST)
    out_path = out_dir / f"{img_path.stem}_mask.png"
    cv2.imwrite(str(out_path), mask_orig)
    print("raw score =", scores[0])
    print("mask logits min/max =", masks.min(), masks.max())
    print(f"[✓] saved {out_path}  (score {scores[0]:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="学習済み .pt ファイル")
    parser.add_argument("--imgs", nargs="+", required=True,
                        help="推論したい画像パス（複数可）")
    parser.add_argument("--out_dir", default="outputs",
                        help="マスク保存先ディレクトリ")
    parser.add_argument("--size", type=int, default=1024,
                        help="リサイズ解像度（学習時と合わせる）")
    args = parser.parse_args()

    # デバイス自動判定
    device = "mps" if torch.backends.mps.is_available() \
             else "cuda" if torch.cuda.is_available() else "cpu"
    predictor = load_model(args.ckpt, cfg="sam2_hiera_s", device=device)
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True, parents=True)

    for path in args.imgs:
        run_inference(predictor, Path(path), out_dir,
                      img_size=args.size, device=device)


if __name__ == "__main__":
    main()