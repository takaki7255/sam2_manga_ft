"""
Train SAM-2 on Manga Balloon COCO Dataset
ディレクトリ例:
 sam2_manga_ft/
 ├ sam2/                  # Meta の公式リポジトリ
 │ └ sam2_configs/__init__.py  ← 空ファイルで OK
 ├ data/
 │   ├ images/            # 元画像
 │   └ annotations/manga_balloon.json
 ├ checkpoints/sam2_hiera_small.pt
 └ train.py               # ← このファイル
"""

import os, json, cv2, numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as maskUtils
from tqdm import tqdm
from contextlib import nullcontext

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import wandb

samples_dir = Path("samples"); samples_dir.mkdir(exist_ok=True)


# ---------------------- Dataset ---------------------- #
class MangaBalloonCOCODataset(Dataset):
    """COCO 形式（1 クラス）の最小実装"""

    def __init__(self, root: str, json_file: str, img_size: int = 1024):
        self.root = Path(root)
        with open(json_file) as f:
            coco = json.load(f)

        self.img_dir = self.root / "images"
        self.anns = coco["annotations"]
        self.imgs = {img["id"]: img for img in coco["images"]}
        self.img_size = img_size

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        meta = self.imgs[ann["image_id"]]
        img_path = self.img_dir / meta["file_name"]

        # 画像読み込み & リサイズ（RGB）
        img = cv2.imread(str(img_path))[:, :, ::-1]
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)

        # polygon → mask → リサイズ
        rles = maskUtils.frPyObjects(ann["segmentation"],
                                     meta["height"], meta["width"])
        mask = maskUtils.decode(maskUtils.merge(rles)).astype(np.float32)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # Torch Tensor へ
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask)[None, ...]

        return {"image": img_t, "mask": mask_t}


# ---------------------- Utility ---------------------- #
def random_click_points(mask_batch: torch.Tensor) -> np.ndarray:
    """(B,1,H,W) のバイナリマスクから 1 点ランダムクリックを返す"""
    pts = []
    for mk in mask_batch:
        ys, xs = torch.nonzero(mk[0], as_tuple=True)
        if len(xs) == 0:              # マスクが空なら中央
            h, w = mk.shape[-2:]
            pts.append([[w // 2, h // 2]])
        else:
            i = torch.randint(0, len(xs), (1,)).item()
            pts.append([[xs[i].item(), ys[i].item()]])
    return np.array(pts, dtype=np.int64)

def sample_points(mask_batch: torch.Tensor, n_pts=4):
    pts = []
    for mk in mask_batch:
        ys, xs = torch.nonzero(mk[0], as_tuple=True)
        if len(xs) == 0:
            h, w = mk.shape[-2:]
            pts.append([[w//2, h//2]])
        else:
            idx = torch.randperm(len(xs))[:n_pts]
            pts.append([[int(xs[i]), int(ys[i])] for i in idx])
    return np.array(pts, dtype=np.int64)   # (B, N, 2)

def load_maybe_legacy_ckpt(path):
    obj = torch.load(path, map_location="cpu")
    return obj["model"] if isinstance(obj, dict) and "model" in obj else obj

def resume_or_initialize(model,
                         optimizer,
                         scaler,
                         ckpt_path: str | None) -> int:
    """
    *None* または存在しない ckpt → 新規学習 (epoch0)。
    dict 形式なら model/optimizer/scaler/epoch を復元。
    state-dict 形式なら model だけロード。
    Returns: start_epoch
    """
    if ckpt_path is None or not Path(ckpt_path).exists():
        print("⚑ No checkpoint. Start new training.")
        return 0

    obj = torch.load(ckpt_path, map_location="cpu")
    # -------- model --------
    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"], strict=False)
    else:
        model.load_state_dict(obj, strict=False)

    # -------- opt / scaler --------
    if isinstance(obj, dict) and "optimizer" in obj:
        optimizer.load_state_dict(obj["optimizer"])
        print("✔ optimizer restored")
    if isinstance(obj, dict) and "scaler" in obj:
        scaler.load_state_dict(obj["scaler"])
        print("✔ GradScaler restored")

    start_epoch = obj["epoch"] if isinstance(obj, dict) and "epoch" in obj else 0
    print(f"▶ Resume from epoch {start_epoch}")
    return start_epoch


# ---------------------- Training ---------------------- #
def train():
    # ハイパラ & パス
    img_size = 1024
    batch_size = 4
    TOTAL_EPOCHS   = 40
    lr = 1e-5
    VIS_EVERY = 500         # 何ステップごとに保存するか
    CLICKS_PER_MASK = 4
    CKPT_PATH      = "checkpoints/sam2_hiera_small.pt"
    MODEL_CFG      = "sam2_hiera_s"
    SAVE_FREQ      = 1          # ★ 何エポックごとに保存するか
    BEST_PATH      = "checkpoints/best_miou.pt"   # ★ ベストモデル保存先
    
    wandb.init(
    project="sam2-manga",            
    name   ="hiera_small_bs4_lr1e-5",
    config = dict(
        img_size   = img_size,
        batch_size = batch_size,
        lr         = lr,
        epochs     = TOTAL_EPOCHS,
        backbone   = "sam2_hiera_small",
    ),
    # sync_tensorboard=True  # TensorBoard 併用ならコメント解除
)

    root_dir = "data"
    ann_file = "data/annotations/manga_balloon.json"

    # デバイス
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    amp_enable = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enable)

    # データローダ
    ds = MangaBalloonCOCODataset(root_dir, ann_file, img_size)
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    # モデル
    model = build_sam2(MODEL_CFG, None, device=device)
    predictor = SAM2ImagePredictor(model)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=4e-5)
    scaler      = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_epoch = resume_or_initialize(model, optimizer, scaler, CKPT_PATH)
    global_step = start_epoch * len(loader)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)  # 4 epoch ごとに 1/10

    global_step, mean_iou = 0, 0.0
    best_miou   = -1.0

    samples_dir = Path("samples"); samples_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = Path("checkpoints"); checkpoints_dir.mkdir(exist_ok=True)

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        pbar = tqdm(loader, desc=f"E{epoch+1}/{TOTAL_EPOCHS}", unit="batch")
        for step, batch in enumerate(pbar):
            mean_iou_epoch = 0.0
            global_step += 1
            imgs   = batch["image"].to(device)
            masks_gt = batch["mask"].to(device)

            # クリック生成
            point_coords = sample_points(masks_gt, n_pts=CLICKS_PER_MASK)   # (B,N,2)
            point_labels = np.ones(point_coords.shape[:2], dtype=np.int64)  # (B,N)

            # 画像を list[np.ndarray] → predictor
            imgs_np = [(img * 255).byte().permute(1, 2, 0).cpu().numpy() for img in imgs]
            predictor.set_image_batch(imgs_np)

            amp_ctx = torch.cuda.amp.autocast(enabled=True) if device == "cuda" else nullcontext()
            with amp_ctx:
                # prompt encoder
                mask_in, coords_t, labels_t, _ = predictor._prep_prompts(
                    point_coords, point_labels,
                    box=None, mask_logits=None, normalize_coords=True)
                sparse_emb, dense_emb = predictor.model.sam_prompt_encoder(
                    points=(coords_t, labels_t),
                    boxes=None, masks=None)

                # mask decoder
                hi_feats = [f[-1].unsqueeze(0)
                            for f in predictor._features["high_res_feats"]]
                low_masks, scores, *_ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"],
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=hi_feats)
                logits = predictor._transforms.postprocess_masks(low_masks, predictor._orig_hw[-1])           # (B,1,H,W) logits
                masks_pred = torch.sigmoid(logits)

                # --- loss ---
                eps = 1e-6
                # Balanced BCE
                pos_pix = masks_gt.mean().item()
                # Balanced BCE ― ロジット版で安全に
                bce = F.binary_cross_entropy_with_logits(logits, masks_gt, weight=masks_gt * (1-pos_pix) + (1-masks_gt) * pos_pix)
                # Dice
                inter = (masks_pred * masks_gt).sum((1,2,3))
                dice = 1 - (2*inter + eps) / (masks_pred.sum((1,2,3))
                                               + masks_gt.sum((1,2,3)) + eps)
                loss = bce + dice.mean()

            # --- backward ---
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- metrics ---
            with torch.no_grad():
                pred_bin = (masks_pred > 0.5).float()
                inter = (pred_bin * masks_gt).sum((1,2,3))
                union = masks_gt.sum((1,2,3)) + pred_bin.sum((1,2,3)) - inter
                iou = (inter / (union + eps)).mean().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.3f}")
            mean_iou_epoch = (mean_iou_epoch * step + iou) / (step + 1) if step else iou

            # --- wandb log (optional) ---
            wandb.log({"train/loss": loss.item(),
                       "train/iou":  iou,
                       "lr": optimizer.param_groups[0]["lr"],
                       "epoch_step": epoch + step/len(loader)},
                      step=global_step)

            # --- Save sample PNGs ---
            if global_step % VIS_EVERY == 0:
                gt_np   = (masks_gt[0,0]*255).cpu().byte().numpy()
                pred_np = (pred_bin[0,0]*255).cpu().byte().numpy()
                cv2.imwrite(str(samples_dir /
                           f"e{epoch+1:02d}_s{global_step}_gt.png"), gt_np)
                cv2.imwrite(str(samples_dir /
                           f"e{epoch+1:02d}_s{global_step}_pred.png"), pred_np)

        # --- epoch end ---
        scheduler.step()

        # ---- ベストモデル判定 ----
        if mean_iou_epoch > best_miou:
            best_miou = mean_iou_epoch
            torch.save({"model": model.state_dict(),
                        "miou":  best_miou,
                        "epoch": epoch + 1},
                    BEST_PATH)
            print(f"✓ New best mIoU {best_miou:.4f}  →  {BEST_PATH}")

        # ---- 周期保存 ----
        if (epoch + 1) % SAVE_FREQ == 0:
            torch.save({
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler":    scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch":     epoch + 1,
                "miou":      mean_iou_epoch,
            }, checkpoints_dir / f"sam2_manga_epoch{epoch+1}.pt")


# ---------------------- Entry ---------------------- #
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    train()
