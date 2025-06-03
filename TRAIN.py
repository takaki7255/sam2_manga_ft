#!/usr/bin/env python
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
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as maskUtils
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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


# ---------------------- Training ---------------------- #
def train():
    # ハイパラ & パス
    img_size = 1024
    batch_size = 4
    num_epochs = 20

    root_dir = "data"
    ann_file = "data/annotations/manga_balloon.json"
    checkpoint_path = "checkpoints/sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s"      # .yaml を付けない

    # デバイス
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    amp_enable = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enable)

    # データローダ
    ds = MangaBalloonCOCODataset(root_dir, ann_file, img_size)
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    # モデル
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    optimizer = torch.optim.AdamW(predictor.model.parameters(),
                                  lr=1e-5, weight_decay=4e-5)

    global_step, mean_iou = 0, 0.0

    for epoch in range(num_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch in pbar:
            global_step += 1
            imgs = batch["image"].to(device)
            masks_gt = batch["mask"].to(device)

            # predictor が受ける形式へ変換 (List[np.ndarray])
            imgs_np = [(img * 255).byte().permute(1, 2, 0).cpu().numpy() for img in imgs]
            # ランダムクリック生成
            pts = random_click_points(masks_gt)
            lbl = np.ones((len(pts), 1), dtype=np.int64)

            from contextlib import nullcontext
            amp_ctx = torch.cuda.amp.autocast(enabled=True) if device == "cuda" else nullcontext()
            with amp_ctx:
                predictor.set_image_batch(imgs_np)

                mask_in, coords_t, labels_t, _ = predictor._prep_prompts(
                    pts, lbl, box=None, mask_logits=None, normalize_coords=True
                )
                sparse_emb, dense_emb = predictor.model.sam_prompt_encoder(
                    points=(coords_t, labels_t), boxes=None, masks=None)

                high_res_feats = [f[-1].unsqueeze(0)
                                  for f in predictor._features["high_res_feats"]]

                low_masks, scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"],
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=high_res_feats)

                masks_pred = predictor._transforms.postprocess_masks(
                    low_masks, predictor._orig_hw[-1])
                masks_pred = torch.sigmoid(masks_pred[:, 0]).unsqueeze(1)  # [B, 1, H, W]

                # 損失
                eps = 1e-6
                seg_loss = (-masks_gt * torch.log(masks_pred + eps) -
                            (1 - masks_gt) * torch.log(1 - masks_pred + eps)).mean()

                inter = (masks_gt * (masks_pred > 0.5)).sum((1, 2, 3))
                union = masks_gt.sum((1, 2, 3)) + \
                    (masks_pred > 0.5).sum((1, 2, 3)) - inter
                iou = inter / (union + eps)
                score_loss = torch.abs(scores[:, 0] - iou).mean()

                loss = seg_loss + 0.05 * score_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            mean_iou = 0.99 * mean_iou + 0.01 * iou.mean().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             iou=f"{mean_iou:.3f}")

        # 各エポックで保存
        torch.save(predictor.model.state_dict(),
                   f"checkpoints/sam2_manga_epoch{epoch+1}.pt")


# ---------------------- Entry ---------------------- #
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    train()
