#!/usr/bin/env python
# infer_click.py  –  手動クリック & 自動フォールバック

import argparse, cv2, numpy as np, torch
from pathlib import Path
from torch.backends.cuda import sdp_kernel, SDPBackend
import torch.cuda.amp as amp
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------- クリック UI ---------- #
def gather_points(img_bgr, scale=1.0):
    """
    画像を表示してクリック点を取得。
    左クリック: 正例 (label=1), 右クリック: 負例 (label=0)
    Enter で確定,  c キーでクリア,  Esc でキャンセル
    Returns: (pts: np.ndarray[N,2], labels: np.ndarray[N])
    """
    pts, lbls = [], []
    win = "click (Enter=done, c=clear)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal img_show, pts, lbls
        if event == cv2.EVENT_LBUTTONDOWN:   # 正例
            pts.append([x, y]); lbls.append(1)
            cv2.circle(img_show, (x,y), 4, (0,255,0), -1)
        elif event == cv2.EVENT_RBUTTONDOWN: # 負例
            pts.append([x, y]); lbls.append(0)
            cv2.circle(img_show, (x,y), 4, (0,0,255), -1)

    img_show = img_bgr.copy()
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, img_show)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):                   # Enter
            break
        elif k in (27, ):                   # Esc
            pts, lbls = [], []
            break
        elif k in (ord('c'), ord('C')):
            pts, lbls = [], []
            img_show = img_bgr.copy()
    cv2.destroyWindow(win)
    return np.array(pts, np.int64)//scale, np.array(lbls, np.int64)

# ---------- util ---------- #
def load_ckpt(model, path):
    obj = torch.load(path, map_location="cpu")
    sd  = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(sd, strict=False)

def grid_points(h,w,n=4):
    g = np.linspace(0.2,0.8,n)
    return np.array([[int(x*w),int(y*h)] for y in g for x in g],np.int64)

# ---------- 推論 1 枚 ---------- #
@torch.no_grad()
def run_single(predictor, img_path:Path, size:int, clicks:int,
               th:float, out_dir:Path):
    bgr = cv2.imread(str(img_path)); h0,w0=bgr.shape[:2]
    scale = size/max(h0,w0); new = cv2.resize(bgr, (int(w0*scale),int(h0*scale)))
    pts,lbls = gather_points(new, scale=scale)
    rgb = cv2.cvtColor(cv2.resize(bgr,(size,size)), cv2.COLOR_BGR2RGB)

    predictor.set_image(rgb)

    if len(pts)==0:           # 自動グリッド
        pts  = grid_points(size,size,clicks)
        lbls = np.ones(len(pts),np.int64)

    masks,scores,_ = predictor.predict(
        point_coords=pts[None], point_labels=lbls[None],
        multimask_output=True, box=None)

    best = masks[np.argmax(scores)]
    mask = (best>th).astype(np.uint8)*255
    mask = cv2.resize(mask,(w0,h0),cv2.INTER_NEAREST)
    out = out_dir/f"{img_path.stem}_mask.png";  cv2.imwrite(str(out),mask)
    print(f"[✓] {img_path.name:15} → {out.name}  score={scores.max():.3f}")

# ---------- main ---------- #
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True)
    ap.add_argument("--imgs",nargs="+",required=True)
    ap.add_argument("--out_dir",default="outputs")
    ap.add_argument("--size",type=int,default=1024)
    ap.add_argument("--clicks",type=int,default=4)
    ap.add_argument("--th",type=float,default=0.5)
    opt=ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else \
          "mps" if torch.backends.mps.is_available() else "cpu"
    flash_ok = dev=="cuda" and torch.cuda.get_device_capability()[0]>=8
    model = build_sam2("sam2_hiera_s",None,device=dev)
    if flash_ok: model.half()
    load_ckpt(model,opt.ckpt); predictor=SAM2ImagePredictor(model)

    out_dir=Path(opt.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    ctx = sdp_kernel(SDPBackend.FLASH_ATTENTION) if flash_ok \
          else sdp_kernel(SDPBackend.MATH)
    amp_ctx = amp.autocast(dtype=torch.float16) if flash_ok else nullcontext()

    with ctx, amp_ctx:
        for p in map(Path,opt.imgs):
            run_single(predictor,p,opt.size,opt.clicks,opt.th,out_dir)

if __name__=="__main__":
    main()
