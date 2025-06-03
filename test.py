import torch, pathlib

old_path = "checkpoints/sam2_manga_epoch20.pt"
new_path = "checkpoints/sam2_manga_epoch20_dict.pt"

state = torch.load(old_path, map_location="cpu")  # OrderedDict
torch.save({"model": state, "epoch": 20}, new_path)
print("✔ 変換完了:", new_path)