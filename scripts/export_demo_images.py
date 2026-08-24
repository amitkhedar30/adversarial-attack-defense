"""
Run this ONCE in your Kaggle session to export a small, self-contained
subset of CIFAR-10 test images for the dashboard.

Why this exists: the deployed dashboard (Streamlit Community Cloud / HF
Spaces) has no access to /kaggle/input, so it needs its own bundled copy
of a few demo images rather than pulling from the full test set live.

Prerequisites: repo already cloned and on sys.path (same as your training
notebook), so `utils.dataset` is importable.

Output: demo_images.pt -- copy this into dashboard/assets/ in your repo.
"""

import torch
from utils.dataset import get_dataloaders

N_IMAGES = 40  # keep this small -- it ships inside the dashboard repo

_, test_loader = get_dataloaders(batch_size=1)

images, labels = [], []
for i, (x, y) in enumerate(test_loader):
    if i >= N_IMAGES:
        break
    images.append(x.squeeze(0))
    labels.append(y.item())

bundle = {
    "images": torch.stack(images),  # [N, 3, 32, 32], values in [0, 1]
    "labels": torch.tensor(labels),
}
torch.save(bundle, "/kaggle/working/demo_images.pt")
print(f"[*] Saved demo_images.pt with {N_IMAGES} images")
print("[*] Copy this file into dashboard/assets/demo_images.pt")
