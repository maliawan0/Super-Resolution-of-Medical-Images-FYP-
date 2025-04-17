# test.py
import os
import torch
from torchvision.utils import save_image
from dataset import SuperResDataset
from model import UNet
from diffusion import Diffusion
from config import *

from torch.utils.data import DataLoader

# Create output folder
os.makedirs("predictions", exist_ok=True)

# Load dataset (inference only needs 1 or few samples)
test_dataset = SuperResDataset(LR_PATH, HR_PATH)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load trained model
model = UNet(IN_CHANNELS, OUT_CHANNELS, BASE_CHANNELS).to(DEVICE)
checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, "latest.pt"))
model.load_state_dict(checkpoint["model"])
model.eval()
print("✅ Loaded trained model from checkpoint")

# Setup diffusion (no training)
diffusion = Diffusion(timesteps=TIMESTEPS, device=DEVICE)

# Run on a few samples
with torch.no_grad():
    for i, (lr_img, hr_img) in enumerate(test_loader):
        lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)

        # Start from random noise and generate SR image using diffusion
        pred = diffusion.sample(model, lr_img)

        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        # Save images for comparison
        save_image(lr_img, f"predictions/sample_{i}_LR.png")
        save_image(pred, f"predictions/sample_{i}_SR.png")
        save_image(hr_img, f"predictions/sample_{i}_HR.png")

        print(f"✅ Saved LR, SR, HR images for sample {i}")

        if i == 4:  # Limit to first 5 samples
            break
