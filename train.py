import os
import torch
torch.autograd.set_detect_anomaly(True)

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from config import *
from dataset import SuperResDataset
from model import UNet
from diffusion import Diffusion

# Create folders
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(SAMPLE_PATH, exist_ok=True)

# Dataset + loader
dataset = SuperResDataset(LR_PATH, HR_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model + diffusion
model = UNet(IN_CHANNELS, OUT_CHANNELS, BASE_CHANNELS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
mse = nn.MSELoss()
diffusion = Diffusion(timesteps=TIMESTEPS, device=DEVICE)

# Optional resume
start_epoch = 0
checkpoint_file = os.path.join(CHECKPOINT_PATH, "latest.pt")
if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"✅ Resumed training from epoch {start_epoch}")

# Training loop
for epoch in range(start_epoch, EPOCHS):
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0

    for lr_img, hr_img in pbar:
        lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
        t = diffusion.sample_timesteps(hr_img.size(0))
        noised_hr, noise = diffusion.noise_image(hr_img, t)
        predicted_noise = model(noised_hr, t)
        loss = mse(noise, predicted_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(MSE=loss.item())

    print(f"🔁 Epoch {epoch+1} done — Avg Loss: {total_loss / len(loader):.4f}")

    # Save checkpoint
    if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }, checkpoint_file)

        print(f"💾 Checkpoint saved at epoch {epoch+1}")

        # (Optional) You can test sample generation from LR image here later if needed
        # Currently disabled since diffusion.sample() now expects an image, not (image_size, n)
        # You can enable sample saving in test.py instead
