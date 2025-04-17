# config.py

# Paths
HR_PATH = "data/HR"
LR_PATH = "data/LR"
SAVE_DIR = "results"
CHECKPOINT_PATH = "checkpoints"
SAMPLE_PATH = "samples"

# Image + Model
IMAGE_SIZE = 512
IN_CHANNELS = 1
OUT_CHANNELS = 1
BASE_CHANNELS = 64

# Training
BATCH_SIZE = 2         # Balanced for 6GB VRAM
EPOCHS = 200
LR = 2e-4              # Faster convergence than 1e-4
SAVE_EVERY = 5         # Save checkpoint every N epochs

# Diffusion
TIMESTEPS = 50         # Good balance between quality & speed
BETA_START = 1e-4
BETA_END = 0.02

# Device
DEVICE = "cuda"
