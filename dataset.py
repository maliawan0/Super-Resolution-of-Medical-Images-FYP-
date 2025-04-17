# dataset.py
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

class SuperResDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.file_names = sorted(os.listdir(lr_dir))

        self.transform = T.Compose([
            T.ToTensor(),  # Converts to [0,1] and shape (1, H, W)
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.file_names[idx])
        hr_path = os.path.join(self.hr_dir, self.file_names[idx])

        lr_image = Image.open(lr_path).convert('L')  # 'L' = grayscale
        hr_image = Image.open(hr_path).convert('L')

        lr_tensor = self.transform(lr_image)
        hr_tensor = self.transform(hr_image)

        return lr_tensor, hr_tensor

