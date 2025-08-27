from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import os
import glob
from PIL import Image
import torch
import torch.nn.functional as FA


class EasyPortrait(Dataset):
    def __init__(self, root_dir: str, type: str, transform=None):
        image_dir = os.path.join(root_dir, "images", type)
        self.image_paths = []
        for ext in ('*.jpeg', '*.jpg', '*.png'):
            self.image_paths += glob.glob(os.path.join(image_dir, ext))
        self.image_paths = sorted(self.image_paths)

        output_dir = os.path.join(root_dir, "annotations", type)
        self.output_paths = []
        for ext in ('*.jpeg', '*.jpg', '*.png'):
            self.output_paths += glob.glob(os.path.join(output_dir, ext))
        self.output_paths = sorted(self.output_paths)
        self.transform = transform
        if not self.image_paths:
            raise RuntimeError(f"No .jpeg images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        output_path = self.output_paths[idx]

        image = Image.open(img_path).convert("RGB")
        output = Image.open(output_path).convert("L")
        output = F.pil_to_tensor(output)

        if self.transform:
            image = self.transform(image)

        output = output.squeeze(0).long()
        output_onehot = FA.one_hot(output, num_classes=9)
        output_onehot = output_onehot.permute(2, 0, 1).float()

        return image, output_onehot


def get_dataloader(root: str, input_size=256, batch_size=16):

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5),
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # --- Datasets and DataLoaders ---
    train_dataset =  EasyPortrait(root_dir=root, type="train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


    val_dataset =  EasyPortrait(root_dir=root, type="val", transform=test_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    test_dataset =  EasyPortrait(root_dir=root, type="test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Found {len(train_dataset)} images for training.")
    print(f"Found {len(val_dataset)} images for validating.")
    print(f"Found {len(test_dataset)} images for testing.")

    return train_loader, val_loader, test_loader

