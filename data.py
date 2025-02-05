import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

MEAN = (0.9214, 0.9470, 0.9204)
STD = (0.0843, 0.0544, 0.0828)

def train_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(MEAN), torch.Tensor(STD))
    ])

def val_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(MEAN), torch.Tensor(STD))
    ])

def prepare_dataloader(dataset_path, transform, batch_size, is_train=True):
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=is_train)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
