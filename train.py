import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from torch.distributed import destroy_process_group

from models import AIMClassificationModel
from utils import set_device, save_checkpoint, ddp_setup
from data import prepare_dataloader, train_transforms, val_transforms
from aim.v2.utils import load_pretrained

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_correct, total = 0.0, 0.0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), 100.0 * running_correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * running_correct / total

def train(rank, world_size, train_path, val_path, save_path, total_epochs, batch_size):
    ddp_setup(rank, world_size)
    device = set_device()

    batch_size_per_gpu = batch_size // world_size
    train_loader = prepare_dataloader(train_path, train_transforms(), batch_size_per_gpu, is_train=True)
    val_loader = prepare_dataloader(val_path, val_transforms(), batch_size_per_gpu, is_train=False)

    base_model = load_pretrained("aimv2-large-patch14-224", backend="torch")
    model = AIMClassificationModel(base_model, num_classes=1)
    model = DDP(model.to(device), device_ids=[rank])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-8, weight_decay=0.0)

    best_acc = 0.0
    for epoch in range(total_epochs):
        print(f"Epoch {epoch + 1}/{total_epochs}")
        train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Validation Accuracy: {val_acc:.2f}%")

        if rank == 0 and val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, epoch, optimizer, best_acc, save_path)

    destroy_process_group()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distributed Training with AIM Model")
    parser.add_argument("--train_path", type=str, default="./datasets/train/", help="Path to training dataset")
    parser.add_argument("--val_path", type=str, default="./datasets/val/", help="Path to validation dataset")
    parser.add_argument("--save_path", type=str, default="./saved_models/model_checkpoint.pth", help="Path to save the best model")
    parser.add_argument("--total_epochs", type=int, default=10, help="Number of total epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for each GPU")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    spawn(
        train,
        args=(world_size, args.train_path, args.val_path, args.save_path, args.total_epochs, args.batch_size),
        nprocs=world_size,
    )
