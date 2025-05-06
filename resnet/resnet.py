import time
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import freeze_support  # for PyInstaller/exes, optional otherwise

def train():
    # 1) Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset,
                               batch_size=32,
                               shuffle=True,
                               num_workers=2,
                               pin_memory=True)

    # 2) Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(train_dataset.classes))
    resnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=0.001)

    # 3) Training loop
    num_epochs = 2
    start = time.perf_counter()
    print(f"🚀 Starting training on {device}")

    for epoch in range(num_epochs):
        resnet.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(avg_loss=running_loss / (pbar.n + 1))

        avg_loss = running_loss / len(train_loader)
        print(f"→ Epoch {epoch+1} finished, avg loss: {avg_loss:.4f}")

    # 4) Final save & summary
    torch.save(resnet.state_dict(), "resnet_custom.pth")
    elapsed = time.perf_counter() - start
    print(f"✅ Training complete in {elapsed/60:.2f} minutes.")

if __name__ == "__main__":
    freeze_support()   # safe on Windows if you ever freeze to an exe
    train()
