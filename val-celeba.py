import numpy as np
from activation_method import (
    register_hooks,
    clear_hooks,
    compute_activation_magnitudes,
)
from rich.progress import track
from torch import nn
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

import sys
import os

sys.path.append(os.path.abspath("./model"))
from resnet import model

# Device Setup
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# config options
num_classes = 40
hooks = []
batch_size = 128

te_transforms = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

train = torchvision.datasets.CelebA(
    root="./data",
    split="train",
    download=True,
    transform=te_transforms,
    target_type="attr",
)

val = torchvision.datasets.CelebA(
    root="./data",
    split="valid",
    download=True,
    transform=te_transforms,
    target_type="attr",
)

# Init & load model
model = model()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Feeding data
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

num_features = model.fc.in_features
# Getting our shiny new layer
model.fc = nn.Linear(num_features, num_classes)

model.load_state_dict(torch.load("model/celeba-via-imagenet.pth", map_location=device, weights_only=True))

model.to(device)

val_loss = 0.0
total = 0
correct = 0

with torch.no_grad():
    for inputs, labels in track(val_loader):
        inputs = inputs.to(device)
        labels = labels.float().to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        predicted = (outputs > 0.5).float()
        total += labels.size(0) * labels.size(1)
        correct += (predicted == labels).sum().item()
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"Val Accuracy: {100 * correct / total:.2f}%")
