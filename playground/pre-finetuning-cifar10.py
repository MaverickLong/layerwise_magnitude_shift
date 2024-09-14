import numpy as np
from rich.progress import track
from torch import nn
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

import sys
import os

sys.path.append(os.path.abspath("../model"))
from resnet import model


# Add noise util
def noise(tensor):
    return tensor + torch.randn(tensor.size()) / 10


# NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# te_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*NORM)])

batch_size = 64

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

te_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

cifar_train = torchvision.datasets.CIFAR10(
    root="../data",
    download=True,
    train=True,
    transform=te_transforms,
)

cifar_train_sub = Subset(cifar_train, range(0, len(cifar_train), 10))

cifar_train_loader = DataLoader(cifar_train_sub, batch_size=batch_size, shuffle=True)

cifar_val = torchvision.datasets.CIFAR10(
    root="../data",
    download=True,
    train=False,
    transform=te_transforms,
)

cifar_val_sub = Subset(cifar_val, range(0, len(cifar_val), 10))

cifar_val_loader = DataLoader(cifar_val_sub, batch_size=batch_size, shuffle=False)

# Init & load model
model = model()
model.load_state_dict(torch.load("../model/cifar10.pth", map_location=device))

num_features = model.fc.in_features
# 10 binary labels
num_classes = 10
# Getting our shiny new layer (not needed in cifar10 case?)
model.fc = nn.Linear(num_features, num_classes)
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

# uncomment these to freeze all layers except fc
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
# the above

# Send model to device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 3

# Validation
model.eval()
val_loss = 0.0
correct = 0
total = 0

val_loss_list = []
val_acc_list = []

print("Initial Validation")
with torch.no_grad():
    for inputs, labels in track(cifar_val_loader):
        inputs = noise(inputs)
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_loss_list.append(val_loss / len(cifar_val_loader))
val_acc_list.append(100 * correct / total)
print(f"Val Loss: {val_loss/len(cifar_val_loader):.4f}")
print(f"Val Accuracy: {100 * correct / total:.2f}%")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in track(cifar_train_loader):
        inputs = noise(inputs)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in track(cifar_val_loader):
            inputs = noise(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss_list.append(val_loss / len(cifar_val_loader))
    val_acc_list.append(100 * correct / total)
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {running_loss/len(cifar_train_loader):.4f}")
    print(f"Val Loss: {val_loss/len(cifar_val_loader):.4f}")
    print(f"Val Accuracy: {100 * correct / total:.2f}%")

print("Loss History:")
print(val_loss_list)
print("Acc History:")
print(val_acc_list)

# Save the fine-tuned model
torch.save(model.state_dict(), "resnet_cifar.pth")
