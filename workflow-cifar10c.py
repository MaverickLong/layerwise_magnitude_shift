import numpy as np
from activation_method import (
    register_hooks,
    clear_hooks,
    compute_activation_magnitudes,
)
from rich.progress import track
from torch import nn
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms

from torch_uncertainty.datasets.classification.cifar.cifar_c import CIFAR10C

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
num_classes = 10
hooks = []
batch_size = 128

FULL_TUNING = False

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*NORM)])

cifarc = CIFAR10C(
    root="./data",
    transform=te_transforms,
    subset="all",
    severity=5,
    download=True,
)

train, val, _ = random_split(cifarc, (1000, 1000, len(cifarc) - 2000))

# Init & load model
model = model()
model.load_state_dict(torch.load("model/cifar10c.pth", map_location=device))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Feeding data
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

num_features = model.fc.in_features
# Getting our shiny new layer
# model.fc = nn.Linear(num_features, num_classes)
# model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

if FULL_TUNING:
    for param in model.parameters():
        param.requires_grad = True
else:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


# Send model to device
model = model.to(device)


# Pre-Training loop
val_loss_list = []
val_acc_list = []

# Excluding Pre-Validation. At least 1.
num_epochs = 10

# Pre-Validation
model.eval()
val_loss = 0.0
correct = 0
total = 0

print("Pre-Validation")
with torch.no_grad():
    for inputs, labels in track(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_loss_list.append(val_loss / len(val_loader))
val_acc_list.append(100 * correct / total)
print(f"Pre-Val Loss: {val_loss/len(val_loader):.4f}")
print(f"Pre-Val Accuracy: {100 * correct / total:.2f}%")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in track(train_loader):
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
        for inputs, labels in track(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss_list.append(val_loss / len(val_loader))
    val_acc_list.append(100 * correct / total)
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {running_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"Val Accuracy: {100 * correct / total:.2f}%")

print("Loss History:")
print(val_loss_list)
print("Acc History:")
print(val_acc_list)

SIGNIFICANT_LOSS_DIFF = 2

if (val_loss_list[0] / val_loss_list[-1]) > SIGNIFICANT_LOSS_DIFF:
    print("Could be considered as significant. Exit for now.")
    exit()

print("Tuning FC is not effective. switch to activation mode.")

model.load_state_dict(torch.load("model/cifar10c.pth", map_location=device))

model.eval()
# Output level shift
register_hooks(model)

with torch.no_grad():
    for data in track(val_loader, description="Eval Cifar10C Activations"):
        inputs, _ = data
        inputs = inputs.to(device)
        _ = model(inputs)

clear_hooks()

layer_magnitudes = compute_activation_magnitudes()

print("\nActivation Magnitudes")
print("-------------------------------------------")

max_name = ""
max_magnitude = 0

for name, magnitude in layer_magnitudes:
    print(name + " " + str(magnitude))
    if magnitude > max_magnitude and name != "fc":
        max_name = name
        max_magnitude = magnitude

for name, param in model.named_parameters():
    if name.startswith(max_name):
        param.requires_grad = True
    else:
        param.requires_grad = False
    print(name + " " + str(param.requires_grad))

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in track(train_loader):
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
        for inputs, labels in track(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss_list.append(val_loss / len(val_loader))
    val_acc_list.append(100 * correct / total)
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {running_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"Val Accuracy: {100 * correct / total:.2f}%")

# Save the fine-tuned model
# torch.save(model.state_dict(), "result.pth")
