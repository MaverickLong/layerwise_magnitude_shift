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

import copy

import sys
import os

sys.path.append(os.path.abspath("./model"))
from resnet import model
from cifar10_flip import ShiftedLabelsCIFAR10

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

cifar_train = ShiftedLabelsCIFAR10(
    root="./data",
    download=True,
    train=True,
    transform=te_transforms,
)

train_indices = torch.randperm(len(cifar_train))[:1000]

train = Subset(cifar_train, train_indices)

cifar_val = ShiftedLabelsCIFAR10(
    root="./data",
    download=True,
    train=False,
    transform=te_transforms,
)

val_indices = torch.randperm(len(cifar_val))[:1000]

val = Subset(cifar_val, val_indices)

# Init & load model
model = model()
model.load_state_dict(torch.load("model/cifar10.pth", map_location=device))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer_fc = torch.optim.Adam(model.parameters(), lr=0.01)

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
num_epochs = 1

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

        optimizer_fc.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_fc.step()

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

SIGNIFICANT_LOSS_RATIO = 1.5
SIGNIFICANT_LOSS_DIFF = 2

if (val_loss_list[0] / val_loss_list[-1]) > SIGNIFICANT_LOSS_RATIO or (
    val_loss_list[0] - val_loss_list[-1]
) > SIGNIFICANT_LOSS_DIFF:
    print("Could be considered as significant. Continuing...")
    optimizer = optimizer_fc
else:
    print("Tuning FC is not effective. switch to activation mode.")

    model.load_state_dict(torch.load("model/cifar10.pth", map_location=device))

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
        print(name + ": " + str(magnitude))
        if magnitude > max_magnitude and name != "fc":
            max_name = name
            max_magnitude = magnitude

    for name, param in model.named_parameters():
        if name.startswith(max_name):
            param.requires_grad = True
        else:
            param.requires_grad = False


# Initialize Variables for EarlyStopping
best_loss = float("inf")
best_model_weights = None
patience = 2

num_epochs = 10

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

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        patience = 2
    else:
        patience -= 1
        if patience == 0:
            break

    val_loss_list.append(val_loss / len(val_loader))
    val_acc_list.append(100 * correct / total)
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {running_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"Val Accuracy: {100 * correct / total:.2f}%")

# Save the fine-tuned model
# torch.save(model.state_dict(), "result.pth")
