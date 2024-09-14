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

# Device Setup
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Helpers


# Function to get balanced subset of (blond_hair, male)
def get_balanced_blond_subset(dataset, num_samples_per_group=None):
    blond_hair_idx = 9
    gender_idx = 20

    # Initialize lists to hold indices of each group
    group_indices = {
        (1, 1): [],  # (blond_hair, male)
        (0, 1): [],  # (not_blond_hair, male)
        (1, 0): [],  # (blond_hair, female)
        (0, 0): [],  # (not_blond_hair, female)
    }

    # Iterate through the dataset and separate indices based on attributes
    for idx, (img, attr) in enumerate(dataset):
        hair_color = attr[blond_hair_idx].item()  # blond_hair
        gender = attr[gender_idx].item()  # gender
        group_indices[(hair_color, gender)].append(idx)

    # Find the minimum number of samples among all groups to balance the dataset
    if num_samples_per_group is None:
        num_samples_per_group = min(len(group) for group in group_indices.values())

    # Get the balanced indices for each group
    balanced_indices = []
    for group, indices in group_indices.items():
        balanced_indices.extend(
            np.random.choice(indices, num_samples_per_group, replace=False)
        )

    # Create a balanced subset of the dataset
    balanced_subset = Subset(dataset, balanced_indices)

    return balanced_subset


# config options
num_classes = 1
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
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 40)
model.load_state_dict(torch.load("model/celeba.pth", map_location=device))

criterion = nn.BCEWithLogitsLoss()
optimizer_fc = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Feeding data
train = get_balanced_blond_subset(train)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

num_features = model.fc.in_features
# Getting our shiny new layer
# We only need ONE output, blond or not blond.
model.fc = nn.Linear(num_features, num_classes)

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
# for param in model.parameters():
#     param.requires_grad = True

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
        labels = labels[:, 9].float().to(device)

        outputs = model(inputs)
        labels = labels.unsqueeze(1)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        predicted = (outputs > 0.5).float()
        total += labels.size(0) * labels.size(1)
        correct += (predicted == labels).sum().item()

print(f"Pre-Validation Result:")
print(f"Val Loss: {val_loss/len(val_loader):.4f}")
print(f"Val Accuracy: {100 * correct / total:.2f}%")

val_loss_list.append(val_loss / len(val_loader))
val_acc_list.append(100 * correct / total)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in track(train_loader):
        inputs = inputs.to(device)
        # Filters the Blond attribute, index number 9.
        labels = labels[:, 9].float().to(device)

        optimizer_fc.zero_grad()
        outputs = model(inputs)
        labels = labels.unsqueeze(1)
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
            labels = labels[:, 9].float().to(device)

            outputs = model(inputs)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0) * labels.size(1)
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

    model.eval()
    # Output level shift
    register_hooks(model)

    with torch.no_grad():
        for data in track(val_loader, description="Eval CelebA Activations"):
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

num_epochs = 10

# Initialize Variables for EarlyStopping
best_loss = float("inf")
best_model_weights = None
patience = 2

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in track(train_loader):
        inputs = inputs.to(device)
        labels = labels[:, 9].float().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.unsqueeze(1)
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
            labels = labels[:, 9].float().to(device)

            outputs = model(inputs)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0) * labels.size(1)
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
