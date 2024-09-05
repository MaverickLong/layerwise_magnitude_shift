import numpy as np
from rich.progress import track
from torch import nn
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from models import model


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

train = torchvision.datasets.CelebA(
    root="./data",
    split="train",
    download=True,
    transform=te_transforms,
    target_type="attr",
)

train_sub = Subset(train, range(0, len(train), 10))

train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)

val = torchvision.datasets.CelebA(
    root="./data",
    split="valid",
    download=True,
    transform=te_transforms,
    target_type="attr",
)

val_sub = Subset(val, range(0, len(val), 10))

val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

# Init & load model
model = model()
model.load_state_dict(torch.load("best_weights.pth", map_location=device))

num_features = model.fc.in_features
# 40 binary labels
num_classes = 40
# Getting our shiny new layer
model.fc = nn.Linear(num_features, num_classes)
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Send model to device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
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
    for inputs, labels in track(val_loader):
        inputs = inputs.to(device)
        labels = labels.float().to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        predicted = (outputs > 0.5).float()
        total += labels.size(0) * labels.size(1)
        correct += (predicted == labels).sum().item()

val_loss_list.append(val_loss / len(val_loader))
val_acc_list.append(100 * correct / total)
print(f"Val Loss: {val_loss/len(val_loader):.4f}")
print(f"Val Accuracy: {100 * correct / total:.2f}%")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in track(train_loader):
        inputs = inputs.to(device)
        labels = labels.float().to(device)

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
            labels = labels.float().to(device)

            outputs = model(inputs)
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

# Save the fine-tuned model
torch.save(model.state_dict(), "resnet_celeba.pth")
