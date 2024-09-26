from activation_method import (
    register_hooks,
    clear_hooks,
    compute_activation_magnitudes,
)
from rich.progress import track
from torch import nn
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torch_uncertainty.datasets.classification.cifar.cifar_c import CIFAR10C

import copy

import os
import sys
sys.path.append(os.path.abspath("./model"))
from resnet import model

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*NORM)])

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
batch_size = 32

experiments = ["full", "layer1", "layer2", "layer3", "fc"]
exp_dict_loss = {i: [] for i in experiments}
exp_dict_acc = {i: [] for i in experiments}

val_source = torchvision.datasets.CIFAR10(
    root = "./data",
    train=False,
    transform=te_transforms,
    download=False
)
val_loader_source = DataLoader(val_source, batch_size=batch_size, shuffle=False)

# Init & load model
cifarc = CIFAR10C(
    root="./data",
    transform=te_transforms,
    subset="all",
    severity=5,
    download=True,
)

train, val, _ = random_split(cifarc, (1000, 5000, len(cifarc) - 6000))

# Init & load model
model = model()
model.load_state_dict(torch.load("model/cifar10.pth", map_location=device, weights_only=True))

# Feeding data
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

# Full-Tuning on train subset first

criterion = nn.CrossEntropyLoss()
optimizer_fc = torch.optim.Adam(model.parameters(), lr=0.005)
optimizer_src = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer_layers = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer = optimizer_layers

model = model.to(device)

model.eval()


### SFT DETERMINATION

# SOURCE

register_hooks(model)

with torch.no_grad():
    for data in track(val_loader_source, description="Eval Source Activations"):
        inputs, _ = data
        inputs = inputs.to(device)
        _ = model(inputs)

clear_hooks()

print("\nSource Activation Magnitudes")
print("-------------------------------------------")

layer_magnitudes_src = compute_activation_magnitudes()

for name, magnitude in layer_magnitudes_src:
    print(name + ": " + str(magnitude))

# TARGET

register_hooks(model)

with torch.no_grad():
    for data in track(train_loader, description="Eval Target Activations"):
        inputs, _ = data
        inputs = inputs.to(device)
        _ = model(inputs)

clear_hooks()

print("\nTarget Activation Magnitudes")
print("-------------------------------------------")

layer_magnitudes_tgt = compute_activation_magnitudes()

for name, magnitude in layer_magnitudes_tgt:
    print(f"{name}: {str(magnitude)}")
    
print("\nDifference")
print("-------------------------------------------")

max_magnitude = 0
max_name = ""

for idx in range(len(layer_magnitudes_src)):
    name, magnitude_src = layer_magnitudes_src[idx]
    _, magnitude_tgt = layer_magnitudes_tgt[idx]
    if name.startswith("layer") or name.startswith("fc"):
        diff = abs(magnitude_tgt - magnitude_src)
        print(f"{name}: {str(diff)}")
        if diff > max_magnitude:
            max_magnitude = diff
            max_name = name

print(f"SFT has chosen: {max_name}")

### SFT DETERMINATION ENDS


for i in experiments:
    print("-------------")
    print(i)
    print("-------------")
    model.load_state_dict(torch.load("model/cifar10.pth", map_location=device, weights_only=True))
    num_epochs = 10
    
    if i == "fc":
        optimizer = optimizer_fc
    else:
        optimizer = optimizer_layers
        
    if i == "full":
        for name, param in model.named_parameters():
            param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if name.startswith(i):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Initialize Variables for EarlyStopping
    best_loss = float("inf")
    best_model_weights = None
    original_patience = 2
    patience = original_patience

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
                
        exp_dict_loss[i].append(val_loss / len(val_loader))
        exp_dict_acc[i].append(100 * correct / total)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {running_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Val Accuracy: {100 * correct / total:.2f}%")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience = original_patience
        else:
            patience -= 1
            if patience == 0:
                break

print(f"SFT has chosen: {max_name}")
for i in experiments:
    print("Result for " + i + ":")
    print("Loss:\t", end = "")
    for loss in exp_dict_loss[i]:
        print(f"{loss:.4f}\t", end = "")
    print()
    print("Acc:\t", end = "")
    for acc in exp_dict_acc[i]:
        print(f"{acc:.2f}\t", end = "")
    print()
# # Save the fine-tuned model
# # torch.save(model.state_dict(), "result.pth")
