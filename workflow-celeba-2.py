from activation_method import (
    register_hooks,
    clear_hooks,
    compute_activation_magnitudes,
)
from rich.progress import track
from torch import nn
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision
import numpy as np

import copy

# Device Setup
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# config options
num_classes = 1
hooks = []
batch_size = 32

experiments = ["full", "layer1", "layer2", "layer3", "fc"]
exp_dict_loss = {i: [] for i in experiments}
exp_dict_acc = {i: [] for i in experiments}

# Init & load model
import os
import sys
sys.path.append(os.path.abspath("./model"))
from resnet import model

model = model()
num_features = model.fc.in_features

te_transforms = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

train_org = torchvision.datasets.CelebA(
    root="./data",
    split="train",
    download=True,
    transform=te_transforms,
    target_type="attr",
)

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
    for idx, (_, attr) in enumerate(dataset):
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

train = get_balanced_blond_subset(train_org)

val = torchvision.datasets.CelebA(
    root="./data",
    split="valid",
    download=True,
    transform=te_transforms,
    target_type="attr",
)

# TARGET LOADERS

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

# Full-Tuning on train subset first

criterion = nn.BCEWithLogitsLoss()
optimizer_fc = torch.optim.Adam(model.parameters(), lr=0.005)
optimizer_src = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer_layers = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer = optimizer_layers

for param in model.parameters():
    param.requires_grad = True

# Initialize Variables for EarlyStopping

model.fc = nn.Linear(num_features, 40)
model.load_state_dict(torch.load("model/celeba.pth", map_location=device, weights_only=True))
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

model.eval()


### SFT DETERMINATION

# SOURCE

register_hooks(model)

with torch.no_grad():
    for data in track(val_loader, description="Eval Source Activations"):
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
    if name.startswith("layer"):
        diff = abs(magnitude_tgt - magnitude_src)
        print(f"{name}: {str(diff)}")
        if diff > max_magnitude:
            max_magnitude = diff
            max_name = name

print(f"SFT has chosen: {max_name}")

### SFT DETERMINATION ENDS


# for i in experiments:
#     print("-------------")
#     print(i)
#     print("-------------")
    
#     model.fc = nn.Linear(num_features, 40)
#     model.load_state_dict(torch.load("model/celeba.pth", map_location=device, weights_only=True))
#     model.fc = nn.Linear(num_features, num_classes)
    
#     model = model.to(device)
    
#     num_epochs = 10
    
#     if i == "fc":
#         optimizer = optimizer_fc
#     else:
#         optimizer = optimizer_layers
        
#     if i == "full":
#         for name, param in model.named_parameters():
#             param.requires_grad = True
#     else:
#         split = i.split("+")
#         for name, param in model.named_parameters():
#             for j in split:
#                 if name.startswith(j):
#                     param.requires_grad = True
#                     break
#                 else:
#                     param.requires_grad = False

#     # Initialize Variables for EarlyStopping
#     best_loss = float("inf")
#     best_model_weights = None
#     original_patience = 2
#     patience = original_patience

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0

#         for inputs, labels in track(train_loader):
#             inputs = inputs.to(device)
#             # Filters the Blond attribute, index number 9.
#             labels = labels[:, 9].float().to(device)

#             optimizer_fc.zero_grad()
#             outputs = model(inputs)
#             labels = labels.unsqueeze(1)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer_fc.step()

#             running_loss += loss.item()

#         # Validation
#         model.eval()
#         val_loss = 0.0
#         correct = 0
#         total = 0

#         with torch.no_grad():
#             for inputs, labels in track(val_loader):
#                 inputs = inputs.to(device)
#                 labels = labels[:, 9].float().to(device)

#                 outputs = model(inputs)
#                 labels = labels.unsqueeze(1)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()

#                 predicted = (outputs > 0.5).float()
#                 total += labels.size(0) * labels.size(1)
#                 correct += (predicted == labels).sum().item()
                
#         exp_dict_loss[i].append(val_loss / len(val_loader))
#         exp_dict_acc[i].append(100 * correct / total)
        
#         print(f"Epoch [{epoch+1}/{num_epochs}]")
#         print(f"Train Loss: {running_loss/len(train_loader):.4f}")
#         print(f"Val Loss: {val_loss/len(val_loader):.4f}")
#         print(f"Val Accuracy: {100 * correct / total:.2f}%")

#         # Early stopping
#         if val_loss < best_loss:
#             best_loss = val_loss
#             best_model_weights = copy.deepcopy(model.state_dict())
#             patience = original_patience
#         else:
#             patience -= 1
#             if patience == 0:
#                 break

# print(f"SFT has chosen: {max_name}")
# for i in experiments:
#     print("Result for " + i + ":")
#     print("Loss:\t", end = "")
#     for loss in exp_dict_loss[i]:
#         print(f"{loss:.4f}\t", end = "")
#     print()
#     print("Acc:\t", end = "")
#     for acc in exp_dict_acc[i]:
#         print(f"{acc:.2f}\t", end = "")
#     print()
# # Save the fine-tuned model
# # torch.save(model.state_dict(), "result.pth")
