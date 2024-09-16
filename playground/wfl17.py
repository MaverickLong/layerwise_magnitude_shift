from activation_method import (
    register_hooks,
    clear_hooks,
    compute_activation_magnitudes,
)
from rich.progress import track
from torch import nn
import torch
from torch.utils.data import DataLoader, Subset
import random
import torchvision.transforms as transforms

import breeds_utils

import copy
import timm

# Device Setup
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# config options
num_classes = 17
hooks = []
batch_size = 64

# Init & load model
model = timm.create_model('resnet26.bt_in1k', pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

data_config = timm.data.resolve_model_data_config(model)
transforms_train = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
transforms_val = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
    
# living17 generation

from robustness.tools.breeds_helpers import make_living17, ClassHierarchy
from robustness import datasets

info_dir = "./data/breeds/"
num_workers = 8
# Directory of local ImageNet
data_dir = "/home/yushan/imagenet/"

hier = ClassHierarchy(info_dir)
ret = make_living17(info_dir, split="rand")
superclasses, subclass_split, label_map = ret

train_subclasses, test_subclasses = subclass_split

# SOURCE DISTRIBUTION GENERATION

dataset_source = datasets.CustomImageNet(data_dir, train_subclasses, transform_train=transforms_train, transform_test=transforms_val)
loaders_source = dataset_source.make_loaders(num_workers, batch_size)
train_loader_source, val_loader_source = loaders_source

# TARGET DISTRIBUTION GENERATION
dataset_target = datasets.CustomImageNet(data_dir, test_subclasses, transform_train=transforms_train, transform_test=transforms_val)
                    
train_subset, val_subset = breeds_utils.make_subsets(
                                    batch_size=batch_size,
                                    transforms=(transforms_train, transforms_val),
                                    data_path=dataset_target.data_path,
                                    dataset=dataset_target.ds_name,
                                    label_mapping=dataset_target.label_mapping,
                                    custom_class=dataset_target.custom_class,
                                    seed=None,
                                    custom_class_args=dataset_target.custom_class_args)

# Organize images by category
category_dict = {i: [] for i in range(num_classes)}  # 17 categories of entity-17

total = len(train_subset)

# Note: This random algorithm is unstable but super fast in this case
# since number of image required is way fewer than number of image in total
while True:
    idx = random.randint(0, total - 1)
    _, label = train_subset[idx]
    if (len(category_dict[label]) < 50 and idx not in category_dict[label]):
        category_dict[label].append(idx)
    if all(len(v) == 50 for v in category_dict.values()):
        break

# for idx in range(len(train_subset)):
#     _, label = train_subset[idx]
#     category_dict[label].append(idx)

# Randomly sample 50 images from each category
selected_indices = []
for category, indices in category_dict.items():
    selected_indices.extend(indices)
    
train_subset = Subset(train_subset, selected_indices)

# TARGET LOADERS

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Full-Tuning on train subset first

criterion = nn.CrossEntropyLoss()
optimizer_fc = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer_src = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for param in model.parameters():
    param.requires_grad = True
    
num_epochs_source = 30

# Initialize Variables for EarlyStopping
# best_loss = float("inf")
# best_model_weights = None
# original_patience = 5
# patience = original_patience

# model = model.to(device)

# for epoch in range(num_epochs_source):
#     model.train()
#     running_loss = 0.0

#     for inputs, labels in track(train_loader_source):
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         optimizer_src.zero_grad()
#         outputs = model(inputs)
        
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer_src.step()

#         running_loss += loss.item()

#     # Validation
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, labels in track(val_loader_source):
#             inputs = inputs.to(device)
#             labels = labels.to(device)    

#             outputs = model(inputs)
            
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()

#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     # Early stoppin
#     if val_loss < best_loss:
#         best_loss = val_loss
#         best_model_weights = copy.deepcopy(model.state_dict())
#         patience = original_patience
#         print("Early-Stop Patience:" + str(patience))
#     else:
#         patience -= 1
#         print("Early-Stop Patience:" + str(patience))
#         if patience == 0:
#             break
#     print(f"Epoch [{epoch+1}/{num_epochs_source}]")
#     print(f"Train Loss: {running_loss/len(train_loader_source):.4f}")
#     print(f"Val Loss: {val_loss/len(val_loader_source):.4f}")
#     print(f"Val Accuracy: {100 * correct / total:.2f}%")

# torch.save(best_model_weights, "model/living17-subpop-via-imagenet.pth")
model.load_state_dict(torch.load("model/living17-subpop-via-imagenet.pth", map_location=device))

model.to(device)
# ...Following by our Fine-Tuning

# num_features = model.fc.in_features

# Getting our shiny new layer
# model.fc = nn.Linear(num_features, num_classes)

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
# for param in model.parameters():
#     param.requires_grad = True


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


print("Tuning FC is not effective. switch to activation mode.")

model.load_state_dict(torch.load("model/living17-subpop-via-imagenet.pth", map_location=device))

model.eval()
# Output level shift
register_hooks(model)

with torch.no_grad():
    for data in track(val_loader, description="Eval Activations"):
        inputs, _ = data
        inputs = inputs.to(device)
        _ = model(inputs)

clear_hooks()

layer_magnitudes = compute_activation_magnitudes()

print("\nActivation Magnitudes")
print("-------------------------------------------")

max_name = []
layer_count = 0
total_magnitude = 0

for name, magnitude in layer_magnitudes:
    print(name + ": " + str(magnitude))
    if name.startswith("layer"):
        total_magnitude += magnitude
        layer_count += 1

avg_magnitude = total_magnitude / layer_count

for name, magnitude in layer_magnitudes:
    if name.startswith("layer") and magnitude > avg_magnitude:
        max_name.append(name)
        
for name, param in model.named_parameters():
    param.requires_grad = False
    for n in max_name:
        if name.startswith(n):
            print("unfreezed " + name)
            param.requires_grad = True
            break

num_epochs = 10

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

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        patience = original_patience
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
