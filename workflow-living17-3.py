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
batch_size = 32

experiments = ["full", "layer1", "layer2", "layer3", "layer4", "fc"]
exp_dict_loss = {i: [] for i in experiments}
exp_dict_acc = {i: [] for i in experiments}

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
category_dict = {i: [] for i in range(num_classes)}  # 30 categories of living17

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
optimizer_layers = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer = optimizer_layers

for param in model.parameters():
    param.requires_grad = True
    
num_epochs_source = 30

# Initialize Variables for EarlyStopping
model.load_state_dict(torch.load("model/living17-subpop-via-imagenet.pth", map_location=device, weights_only=True))

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
        diff = magnitude_tgt - magnitude_src
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
    model.load_state_dict(torch.load("model/living17-subpop-via-imagenet.pth", map_location=device, weights_only=True))
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
    best_acc = 0
    # best_model_weights = None
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
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
    
        exp_dict_loss[i].append(val_loss)
        exp_dict_acc[i].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {running_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.2f}%")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            # best_model_weights = copy.deepcopy(model.state_dict())
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
