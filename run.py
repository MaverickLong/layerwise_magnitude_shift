import numpy as np
from rich.progress import track
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from cifar10_flip import ShiftedLabelsCIFAR10

from models import model

hooks = []
batch_size = 128

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*NORM)])

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=te_transforms
)
cifar10_shifted_testset = ShiftedLabelsCIFAR10()

testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
cifar10_shifted_testloader = DataLoader(
    cifar10_shifted_testset, batch_size=batch_size, shuffle=False
)

model = model().to(device)
print(device)
if device == "cpu":
    model.load_state_dict(
        torch.load("best_weights.pth", map_location=torch.device("cpu"))
    )
elif device == "mps":
    model.load_state_dict(
        torch.load("best_weights.pth", map_location=torch.device("mps"))
    )
else:
    model.load_state_dict(torch.load("best_weights.pth"))


def register_hooks():
    global hooks
    for name, layer in model.named_children():
        hook = layer.register_forward_hook(get_activation(name))
        hooks.append(hook)


def clear_hooks():
    global hooks
    hooks = []


def get_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu().numpy()

    return hook


def noise(tensor):
    return tensor + torch.randn(tensor.size())


activations = {name: None for name, _ in model.named_children()}

register_hooks()
model.eval()
activations_original = {}
with torch.no_grad():
    for data in track(testloader, description="Eval..."):
        inputs, _ = data
        inputs = inputs.to(device)
        _ = model(inputs)
for name in activations.keys():
    activations_original[name] = activations[name]
for hook in hooks:
    hook.remove()

clear_hooks()

register_hooks()
activations_noise = {}

with torch.no_grad():
    for data in track(testloader, description="Eval IlS"):
        inputs, _ = data
        inputs = noise(inputs)
        inputs = inputs.to(device)
        _ = model(inputs)
for name in activations.keys():
    activations_noise[name] = activations[name]
for hook in hooks:
    hook.remove()

clear_hooks()

# Output level shift
register_hooks()
activations_shifted = {}

with torch.no_grad():
    for data in track(cifar10_shifted_testloader, description="Eval OlS"):
        inputs, _ = data
        inputs = inputs.to(device)
        _ = model(inputs)
for name in activations.keys():
    activations_shifted[name] = activations[name]
for hook in hooks:
    hook.remove()

clear_hooks()


def compute_activation_magnitudes(activations):
    magnitudes = []
    for name, activation in activations.items():
        if activation is not None:
            magnitude = np.abs(activation).mean()
            magnitudes.append((name, magnitude))
    return magnitudes


layer_magnitudes = compute_activation_magnitudes(activations_original)
layer_magnitudes_IlS = compute_activation_magnitudes(activations_noise)
layer_magnitudes_OlS = compute_activation_magnitudes(activations_shifted)

print("Activation Magnitudes:")
print("-------------------------------------------")
for name, magnitude in layer_magnitudes:
    print(f"{name}: {magnitude:.4f}")

print("\nActivation Magnitudes w Input-level Shift")
print("-------------------------------------------")
for name, magnitude in layer_magnitudes_IlS:
    print(f"{name}: {magnitude:.4f}")

print("\nActivation Magnitudes w Output-level Shift")
print("-------------------------------------------")
for name, magnitude in layer_magnitudes_OlS:
    print(f"{name}: {magnitude:.4f}")
