import numpy as np
from torch import randn

hooks = []
activations = {}


def register_hooks(model):
    global hooks, activations
    hooks = []
    activations = {name: None for name, _ in model.named_children()}
    for name, layer in model.named_children():
        hook = layer.register_forward_hook(get_activation(name))
        hooks.append(hook)


def clear_hooks():
    global hooks
    for hook in hooks:
        hook.remove()


def get_activation(name):
    global activations

    def hook(module, input, output):
        activations[name] = output.detach().cpu().numpy()

    return hook


def compute_activation_magnitudes():
    global activations
    magnitudes = []
    for name, activation in activations.items():
        if activation is not None:
            magnitude = np.abs(activation).mean()
            magnitudes.append((name, magnitude))
    return magnitudes


def noise(tensor):
    return tensor + randn(tensor.size()) / 10
