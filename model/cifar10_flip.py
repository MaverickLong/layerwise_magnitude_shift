import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*NORM)])


class ShiftedLabelsCIFAR10:

    def __init__(
        self, root="../data", train=False, download=True, transform=te_transforms
    ):
        self.original_dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        new_label = (label + 1) % 10
        return image, new_label


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    dataloader = DataLoader(
        ShiftedLabelsCIFAR10(), batch_size=4, shuffle=True, num_workers=2
    )

    # Get a batch of training data
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Classes in CIFAR-10
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # Print images
    imshow(torchvision.utils.make_grid(images))
    # Print labels
    print(" ".join("%5s" % classes[labels[j]] for j in range(4)))
