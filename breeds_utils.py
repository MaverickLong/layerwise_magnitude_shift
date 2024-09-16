import os

from robustness.tools import folder

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

import numpy as np
from torch.utils.data import Subset

def make_subsets(batch_size, transforms, data_path, data_aug=True,
                custom_class=None, dataset="", label_mapping=None, subset=None,
                subset_type='rand', subset_start=0, val_batch_size=None,
                only_val=False, seed=1, custom_class_args=None):
    '''
    This is an alternative version to make_loaders, which returns only the subsets.
    This enables more editing freedom for the dataset, i.e. subset slicing.
    '''
    print(f"==> Preparing dataset {dataset}..")
    transform_train, transform_test = transforms
    if not data_aug:
        transform_train = transform_test

    if not val_batch_size:
        val_batch_size = batch_size

    if not custom_class:
        train_path = os.path.join(data_path, 'train')
        test_path = os.path.join(data_path, 'val')
        if not os.path.exists(test_path):
            test_path = os.path.join(data_path, 'test')

        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        if not only_val:
            train_set = folder.ImageFolder(root=train_path, transform=transform_train,
                                           label_mapping=label_mapping)
        test_set = folder.ImageFolder(root=test_path, transform=transform_test,
                                      label_mapping=label_mapping)
    else:
        if custom_class_args is None: custom_class_args = {}
        if not only_val:
            train_set = custom_class(root=data_path, train=True, download=True, 
                                transform=transform_train, **custom_class_args)
        test_set = custom_class(root=data_path, train=False, download=True, 
                                transform=transform_test, **custom_class_args)

    if not only_val:
        attrs = ["samples", "train_data", "data"]
        vals = {attr: hasattr(train_set, attr) for attr in attrs}
        assert any(vals.values()), f"dataset must expose one of {attrs}"
        train_sample_count = len(getattr(train_set,[k for k in vals if vals[k]][0]))

    if (not only_val) and (subset is not None) and (subset <= train_sample_count):
        assert not only_val
        if subset_type == 'rand':
            rng = np.random.RandomState(seed)
            subset = rng.choice(list(range(train_sample_count)), size=subset+subset_start, replace=False)
            subset = subset[subset_start:]
        elif subset_type == 'first':
            subset = np.arange(subset_start, subset_start + subset)
        else:
            subset = np.arange(train_sample_count - subset, train_sample_count)

        train_set = Subset(train_set, subset)

    if only_val:
        return None, test_set

    return train_set, test_set