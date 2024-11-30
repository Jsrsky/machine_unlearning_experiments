import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split

SEED = 42
BATCH_SIZE = 32

def set_seed(seed=SEED):

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    np.random.seed(seed)

    random.seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

def init_dataloaders(datasets, info_file_path, val_ratio=0.2, batch_size=BATCH_SIZE):

    print('Prepare DataLoaders...')

    dataset, test_dataset = datasets
    # dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    classes = dataset.classes

    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size

    set_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    splits = {
        "train_indices": [{"index": idx, "class": dataset[idx][1]} for idx in train_dataset.indices],
        "val_indices": [{"index": idx, "class": dataset[idx][1]} for idx in val_dataset.indices],
        "test_indices": [{"index": idx, "class": dataset[idx][1]} for idx in range(len(test_dataset))]
    }
    
    with open(info_file_path, "w") as f:
        json.dump(splits, f)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Done preparing DataLoaders.')

    return train_loader, val_loader, test_loader