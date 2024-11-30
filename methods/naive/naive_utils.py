import json
import random
from torch.utils.data import DataLoader, Subset

from utils.utils import set_seed


def select_samples_to_unlearn(data_splits_file, unlearn_samples_output_file, unlearn_ratio=0.1):
    
    # Load data splits
    with open(data_splits_file, "r") as f:
        splits = json.load(f)

    # Combine train and validation indices
    combined_indices = splits["train_indices"] + splits["val_indices"]

    set_seed()
    unlearn_count = int(unlearn_ratio * len(combined_indices))
    unlearn_indices = random.sample(combined_indices, unlearn_count)

    # Save unlearn indices
    with open(unlearn_samples_output_file, "w") as f:
        json.dump(unlearn_indices, f)

    print(f"Unlearn indices saved to {unlearn_samples_output_file}")



def update_splits_after_unlearning(data_splits_file, unlearn_samples_file, output_file):
    
    # Load data splits and unlearn indices
    with open(data_splits_file, "r") as f:
        splits = json.load(f)
    with open(unlearn_samples_file, "r") as f:
        unlearn_indices = json.load(f)

    # Extract unlearn indices
    unlearn_indices_set = {entry["index"] for entry in unlearn_indices}

    # Update splits
    updated_splits = {
        "train_indices": [
            entry for entry in splits["train_indices"] if entry["index"] not in unlearn_indices_set
        ],
        "val_indices": [
            entry for entry in splits["val_indices"] if entry["index"] not in unlearn_indices_set
        ],
        "test_indices": splits["test_indices"]  # Test set remains unchanged
    }

    # Save updated splits
    with open(output_file, "w") as f:
        json.dump(updated_splits, f)

    print(f"Updated splits saved to {output_file}")



def recreate_dataloaders(data_splits_file, dataset, batch_size=32):

    print('Recreating DataLoaders...')
    
    # Load updated splits
    with open(data_splits_file, "r") as f:
        splits = json.load(f)

    classes = dataset.classes

    # Extract indices
    train_indices = [entry["index"] for entry in splits["train_indices"]]
    val_indices = [entry["index"] for entry in splits["val_indices"]]
    test_indices = [entry["index"] for entry in splits["test_indices"]]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Done recreating DataLoaders.')
    
    return train_loader, val_loader, test_loader, classes