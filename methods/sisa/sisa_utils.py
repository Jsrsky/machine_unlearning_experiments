import json
from torch.utils.data import DataLoader, Subset, random_split

from utils.utils import set_seed

def create_sisa_structure(dataset, shards=3, slices_per_shard=5):
    """
    Create the SISA structure for the MNIST dataset using random_split.
    - 3 shards, each with 5 slices.
    - Save indices and classes for each sample.

    Args:
        dataset: PyTorch dataset object (MNIST training set).
        shards: Number of shards (default: 3).
        slices_per_shard: Number of slices per shard (default: 5).

    Returns:
        sisa_structure: Dictionary representing the SISA structure.
    """
    # Total size of the dataset

    shard_size = len(dataset) // shards

    shard_sizes = [shard_size] * shards

    shard_sizes[-1] += len(dataset) % shards
    

    # Split dataset into shards
    set_seed()
    shards = random_split(dataset, shard_sizes)
    sisa_structure = {}

    for shard_id, shard in enumerate(shards):

        shard_indices = shard.indices

        # Get the size of each slice within the shard
        slice_size = len(shard) // slices_per_shard

        slice_sizes = [slice_size] * slices_per_shard

        slice_sizes[-1] += len(shard) % slices_per_shard

        # Split the shard into slices
        set_seed()
        slices = random_split(shard, slice_sizes)
        sisa_structure[f"shard_{shard_id}"] = {}

        # Save indices and classes for each slice
        for slice_id, slice_data in enumerate(slices):

            slice_indices = [shard_indices[idx] for idx in slice_data.indices]

            slice_classes = [dataset[idx][1] for idx in slice_indices]

            sisa_structure[f"shard_{shard_id}"][f"slice_{slice_id}"] = {
                "indices": slice_indices,
                "classes": slice_classes,
            }

    filename = 'sisa_structure.json'
    with open(filename, "w") as f:
        json.dump(sisa_structure, f, indent=4)
    print(f"SISA structure saved to {filename}")


def recreate_sisa_dataloaders(datasets, json_file, batch_size=32, val_ratio=0.1):
    """
    Recreates the SISA structure from a JSON file and prepares DataLoaders for each slice.
    Splits each slice into 90% training and 10% validation data.

    Args:
        json_file (str): Path to the JSON file containing the SISA structure.
        dataset (Dataset): The original dataset to recreate slices from.
        batch_size (int): Batch size for the DataLoaders.
        val_ratio (float): Proportion of each slice to use as the validation set.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary with the structure {shard -> slice -> {"train": DataLoader, "val": DataLoader}}.
    """
    # Load the SISA structure

    dataset, test_dataset = datasets

    with open(json_file, 'r') as f:
        sisa_structure = json.load(f)

    dataloaders = {}

    # Iterate over shards and slices
    for shard_id, shard_data in sisa_structure.items():
        dataloaders[shard_id] = {}
        
        for slice_id, slice_data in shard_data.items():
            # Extract indices for this slice
            slice_indices = slice_data["indices"]

            # Create a subset from the dataset
            slice_subset = Subset(dataset, slice_indices)

            # Split into training and validation sets
            val_size = int(len(slice_subset) * val_ratio)
            train_size = len(slice_subset) - val_size
            train_subset, val_subset = random_split(slice_subset, [train_size, val_size])

            # Create DataLoaders
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # Store DataLoaders in the structure
            dataloaders[shard_id][slice_id] = {
                "train": train_loader,
                "val": val_loader
            }

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataloaders["test"] = test_loader

    return dataloaders