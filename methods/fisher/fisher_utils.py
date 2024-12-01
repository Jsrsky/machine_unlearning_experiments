import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from utils.utils import DEVICE

def compute_fisher_matrix(model, dataloader, loss_fn):

    fisher_matrix = {name: torch.zeros_like(param, device=DEVICE) for name, param in model.named_parameters() if param.requires_grad}

    model.eval()

    for inputs, targets in tqdm(dataloader, desc=f"Calculating FIM..."):

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        model.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_matrix[name] += param.grad.data ** 2

    # Normalize FIM by dataset size
    for name in fisher_matrix:
        fisher_matrix[name] /= len(dataloader.dataset)

    return fisher_matrix

def fisher_unlearning(model, unlearn_loader, fisher_matrix, loss_fn, sigma=1.0):
    """Performs Fisher Information Matrix (FIM) unlearning."""
    model.train()

    for inputs, targets in tqdm(unlearn_loader, desc="Unlearning Process..."):

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        model.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, targets)
        
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Perform Newton step
                param.data -= fisher_matrix[name] ** -1 * param.grad.data
                # Inject Gaussian noise
                noise = sigma * torch.randn_like(param.data) / (fisher_matrix[name] ** 0.5 + 1e-8)
                param.data += noise
    return model


def create_unlearning_dataloader(unlearn_file, dataset, batch_size=32):
    
    with open(unlearn_file, "r") as f:
        unlearn_samples = json.load(f)

    unlearn_indices = [entry["index"] for entry in unlearn_samples]

    unlearn_dataset = Subset(dataset, unlearn_indices)

    unlearn_loader = DataLoader(unlearn_dataset, batch_size=batch_size, shuffle=False)
    
    return unlearn_loader