import torch
from torch.utils.data import DataLoader, Subset
import json
from utils.utils import DEVICE
def compute_gradient(model, criterion, inputs, targets):
    """
    Compute the gradient of the loss with respect to model parameters
    for a given batch of inputs and targets.
    """
    model.zero_grad()
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    grad_vec = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return grad_vec


def compute_hessian(model, criterion, data_loader):
    """
    Compute the Hessian of the loss with respect to model parameters
    over the given data loader.
    """
    model.zero_grad()
    num_params = sum(p.numel() for p in model.parameters())
    H = torch.zeros(num_params, num_params, device=DEVICE)

    for inputs, targets in data_loader:
        grad = compute_gradient(model, criterion, inputs, targets)
        for i in range(num_params):
            model.zero_grad()
            grad[i].backward(retain_graph=True)
            h_row = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            H[i] += h_row

    return H


def invert_hessian(H, damping=1e-6):
    """
    Invert the Hessian matrix with optional damping for numerical stability.
    """
    num_params = H.shape[0]
    I = torch.eye(num_params, device=H.device)
    H_damped = H + damping * I
    return torch.linalg.inv(H_damped)


def parameters_to_vector(model):
    """
    Flatten all model parameters into a single vector.
    """
    return torch.cat([p.view(-1) for p in model.parameters()])


def vector_to_parameters(vec, model):
    """
    Map a single parameter vector back into the model parameters.
    """
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.copy_(vec[pointer:pointer + num_param].view(param.size()))
        pointer += num_param


def influence_unlearn(model, criterion, samples_to_unlearn_loader, remaining_loader, damping=1e-5):
    """
    Perform influence-based unlearning by removing the effect of a specific set of data (D_u)
    from the model parameters.

    Parameters:
        - model: PyTorch model
        - criterion: Loss function
        - samples_to_unlearn_loader: DataLoader for D_u (data to forget)
        - remaining_loader: DataLoader for D' (remaining data)
        - damping: Damping term for Hessian inversion
    """

    model.eval()

    # Step 1: Compute the gradient of the unlearn data (Delta_u)
    grad_u = torch.zeros_like(parameters_to_vector(model))
    for inputs, targets in samples_to_unlearn_loader:
        grad_u += compute_gradient(model, criterion, inputs, targets)

    # Step 2: Compute the Hessian of the remaining data (H)
    H = compute_hessian(model, criterion, remaining_loader)

    # Step 3: Invert the Hessian (H^-1)
    H_inv = invert_hessian(H, damping=damping)

    # Step 4: Compute the parameter update (H^-1 * Delta_u)
    update = torch.matmul(H_inv, grad_u)

    # Step 5: Update model parameters (theta^u = theta - update)
    theta_vec = parameters_to_vector(model)
    new_theta_vec = theta_vec - update
    vector_to_parameters(new_theta_vec, model)

    return model

def create_unlearning_dataloader(unlearn_file, dataset, batch_size=32):
    
    with open(unlearn_file, "r") as f:
        unlearn_samples = json.load(f)

    unlearn_indices = [entry["index"] for entry in unlearn_samples]

    unlearn_dataset = Subset(dataset, unlearn_indices)

    unlearn_loader = DataLoader(unlearn_dataset, batch_size=batch_size, shuffle=False)
    
    return unlearn_loader
