import json

from tqdm.notebook import tqdm
from loguru import logger
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from utils.utils import DEVICE

def create_unlearning_dataloader(unlearn_file, dataset, batch_size=32):
    
    with open(unlearn_file, "r") as f:
        unlearn_samples = json.load(f)

    unlearn_indices = [entry["index"] for entry in unlearn_samples]

    unlearn_dataset = Subset(dataset, unlearn_indices)

    unlearn_loader = DataLoader(unlearn_dataset, batch_size=batch_size, shuffle=False)
    
    return unlearn_loader

def compute_full_hessian(model: nn.Module, loss_fn, remain_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Computes the full Hessian of the average loss over the entire remaining dataset.
    Assumes that all remaining data can be concatenated into one batch.
    """
    # Concatenate all remaining samples into one batch.
    all_inputs, all_targets = [], []
    for batch in remain_loader:
        inputs, targets = batch
        all_inputs.append(inputs)
        all_targets.append(targets)
    all_inputs = torch.cat(all_inputs, dim=0).to(device)
    all_targets = torch.cat(all_targets, dim=0).to(device)

    # Define a function f(theta) that returns the loss computed on all remaining data.
    def f(theta_flat):
        vector_to_parameters(theta_flat, model.parameters())
        outputs = model(all_inputs)
        loss = loss_fn(outputs, all_targets)
        return loss

    theta_flat = parameters_to_vector(model.parameters())
    # Compute the full Hessian using PyTorch's autograd.functional.hessian.
    # The result is a tensor of shape (n_params, n_params)
    H = torch.autograd.functional.hessian(f, theta_flat, vectorize=True)
    return H

def influence_unlearn_direct(model: nn.Module,
                             loss_fn,
                             unlearn_loader: DataLoader,
                             remain_loader: DataLoader,
                             device: torch.device =DEVICE,
                             damping: float = 1e-5) -> nn.Module:
    """
    Influence unlearning using full Hessian inversion.
    
    For each mini-batch in the unlearn_loader, the method:
      1. Computes the gradient on the deletion batch (Δ).
      2. Computes the full Hessian H on the remaining data.
      3. Inverts H (with damping for numerical stability) and computes δ = H⁻¹ Δ.
      4. Updates model parameters: θ ← θ + δ.
    """
    model.train()
    for batch_idx, (inputs_u, targets_u) in enumerate(unlearn_loader, 1):
        inputs_u, targets_u = inputs_u.to(device), targets_u.to(device)
        
        # (Step 5b) Compute gradient Δ on the deletion mini-batch.
        model.zero_grad()
        outputs_u = model(inputs_u)
        loss_u = loss_fn(outputs_u, targets_u)
        grad = torch.autograd.grad(loss_u, model.parameters(), create_graph=False)
        flat_grad = parameters_to_vector(grad).detach()

        # (Step 5c) Compute the full Hessian H on the remaining data.
        H = compute_full_hessian(model, loss_fn, remain_loader, device)

        # (Step 5d) Add damping and compute the update: δ = H⁻¹ Δ.
        H_damped = H + damping * torch.eye(H.size(0), device=device)
        delta = torch.linalg.solve(H_damped, flat_grad)
        
        # Update model parameters.
        flat_params = parameters_to_vector(model.parameters())
        new_flat_params = flat_params + delta
        vector_to_parameters(new_flat_params, model.parameters())
        
        print(f"Direct method: Processed unlearning batch {batch_idx}/{len(unlearn_loader)}.")
    
    return model

def compute_hvp_vectorized(model: nn.Module, loss_fn, data_loader: DataLoader, vector: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Computes the average Hessian–vector product (H*v) over the data in data_loader.
    This function works with a flattened parameter representation.
    """
    hvp_total = torch.zeros_like(vector)
    total_samples = 0
    num_batches = len(data_loader)

    # logger.info("Starting HVP computation over {} batches.", num_batches)

    for batch_idx, batch in enumerate(tqdm(data_loader, desc="HVP Batches"), 1):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        total_samples += batch_size

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Compute first-order gradients (flattened)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_grads = parameters_to_vector(grads)
        # Compute the dot product with the input vector
        grad_dot_vector = torch.dot(flat_grads, vector)
        # Compute the second derivative (Hessian–vector product)
        hvp = torch.autograd.grad(grad_dot_vector, model.parameters(), retain_graph=False)
        flat_hvp = torch.cat([h.reshape(-1) for h in hvp]).detach()
        hvp_total += flat_hvp * batch_size

    # logger.info("Completed HVP computation over {} samples.", total_samples)
    return hvp_total / total_samples


def influence_unlearn(
    model: nn.Module,
    loss_fn,
    unlearn_loader: DataLoader,
    remain_loader: DataLoader,
    device: torch.device = DEVICE,
    cg_iterations: int = 50,
    tol: float = 1e-10,
) -> nn.Module:
    """
    Implements the Influence unlearning update using an iterative method that leverages Hessian–vector products (HVPs)
    and SciPy's conjugate gradient solver to approximate H⁻¹ Δ.

    For each mini-batch in unlearn_loader:
      1. Compute Δ = ∇L(θ, D_u_i).
      2. Wrap the HVP function (computed on remain_loader) into a SciPy LinearOperator.
      3. Use scipy.sparse.linalg.cg to solve H δ = Δ.
      4. Update model parameters: θ ← θ + δ.
    """
    model.train()
    num_unlearn_batches = len(unlearn_loader)
    logger.info("Starting influence unlearning over {} unlearn batches.", num_unlearn_batches)
    for batch_idx, (inputs_u, targets_u) in enumerate(tqdm(unlearn_loader, desc="Unlearning Batches"), 1):
        inputs_u, targets_u = inputs_u.to(device), targets_u.to(device)

        # (Step 5b) Compute gradient Δ on the unlearn mini-batch.
        model.zero_grad()
        outputs_u = model(inputs_u)
        loss_u = loss_fn(outputs_u, targets_u)
        grad = torch.autograd.grad(loss_u, model.parameters(), create_graph=True)
        flat_grad = parameters_to_vector(grad).detach()

        # Determine the number of parameters
        n_params = flat_grad.numel()
        logger.info("Processing unlearn batch {}/{}. n_params: {}", batch_idx, num_unlearn_batches, n_params)

        # (Step 5c) Wrap the HVP function using SciPy's LinearOperator.
        # The CG solver expects functions that operate on NumPy arrays.
        def hvp_scipy(v_np):
            # Convert the input vector to a PyTorch tensor on the appropriate device.
            v_torch = torch.from_numpy(v_np).to(device).float()
            hvp = compute_hvp_vectorized(model, loss_fn, remain_loader, v_torch, device)
            # Return as a NumPy array (on CPU)
            return hvp.cpu().numpy()

        linear_operator = spla.LinearOperator((n_params, n_params), matvec=hvp_scipy)

        # (Step 5d) Use SciPy's CG solver to solve H δ = Δ.
        # Convert the right-hand side to a NumPy array.
        b_np = flat_grad.cpu().numpy()
        delta_np, info = spla.cg(linear_operator, b_np, tol=tol, maxiter=cg_iterations)
        if info != 0:
            logger.warning("CG solver did not converge for batch {}/{} (info={}).", batch_idx, num_unlearn_batches, info)

        # Convert the solution back to a PyTorch tensor.
        delta = torch.from_numpy(delta_np).to(device)

        # Update model parameters: θ ← θ + δ.
        flat_params = parameters_to_vector(model.parameters())
        new_flat_params = flat_params + delta
        vector_to_parameters(new_flat_params, model.parameters())

        logger.info("Processed unlearn batch {}/{}.", batch_idx, num_unlearn_batches)
    logger.info("Completed influence unlearning.")
    return model
