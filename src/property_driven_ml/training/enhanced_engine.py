"""Enhanced training engine for unified constraint architecture.

This module provides training and evaluation functions that work with
the new InputRegionConstraint architecture, eliminating the need for
BoundedDataset classes.
"""

import torch
import torch.nn.functional as F

import property_driven_ml.logics as logics
import property_driven_ml.training as training
from property_driven_ml.utils import maybe
from property_driven_ml.training.epoch_info import EpochInfoTrain, EpochInfoTest
from property_driven_ml.constraints.unified_constraints import InputRegionConstraint
from property_driven_ml.training.enhanced_attacks import EnhancedAttack


def train_enhanced(
    N: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer,
    oracle: EnhancedAttack,
    grad_norm: training.GradNorm,
    logic: logics.Logic,
    constraint: InputRegionConstraint,
    with_dl: bool,
    is_classification: bool,
    denorm_scale: None | torch.Tensor = None,
) -> EpochInfoTrain:
    """Train the model for one epoch with unified property-driven learning.

    Args:
        N: Neural network model to train.
        device: Computing device (CPU or GPU).
        train_loader: Training data loader (regular dataset, not BoundedDataset).
        optimizer: Model optimizer.
        oracle: Enhanced attack oracle that works with InputRegionConstraint.
        grad_norm: Gradient normalization handler.
        logic: Logic system for constraint evaluation.
        constraint: InputRegionConstraint providing both bounds and constraint logic.
        with_dl: Whether to use property-driven learning.
        is_classification: Whether this is a classification task.
        denorm_scale: Denormalization coefficient for output images and loss.

    Returns:
        Training epoch information including metrics and sample images.
    """
    avg_pred_metric, avg_pred_loss = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )
    avg_constr_acc, avg_constr_sec, avg_constr_loss, avg_random_loss = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )

    N.train()

    # Keep track of sample images from first batch
    sample_x, sample_random, sample_adv = None, None, None

    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        x, y_target = data.to(device), target.to(device)

        # Get bounds dynamically from the constraint
        lo, hi = constraint.get_input_bounds(x)

        # forward pass
        y = N(x)

        if is_classification:
            # loss + prediction accuracy calculation
            loss = F.cross_entropy(y, y_target)
            correct = torch.mean(torch.argmax(y, dim=1).eq(y_target).float())
            avg_pred_metric += correct
        else:
            # loss calculation for regression
            loss = F.mse_loss(y, y_target)
            rmse = torch.sqrt(loss)
            rmse = (denorm_scale * rmse.cpu()).squeeze()
            avg_pred_metric += rmse

        # get random + adversarial samples using enhanced methods
        with torch.no_grad():
            random = oracle.uniform_random_sample((lo, hi))

        # Use enhanced attack method that gets bounds from constraint
        adv = oracle.attack_enhanced(N, x, y_target, constraint)

        # Store sample images from first batch
        if batch_idx == 1:
            sample_x = x[:4].detach()
            sample_random = random[:4].detach()
            sample_adv = adv[:4].detach()

        # forward pass for constraint accuracy (constraint satisfaction on random samples)
        with torch.no_grad():
            loss_random, sat_random = constraint.eval(
                N, x, random, y_target, logic, reduction="mean"
            )

        # forward pass for constraint security (constraint satisfaction on adversarial samples)
        with maybe(torch.no_grad(), not with_dl):
            loss_adv, sat_adv = constraint.eval(
                N, x, adv, y_target, logic, reduction="mean"
            )

        avg_pred_loss += loss.detach()
        avg_constr_acc += sat_random.detach()
        avg_constr_sec += sat_adv.detach()
        avg_constr_loss += loss_adv.detach()
        avg_random_loss += loss_random.detach()

        # Apply gradient normalization if using deep learning
        if with_dl:
            grad_norm.balance(loss, loss_adv)

        # Total loss calculation (same as before)
        if with_dl:
            total_loss = grad_norm.weights[0] * loss + grad_norm.weights[1] * loss_adv
        else:
            total_loss = loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Apply gradient normalization renormalization
    if with_dl:
        grad_norm.renormalise()

    n_samples = len(train_loader)
    avg_pred_metric /= n_samples
    avg_pred_loss /= n_samples
    avg_constr_acc /= n_samples
    avg_constr_sec /= n_samples
    avg_constr_loss /= n_samples
    avg_random_loss /= n_samples

    return EpochInfoTrain(
        pred_metric=avg_pred_metric.cpu().numpy(),
        pred_loss=avg_pred_loss.cpu().numpy(),
        constr_acc=avg_constr_acc.cpu().numpy(),
        constr_sec=avg_constr_sec.cpu().numpy(),
        constr_loss=avg_constr_loss.cpu().numpy(),
        random_loss=avg_random_loss.cpu().numpy(),
        pred_loss_weight=grad_norm.weights[0].item() if with_dl else 1.0,
        constr_loss_weight=grad_norm.weights[1].item() if with_dl else 0.0,
        input_img=sample_x.cpu().numpy() if sample_x is not None else None,
        adv_img=sample_adv.cpu().numpy() if sample_adv is not None else None,
        random_img=sample_random.cpu().numpy() if sample_random is not None else None,
    )


def test_enhanced(
    N: torch.nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    oracle: EnhancedAttack,
    logic: logics.Logic,
    constraint: InputRegionConstraint,
    is_classification: bool,
    denorm_scale: None | torch.Tensor = None,
) -> EpochInfoTest:
    """Test the model with unified property-driven evaluation.

    Args:
        N: Neural network model to evaluate.
        device: Computing device (CPU or GPU).
        test_loader: Test data loader (regular dataset, not BoundedDataset).
        oracle: Enhanced attack oracle that works with InputRegionConstraint.
        logic: Logic system for constraint evaluation.
        constraint: InputRegionConstraint providing both bounds and constraint logic.
        is_classification: Whether this is a classification task.
        denorm_scale: Denormalization coefficient for output images and loss.

    Returns:
        Test epoch information including metrics and sample images.
    """
    avg_pred_metric, avg_pred_loss = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )
    avg_constr_acc, avg_constr_sec, avg_constr_loss, avg_random_loss = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )

    N.eval()

    # Keep track of sample images from first batch
    sample_x, sample_random, sample_adv = None, None, None

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader, start=1):
            x, y_target = data.to(device), target.to(device)

            # Get bounds dynamically from the constraint
            lo, hi = constraint.get_input_bounds(x)

            # forward pass
            y = N(x)

            if is_classification:
                # loss + prediction accuracy calculation
                loss = F.cross_entropy(y, y_target)
                correct = torch.mean(torch.argmax(y, dim=1).eq(y_target).float())
                avg_pred_metric += correct
            else:
                # loss calculation for regression
                loss = F.mse_loss(y, y_target)
                rmse = torch.sqrt(loss)
                rmse = (denorm_scale * rmse.cpu()).squeeze()
                avg_pred_metric += rmse

            # get random + adversarial samples using enhanced methods
            random = oracle.uniform_random_sample((lo, hi))

            # Use enhanced attack method that gets bounds from constraint
            adv = oracle.attack_enhanced(N, x, y_target, constraint)

            # Store sample images from first batch
            if batch_idx == 1:
                sample_x = x[:4].detach()
                sample_random = random[:4].detach()
                sample_adv = adv[:4].detach()

            # Constraint evaluation
            loss_random, sat_random = constraint.eval(
                N, x, random, y_target, logic, reduction="mean"
            )
            loss_adv, sat_adv = constraint.eval(
                N, x, adv, y_target, logic, reduction="mean"
            )

            avg_pred_loss += loss
            avg_constr_acc += sat_random
            avg_constr_sec += sat_adv
            avg_constr_loss += loss_adv
            avg_random_loss += loss_random

    n_samples = len(test_loader)
    avg_pred_metric /= n_samples
    avg_pred_loss /= n_samples
    avg_constr_acc /= n_samples
    avg_constr_sec /= n_samples
    avg_constr_loss /= n_samples
    avg_random_loss /= n_samples

    return EpochInfoTest(
        pred_metric=avg_pred_metric.cpu().numpy(),
        pred_loss=avg_pred_loss.cpu().numpy(),
        constr_acc=avg_constr_acc.cpu().numpy(),
        constr_sec=avg_constr_sec.cpu().numpy(),
        constr_loss=avg_constr_loss.cpu().numpy(),
        random_loss=avg_random_loss.cpu().numpy(),
        input_img=sample_x.cpu().numpy() if sample_x is not None else None,
        adv_img=sample_adv.cpu().numpy() if sample_adv is not None else None,
        random_img=sample_random.cpu().numpy() if sample_random is not None else None,
    )


def create_enhanced_constraint_from_specs(
    device: torch.device, input_region_spec: str, output_constraint_spec: str, **kwargs
) -> InputRegionConstraint:
    """Create an InputRegionConstraint from specification strings.

    This function provides a bridge between the old string-based configuration
    and the new unified constraint architecture.

    Args:
        device: PyTorch device for tensor computations.
        input_region_spec: String specification of input region (e.g., "EpsilonBall").
        output_constraint_spec: String specification of output constraint.
        **kwargs: Additional parameters for constraint creation.

    Returns:
        InputRegionConstraint combining the specified input region and output constraint.
    """
    from ..constraints.unified_constraints import (
        StandardRobustnessWithInputRegion,
        LipschitzRobustnessWithInputRegion,
    )

    # Parse input region and output constraint specifications
    if (
        input_region_spec == "EpsilonBall"
        and output_constraint_spec == "StandardRobustness"
    ):
        return StandardRobustnessWithInputRegion(
            device=device,
            delta=kwargs.get("delta", 0.1),
            eps=kwargs.get("eps", 0.3),
            transform=kwargs.get("transform", None),
        )
    elif (
        input_region_spec == "EpsilonBall"
        and output_constraint_spec == "LipschitzRobustness"
    ):
        return LipschitzRobustnessWithInputRegion(
            device=device,
            L=kwargs.get("L", 1.0),
            eps=kwargs.get("eps", 0.3),
            transform=kwargs.get("transform", None),
        )
    else:
        raise ValueError(
            f"Unsupported combination: {input_region_spec} + {output_constraint_spec}"
        )
