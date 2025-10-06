"""Training and evaluation engine for property-driven machine learning.

This module contains the core training and testing functions that handle
the property-driven learning loop with constraint evaluation and adversarial training.
"""

import numpy as np
import torch
import torch.nn.functional as F

import property_driven_ml.logics as logics
import property_driven_ml.constraints as constraints
import property_driven_ml.training as training
from property_driven_ml.utils import maybe
from property_driven_ml.training.epoch_info import EpochInfoTrain, EpochInfoTest
from property_driven_ml.training.mode import Mode


def train(  # TODO: add task loss function as an argument
    N: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer,
    oracle: training.Attack,
    grad_norm: training.GradNorm,
    logic: logics.Logic,
    constraint: constraints.Constraint,
    with_dl: bool,
    mode: Mode,
    denorm_scale: None | torch.Tensor = None,
) -> EpochInfoTrain:
    """Train the model for one epoch with property-driven learning.

    Args:
        N: Neural network model to train.
        device: Computing device (CPU or GPU).
        train_loader: Training data loader.
        optimizer: Model optimizer.
        oracle: Attack oracle for generating adversarial examples.
        grad_norm: Gradient normalization handler.
        logic: Logic system for constraint evaluation.
        constraint: Constraint to enforce during training.
        with_dl: Whether to use property-driven learning.
        mode: The training mode, i.e. multi-class classification, multi-label classification, or regression.
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

    for _, (data, target) in enumerate(train_loader, start=1):
        x, y_target = (
            data.to(device),
            target.to(device),
        )

        # forward pass
        y = N(x)

        if mode is Mode.MultiClassClassification:
            # loss + prediction accuracy calculation
            loss = F.cross_entropy(y, y_target)
            correct = torch.mean(torch.argmax(y, dim=1).eq(y_target).float())
            avg_pred_metric += correct
        elif mode is Mode.MultiLabelClassification:
            # loss + 0/1 accuracy
            loss = F.binary_cross_entropy_with_logits(y, y_target)
            # TODO: ugly!
            pred = (
                y > constraint.postcondition.delta
            ).float()  # no sigmoid to make verification easier
            correct = torch.mean(torch.all(pred == y_target, dim=1).float())
            avg_pred_metric += correct
        elif mode is Mode.Regression:
            # loss calculation for regression
            loss = F.mse_loss(y, y_target)
            rmse = torch.sqrt(loss)
            rmse = (denorm_scale * rmse.cpu()).squeeze()
            avg_pred_metric += rmse
        else:  # TODO: can this happen?
            assert False, f"mode {mode} not supported!"  # nosec

        adv = oracle.attack(N, x, y_target, constraint)

        # forward pass for constraint accuracy (constraint satisfaction on random samples)
        with torch.no_grad():
            loss_random, sat_random = constraint.eval(
                N, x, None, y_target, logic, reduction="mean"
            )

        # forward pass for constraint security (constraint satisfaction on adversarial samples)
        with maybe(torch.no_grad(), not with_dl):
            loss_adv, sat_adv = constraint.eval(
                N, x, adv, y_target, logic, reduction="mean"
            )

        optimizer.zero_grad(set_to_none=True)

        if not with_dl:
            loss.backward()
            optimizer.step()
        else:
            grad_norm.balance(loss, loss_adv)

        avg_pred_loss += loss
        avg_constr_acc += sat_random
        avg_constr_sec += sat_adv
        avg_constr_loss += loss_adv
        avg_random_loss += loss_random

        # save one original image, random sample, and adversarial sample image (for debugging, inspecting attacks)
        i = np.random.randint(0, x.size(0))
        # Generate random sample from constraint for visualization
        random_sample = constraint.uniform_sample(x[i : i + 1], 1).squeeze(0)
        images = dict()
        images["input"], images["random"], images["adv"] = x[i], random_sample, adv[i]

    if with_dl:
        grad_norm.renormalise()

    return EpochInfoTrain(
        pred_metric=avg_pred_metric.item() / len(train_loader),
        constr_acc=avg_constr_acc.item() / len(train_loader),
        constr_sec=avg_constr_sec.item() / len(train_loader),
        pred_loss=avg_pred_loss.item() / len(train_loader),
        random_loss=avg_random_loss.item() / len(train_loader),
        constr_loss=avg_constr_loss.item() / len(train_loader),
        pred_loss_weight=grad_norm.weights[0].item(),
        constr_loss_weight=grad_norm.weights[1].item(),
        input_img=images["input"],  # type: ignore
        adv_img=images["adv"],  # type: ignore
        random_img=images["random"],  # type: ignore
    )


def test(
    N: torch.nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    oracle: training.Attack,
    logic: logics.Logic,
    constraint: constraints.Constraint,
    mode: Mode,
    denorm_scale: None | torch.Tensor = None,
) -> EpochInfoTest:
    """Evaluate the model on test data.

    Args:
        N: Neural network model to evaluate.
        device: Computing device (CPU or GPU).
        test_loader: Test data loader.
        oracle: Attack oracle for generating adversarial examples.
        logic: Logic system for constraint evaluation.
        constraint: Constraint to evaluate.
        mode: The training mode, i.e. multi-class classification, multi-label classification, or regression.
        denorm_scale: Denormalization coefficient for output images and loss.

    Returns:
        Test epoch information including metrics and sample images.
    """
    correct, constr_acc, constr_sec = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )
    avg_pred_loss, avg_constr_loss, avg_random_loss = (
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    )

    total_samples = 0

    N.eval()

    for _, (data, target) in enumerate(test_loader, start=1):
        x, y_target = (
            data.to(device),
            target.to(device),
        )
        total_samples += x.size(0)

        with torch.no_grad():
            # forward pass
            y = N(x)

            if mode is Mode.MultiClassClassification:
                avg_pred_loss += F.cross_entropy(y, y_target, reduction="sum")
                pred = y.max(dim=1, keepdim=True)[1]
                correct += pred.eq(y_target.view_as(pred)).sum()
            elif mode is Mode.MultiLabelClassification:
                avg_pred_loss += F.binary_cross_entropy_with_logits(
                    y, y_target, reduction="sum"
                )
                # TODO: ugly!
                pred = (y > constraint.postcondition.delta).float()
                correct += torch.sum(torch.all(pred == y_target, dim=1).float())
            elif mode is Mode.Regression:
                avg_pred_loss += F.mse_loss(y, y_target, reduction="sum")
            else:  # TODO: can this happen?
                assert False, f"mode {mode} not supported!"  # nosec

        # get adversarial samples (requires grad)
        adv = oracle.attack(N, x, y_target, constraint)

        # forward passes for constraint accuracy (constraint satisfaction on random samples) + constraint security (constraint satisfaction on adversarial samples)
        with torch.no_grad():
            loss_random, sat_random = constraint.eval(
                N, x, None, y_target, logic, reduction="sum"
            )
            loss_adv, sat_adv = constraint.eval(
                N, x, adv, y_target, logic, reduction="sum"
            )

            constr_acc += sat_random
            constr_sec += sat_adv

            avg_random_loss += loss_random
            avg_constr_loss += loss_adv

        # save one original image, random sample, and adversarial sample image (for debugging, inspecting attacks)
        i = np.random.randint(0, x.size(0))
        # Generate random sample from constraint for visualization
        random_sample = constraint.uniform_sample(x[i : i + 1], 1).squeeze(0)
        images = dict()
        images["input"], images["random"], images["adv"] = x[i], random_sample, adv[i]

    if mode is Mode.MultiClassClassification or Mode.MultiLabelClassification:
        pred_acc = correct.item() / total_samples
    elif mode is Mode.Regression:
        rmse = torch.sqrt(avg_pred_loss / total_samples)
        rmse = (denorm_scale * rmse.cpu()).item()
    else:  # TODO: can this happen?
        assert False, f"mode {mode} not supported!"  # nosec

    return EpochInfoTest(
        pred_metric=pred_acc
        if mode is Mode.MultiClassClassification or Mode.MultiLabelClassification
        else rmse,  # type: ignore
        constr_acc=constr_acc.item() / total_samples,
        constr_sec=constr_sec.item() / total_samples,
        pred_loss=avg_pred_loss.item() / total_samples,
        random_loss=avg_random_loss.item() / total_samples,
        constr_loss=avg_constr_loss.item() / total_samples,
        input_img=images["input"],  # type: ignore
        adv_img=images["adv"],  # type: ignore
        random_img=images["random"],  # type: ignore
    )
