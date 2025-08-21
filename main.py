import argparse
import time
import os
import csv
import sys

import numpy as np
import onnx

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from examples.alsomitra_dataset import AlsomitraDataset
from examples.models import AlsomitraNet

from property_driven_ml.utils import (
    CreateEpsilonBall,
    CreateAlsomitraInputRegion,
    CreateStandardRobustnessConstraint,
    CreateLipschitzRobustnessConstraint,
    CreateAlsomitraOutputConstraint,
)
from property_driven_ml.datasets import create_dataset
from property_driven_ml.utils.visualization import save_epoch_images

# Import from the property_driven_ml package
import property_driven_ml.logics as logics
import property_driven_ml.constraints as constraints
import property_driven_ml.training as training
from property_driven_ml.utils import safe_call
from property_driven_ml.training import EpochInfoTrain, train, test


def main():
    """Main training script for property-driven machine learning."""
    logics_list: list[logics.Logic] = [
        logics.DL2(),
        logics.GoedelFuzzyLogic(),
        logics.KleeneDienesFuzzyLogic(),
        logics.LukasiewiczFuzzyLogic(),
        logics.ReichenbachFuzzyLogic(),
        logics.GoguenFuzzyLogic(),
        logics.ReichenbachSigmoidalFuzzyLogic(),
        logics.YagerFuzzyLogic(),
        logics.STL(),
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument(
        "--epochs", type=int, required=True, help="number of epochs to train for"
    )
    parser.add_argument(
        "--data-set", type=str, required=True, choices=["mnist", "alsomitra", "gtsrb"]
    )
    parser.add_argument(
        "--input-region",
        type=str,
        required=True,
        help="the input region induced by the precondition P(x)",
    )
    parser.add_argument(
        "--output-constraint",
        type=str,
        required=True,
        help="the output constraint given by Q(f(x))",
    )
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument(
        "--oracle",
        type=str,
        default="apgd",
        choices=["pgd", "apgd"],
        help="standard PGD or AutoPGD",
    )
    parser.add_argument(
        "--oracle-steps", type=int, default=20, help="number of PGD iterations"
    )
    parser.add_argument(
        "--oracle-restarts", type=int, default=10, help="number of PGD random restarts"
    )
    parser.add_argument("--pgd-step-size", type=float, default=0.03)
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="number of epochs to wait before introducing constraint loss",
    )
    parser.add_argument(
        "--logic",
        type=str,
        default=None,
        choices=[logic.name for logic in logics_list],
        help="the differentiable logic to use for training with the constraint, or None",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="directory in which to save .onnx and .csv files",
    )
    parser.add_argument("--initial-dl-weight", type=float, default=1.0)
    parser.add_argument(
        "--grad-norm-alpha",
        type=float,
        default=0.12,
        help="restoring force for GradNorm",
    )
    parser.add_argument(
        "--grad-norm-lr",
        type=float,
        default=None,
        help="learning rate for GradNorm weights, equal to --lr if not specified",
    )
    parser.add_argument(
        "--save-onnx", action="store_true", help="save .onnx file after training"
    )
    parser.add_argument(
        "--save-imgs",
        action="store_true",
        help="save one input image, random image, and adversarial image per epoch",
    )
    args = parser.parse_args()

    kwargs = {"batch_size": args.batch_size}

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")

        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if (
            os.name != "nt"
        ):  # NOTE: on Windows, our EpsilonBall implementation cannot be pickled
            kwargs.update({"num_workers": 4, "pin_memory": True})
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.logic is None:
        logic = logics_list[0]  # need some logic loss for oracle even for baseline
        is_baseline = True
    else:
        logic = next(logic for logic in logics_list if logic.name == args.logic)
        is_baseline = False

    ### Set up dataset ###

    temp_train_loader, temp_test_loader, N, (mean, std) = create_dataset(
        args.data_set, args.batch_size
    )

    # Extract the underlying datasets from the DataLoaders for constraint creation
    dataset_train = temp_train_loader.dataset
    dataset_test = temp_test_loader.dataset

    # Move model to device
    N = N.to(device)

    ### Parse input constraint ###

    # Handle input constraint creation using centralized factories
    if args.input_region == "EpsilonBall":
        train_factory, test_factory = CreateEpsilonBall(args.epsilon)
        wrapper_train = train_factory(dataset_train, mean, std)
        wrapper_test = test_factory(dataset_test, mean, std)
    elif args.input_region == "AlsomitraInputRegion":
        # Use centralized factory for AlsomitraInputRegion
        train_factory, test_factory = CreateAlsomitraInputRegion()

        # Handle case where datasets might be Subset from random_split
        from torch.utils.data import Subset

        if isinstance(dataset_train, Subset):
            if isinstance(dataset_train.dataset, AlsomitraDataset):
                train_dataset_for_constraint = dataset_train.dataset
            else:
                raise ValueError(
                    f"Expected AlsomitraDataset in Subset, got {type(dataset_train.dataset)}"
                )
        elif isinstance(dataset_train, AlsomitraDataset):
            train_dataset_for_constraint = dataset_train
        else:
            raise ValueError(f"Expected AlsomitraDataset, got {type(dataset_train)}")

        if isinstance(dataset_test, Subset):
            if isinstance(dataset_test.dataset, AlsomitraDataset):
                test_dataset_for_constraint = dataset_test.dataset
            else:
                raise ValueError(
                    f"Expected AlsomitraDataset in Subset, got {type(dataset_test.dataset)}"
                )
        elif isinstance(dataset_test, AlsomitraDataset):
            test_dataset_for_constraint = dataset_test
        else:
            raise ValueError(f"Expected AlsomitraDataset, got {type(dataset_test)}")

        wrapper_train = train_factory(train_dataset_for_constraint, mean, std)
        wrapper_test = test_factory(test_dataset_for_constraint, mean, std)
    else:
        raise ValueError(f"Unsupported input region: {args.input_region}")

    train_loader = torch.utils.data.DataLoader(
        wrapper_train, shuffle=True, drop_last=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        wrapper_test, shuffle=False, drop_last=True, **kwargs
    )

    print(
        f"len(dataset_train)={len(wrapper_train)} len(dataset_test)={len(wrapper_test)}"
    )
    print(f"len(train_loader)={len(train_loader)} len(test_loader)={len(test_loader)}")

    ### Parse output constraint ###

    def CreateGroupConstraint(delta: float) -> constraints.GroupConstraint:
        """Create a group constraint specific to the GTSRB dataset.

        This function is kept local since it contains dataset-specific logic
        for defining traffic sign groups that is not suitable for the general factory.
        """
        if not args.data_set == "gtsrb":
            raise ValueError("groups are only defined for GTSRB")

        groups: list[list[int]] = [
            [*range(6), 7, 8],  # speed limit signs
            [9, 10, 15, 16],  # other prohibitory signs
            [12, 13, 14, 17],  # unique signs
            [11, *range(18, 32)],  # danger signs
            [*range(33, 41)],  # mandatory signs
            [6, 32, 41, 42],  # derestriction signs
        ]

        return constraints.GroupConstraint(device, groups, delta)

    output_allowed = {
        "StandardRobustness": CreateStandardRobustnessConstraint,
        "LipschitzRobustness": CreateLipschitzRobustnessConstraint,
        "AlsomitraOutputConstraint": CreateAlsomitraOutputConstraint,
        "Groups": CreateGroupConstraint,  # Keep local since it has dataset-specific logic
    }

    constraint: constraints.Constraint = safe_call(
        args.output_constraint, output_allowed
    )

    ### Set up PGD, ADAM, GradNorm ###

    # Get a sample from a temporary DataLoader to determine input shape
    temp_loader = DataLoader(dataset_train, batch_size=1, shuffle=False)
    x0, _ = next(iter(temp_loader))
    x0 = x0[0]  # Remove batch dimension

    if args.oracle == "pgd":
        oracle_train = training.PGD(
            x0,
            logic,
            device,
            args.oracle_steps,
            args.oracle_restarts,
            args.pgd_step_size,
            mean,
            std,
        )
        oracle_test = training.PGD(
            x0,
            logics_list[0],
            device,
            args.oracle_steps,
            args.oracle_restarts,
            args.pgd_step_size,
            mean,
            std,
        )
    else:
        oracle_train = training.APGD(
            x0, logic, device, args.oracle_steps, args.oracle_restarts, mean, std
        )
        oracle_test = training.APGD(
            x0,
            logics_list[0],
            device,
            args.oracle_steps,
            args.oracle_restarts,
            mean,
            std,
        )

    optimizer = optim.AdamW(N.parameters(), lr=args.lr, weight_decay=1e-4)

    grad_norm = training.GradNorm(
        N,
        device,
        optimizer,
        lr=args.grad_norm_lr if args.grad_norm_lr is not None else args.lr,
        alpha=args.grad_norm_alpha,
        initial_dl_weight=args.initial_dl_weight,
    )

    ### Set up folders for results and PGD images ###

    if args.experiment_name is None:
        if isinstance(constraint, constraints.StandardRobustnessConstraint):
            folder = "standard-robustness"
        elif isinstance(constraint, constraints.LipschitzRobustnessConstraint):
            folder = "lipschitz-robustness"
        elif isinstance(constraint, constraints.GroupConstraint):
            folder = "group-constraint"
        else:
            raise ValueError(f"unknown constraint {constraint}!")
    else:
        folder = args.experiment_name

    folder_name = f"{args.results_dir}/{folder}/{args.data_set}"
    file_name = f"{folder_name}/{logic.name if not is_baseline else 'Baseline'}"

    report_file_name = f"{file_name}.csv"
    model_file_name = f"{file_name}.onnx"

    os.makedirs(folder_name, exist_ok=True)

    if args.save_imgs:
        save_dir = f"../saved_imgs/{folder}/{args.data_set}/{logic.name if not is_baseline else 'Baseline'}"

    ### Start training ###

    print(f"using device {device}")
    print(
        f"#model parameters: {sum(p.numel() for p in N.parameters() if p.requires_grad)}"
    )

    with open(report_file_name, "w", buffering=1, newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        csvfile.write(f"#{sys.argv}\n")
        writer.writerow(
            [
                "Epoch",
                "Train-P-Loss",
                "Train-R-Loss",
                "Train-C-Loss",
                "Train-P-Loss-Weight",
                "Train-C-Loss-Weight",
                "Train-P-Metric",
                "Train-C-Acc",
                "Train-C-Sec",
                "Test-P-Loss",
                "Test-R-Loss",
                "Test-C-Loss",
                "Test-P-Metric",
                "Test-C-Acc",
                "Test-C-Sec",
                "Train-Time",
                "Test-Time",
            ]
        )

        for epoch in range(0, args.epochs + 1):
            start = time.time()

            if epoch > 0:
                with_dl = (epoch > args.delay) and (not is_baseline)
                train_info = train(
                    N,
                    device,
                    train_loader,
                    optimizer,
                    oracle_train,
                    grad_norm,
                    logic,
                    constraint,
                    with_dl,
                    is_classification=not isinstance(N, AlsomitraNet),
                )  # TODO: better check?
                train_time = time.time() - start

                if args.save_imgs:
                    save_epoch_images(train_info, epoch, save_dir, mean, std)

                print(
                    f"Epoch {epoch}/{args.epochs}\t {args.output_constraint if args.experiment_name is None else args.experiment_name} on {args.data_set}, {logic.name if not is_baseline else 'Baseline'} \t TRAIN \t P-Metric: {train_info.pred_metric:.6f} \t C-Acc: {train_info.constr_acc:.2f}\t C-Sec: {train_info.constr_sec:.2f}\t P-Loss: {train_info.pred_loss:.2f}\t R-Loss: {train_info.random_loss:.2f}\t DL-Loss: {train_info.constr_loss:.2f}\t Time (Train) [s]: {train_time:.1f}"
                )
            else:
                train_info = EpochInfoTrain(
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, None, None, None
                )
                train_time = 0.0

            test_info = test(
                N,
                device,
                test_loader,
                oracle_test,
                logic,
                constraint,
                is_classification=not isinstance(N, AlsomitraNet),
            )
            test_time = time.time() - start - train_time

            if args.save_imgs:
                save_epoch_images(test_info, epoch, save_dir, mean, std)

            writer.writerow(
                [
                    epoch,
                    train_info.pred_loss,
                    train_info.random_loss,
                    train_info.constr_loss,
                    train_info.pred_loss_weight,
                    train_info.constr_loss_weight,
                    train_info.pred_metric,
                    train_info.constr_acc,
                    train_info.constr_sec,
                    test_info.pred_loss,
                    test_info.random_loss,
                    test_info.constr_loss,
                    test_info.pred_metric,
                    test_info.constr_acc,
                    test_info.constr_sec,
                    train_time,
                    test_time,
                ]
            )

            print(
                f"Epoch {epoch}/{args.epochs}\t {args.output_constraint if args.experiment_name is None else args.experiment_name} on {args.data_set}, {logic.name if not is_baseline else 'Baseline'} \t TEST \t P-Metric: {test_info.pred_metric:.6f}\t C-Acc: {test_info.constr_acc:.2f}\t C-Sec: {test_info.constr_sec:.2f}\t P-Loss: {test_info.pred_loss:.2f}\t R-Loss: {test_info.random_loss:.2f}\t DL-Loss: {test_info.constr_loss:.2f}\t Time (Test) [s]: {test_time:.1f}"
            )
            print("===")

    if args.save_onnx:
        x, _, _, _ = next(iter(train_loader))
        dummy_input = torch.randn(args.batch_size, *x.shape[1:], requires_grad=True).to(
            device=device
        )

        torch.onnx.export(
            N.eval(),
            (dummy_input,),  # Wrap in tuple as expected
            model_file_name,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        onnx_model = onnx.load(model_file_name)
        onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    main()
