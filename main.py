import argparse
import time
import os
import csv
import sys

import numpy as np
import onnx

import torch
import torch.optim as optim

from examples.datasets.alsomitra import AlsomitraDataset
from examples.models import AlsomitraNet

from property_driven_ml.constraints import (
    StandardRobustnessConstraint,
    OppositeFacesConstraint,
)
from examples.datasets import create_dataset
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
        "--dataset",
        type=str,
        required=True,
        choices=["mnist", "alsomitra", "dice", "gtsrb"],
    )
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument(
        "--constraint",
        type=str,
        default="StandardRobustness",
        choices=["StandardRobustness", "OppositeFaces"],  # Will add more later
        help="which constraint to use",
    )
    parser.add_argument(
        "--oracle",
        type=str,
        default="apgd",
        choices=["pgd", "apgd"],
        help="attack oracle: standard PGD or AutoPGD",
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

    train_loader, test_loader, N, (mean, std), mode = create_dataset(
        args.dataset, args.batch_size
    )

    # Extract the underlying datasets from the DataLoaders for constraint creation
    N = N.to(device)

    # Handle input constraint creation using centralized factories

    # Define allowed constraint classes
    output_allowed = {
        "StandardRobustness": StandardRobustnessConstraint,
        "OppositeFaces": OppositeFacesConstraint,
        # "LipschitzRobustness": CreateLipschitzRobustnessConstraint,
        # "AlsomitraOutputConstraint": CreateAlsomitraOutputConstraint,
        # "Groups": CreateGroupConstraint,  # Keep local since it has dataset-specific logic
    }

    # Get constraint class from safe mapping
    constraint_class = safe_call(args.constraint, output_allowed)

    # Instantiate constraint with proper parameters based on type
    if constraint_class == StandardRobustnessConstraint:
        constraint: constraints.Constraint = StandardRobustnessConstraint(
            device=device,
            epsilon=0.3,  # Default epsilon for standard robustness on MNIST TODO: how can this be changed from the command line?
            delta=0.05,  # Default delta for standard robustness on MNIST
            std=std,  # epsilon is specified in terms of [0, 1] for MNIST but mean / std normalisation changes their domain
        )
    elif constraint_class == OppositeFacesConstraint:
        constraint: constraints.Constraint = OppositeFacesConstraint(
            device=device,
            epsilon=24 / 255,  # TODO: how can this be changed from the command line?
            delta=1.0,  # Default delta for OppositeFaces constraint
            std=std,  # epsilon is specified in terms of [0, 255] for dice images but mean / std normalisation changes their domain
        )
    else:
        raise NotImplementedError(f"Unhandeled constraint type: {constraint_class}")

    ### Set up PGD, ADAM, GradNorm ###
    if args.oracle == "pgd":
        oracle_train = training.PGD(
            logic,
            device,
            args.oracle_steps,
            args.oracle_restarts,
            args.pgd_step_size,
            mean,
            std,
        )
        oracle_test = training.PGD(
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
            logic, device, args.oracle_steps, args.oracle_restarts, mean, std
        )
        oracle_test = training.APGD(
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
        elif isinstance(constraint, constraints.OppositeFacesConstraint):
            folder = "opposite-faces"
        # elif isinstance(constraint, constraints.LipschitzRobustnessConstraint): # TODO uncomment when implemented
        # folder = "lipschitz-robustness"
        # elif isinstance(constraint, constraints.GroupConstraint):
        # folder = "group-constraint"
        else:
            raise ValueError(f"unknown constraint {constraint}!")
    else:
        folder = args.experiment_name

    folder_name = f"{args.results_dir}/{folder}/{args.dataset}"
    file_name = f"{folder_name}/{logic.name if not is_baseline else 'Baseline'}"

    report_file_name = f"{file_name}.csv"
    model_file_name = f"{file_name}.onnx"

    os.makedirs(folder_name, exist_ok=True)

    if args.save_imgs:
        save_dir = f"../saved_imgs/{folder}/{args.dataset}/{logic.name if not is_baseline else 'Baseline'}"

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
                if not isinstance(N, AlsomitraNet):
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
                        mode,
                    )
                else:
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
                        mode,  # TODO: or hardcode Mode.Regression here?
                        denorm_scale=AlsomitraDataset.S_out,
                    )
                train_time = time.time() - start

                if args.save_imgs:
                    save_epoch_images(train_info, epoch, save_dir, mean, std)  # type: ignore

                print(
                    f"Epoch {epoch}/{args.epochs}\t {args.constraint if args.experiment_name is None else args.experiment_name} on {args.dataset}, {logic.name if not is_baseline else 'Baseline'} \t TRAIN \t P-Metric: {train_info.pred_metric:.6f} \t C-Acc: {train_info.constr_acc:.2f}\t C-Sec: {train_info.constr_sec:.2f}\t P-Loss: {train_info.pred_loss:.2f}\t R-Loss: {train_info.random_loss:.2f}\t DL-Loss: {train_info.constr_loss:.2f}\t Time (Train) [s]: {train_time:.1f}"
                )
            else:
                train_info = EpochInfoTrain(
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, None, None, None, None
                )
                train_time = 0.0

            test_info = test(
                N,
                device,
                test_loader,
                oracle_test,
                logic,
                constraint,
                mode,
            )
            test_time = time.time() - start - train_time

            if args.save_imgs:
                save_epoch_images(test_info, epoch, save_dir, mean, std)  # type: ignore

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
                f"Epoch {epoch}/{args.epochs}\t {args.constraint if args.experiment_name is None else args.experiment_name} on {args.dataset}, {logic.name if not is_baseline else 'Baseline'} \t TEST \t P-Metric: {test_info.pred_metric:.6f}\t C-Acc: {test_info.constr_acc:.2f}\t C-Sec: {test_info.constr_sec:.2f}\t P-Loss: {test_info.pred_loss:.2f}\t R-Loss: {test_info.random_loss:.2f}\t DL-Loss: {test_info.constr_loss:.2f}\t Time (Test) [s]: {test_time:.1f}"
            )
            print("===")

    if args.save_onnx:
        x, _ = next(iter(train_loader))
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
