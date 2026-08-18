import argparse
import time
import os
import csv
import sys

import numpy as np
import onnx

import torch
import torch.optim as optim

from property_driven_ml.constraints import (
    StandardRobustnessConstraint,
    StrongClassificationRobustnessConstraint,
    ClassificationRobustnessConstraint,
    ExactlyOnePerPairConstraint,
    NotBothConstraint,
    ClothingFootwearConstraint,
    AlsomitraProperty1Constraint,
    AlsomitraProperty2Constraint,
    AlsomitraProperty3Constraint,
)
from examples.datasets import create_dataset
from property_driven_ml.utils.visualization import save_epoch_images

# Import from the property_driven_ml package
import property_driven_ml.logics as logics
import property_driven_ml.constraints as constraints
import property_driven_ml.training as training
from property_driven_ml.utils import safe_call
from property_driven_ml.training import EpochInfoTrain, train, test

# torch.autograd.set_detect_anomaly(True)


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
        logics.STL(1),
        logics.STL(2),
        logics.STL(5),
        logics.STL(10),
        logics.STL(20),
        logics.QLL(1),
        logics.QLL(2),
        logics.QLL(5),
        logics.QLL(10),
        logics.QLL(20),
        logics.QLL(50),
        logics.QLL(100),
        logics.LeakyLogic(),
        logics.RealProductLogic(),
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument(
        "--epsilon", type=float, default=0.3, help="epsilon value for epsilon-ball"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="delta value for probabilistic constraints",
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="number of epochs to train for"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mnist", "fashion", "alsomitra", "dice", "gtsrb"],
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default="StandardRobustness",
        choices=[
            "StandardRobustness",
            "StrongClassificationRobustness",
            "ClassificationRobustness",
            "NotBoth",
            "ClothingFootwear",
            "ExactlyOnePerPair",
            "AlsomitraProperty1",
            "AlsomitraProperty2",
            "AlsomitraProperty3",
            "AlsomitraProperty4",
        ],  # Will add more later
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
    parser.add_argument(
        "--seed",
        type=int,
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    kwargs = {"batch_size": args.batch_size}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")

        torch.cuda.manual_seed(args.seed)
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

    common_logic = logics.QLL()

    if args.logic is None:
        logic = common_logic
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
        "StrongClassificationRobustness": StrongClassificationRobustnessConstraint,
        "ClassificationRobustness": ClassificationRobustnessConstraint,
        "ExactlyOnePerPair": ExactlyOnePerPairConstraint,
        "NotBoth": NotBothConstraint,
        "ClothingFootwear": ClothingFootwearConstraint,
        # "LipschitzRobustness": CreateLipschitzRobustnessConstraint,
        "AlsomitraProperty1": AlsomitraProperty1Constraint,
        "AlsomitraProperty2": AlsomitraProperty2Constraint,
        "AlsomitraProperty3": AlsomitraProperty3Constraint,
        # "Groups": CreateGroupConstraint,  # Keep local since it has dataset-specific logic
    }

    # Get constraint class from safe mapping
    constraint_class = safe_call(args.constraint, output_allowed)

    # Instantiate constraint with proper parameters based on type
    if constraint_class == StandardRobustnessConstraint:
        constraint: constraints.Constraint = StandardRobustnessConstraint(
            device=device,
            epsilon=args.epsilon,
            delta=args.delta,
            std=std,  # epsilon is specified in terms of [0, 1] for MNIST but mean / std normalisation changes their domain
        )
    elif constraint_class == StrongClassificationRobustnessConstraint:
        constraint: constraints.Constraint = StrongClassificationRobustnessConstraint(
            device=device,
            epsilon=args.epsilon,
            delta=args.delta,
            std=std,  # epsilon is specified in terms of [0, 1] for MNIST but mean / std normalisation changes their domain
        )
    elif constraint_class == ClassificationRobustnessConstraint:
        constraint: constraints.Constraint = ClassificationRobustnessConstraint(
            device=device,
            epsilon=args.epsilon,
            std=std,  # epsilon is specified in terms of [0, 1] for MNIST but mean / std normalisation changes their domain
        )
    elif constraint_class == ExactlyOnePerPairConstraint:
        constraint: constraints.Constraint = ExactlyOnePerPairConstraint(
            device=device,
            epsilon=args.epsilon,
            std=std,  # epsilon is specified in terms of [0, 255] for dice images but mean / std normalisation changes their domain
        )
    elif constraint_class == NotBothConstraint:
        constraint: constraints.Constraint = NotBothConstraint(
            device=device,
            epsilon=args.epsilon,
            std=std,  # epsilon is specified in terms of [0, 255] for dice images but mean / std normalisation changes their domain
        )
    elif constraint_class == ClothingFootwearConstraint:
        constraint: constraints.Constraint = ClothingFootwearConstraint(
            device=device,
            epsilon=args.epsilon,
            std=std,  # epsilon is specified in terms of [0, 1] for MNIST but mean / std normalisation changes their domain
        )
    elif constraint_class == AlsomitraProperty1Constraint:
        constraint: constraints.Constraint = AlsomitraProperty1Constraint(device=device)
    elif constraint_class == AlsomitraProperty2Constraint:
        constraint: constraints.Constraint = AlsomitraProperty2Constraint(device=device)
    elif constraint_class == AlsomitraProperty3Constraint:
        constraint: constraints.Constraint = AlsomitraProperty3Constraint(device=device)
    else:
        raise NotImplementedError(f"Unhandled constraint type: {constraint_class}")

    ### Set up PGD, ADAM ###
    train_steps = args.oracle_steps
    train_restarts = args.oracle_restarts  # // 2

    test_steps = args.oracle_steps
    test_restarts = args.oracle_restarts

    def make_oracle(logic, steps, restarts):
        if args.oracle == "pgd":
            return training.PGD(
                logic,
                device,
                steps,
                restarts,
                args.pgd_step_size,
                mean,
                std,
            )
        else:
            return training.APGD(
                logic,
                device,
                steps,
                restarts,
                mean,
                std,
            )

    oracle_self_train = make_oracle(logic, train_steps, train_restarts)
    oracle_self_test = make_oracle(logic, test_steps, test_restarts)
    oracle_common_test = make_oracle(common_logic, test_steps, test_restarts)

    optimizer = optim.AdamW(N.parameters(), lr=args.lr, weight_decay=1e-4)

    ### Set up folders for results and PGD images ###

    if isinstance(constraint, constraints.StandardRobustnessConstraint):
        folder = "standard-robustness"
    elif isinstance(constraint, constraints.StrongClassificationRobustnessConstraint):
        folder = "strong-classification-robustness"
    elif isinstance(constraint, constraints.ClassificationRobustnessConstraint):
        folder = "classification-robustness"
    elif isinstance(constraint, constraints.ExactlyOnePerPairConstraint):
        folder = "exactly-one"
    elif isinstance(constraint, constraints.NotBothConstraint):
        folder = "not-both"
    elif isinstance(constraint, constraints.ClothingFootwearConstraint):
        folder = "clothing-footwear"
    else:
        raise ValueError(f"unknown constraint {constraint}!")

    folder_name = f"{args.results_dir}/{folder}/{args.dataset}/{args.seed}"
    file_name = f"{folder_name}/{logic.name if not is_baseline else 'Baseline'}"

    report_file_name = f"{file_name}.csv"
    model_file_name = f"{file_name}.onnx"

    os.makedirs(folder_name, exist_ok=True)

    if args.save_imgs:
        save_dir = f"../saved_imgs/{folder}/{args.dataset}/{args.seed}/{logic.name if not is_baseline else 'Baseline'}"

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
                "Train-P-Grad",
                "Train-C-Grad",
                "Train-C-Weight",
                "Train-Weighted-C-Grad",
                "Train-Grad-Ratio",
                "Train-P-Metric",
                "Train-C-Acc",
                "Train-C-Sec",
                "Test-P-Loss",
                "Test-R-Loss",
                "Test-C-Loss-self",
                "Test-C-Loss-common",
                "Test-P-Metric",
                "Test-C-Acc",
                "Test-C-Sec-self",
                "Test-C-Sec-common",
                "Train-Time",
                "Test-Time",
            ]
        )

        for epoch in range(0, args.epochs + 1):
            start = time.time()

            if epoch > 0:
                with_dl = (epoch > args.delay) and (not is_baseline)

                train_info = train(
                    epoch,
                    N,
                    device,
                    train_loader,
                    optimizer,
                    oracle_self_train,
                    logic,
                    constraint,
                    with_dl,
                    mode,
                    args.alpha,
                )
                train_time = time.time() - start

                if args.save_imgs:
                    save_epoch_images(train_info, epoch, save_dir, mean, std)  # type: ignore

                print(
                    f"Epoch {epoch}/{args.epochs}\t "
                    f"{args.constraint} "
                    f"on {args.dataset}, {logic.name if not is_baseline else 'Baseline'}\t TRAIN\t "
                    f"Time [s]: {train_time:.1f}\t "
                    f"P-Metric: {train_info.pred_metric:.4f}\t "
                    f"C-Acc: {f'{train_info.constr_acc:.4f}' if train_info.constr_acc is not None else '   n/a'}\t "
                    f"C-Sec: {f'{train_info.constr_sec:.4f}' if train_info.constr_sec is not None else '   n/a'}\t "
                    f"P-Loss: {train_info.pred_loss:.2f}\t "
                    f"R-Loss: {f'{train_info.random_loss:.6f}' if train_info.random_loss is not None else ' n/a'}\t "
                    f"C-Loss: {f'{train_info.constr_loss:.6f}' if train_info.constr_loss is not None else ' n/a'}\t "
                    f"P-Grad: {f'{train_info.pred_grad_norm:.2e}' if train_info.pred_grad_norm is not None else ' n/a'}\t "
                    f"C-Grad: {f'{train_info.constr_grad_norm:.2e}' if train_info.constr_grad_norm is not None else ' n/a'}\t "
                    f"Grad-Ratio: {f'{train_info.grad_ratio:.2f}' if train_info.grad_ratio is not None else ' n/a'}"
                )
            else:
                train_info = EpochInfoTrain(
                    pred_metric=0.0, pred_loss=0.0, constr_sec=0.0
                )
                train_time = 0.0

            test_info = test(
                epoch,
                N,
                device,
                test_loader,
                oracle_self_test,
                oracle_common_test,
                logic,
                constraint,
                is_baseline,
                mode,
            )
            test_time = time.time() - start - train_time

            if args.save_imgs:
                save_epoch_images(test_info, epoch, save_dir, mean, std)  # type: ignore

            writer.writerow(
                [
                    epoch,
                    train_info.pred_loss,
                    -1,  # random loss, we don't evaluate that during training for performance
                    train_info.constr_loss,
                    train_info.pred_grad_norm
                    if train_info.pred_grad_norm is not None
                    else -1,
                    train_info.constr_grad_norm
                    if train_info.constr_grad_norm is not None
                    else -1,
                    train_info.constr_loss_weight
                    if train_info.constr_loss_weight is not None
                    else -1,
                    train_info.weighted_constr_grad_norm
                    if train_info.weighted_constr_grad_norm is not None
                    else -1,
                    train_info.grad_ratio if train_info.grad_ratio is not None else -1,
                    train_info.pred_metric,
                    -1,  # c acc, we don't evaluate that during training for performance
                    train_info.constr_sec,
                    test_info.pred_loss,
                    test_info.random_loss,
                    test_info.constr_loss_self if not is_baseline else -1,
                    test_info.constr_loss_common,
                    test_info.pred_metric,
                    test_info.constr_acc,
                    test_info.constr_sec_self if not is_baseline else -1,
                    test_info.constr_sec_common,
                    train_time,
                    test_time,
                ]
            )

            print(
                f"Epoch {epoch}/{args.epochs}\t "
                f"{args.constraint} "
                f"on {args.dataset}, {logic.name if not is_baseline else 'Baseline'}\t TEST\t "
                f"Time [s]: {test_time:.1f}\t "
                f"P-Metric: {test_info.pred_metric:.4f}\t "
                f"C-Acc: {test_info.constr_acc:.4f}\t "
                f"C-Sec (self): {f'{test_info.constr_sec_self:.4f}' if not is_baseline else 'n/a'}\t "
                f"C-Sec (common): {test_info.constr_sec_common:.4f}\t "
                f"P-Loss: {test_info.pred_loss:.2f}\t "
                f"R-Loss: {test_info.random_loss:.6f}\t "
                f"C-Loss (self): {f'{test_info.constr_loss_self:.6f}' if not is_baseline else 'n/a'}\t "
                f"C-Loss (common): {test_info.constr_loss_common:.6f}"
            )
            print("===")

    if args.save_onnx:
        x, _ = next(iter(train_loader))
        dummy_input = torch.randn(args.batch_size, *x.shape[1:], requires_grad=True).to(
            device=device
        )

        torch.onnx.export(
            N.eval(),
            (dummy_input,),
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
