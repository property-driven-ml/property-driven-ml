from __future__ import print_function

from collections import namedtuple

import argparse
import ast

import time
import os
import csv
import sys

import numpy as np

import onnx

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import random_split

from torchvision import datasets, transforms
from torchvision.utils import save_image

from examples.alsomitra_dataset import AlsomitraDataset, AlsomitraInputRegion
from examples.models import MnistNet, AlsomitraNet, GTSRBNet

# Import from the property_driven_ml package
import property_driven_ml.logics as logics
import property_driven_ml.constraints as constraints
import property_driven_ml.training as training
from property_driven_ml import maybe

EpochInfoTrain = namedtuple('EpochInfoTrain', 'pred_metric constr_acc constr_sec pred_loss random_loss constr_loss pred_loss_weight constr_loss_weight input_img adv_img random_img')
EpochInfoTest = namedtuple('EpochInfoTest', 'pred_metric constr_acc constr_sec pred_loss random_loss constr_loss input_img adv_img random_img')

# --- Minimal safe helpers to replace eval on user-controlled strings ---
def _is_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, str, type(None)))
    if isinstance(node, ast.Tuple):
        return all(_is_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Name) and node.id == 'inf':
        return True
    return False

def _literal_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_literal_value(elt) for elt in node.elts)
    if isinstance(node, ast.Name) and node.id == 'inf':
        return float('inf')
    raise ValueError('Only literal values are allowed in arguments')

def safe_call(expr: str, allowed: dict):
    tree = ast.parse(expr, mode='eval')
    if not isinstance(tree, ast.Expression) or not isinstance(tree.body, ast.Call):
        raise ValueError('Only simple function calls are allowed')
    call = tree.body
    if not isinstance(call.func, ast.Name):
        raise ValueError('Only direct calls to allowed functions are permitted')
    func_name = call.func.id
    if func_name not in allowed:
        raise ValueError(f"Function '{func_name}' is not allowed")
    if not all(_is_literal(a) for a in call.args):
        raise ValueError('Only literal positional arguments are allowed')
    if not all(isinstance(kw, ast.keyword) and kw.arg and _is_literal(kw.value) for kw in call.keywords):
        raise ValueError('Only literal keyword arguments are allowed')
    args = [_literal_value(a) for a in call.args]
    kwargs = {kw.arg: _literal_value(kw.value) for kw in call.keywords}
    return allowed[func_name](*args, **kwargs)

def _eval_arith(node: ast.AST, context: dict) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == 'inf':
            return float('inf')
        if node.id in context:
            return float(context[node.id])
        raise ValueError(f"Unknown variable '{node.id}' in bounds")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_arith(node.operand, context)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        l = _eval_arith(node.left, context)
        r = _eval_arith(node.right, context)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        return l / r
    raise ValueError('Only basic arithmetic (+,-,*,/) with numbers and variables is allowed in bounds')

def safe_bounds(expr: str, context: dict) -> tuple[float, float]:
    tree = ast.parse(expr, mode='eval')
    if not isinstance(tree, ast.Expression) or not isinstance(tree.body, ast.Tuple) or len(tree.body.elts) != 2:
        raise ValueError('Bounds must be a tuple expression of the form (lo, hi)')
    lo_node, hi_node = tree.body.elts
    lo = _eval_arith(lo_node, context)
    hi = _eval_arith(hi_node, context)
    return (lo, hi)

def train(N: torch.nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader, optimizer, oracle: training.Attack, grad_norm: training.GradNorm, logic: logics.Logic, constraint: constraints.Constraint, with_dl: bool, is_classification: bool) -> EpochInfoTrain:
    avg_pred_metric, avg_pred_loss = torch.tensor(0., device=device), torch.tensor(0., device=device)
    avg_constr_acc, avg_constr_sec, avg_constr_loss, avg_random_loss = torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0., device=device)

    images = { 'input': None, 'random': None, 'adv': None}

    N.train()

    for _, (data, target, lo, hi) in enumerate(train_loader, start=1):
        x, y_target, lo, hi = data.to(device), target.to(device), lo.to(device), hi.to(device)

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
            rmse = (AlsomitraDataset.S_out * rmse.cpu()).squeeze() # TODO: hmm, explain denormalise
            avg_pred_metric += rmse

        # get random + adversarial samples
        with torch.no_grad():
            random = oracle.uniform_random_sample(lo, hi)

        adv = oracle.attack(N, x, y_target, lo, hi, constraint)

        # forward pass for constraint accuracy (constraint satisfaction on random samples)
        with torch.no_grad():
            loss_random, sat_random = constraint.eval(N, x, random, y_target, logic, reduction='mean')

        # forward pass for constraint security (constraint satisfaction on adversarial samples)
        with maybe(torch.no_grad(), not with_dl):
            loss_adv, sat_adv = constraint.eval(N, x, adv, y_target, logic, reduction='mean')

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
        images['input'], images['random'], images['adv'] = x[i], random[i], adv[i]

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
        input_img=images['input'],
        adv_img=images['adv'],
        random_img=images['random']
    )

def test(N: torch.nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader, oracle: training.Attack, logic: logics.Logic, constraint: constraints.Constraint, is_classification: bool) -> EpochInfoTest:
    correct, constr_acc, constr_sec = torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0., device=device)
    avg_pred_loss, avg_constr_loss, avg_random_loss = torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0., device=device)

    total_samples = 0

    images = { 'input': None, 'random': None, 'adv': None}

    N.eval()

    for _, (data, target, lo, hi) in enumerate(test_loader, start=1):
        x, y_target, lo, hi = data.to(device), target.to(device), lo.to(device), hi.to(device)
        total_samples += x.size(0)

        with torch.no_grad():
            # forward pass
            y = N(x)

            if is_classification:
                avg_pred_loss += F.cross_entropy(y, y_target, reduction='sum')
                pred = y.max(dim=1, keepdim=True)[1]
                correct += pred.eq(y_target.view_as(pred)).sum()
            else:
                avg_pred_loss += F.mse_loss(y, y_target, reduction='sum')

            # get random samples (no grad)
            random = oracle.uniform_random_sample(lo, hi)

        # get adversarial samples (requires grad)
        adv = oracle.attack(N, x, y_target, lo, hi, constraint)

        # forward passes for constraint accuracy (constraint satisfaction on random samples) + constraint security (constraint satisfaction on adversarial samples)
        with torch.no_grad():
            loss_random, sat_random = constraint.eval(N, x, random, y_target, logic, reduction='sum')
            loss_adv, sat_adv = constraint.eval(N, x, adv, y_target, logic, reduction='sum')

            constr_acc += sat_random
            constr_sec += sat_adv

            avg_random_loss += loss_random
            avg_constr_loss += loss_adv

        # save one original image, random sample, and adversarial sample image (for debugging, inspecting attacks)
        i = np.random.randint(0, x.size(0))
        images['input'], images['random'], images['adv'] = x[i], random[i], adv[i]

    if is_classification:
        pred_acc = correct.item() / total_samples
    else:
        rmse = torch.sqrt(avg_pred_loss / total_samples)
        rmse = (AlsomitraDataset.S_out * rmse.cpu()).item()

    return EpochInfoTest(
        pred_metric=pred_acc if is_classification else rmse,
        constr_acc=constr_acc.item() / total_samples,
        constr_sec=constr_sec.item() / total_samples,
        pred_loss=avg_pred_loss.item() / total_samples,
        random_loss=avg_random_loss.item() / total_samples,
        constr_loss=avg_constr_loss.item() / total_samples,
        input_img=images['input'],
        adv_img=images['adv'],
        random_img=images['random']
    )

def main():
    logics_map = {
        'DL2': logics.DL2(),
        'GD': logics.GoedelFuzzyLogic(),
        'KD': logics.KleeneDienesFuzzyLogic(),
        'LK': logics.LukasiewiczFuzzyLogic(),
        'RC': logics.ReichenbachFuzzyLogic(),
        'GG': logics.GoguenFuzzyLogic(),
        'RCS': logics.ReichenbachSigmoidalFuzzyLogic(),
        'YG': logics.YagerFuzzyLogic(),
        'STL': logics.STL(),
    }

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
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train for')
    parser.add_argument('--data-set', type=str, required=True, choices=['mnist', 'alsomitra', 'gtsrb'])
    parser.add_argument('--input-region', type=str, required=True, help='the input region induced by the precondition P(x)')
    parser.add_argument('--output-constraint', type=str, required=True, help='the output constraint given by Q(f(x))')
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--oracle', type=str, default='apgd', choices=['pgd', 'apgd'], help='standard PGD or AutoPGD')
    parser.add_argument('--oracle-steps', type=int, default=20, help='number of PGD iterations')
    parser.add_argument('--oracle-restarts', type=int, default=10, help='number of PGD random restarts')
    parser.add_argument('--pgd-step-size', type=float, default=.03)
    parser.add_argument('--delay', type=int, default=0, help='number of epochs to wait before introducing constraint loss')
    parser.add_argument('--logic', type=str, default=None, choices=[l.name for l in logics_list], help='the differentiable logic to use for training with the constraint, or None')
    parser.add_argument('--results-dir', type=str, default='results', help='directory in which to save .onnx and .csv files')
    parser.add_argument('--initial-dl-weight', type=float, default=1.)
    parser.add_argument('--grad-norm-alpha', type=float, default=.12, help='restoring force for GradNorm')
    parser.add_argument('--grad-norm-lr', type=float, default=None, help='learning rate for GradNorm weights, equal to --lr if not specified')
    parser.add_argument('--save-onnx', action='store_true', help='save .onnx file after training')
    parser.add_argument('--save-imgs', action='store_true', help='save one input image, random image, and adversarial image per epoch')
    args = parser.parse_args()

    kwargs = { 'batch_size': args.batch_size }

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        device = torch.device('cuda')

        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if os.name != 'nt': # NOTE: on Windows, our EpsilonBall implementation cannot be pickled
            kwargs.update({ 'num_workers': 4, 'pin_memory': True })
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if args.logic == None:
        logic = logics_list[0] # need some logic loss for oracle even for baseline
        is_baseline = True
    else:
        logic = next(l for l in logics_list if l.name == args.logic)
        is_baseline = False

    ### Set up dataset ###

    if args.data_set == 'mnist':
        mean, std = (.1307,), (.3081,)

        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        dataset_train = datasets.MNIST('data', train=True, download=True, transform=transform_train)
        dataset_test = datasets.MNIST('data', train=False, download=True, transform=transform_test)

        N = MnistNet().to(device)
    elif args.data_set == 'alsomitra':
        mean, std = (0.,), (1.,)

        dataset = AlsomitraDataset('examples/alsomitra_data_680.csv')
        dataset_train, dataset_test = random_split(dataset, [.9, .1])

        N = AlsomitraNet().to(device)
    elif args.data_set == 'gtsrb':
        mean, std = (.3403, .3121, .3214), (.2724, .2608, .2669)

        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=5, translate=(.1, .1)),
            transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        dataset_train = datasets.GTSRB('data', split="train", download=True, transform=transform_train)
        dataset_test = datasets.GTSRB('data', split="test", download=True, transform=transform_test)

        N = GTSRBNet().to(device)

    ### Parse input constraint ###

    def CreateEpsilonBall(eps: float) -> tuple[constraints.EpsilonBall, constraints.EpsilonBall]:
        train_constraint, test_constraint = constraints.EpsilonBall(dataset_train, eps, mean, std), constraints.EpsilonBall(dataset_test, eps, mean, std)
        return train_constraint, test_constraint

    def CreateAlsomitraInputRegion(v_x: str | None = None, v_y: str | None = None, omega: str | None = None, theta: str | None = None, x: str | None = None, y: str | None = None) -> tuple[AlsomitraInputRegion, AlsomitraInputRegion]:
        def bounds_fn(input: torch.Tensor):
            # store the current data point
            context = {
                'v_x': input[0],
                'v_y': input[1],
                'omega': input[2],
                'theta': input[3],
                'x': input[4],
                'y': input[5],
                'inf': np.inf
            }

            # evaluate the user specified bounds (which may refer to current data point values like 'x' in 'y >= 0.5 - x')
            variables = ['v_x', 'v_y', 'omega', 'theta', 'x', 'y']
            expressions = [v_x, v_y, omega, theta, x, y]

            bounds = {
                var: (-np.inf, np.inf) if expr is None else safe_bounds(expr, context)
                for var, expr in zip(variables, expressions)
            }

            lo = torch.tensor([bounds[var][0] for var in variables])
            hi = torch.tensor([bounds[var][1] for var in variables])

            return (lo, hi)

        train_constraint, test_constraint = AlsomitraInputRegion(dataset_train, bounds_fn, mean, std), AlsomitraInputRegion(dataset_test, bounds_fn, mean, std)
        return train_constraint, test_constraint
    
    input_allowed = {
        'EpsilonBall': CreateEpsilonBall,
        'AlsomitraInputRegion': CreateAlsomitraInputRegion,
    }

    wrapper_train, wrapper_test = safe_call(args.input_region, input_allowed)

    train_loader = torch.utils.data.DataLoader(wrapper_train, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(wrapper_test, shuffle=False, drop_last=True, **kwargs)

    print(f'len(dataset_train)={len(wrapper_train)} len(dataset_test)={len(wrapper_test)}')
    print(f'len(train_loader)={len(train_loader)} len(test_loader)={len(test_loader)}')

    ### Parse output constraint ###

    def CreateStandardRobustnessConstraint(delta: float) -> constraints.StandardRobustnessConstraint:
        return constraints.StandardRobustnessConstraint(device, delta)
    
    def CreateLipschitzRobustnessConstraint(L: float) -> constraints.LipschitzRobustnessConstraint:
        return constraints.LipschitzRobustnessConstraint(device, L)
    
    def CreateAlsomitraOutputConstraint(e_x: tuple[float, float]) -> constraints.AlsomitraOutputConstraint:
        lo, hi = e_x
        return constraints.AlsomitraOutputConstraint(device, None if lo is None else AlsomitraDataset.normalise_output(lo).squeeze(), None if hi is None else AlsomitraDataset.normalise_output(hi).squeeze())
    
    def CreateGroupConstraint(delta: float) -> constraints.GroupConstraint:
        assert args.data_set == 'gtsrb', 'groups are only defined for GTSRB'

        groups: list[list[int]] = [
            [*range(6), 7, 8],    # speed limit signs
            [9, 10, 15, 16],      # other prohibitory signs
            [12, 13, 14, 17],     # unique signs
            [11, *range(18, 32)], # danger signs
            [*range(33, 41)],     # mandatory signs
            [6, 32, 41, 42]       # derestriction signs
        ]

        return constraints.GroupConstraint(device, groups, delta)

    output_allowed = {
        'StandardRobustness': CreateStandardRobustnessConstraint,
        'LipschitzRobustness': CreateLipschitzRobustnessConstraint,
        'AlsomitraOutputConstraint': CreateAlsomitraOutputConstraint,
        'Groups': CreateGroupConstraint,
    }

    constraint: constraints.Constraint = safe_call(args.output_constraint, output_allowed)

    ### Set up PGD, ADAM, GradNorm ###

    x0, _ = dataset_train[0]

    if args.oracle == 'pgd':
        oracle_train = training.PGD(x0, logic, device, args.oracle_steps, args.oracle_restarts, args.pgd_step_size, mean, std)
        oracle_test = training.PGD(x0, logics_list[0], device, args.oracle_steps, args.oracle_restarts, args.pgd_step_size, mean, std)
    else:
        oracle_train = training.APGD(x0, logic, device, args.oracle_steps, args.oracle_restarts, mean, std)
        oracle_test = training.APGD(x0, logics_list[0], device, args.oracle_steps, args.oracle_restarts, mean, std)

    optimizer = optim.AdamW(N.parameters(), lr=args.lr, weight_decay=1e-4)

    grad_norm = training.GradNorm(N, device, optimizer, lr=args.grad_norm_lr if args.grad_norm_lr is not None else args.lr, alpha=args.grad_norm_alpha, initial_dl_weight=args.initial_dl_weight)

    ### Set up folders for results and PGD images ###

    if args.experiment_name == None:
        if isinstance(constraint, constraints.StandardRobustnessConstraint):
            folder = 'standard-robustness'
        elif isinstance(constraint, LipschitzRobustnessConstraint):
            folder = 'lipschitz-robustness'
        elif isinstance(constraint, GroupConstraint):
            folder = 'group-constraint'
        else:
            assert False, f'unknown constraint {constraint}!'
    else:
        folder = args.experiment_name

    folder_name = f'{args.results_dir}/{folder}/{args.data_set}'
    file_name = f'{folder_name}/{logic.name if not is_baseline else "Baseline"}'

    report_file_name = f'{file_name}.csv'
    model_file_name = f'{file_name}.onnx'

    os.makedirs(folder_name, exist_ok=True)

    if args.save_imgs:
        save_dir = f'../saved_imgs/{folder}/{args.data_set}/{logic.name if not is_baseline else "Baseline"}'
        os.makedirs(save_dir, exist_ok=True)

    def save_imgs(info: EpochInfoTrain | EpochInfoTest, epoch):
        if not args.save_imgs:
            return
        
        def denormalise(x: torch.Tensor) -> torch.Tensor:
            return x * torch.as_tensor(std, device=x.device).view(-1, 1, 1) + torch.as_tensor(mean, device=x.device).view(-1, 1, 1)

        def save_img(img: torch.Tensor, name: str):
            save_image(denormalise(img), os.path.join(save_dir, name))

        if isinstance(info, EpochInfoTrain):
            prefix = 'train'
        else:
            prefix = 'test'

        save_img(info.input_img, f'{epoch}-{prefix}_input.png')
        save_img(info.adv_img, f'{epoch}-{prefix}_adv.png')
        save_img(info.random_img, f'{epoch}-{prefix}_random.png')

    ### Start training ###

    print(f'using device {device}')
    print(f'#model parameters: {sum(p.numel() for p in N.parameters() if p.requires_grad)}')

    with open(report_file_name, 'w', buffering=1, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        csvfile.write(f'#{sys.argv}\n')
        writer.writerow(['Epoch', 'Train-P-Loss', 'Train-R-Loss', 'Train-C-Loss', 'Train-P-Loss-Weight', 'Train-C-Loss-Weight', 'Train-P-Metric', 'Train-C-Acc', 'Train-C-Sec', 'Test-P-Loss', 'Test-R-Loss', 'Test-C-Loss', 'Test-P-Metric', 'Test-C-Acc', 'Test-C-Sec', 'Train-Time', 'Test-Time'])

        for epoch in range(0, args.epochs + 1):
            start = time.time()

            if epoch > 0:
                with_dl = (epoch > args.delay) and (not is_baseline)
                train_info = train(N, device, train_loader, optimizer, oracle_train, grad_norm, logic, constraint, with_dl, is_classification=not isinstance(N, AlsomitraNet)) # TODO: better check?
                train_time = time.time() - start

                save_imgs(train_info, epoch)

                print(f'Epoch {epoch}/{args.epochs}\t {args.output_constraint if args.experiment_name == None else args.experiment_name} on {args.data_set}, {logic.name if not is_baseline else "Baseline"} \t TRAIN \t P-Metric: {train_info.pred_metric:.6f} \t C-Acc: {train_info.constr_acc:.2f}\t C-Sec: {train_info.constr_sec:.2f}\t P-Loss: {train_info.pred_loss:.2f}\t R-Loss: {train_info.random_loss:.2f}\t DL-Loss: {train_info.constr_loss:.2f}\t Time (Train) [s]: {train_time:.1f}')
            else:
                train_info = EpochInfoTrain(0., 0., 0., 0., 0., 0., 1., 1., None, None, None)
                train_time = 0.

            test_info = test(N, device, test_loader, oracle_test, logic, constraint, is_classification=not isinstance(N, AlsomitraNet))
            test_time = time.time() - start - train_time

            save_imgs(test_info, epoch)

            writer.writerow([epoch, \
                             train_info.pred_loss, train_info.random_loss, train_info.constr_loss, train_info.pred_loss_weight, train_info.constr_loss_weight, train_info.pred_metric, train_info.constr_acc, train_info.constr_sec, \
                             test_info.pred_loss, test_info.random_loss, test_info.constr_loss, test_info.pred_metric, test_info.constr_acc, test_info.constr_sec, \
                             train_time, test_time])

            print(f'Epoch {epoch}/{args.epochs}\t {args.output_constraint if args.experiment_name == None else args.experiment_name} on {args.data_set}, {logic.name if not is_baseline else "Baseline"} \t TEST \t P-Metric: {test_info.pred_metric:.6f}\t C-Acc: {test_info.constr_acc:.2f}\t C-Sec: {test_info.constr_sec:.2f}\t P-Loss: {test_info.pred_loss:.2f}\t R-Loss: {test_info.random_loss:.2f}\t DL-Loss: {test_info.constr_loss:.2f}\t Time (Test) [s]: {test_time:.1f}')
            print(f'===')

    if args.save_onnx:
        x, _, _, _ = next(iter(train_loader))

        torch.onnx.export(
            N.eval(),
            torch.randn(args.batch_size, *x.shape[1:], requires_grad=True).to(device=device),
            model_file_name,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': { 0: 'batch_size' }, 'output': { 0: 'batch_size' }},
        )

        onnx_model = onnx.load(model_file_name)
        onnx.checker.check_model(onnx_model)

if __name__ == '__main__':
    main()