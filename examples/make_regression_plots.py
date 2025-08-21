from __future__ import print_function

import numpy as np

import torch
from torch.utils.data import random_split

from alsomitra_dataset import AlsomitraDataset

import onnxruntime as ort

import pandas as pd

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")

    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dataset = AlsomitraDataset("alsomitra_data_680.csv")
_, dataset_test = random_split(dataset, [0.9, 0.1])
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

for file_name in ["Baseline", "DL2", "GD"]:
    folder = "../results/LipschitzRobustness/alsomitra/"

    session = ort.InferenceSession(f"{folder}{file_name}.onnx")

    input_name = session.get_inputs()[0].name

    results = []

    for data, target in test_loader:
        outputs = session.run(None, {input_name: data.numpy()})

        results.append(
            {
                "predicted": AlsomitraDataset.denormalise_output(
                    outputs[0][0][0]
                ).item(),
                "target": AlsomitraDataset.denormalise_output(target).item(),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(f"{folder}{file_name}-RegressionPlot.csv", index=False)
