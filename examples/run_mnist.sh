#!/bin/bash
python ../main.py --dataset=mnist --epochs=100 --batch-size=2048 --lr=1e-3 --save-imgs --oracle-steps=5 --oracle-restarts=0 --constraint="StandardRobustness" --experiment-name="StandardRobustness-small"
python ../main.py --dataset=mnist --epochs=100 --batch-size=2048 --lr=1e-3 --save-imgs --oracle-steps=5 --oracle-restarts=0 --constraint="StandardRobustness" --experiment-name="StandardRobustness-small" --logic=DL2
python ../main.py --dataset=mnist --epochs=100 --batch-size=2048 --lr=1e-3 --save-imgs --oracle-steps=5 --oracle-restarts=0 --constraint="StandardRobustness" --experiment-name="StandardRobustness-small" --logic=GD
python ../main.py --dataset=mnist --epochs=100 --batch-size=2048 --lr=1e-3 --save-imgs --oracle-steps=5 --oracle-restarts=0 --constraint="StandardRobustness" --experiment-name="StandardRobustness-small" --logic=LL
