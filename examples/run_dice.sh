#!/bin/bash
python ../main.py --dataset=dice --epochs=100 --batch-size=8 --lr=5e-4 --save-onnx --oracle-steps=70 --oracle-restarts=50 --constraint=OppositeFaces --experiment-name=OppositeFaces --save-imgs
python ../main.py --dataset=dice --epochs=100 --batch-size=8 --lr=5e-4 --save-onnx --oracle-steps=70 --oracle-restarts=50 --constraint=OppositeFaces --experiment-name=OppositeFaces --save-imgs --logic=DL2
python ../main.py --dataset=dice --epochs=100 --batch-size=8 --lr=5e-4 --save-onnx --oracle-steps=70 --oracle-restarts=50 --constraint=OppositeFaces --experiment-name=OppositeFaces --save-imgs --logic=LL

# PYTHONPATH=.. python dice_to_idx.py

#  4/255 = 0.0156862745
#  8/255 = 0.03137254901
# 16/255 = 0.06274509803
vehicle verify --no-sat-print -s dice.vcl -n classifier:results/OppositeFaces-full/dice/Baseline.onnx -p epsilon:0.03137254901 -d images:dice-images-68.idx -v Marabou -a "--timeout=300" -l "/home/thomasflinkow/Marabou/build/bin/Marabou"
vehicle verify --no-sat-print -s dice.vcl -n classifier:results/OppositeFaces-full/dice/DL2.onnx -p epsilon:0.03137254901 -d images:dice-images-68.idx -v Marabou -a "--timeout=300" -l "/home/thomasflinkow/Marabou/build/bin/Marabou"
vehicle verify --no-sat-print -s dice.vcl -n classifier:results/OppositeFaces-full/dice/LL.onnx -p epsilon:0.03137254901 -d images:dice-images-68.idx -v Marabou -a "--timeout=300" -l "/home/thomasflinkow/Marabou/build/bin/Marabou"
