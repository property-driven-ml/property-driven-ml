#!/bin/bash

### MNIST
COMMON_ARGS="--data-set=mnist --epochs=100 --batch-size=2048 --lr=1e-3 --save-imgs"
PGD_ARGS="--oracle-steps=20 --oracle-restarts=30"

# MNIST Property: StandardRobustness
INPUT_PROPERTY="EpsilonBall(eps=0.3)"
OUTPUT_PROPERTY="StandardRobustness(delta=0.05)"

for LOGIC in "" "--logic=DL2 --initial-dl-weight=1.5" "--logic=GD --initial-dl-weight=1.5"; do
  python main.py $COMMON_ARGS $PGD_ARGS \
    --input-region="$INPUT_PROPERTY" \
    --output-constraint="$OUTPUT_PROPERTY" \
    $LOGIC \
    --experiment-name="StandardRobustness"
done

### ALSOMITRA
COMMON_ARGS="--data-set=alsomitra --epochs=100 --batch-size=64 --lr=1e-3"
PGD_ARGS="--oracle-steps=50 --oracle-restarts=80"

# Alsomitra Property 1: (y >= 2 - x) => (e_x >= 0.187)
INPUT_PROPERTY="AlsomitraInputRegion(y='(2 - x, inf)')"
OUTPUT_PROPERTY="AlsomitraOutputConstraint(e_x=(0.187, inf))"

for LOGIC in "" "--logic=DL2" "--logic=GD"; do
  python main.py $COMMON_ARGS $PGD_ARGS \
    --input-region="$INPUT_PROPERTY" \
    --output-constraint="$OUTPUT_PROPERTY" \
    $LOGIC \
    --experiment-name="AlsomitraProperty1"
done

# Alsomitra Property 2: (-2 - x <= y <= 2 - x) AND (-0.786 <= theta <= -0.747) => (0.184 <= e_x <= 0.19)
INPUT_PROPERTY="AlsomitraInputRegion(theta='(-0.786, -0.747)', y='(-2 - x, 2 - x)')"
OUTPUT_PROPERTY="AlsomitraOutputConstraint(e_x=(0.184, 0.19))"

for LOGIC in "" "--logic=DL2" "--logic=GD" "--logic=LK" "--logic=RC" "--logic=YG"; do
  python main.py $COMMON_ARGS $PGD_ARGS \
    --input-region="$INPUT_PROPERTY" \
    --output-constraint="$OUTPUT_PROPERTY" \
    $LOGIC \
    --experiment-name="AlsomitraProperty2"
done

# Alsomitra Property 3: (-x <= y <= 2 - x) AND (omega <= -0.12) AND (v_y <= -0.3) => (e_x <= 0.187)
INPUT_PROPERTY="AlsomitraInputRegion(v_y='(-inf, -0.3)', omega='(-inf, -0.12)', y='(-x, 2 - x)')"
OUTPUT_PROPERTY="AlsomitraOutputConstraint(e_x=(-inf, 0.187))"

for LOGIC in "" "--logic=DL2" "--logic=GD"; do
  python main.py $COMMON_ARGS $PGD_ARGS \
    --input-region="$INPUT_PROPERTY" \
    --output-constraint="$OUTPUT_PROPERTY" \
    $LOGIC \
    --experiment-name="AlsomitraProperty3"
done

# Alsomitra Property 4: (-2 - x <= y <= 2 - x) => LipschitzRobustness(L=0.3)
INPUT_PROPERTY="AlsomitraInputRegion(y='(-2 - x, 2 - x)')"
OUTPUT_PROPERTY="LipschitzRobustness(L=0.3)"

for LOGIC in "" "--logic=DL2" "--logic=GD"; do
  python main.py $COMMON_ARGS $PGD_ARGS \
    --input-region="$INPUT_PROPERTY" \
    --output-constraint="$OUTPUT_PROPERTY" \
    $LOGIC \
    --experiment-name="LipschitzRobustness" \
    --save-onnx
done