#!/bin/bash

### MNIST - Standard Robustness
COMMON_ARGS="--dataset=mnist --epochs=100 --batch-size=2048 --lr=1e-3 --save-imgs"
PGD_ARGS="--oracle-steps=20 --oracle-restarts=30"

# Standard Robustness with epsilon ball (eps=0.3, delta=0.05 are built into constraint)
for LOGIC in "" "--logic=DL2 --initial-dl-weight=1.5" "--logic=GD --initial-dl-weight=1.5"; do
  uv run python ../main.py $COMMON_ARGS $PGD_ARGS \
    --constraint="StandardRobustness" \
    $LOGIC \
    --experiment-name="StandardRobustness"
done

### ALSOMITRA - Examples (these constraints are not yet implemented in the new API)
COMMON_ARGS="--dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3"
PGD_ARGS="--oracle-steps=50 --oracle-restarts=80"

# Note: The following Alsomitra constraints need to be reimplemented for the new unified API
# They were using the old input-region/output-constraint separation which has been removed

echo "=========================================="
echo "WARNING: Alsomitra constraint examples are disabled"
echo "The following constraints need to be reimplemented for the new unified constraint API:"
echo "- AlsomitraPrecondition + AlsomitraPostcondition"
echo "- LipschitzRobustness constraint"
echo "=========================================="

# # Alsomitra Property 1: (y >= 2 - x) => (e_x >= 0.187)
# # TODO: Need to implement AlsomitraProperty1Constraint class
# for LOGIC in "" "--logic=DL2" "--logic=GD"; do
#   uv run python ../main.py $COMMON_ARGS $PGD_ARGS \
#     --constraint="AlsomitraProperty1" \
#     $LOGIC \
#     --experiment-name="AlsomitraProperty1"
# done

# # Alsomitra Property 2: (-2 - x <= y <= 2 - x) AND (-0.786 <= theta <= -0.747) => (0.184 <= e_x <= 0.19)
# # TODO: Need to implement AlsomitraProperty2Constraint class
# for LOGIC in "" "--logic=DL2" "--logic=GD" "--logic=LK" "--logic=RC" "--logic=YG"; do
#   uv run python ../main.py $COMMON_ARGS $PGD_ARGS \
#     --constraint="AlsomitraProperty2" \
#     $LOGIC \
#     --experiment-name="AlsomitraProperty2"
# done

# # Alsomitra Property 3: (-x <= y <= 2 - x) AND (omega <= -0.12) AND (v_y <= -0.3) => (e_x <= 0.187)
# # TODO: Need to implement AlsomitraProperty3Constraint class
# for LOGIC in "" "--logic=DL2" "--logic=GD"; do
#   uv run python ../main.py $COMMON_ARGS $PGD_ARGS \
#     --constraint="AlsomitraProperty3" \
#     $LOGIC \
#     --experiment-name="AlsomitraProperty3"
# done

# # Alsomitra Property 4: (-2 - x <= y <= 2 - x) => LipschitzRobustness(L=0.3)
# # TODO: Need to implement LipschitzRobustnessConstraint class
# for LOGIC in "" "--logic=DL2" "--logic=GD"; do
#   uv run python ../main.py $COMMON_ARGS $PGD_ARGS \
#     --constraint="LipschitzRobustness" \
#     $LOGIC \
#     --experiment-name="LipschitzRobustness" \
#     --save-onnx
# done
