#!/bin/bash

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <quantization> <bitlinear_implementation> <loss_function>"
  exit 1
fi

quantization="$1"
bitlinear="$2"
loss_function="$3"
lrs=(1e-6 1e-5 1e-4 1e-3 1e-2)

for lr in "${lrs[@]}"; do
  uv run python -m src.main fit \
    --model QuantizedSmolModel \
    --data AmberDataModule \
    --data.chunks "2" \
    --trainer.max_steps 500 \
    --model.quantization "$quantization" \
    --model.bitlinear_implementation "$bitlinear" \
    --model.loss_function "$loss_function" \
    --model.lr "$lr"
done
