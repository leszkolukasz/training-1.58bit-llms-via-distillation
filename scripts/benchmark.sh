#!/bin/bash

set -e

configs=(
  # "1b BitNet CrossEntropy"
  # "1_58b BitNet CrossEntropy"
  # "1_58b FBI CrossEntropy"
  # "1_58b OneBit CrossEntropy"
  # "1_58b OneBit KL"
  "1_58b OneBit CAKL"
  # "1_58b OneBit Wasserstein"
)

for config in "${configs[@]}"; do
  read -r quantization bitlinear_impl loss_function <<< "$config"

  printf "%s\n%s\n%s\n" "$quantization" "$bitlinear_impl" "$loss_function" > input.txt

  uv run python -m src.hf_model.register_and_test
  benchmark_output=$(uv run python -m src.hf_model.benchmark)

  name="quant_${quantization}_impl_${bitlinear_impl}_loss_${loss_function}"
  FILE="./data/benchmark.md"
  
  awk -v str="$name" -v value="$benchmark_output" '
  index($0, str) && !found {
      print
      printf "\n%s\n", value
      found=1
      next
  }
  { print }
  END {
    if (!found) {
      printf "\n# %s\n", str
      printf "\n%s\n", value
    }
  }
  ' "$FILE" > "${FILE}.tmp" && mv "${FILE}.tmp" "$FILE"

done