#!/bin/zsh

MODEL=${1:-"qwen2.5-coder:1.5b"}
N=${2:-164}
OFFSET=${3:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/opt/homebrew/Caskroom/miniconda/base/envs/trial-first/bin/python"

echo "Model: $MODEL | Problems: $N | Offset: $OFFSET"
echo "========================================"

cd "$SCRIPT_DIR"
mkdir -p results

for METHOD in baseline cot self_repair trial_first; do
    echo ""
    echo ">>> $METHOD"
    "$PYTHON" evaluate.py --method "$METHOD" --model "$MODEL" --n "$N" --offset "$OFFSET"
done

echo ""
echo "========================================"
echo "Summary:"
"$PYTHON" analyze.py
