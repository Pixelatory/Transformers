#!/bin/bash

# Get the directory of the script.
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Check if we are in the top-level directory.
if [[ -d "$SCRIPT_DIR/tests" && -d "$SCRIPT_DIR/transformers" ]]; then
  BASE_DIR="$SCRIPT_DIR"
else
  echo "Error: Script must be run from the top-level directory containing 'tests' and 'transformers'."
  exit 1
fi

# Change to the top-level directory.
cd "$BASE_DIR" || exit

export PYTHONPATH="$BASE_DIR"

# Run python tests.
python -m unittest discover -v -s tests -p "*test*.py"
