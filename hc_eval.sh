#!/bin/bash

# runner command: "source bench.sh"

cd "$HOME/volume/hecate-compiler" || { echo "Failed to change directory to hecate-compiler"; exit 1; }

# Set the working directory to the location of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export HECATE="$DIR"
echo "HECATE DIR: $HECATE"

# Activate the virtual environment and load configuration
echo "Activating the virtual environment and loading configuration"
source .venv/bin/activate
source config.sh

echo "========================================="
echo "Building the Hecate optimizer"
cmake -S . -B build -DMLIR_ROOT=$HOME/install/MLIR \
                -DSEAL_ROOT=$HOME/install/SEAL \
                -DHEaaN_ROOT=$HOME/install/heaan \
                -DHEonGPU_ROOT=$HOME/install/HEonGPU \
                -DCMAKE_C_COMPILER=clang \
                -DCMAKE_CXX_COMPILER=clang++ \
                -DLLVM_EXTERNAL_LIT=$HOME/volume/llvm-project/build/bin/llvm-lit \
                -DCMAKE_BUILD_TYPE=Release \
                -DENABLE_PRINT_OPSTATS=OFF
cmake --build build -j$(nproc)

echo "successfully built the Hecate optimizer"
echo "========================================="

# cd $HECATE/python/hecate
# python setup.py sdist --format=tar
# pip uninstall -y hecate
# pip install dist/hecate-0.0.1.tar

# cd $HECATE/python/poly
# python setup.py sdist --format=tar
# pip uninstall -y poly
# pip install dist/poly-0.0.1.tar

# cd $HECATE/python/hetorch
# python setup.py sdist --format=tar
# pip uninstall -y hetorch
# pip install dist/hetorch-0.0.1.tar

echo "successfully installed the Hecate"
echo "========================================="
cd

echo "Successfully installed the Hecate Python packages"
echo "=========================================="
echo "Setup complete! Available commands:"
echo ""
echo "Workflow commands:"
echo "  hc-trace        - Trace benchmarks (requires --opt flag)"
echo "  hc-opt          - Run optimizer on traced files (requires --opt flag)"
echo "  hc-test         - Test optimized files (requires --opt flag)"
echo "  hc-tot          - Run all three steps (trace, opt, test)"
echo "  hc-eval         - Run evaluation with logging"
echo ""
echo "Archived (legacy) commands:"
echo "  hc-trace        - Run archived benchmarks (without --opt flag)"
echo "  hc-test         - Run archived tests (without --opt flag)"
echo "  Note: These are automatically called by hc-trace/hc-opt/hc-test when --opt is NOT provided"
echo ""
echo "Example usage:"
echo "  Default flags: --opt dacapo --waterline 40 --library HEONGPU --hardware GPU --epochs 1"
echo "  hc-trace MLP --opt dacapo"
echo "  hc-opt MLP --opt dacapo"
echo "  hc-test MLP --opt dacapo"
echo "  hc-tot MLP --opt dacapo"
echo "  hc-eval MLP --opt dacapo"
echo ""
echo "Archived examples:"
echo "  hc-trace ResNet [additional-args]"
echo "  hbt dacapo 40 ResNet HEONGPU GPU [additional-args]"
echo "  hc-test dacapo 40 ResNet HEONGPU GPU [additional-args]"
echo ""
echo "=========================================="
