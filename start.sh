#!/bin/bash

# runner command: "source start.sh"

# Set the working directory to the location of this script
export HOME=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )
export HECATE=$HOME/volume/hecate-compiler
export CC="clang"
export CXX="clang++"

echo "HECATE DIR: $HECATE"

# Change to HECATE directory
cd "$HECATE" || { echo "Failed to change directory to hecate-compiler"; exit 1; }

# Activate the virtual environment
echo "=========================================="
echo "Activating the virtual environment"
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found at .venv"
    exit 1
fi
source .venv/bin/activate

# Load configuration (creates directories and defines functions)
source config.sh

# Build the Hecate optimizer
echo "=========================================="
echo "Building the Hecate optimizer"
cmake -S . -B build -DMLIR_ROOT=$HOME/install/MLIR \
                -DSEAL_ROOT=$HOME/install/SEAL \
                -DHEaaN_ROOT=$HOME/install/heaan \
                -DHEonGPU_ROOT=$HOME/install/HEonGPU \
                -DCMAKE_C_COMPILER=clang \
                -DCMAKE_CXX_COMPILER=clang++ \
                -DLLVM_EXTERNAL_LIT=$HOME/volume/llvm-project/build/bin/llvm-lit \
                -DCMAKE_BUILD_TYPE=Release \
                -DENABLE_PRINT_OPSTATS=ON

cmake --build build -j$(nproc)

echo "Successfully built the Hecate optimizer"
echo "=========================================="

# Install Python packages
echo "Installing Hecate Python packages"

cd $HECATE/python/hecate
python setup.py sdist --format=tar
pip uninstall -y hecate 2>/dev/null || true
pip install dist/hecate-0.0.1.tar

cd $HECATE/python/poly
python setup.py sdist --format=tar
pip uninstall -y poly 2>/dev/null || true
pip install dist/poly-0.0.1.tar

cd $HECATE/python/hetorch
python setup.py sdist --format=tar
pip uninstall -y hetorch 2>/dev/null || true
pip install dist/hetorch-0.0.1.tar

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
