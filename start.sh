#!/bin/bash

# runner command: "source start.sh"

# Set the working directory to the location of this script
export HOME=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )
export HECATE=$HOME/hecate-compiler

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
cmake -S . -B build -DMLIR_ROOT=$HOME/models/install/ \
                -DHEonGPU_ROOT=$HOME/install/HEonGPU \
                -DCMAKE_C_COMPILER=clang \
                -DCMAKE_CXX_COMPILER=clang++ \
                -DLLVM_EXTERNAL_LIT=$HOME/models/install/bin/llvm-lit \
                -DCMAKE_BUILD_TYPE=Release

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
pip uninstall -y hetorch 2>/dev/null || true
pip install -e .

cd

echo "Successfully installed the Hecate Python packages"
echo "=========================================="
echo "Setup complete! Available commands:"
echo ""
echo "Workflow commands:"
echo "  hc-trace        - Trace benchmarks (default: --opt dacapo)"
echo "  hc-opt          - Run optimizer on traced files (default: --opt dacapo)"
echo "  hc-test         - Test optimized files (default: --opt dacapo)"
echo "  hc-tot          - Run all three steps (trace, opt, test)"
echo "  hc-eval         - Run evaluation with logging"
echo ""
echo "Archived (legacy) commands:"
echo "  hc-trace        - Run archived benchmarks"
echo "  hc-test         - Run archived tests"
echo "  Note: These work seamlessly with defaults."
echo ""
echo "Example usage:"
echo "  Default flags: --opt dacapo --waterline 40 --library HEONGPU --hardware GPU --epochs 1"
echo "  hc-trace MLP"
echo "  hc-opt MLP"
echo "  hc-test MLP"
echo "  hc-all MLP"
echo "  hc-eval MLP"
echo "  hc-trace MLP --opt dacapo  # Explicitly specifying opt"
echo ""
echo "Archived examples:"
echo "  hc-trace ResNet [additional-args]"
echo "  hbt dacapo 40 ResNet HEONGPU GPU [additional-args]"
echo "  hc-test dacapo 40 ResNet HEONGPU GPU [additional-args]"
echo ""
echo "=========================================="
