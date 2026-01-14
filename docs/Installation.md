## Installation
### Requirements 
```
Ninja   
git  
cmake >= 3.22.1  
python >= 3.10  
clang,clang++ >= 14.0.0  
```

### Install MLIR 
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-18.1.2
cmake -GNinja -Bbuild \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_TARGETS_TO_BUILD=host -DCMAKE_INSTALL_PREFIX=<MLIR_INSTALL>\
  llvm
cmake --build build
sudo cmake --install build
cd .. 
```
### Install SEAL 
```bash
git clone https://github.com/microsoft/SEAL.git
cd SEAL
git checkout 4.0.0
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<SEAL_INSTALL>
cmake --build build
sudo cmake --install build
cd .. 
```
### Install HEonGPU 
CMAKE_CUDA_ARCHITECTURES: Ampere (86), Ada (89), Hopper (90), Blackwell (120)
```bash
git clone https://github.com/corelab-src/HEonGPU.git
git checkout corelab
cd HEonGPU
cmake -S . -B build -D CMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_INSTALL_PREFIX=<HEonGPU_INSTALL>
cmake --build build
sudo cmake --install build
cd .. 
```

### Build Hecate 
```bash
git clone <this-repository>
cd <this-repository>
cmake -S . -B build \
  -DMLIR_ROOT=<MLIR_INSTALL> -DSEAL_ROOT=<SEAL_INSTALL> \
  -DHEonGPU_ROOT=<HEonGPU_INSTALL> -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
cmake --build build 
```
### Configure Hecate 
```bash
python3 -m venv .venv
# if use nvidia-docker, use this command
python3 -m venv --system-site-packages .venv

source .venv/bin/activate
source config.sh 
```

### Install Hecate Python Binding 
```bash
pip install -r requirements.txt
./install.sh
```

