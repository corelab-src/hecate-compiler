#!/bin/bash
scriptPATH=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )
deployPATH=${scriptPATH}/../..
currentPATH=$PWD
installPATH=${deployPATH}/install

function build_llvm(){
  echo -e "\033[1;32m======Build llvm-project======\033[0m"
  cd ${deployPATH}/llvm-project
  cmake -GNinja -Bbuild \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DCMAKE_INSTALL_PREFIX=${installPATH}/MLIR \
    llvm
  cmake --build build
  cmake --install build
}

function build_SEAL(){
  echo -e "\n\033[1;32m======Build SEAL======\033[0m"
  cd ${deployPATH}/SEAL
  cmake -S . -B build -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX=${installPATH}/SEAL
  cmake --build build
  cmake --install build
}

function build_HEaaN(){
  echo -e "\n\033[1;32m======Build HEaaN======\033[0m"
  cd ${deployPATH}/HEaaN
  cmake -Bbuild -DBUILD_WITH_CUDA=ON \
    -DBUILD_EXAMPLES=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUILD_DOCUMENTATION=OFF \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX=${installPATH}/heaan
  cd build
  ninja
  ninja install
}

function build_HEonGPU(){
  echo -e "\n\033[1;32m======Build HEonGPU======\033[0m"
  if [ -z "$2" ]; then
    echo 'Please insert correct argument number for GPU Architecture'
    echo 'Possible GPU Architecture and argument number.'
    echo 'Volta = 70, 72'
    echo 'Turing = 75'
    echo 'Ampere = 80, 86'
    echo 'Ada = 89, 90'
    echo 'ex) ./build.sh heon 86'
    echo 'Build again from HEonGPU.'
    exit 1
  else
    echo "Build as $2 Architecture."
    cd ${deployPATH}/HEonGPU
    cmake -S . -D CMAKE_CUDA_ARCHITECTURES=$2 \
      -D CMAKE_INSTALL_PREFIX=${installPATH}/HEON \
      -D HEonGPU_BUILD_BENCHMARKS=ON \
      -D HEonGPU_BUILD_TESTS=ON \
      -D HEonGPU_BUILD_EXAMPLES=ON \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -B build
    cmake --build ./build/
    cmake --install build
  fi
}

function build_Hecate(){
  cd ${scriptPATH}/../
  echo -e "\n\033[1;32m======Build Hecate-Compiler======\033[0m"
  cmake -S . -B build -DMLIR_ROOT=${installPATH}/MLIR \
    -DSEAL_ROOT=${installPATH}/SEAL \
    -DHEonGPU_ROOT=${installPATH}/HEON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_FLAGS="-w"
  cmake --build build
}

function check_Hecate(){
  cd ${scriptPATH}/../
  echo -e "\n\033[1;32m======Check Hecate-Compiler======\033[0m"
  cmake -S . -B build -DLLVM_EXTERNAL_LIT=${deployPATH}/llvm-project/build/bin/llvm-lit
  #cmake -S . -B build -DMLIR_ROOT=${installPATH}/MLIR \
  #  -DSEAL_ROOT=${installPATH}/SEAL \
  #  -DHEaaN_ROOT=${installPATH}/heaan \
  #  -DHEonGPU_ROOT=${installPATH}/HEON \
  #  -DCMAKE_C_COMPILER=clang \
  #  -DCMAKE_CXX_COMPILER=clang++ \
  #  -DLLVM_EXTERNAL_LIT=${deployPATH}/llvm-project/build/bin/llvm-lit
  cmake --build build --target check-hecate
}

function set_ENV(){
  cd ${scriptPATH}/../
  echo -e "\n\033[1;32m======Set Python3 environment======\033[0m"
  python3 -m venv .venv
  source .venv/bin/activate
  source config.sh

  pip install -r requirements.txt
  ./install.sh
}


if [ $1 == 'llvm' ];then
  build_llvm
elif [ $1 == 'seal' ];then
  build_SEAL
elif [ $1 == 'heaan' ];then
  build_HEaaN
elif [ $1 == 'heon' ];then
  build_HEonGPU "$@"
elif [ $1 == 'hecate' ];then
  build_Hecate
elif [ $1 == 'check' ];then
  check_Hecate
elif [ $1 == 'env' ];then
  set_ENV
elif [ $1 == 'all' ];then
  build_llvm
  build_SEAL
  build_HEaaN
  build_HEonGPU "$@"
  build_Hecate
  set_ENV
else
  echo "possible command : llvm, seal, heaan, heon, hecate, check, env, \033[1;31mall\033[0m"
fi
cd ${currentPATH}
