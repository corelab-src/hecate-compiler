scriptPATH=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )
deployPATH=${scriptPATH}/../..
currentPATH=$PWD

echo -e "\033[1;32m======Clone llvm-project git======\033[0m"
git clone https://github.com/llvm/llvm-project.git ${deployPATH}/llvm-project
cd ${deployPATH}/llvm-project
git checkout llvmorg-18.1.2
cd ${deployPATH}

echo -e "\033[1;32m======Clone SEAL git project======\033[0m"
git clone https://github.com/microsoft/SEAL.git ${deployPATH}/SEAL
cd ${deployPATH}/SEAL
git checkout 4.0.0
cd ${deployPATH}

echo -e "\033[1;32m======Clone HEonGPU git project======\033[0m"
git clone https://github.com/corelab-src/HEonGPU.git ${deployPATH}/HEonGPU
cd ${deployPATH}/HEonGPU
git checkout corelab
git submodule update --init --recursive
cd ${deployPATH}

cd ${currentPATH}

