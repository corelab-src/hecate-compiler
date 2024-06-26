scriptPATH=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )
deployPATH=${scriptPATH}/../..
currentPATH=$PWD

echo -e "\033[1;32m======Clone llvm-project git======\033[0m"
git clone https://github.com/llvm/llvm-project.git ${deployPATH}/llvm-project
cd ${deployPATH}/llvm-project
git checkout llvmorg-18.1.2

# echo -e "\033[1;32m======Clone SEAL git project======\033[0m"
# git clone https://github.com/microsoft/SEAL.git ${deployPATH}/SEAL
# cd ${deployPATH}/SEAL
# git checkout 4.0.0


echo -e "\033[1;32m======Clone HEaaN git project======\033[0m"
echo -e "\033[1;31m===Need to Download the HEaaN library======\033[0m"

#echo -e "\033[1;32m======Clone OpenFHE git project======\033[0m"
#git clone https://github.com/openfheorg/openfhe-development.git ${deployPATH}/openfhe-development
#cd ${deployPATH}/openfhe-development
#git checkout v1.1.1

# cd ${scriptPATH}
# echo -e "\033[1;32m======Build Docker Image======\033[0m"
# ${scriptPATH}/dockerbuild.sh

cd ${currentPATH}

