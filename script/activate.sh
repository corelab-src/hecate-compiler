scriptPATH=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )
deployPATH=${scriptPATH}/../..
currentPATH=$PWD
installPATH=${deployPATH}/install

echo -e "\033[1;32m======Activate python3.10 and config======\033[0m"
cd ${scriptPATH}/../
source .venv/bin/activate
source config.sh

cd ${currentPATH}
