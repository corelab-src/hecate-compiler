scriptPATH=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )
deployPATH=${scriptPATH}/../..
currentPATH=$PWD

echo -e "\033[1;32m======Download Models======\033[0m"
cd ${scriptPATH}/../examples/data/
wget http://corelab.or.kr/~corelab/data/ResNet20_ReLU.pt
wget http://corelab.or.kr/~corelab/data/ResNet20_SiLU.pt
wget http://corelab.or.kr/~corelab/data/AlexNet_ReLU_MaxPool.pt
wget http://corelab.or.kr/~corelab/data/AlexNet_SiLU_AvgPool.pt
wget http://corelab.or.kr/~corelab/data/MobileNet_ReLU.pt
wget http://corelab.or.kr/~corelab/data/MobileNet_SiLU.pt
wget http://corelab.or.kr/~corelab/data/SqueezeNet_ReLU_MaxPool.pt
wget http://corelab.or.kr/~corelab/data/SqueezeNet_SiLU_AvgPool.pt
wget http://corelab.or.kr/~corelab/data/VGG16_ReLU_MaxPool.pt
wget http://corelab.or.kr/~corelab/data/VGG16_SiLU_AvgPool.pt

cd ${currentPATH}

