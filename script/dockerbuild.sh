scriptPATH=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )
imageName=hedl_deploy_gpu


echo -e "\033[1;32m======Build ${imageName} container======\033[0m"
docker build --build-arg username_=${USER} --build-arg uid_=$(id -u) --build-arg gid_=$(id -g) -t ${imageName} ${scriptPATH}
