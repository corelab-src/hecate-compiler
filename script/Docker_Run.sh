containername=hecate
imageName=hecate-compiler

function dockerrun_f() {
  docker run -it --gpus all --name ${containername} --hostname deploydocker --volume ${scriptPATH}/../../:/home/$USER/volume:rw ${imageName}:latest
}

scriptPATH=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )

echo -e "\033[1;32m======Run ${containername} container======\033[0m"
CID=$(docker ps -q -f status=running -f name=^/${containername}$)
CID2=$(docker ps -a -q -f name=^/${containername}$)
if [ ${CID} ]; then
  echo -e "\033[1;31m===${containername} already exist and is running===\033[0m"
  while true; do
    read -p "Reset? (y/N), if you want execute, press 'S' : " yn
    case $yn in
      [Yy]* ) docker stop ${containername}; docker rm ${containername}; dockerrun_f; break;;
      [Ss]* ) docker exec -it ${containername} bash; break;;
      * ) exit;;
    esac
  done
elif [ ${CID2} ]; then
  echo -e "\033[1;31m===${containername} already exist===\033[0m"
  while true; do
    read -p "Reset? (y/N), if you want execute, press 'S' : " yn
    case $yn in
      [Yy]* ) docker rm ${containername}; dockerrun_f; break;;
      [Ss]* ) docker start ${containername}; docker exec -it ${containername} bash; break;;
      * ) exit;;
    esac
  done
else
  dockerrun_f
fi
