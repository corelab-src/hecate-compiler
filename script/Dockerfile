#FROM ubuntu:22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update
RUN apt-get -y install sudo
#RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get -y install ninja-build
RUN apt-get -y install python3 python3-dev python3-pip
RUN apt-get -y install python3.10-venv
#RUN apt-get -y install cmake
RUN apt-get -y install libstdc++-12-dev libomp-dev
RUN apt-get -y install clang-format clang-14 clang++-14
RUN apt-get -y install clang

RUN apt-get -y install vim git wget
RUN apt-get -y install libssl-dev libgmp-dev
RUN wget https://cmake.org/files/v3.30/cmake-3.30.5.tar.gz
RUN tar -xf cmake-3.30.5.tar.gz
RUN cd cmake-3.30.5 && ./bootstrap --prefix=/usr/local
RUN cd cmake-3.30.5 && make && make install

ARG username_
ENV username=${username_}
ARG uid_
ENV uid=${uid_}
ARG gid_
ENV gid=${gid_}
# replace 1000 with your user / group id
#run export uid=1000 gid=1000 && \
run mkdir -p /home/${username} && \
    echo "${username}:x:${uid}:${gid}:${username},,,:/home/${username}:/bin/bash" >> /etc/passwd && \
    echo "${username}:x:${uid}:" >> /etc/group && \
    echo "${username}:${username}" | chpasswd && \
    usermod -aG sudo  ${username} && \
    echo "${username} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${username} && \
    chmod 0440 /etc/sudoers.d/${username} && \
    chown ${uid}:${gid} -R /home/${username}

USER $username_
ENV HOME /home/$username_
ENV CC="clang"
ENV CXX="clang++"
ENV PATH="/usr/lib/ccache:/usr/local/cuda/bin:/usr/local/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin:~/.local/bin"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
ENV CMAKE_GENERATOR=Ninja
WORKDIR $HOME
CMD /bin/bash

