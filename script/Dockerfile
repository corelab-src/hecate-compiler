#FROM ubuntu:22.04
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN apt-get update
RUN apt-get -y install sudo
RUN apt-get -y install ninja-build
RUN apt-get -y install python3 python3-dev python3-pip
RUN apt-get -y install python3.10-venv
RUN apt-get -y install cmake
RUN apt-get -y install libstdc++-12-dev libomp-dev
RUN apt-get -y install clang-format clang-14 clang++-14
RUN apt-get -y install clang
#RUN apt-get -y install clang-format clang-14 clang++-14

RUN apt-get -y install vim git

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
ENV CMAKE_GENERATOR=Ninja
WORKDIR $HOME
CMD /bin/bash

