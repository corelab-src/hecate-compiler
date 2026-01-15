################################################
# Cuda 12 ver.
################################################
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS cuda12
RUN apt-get update
RUN apt-get -y install sudo
#RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get -y install ninja-build vim git wget
RUN apt-get -y install python3 python3-dev python3-pip python3.10-venv
RUN apt-get -y install libstdc++-12-dev libssl-dev libgmp-dev libomp-dev libntl-dev
RUN apt-get -y install clang-format clang-14 clang++-14 clang

RUN wget https://cmake.org/files/v3.30/cmake-3.30.5.tar.gz
RUN tar -xf cmake-3.30.5.tar.gz
RUN cd cmake-3.30.5 && ./bootstrap --prefix=/usr/local
RUN cd cmake-3.30.5 && make && make install

COPY entrypoint.sh /opt/nvidia/entrypoint.d/99-custom-entrypoint.sh
RUN chmod +x /opt/nvidia/entrypoint.d/99-custom-entrypoint.sh

ENV CC="clang"
ENV CXX="clang++"
#ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
#RUN echo 'PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/lib/ccache:/usr/local/cuda/bin"' > /etc/environment
ENV CMAKE_GENERATOR=Ninja

ENV DEFAULT_USERNAME=hecate
ENV DEFAULT_UID=1000
ENV DEFAULT_GID=1000

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["/bin/bash"]

################################################
# Cuda 13 ver.
################################################
FROM nvcr.io/nvidia/pytorch:25.12-py3 AS cuda13

# Install additional dependencies
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        sudo build-essential \
        clang-format clang-18 clang++-18 clang \
        libstdc++-13-dev libssl-dev libgmp-dev libomp-dev

RUN test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
RUN apt-get update
RUN apt-get -y install cmake

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy entrypoint script
COPY entrypoint.sh /opt/nvidia/entrypoint.d/99-custom-entrypoint.sh
RUN chmod +x /opt/nvidia/entrypoint.d/99-custom-entrypoint.sh

# Environment variables
ENV CC="clang"
ENV CXX="clang++"
ENV PATH="/usr/lib/ccache:/usr/local/cuda/bin:/usr/local/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
ENV CMAKE_GENERATOR=Ninja

# Default user settings (used when HOST_* env vars are not provided)
ENV DEFAULT_USERNAME=hecate
ENV DEFAULT_UID=1000
ENV DEFAULT_GID=1000

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["/bin/bash"]
