FROM nvcr.io/nvidia/pytorch:25.12-py3

# Install additional dependencies
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        sudo \
        clang-format clang-18 clang++-18 clang \
        libstdc++-13-dev libssl-dev libgmp-dev libomp-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

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

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]