## Docker Installation

The easiest way to get started is using Docker.

#### Build Docker Image
```bash
cd <Project_root>
./script/Docker_Build.sh

# Or with custom image name/tag
cd <Project_root>
IMAGE_NAME=hecate IMAGE_TAG=v1.0 TARGET=cuda13 ./script/Docker_Build.sh
```

#### Run Container
```bash
cd <Project_root>
docker run --gpus all -it \
  -e HOST_USERNAME=$(whoami) \
  -e HOST_UID=$(id -u) \
  -e HOST_GID=$(id -g) \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --name hecate \
  -v "$(pwd)/../:/home/$(whoami)/volume/" \
  hecate-compiler:latest
```

> **Note**: The `HOST_*` environment variables map your host user to the container, solving volume permission issues.

---

## 📁 Files

| File | Description |
|------|-------------|
| `Dockerfile` | Docker image definition |
| `entrypoint.sh` | Runtime user creation script |
| `Docker_Build.sh` | Image build script |

---

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST_USERNAME` | Username inside container | `hecate` |
| `HOST_UID` | User UID | `1000` |
| `HOST_GID` | User GID | `1000` |
| `IMAGE_NAME` | Docker image name | `hecate-compiler` |
| `IMAGE_TAG` | Docker image tag | `latest` |

---

