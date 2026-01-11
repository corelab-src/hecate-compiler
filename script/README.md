# Hecate Compiler Docker Scripts

This folder contains Docker-related scripts for Hecate Compiler development and CI/CD.

## 🐳 Quick Start

### 1. Build Docker Image

```bash
# From anywhere (project root recommended)
./script/dockerbuild.sh
```

Custom image name/tag:
```bash
IMAGE_NAME=my-hecate IMAGE_TAG=v1.0 ./script/dockerbuild.sh
```

### 2. Run Container (Local Development)

```bash
docker run --gpus all -it \
  -e HOST_USERNAME=$(whoami) \
  -e HOST_UID=$(id -u) \
  -e HOST_GID=$(id -g) \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --name hecate \
  -v "$(pwd):/home/$(whoami)/hecate-compiler" \
  hecate-compiler:latest
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `Dockerfile` | Docker image definition (base: `nvcr.io/nvidia/pytorch:25.06-py3`) |
| `entrypoint.sh` | Runtime user creation script |
| `dockerbuild.sh` | Image build script |

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

## 🔄 CI/CD vs Local Development

### CI/CD Mode
Run without environment variables to use default user (`hecate:1000:1000`):
```bash
docker run --gpus all -it hecate-compiler:latest
```

### Local Development Mode
Pass `HOST_*` environment variables to match your host user:
```bash
docker run --gpus all -it \
  -e HOST_USERNAME=$(whoami) \
  -e HOST_UID=$(id -u) \
  -e HOST_GID=$(id -g) \
  -v "/path/to/project:/home/$(whoami)/project" \
  hecate-compiler:latest
```

Benefits of this approach:
- ✅ No volume permission issues
- ✅ One image works for both local and CI/CD
- ✅ Team members can share the same image

---

## 🔒 Security Notes

The `HOST_UID`/`HOST_GID` approach is **safe for development** because:

1. **No privilege escalation**: The container user only has the same permissions as your host user
2. **Volume isolation**: Only mounted directories are accessible
3. **Passwordless sudo**: Only affects the container, not the host system
4. **Runtime user creation**: User info is not baked into the image

> **Production recommendation**: For production deployments, consider using fixed non-root users and proper orchestration (Kubernetes securityContext, etc.)
