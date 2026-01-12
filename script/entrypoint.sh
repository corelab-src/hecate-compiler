#!/bin/bash
set -euo pipefail

# Runtime user creation for both local development and CI/CD
# - Local: pass HOST_USERNAME, HOST_UID, HOST_GID as env vars
# - CI/CD: uses default values (hecate:1000:1000)

USERNAME=${HOST_USERNAME:-${DEFAULT_USERNAME:-hecate}}
USER_UID=${HOST_UID:-${DEFAULT_UID:-1000}}
USER_GID=${HOST_GID:-${DEFAULT_GID:-1000}}

# Create group if it doesn't exist
if ! getent group "$USER_GID" >/dev/null 2>&1; then
    groupadd -g "$USER_GID" "$USERNAME" 2>/dev/null || true
fi

# Create user if it doesn't exist
if ! id -u "$USERNAME" >/dev/null 2>&1; then
    useradd -M -s /bin/bash -u "$USER_UID" -g "$USER_GID" "$USERNAME" 2>/dev/null || true
    # Grant passwordless sudo
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/"$USERNAME"
    chmod 440 /etc/sudoers.d/"$USERNAME"
fi

# Setup home directory
HOME_DIR="/home/$USERNAME"
if [ ! -d "$HOME_DIR" ]; then
    mkdir -p "$HOME_DIR"
fi
chown "$USER_UID:$USER_GID" "$HOME_DIR"

# Set working directory
cd "$HOME_DIR"

# Execute command as the user
if [ $# -eq 0 ]; then
    exec runuser -u "$USERNAME" -- bash --login
else
    exec runuser -u "$USERNAME" -- "$@"
fi
