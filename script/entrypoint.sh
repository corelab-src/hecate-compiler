#!/bin/bash
set -euo pipefail

if [ -z "${HOST_UID:-}" ] && [ -z "${HOST_GID:-}" ] && [ -z "${HOST_USERNAME:-}" ]; then
  exec "${@:-/bin/bash}"
fi

USERNAME=${HOST_USERNAME:-${DEFAULT_USERNAME:-hecate}}
USER_UID=${HOST_UID:-${DEFAULT_UID:-1000}}
USER_GID=${HOST_GID:-${DEFAULT_GID:-1000}}

if [ "$(id -u)" -ne 0 ]; then
  echo "[entrypoint] Not root; cannot create users. Running as current user: $(id -un)"
  exec "${@:-/bin/bash}"
fi

# Create group if it doesn't exist
if ! getent group | awk -F: -v gid="$USER_GID" '$3==gid{found=1} END{exit !found}'; then
    groupadd -g "$USER_GID" "$USERNAME"
fi

# Create user if it doesn't exist
existing_user_by_uid="$(getent passwd "$USER_UID" | cut -d: -f1 || true)"

if [ -n "$existing_user_by_uid" ]; then
  echo "[entrypoint] UID $USER_UID already exists as '$existing_user_by_uid'. Using it."
  USERNAME="$existing_user_by_uid"
else
  if ! id -u "$USERNAME" >/dev/null 2>&1; then
    useradd -M -s /bin/bash -u "$USER_UID" -g "$USER_GID" "$USERNAME"
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/"$USERNAME"
    chmod 440 /etc/sudoers.d/"$USERNAME"
  fi
fi

# Setup home directory
HOME_DIR="/home/$USERNAME"
if [ ! -d "$HOME_DIR" ]; then
  mkdir -p "$HOME_DIR"
fi
chown "$USER_UID:$USER_GID" "$HOME_DIR"

if [ ! -f "$HOME_DIR/.bashrc" ]; then
  cp /etc/skel/.bashrc "$HOME_DIR/"
  chown "$USER_UID:$USER_GID" "$HOME_DIR/.bashrc"
fi

if [ ! -f "$HOME_DIR/.profile" ]; then
  cp /etc/skel/.profile "$HOME_DIR/"
  chown "$USER_UID:$USER_GID" "$HOME_DIR/.profile"
fi


# Set working directory
cd "$HOME_DIR"

# Execute command as the user
if [ $# -eq 0 ]; then
    exec runuser -u "$USERNAME" -- bash --login
else
    exec runuser -u "$USERNAME" -- "$@"
fi
