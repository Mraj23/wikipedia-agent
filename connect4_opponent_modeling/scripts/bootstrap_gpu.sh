#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PONS_REPO_URL="${PONS_REPO_URL:-https://github.com/PascalPons/connect4.git}"
PONS_BOOK_URL="${PONS_BOOK_URL:-https://github.com/PascalPons/connect4/releases/download/book/7x6.book}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}"

log() {
  echo "[bootstrap] $*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

maybe_install_apt_deps() {
  if [[ "$INSTALL_APT_DEPS" != "1" ]]; then
    return
  fi

  if ! need_cmd apt-get; then
    return
  fi

  local missing=0
  for cmd in git make g++ wget; do
    if ! need_cmd "$cmd"; then
      missing=1
      break
    fi
  done

  if [[ "$missing" == "0" ]]; then
    return
  fi

  local sudo_cmd=""
  if [[ "$(id -u)" != "0" ]]; then
    if need_cmd sudo; then
      sudo_cmd="sudo"
    else
      log "Missing build dependencies and no sudo available. Install: build-essential git wget"
      exit 1
    fi
  fi

  log "Installing system dependencies with apt-get"
  ${sudo_cmd} apt-get update
  DEBIAN_FRONTEND=noninteractive ${sudo_cmd} apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    wget
}

create_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  log "Upgrading pip tooling"
  python -m pip install --upgrade pip setuptools wheel
}

install_python_deps() {
  log "Installing Python dependencies"
  python -m pip install -r "$ROOT_DIR/requirements-gpu.txt"
  python -m pip install -e "$ROOT_DIR"
}

download_book() {
  if [[ -f "$ROOT_DIR/7x6.book" ]]; then
    log "Opening book already present"
    return
  fi

  log "Downloading Pascal Pons opening book"
  wget -O "$ROOT_DIR/7x6.book" "$PONS_BOOK_URL"
}

build_solver() {
  if [[ -x "$ROOT_DIR/connect4_solver" ]]; then
    log "Solver binary already present"
    return
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT

  log "Cloning Pascal Pons solver source"
  git clone --depth 1 "$PONS_REPO_URL" "$tmp_dir/connect4"

  log "Building solver"
  make -C "$tmp_dir/connect4" c4solver
  cp "$tmp_dir/connect4/c4solver" "$ROOT_DIR/connect4_solver"
  chmod +x "$ROOT_DIR/connect4_solver"
}

main() {
  log "Project root: $ROOT_DIR"
  maybe_install_apt_deps
  create_venv

  # shellcheck disable=SC1091
  source "$ROOT_DIR/scripts/gpu_env.sh"
  install_python_deps
  download_book
  build_solver

  log "Running setup verification"
  python "$ROOT_DIR/scripts/verify_setup.py" --expect-gpu --expect-vllm --expect-wandb

  cat <<EOF

Bootstrap complete.

Next commands:
  source "$VENV_DIR/bin/activate"
  source "$ROOT_DIR/scripts/gpu_env.sh"
  bash "$ROOT_DIR/scripts/run_preliminary.sh"
EOF
}

main "$@"
