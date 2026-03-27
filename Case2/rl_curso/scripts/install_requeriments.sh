#!/usr/bin/env bash
set -euo pipefail

# install_requirements.sh
# 
# Run:
#  $ sudo chmod +x install_requirements.sh
#  $ ./install_requirements.sh
#
# Env vars:
#   ROS_DISTRO=humble
#   VENV_DIR=$HOME/venvs/robot_nav_rl
#   SKIP_UPGRADE=1
#   TORCH_CPU_ONLY=1
#   FORCE_ROS_INSTALL=1      # install ROS/Gazebo even if detected
#   FORCE_VENV_RECREATE=1    # delete and recreate venv

ROS_DISTRO="${ROS_DISTRO:-humble}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/robot_nav_rl}"
SKIP_UPGRADE="${SKIP_UPGRADE:-0}"
TORCH_CPU_ONLY="${TORCH_CPU_ONLY:-0}"
FORCE_ROS_INSTALL="${FORCE_ROS_INSTALL:-0}"
FORCE_VENV_RECREATE="${FORCE_VENV_RECREATE:-0}"

log()  { echo -e "\n\033[1;32m[install]\033[0m $*"; }
warn() { echo -e "\n\033[1;33m[warn]\033[0m $*"; }
die()  { echo -e "\n\033[1;31m[error]\033[0m $*" >&2; exit 1; }

if [[ "$(id -u)" -eq 0 ]]; then
  die "Não rode como root. Rode como usuário normal com sudo."
fi

have_pkg() { dpkg -s "$1" >/dev/null 2>&1; }

ROS_INSTALLED=0
GAZEBO_INSTALLED=0

if have_pkg "ros-${ROS_DISTRO}-desktop" || have_pkg "ros-${ROS_DISTRO}-ros-base"; then
  ROS_INSTALLED=1
fi
if have_pkg "ros-${ROS_DISTRO}-gazebo-ros-pkgs" && have_pkg "gazebo"; then
  GAZEBO_INSTALLED=1
fi

log "Detecção: ROS_INSTALLED=${ROS_INSTALLED}, GAZEBO_INSTALLED=${GAZEBO_INSTALLED}, FORCE_ROS_INSTALL=${FORCE_ROS_INSTALL}"

log "Base tools (curl + python venv)..."
sudo apt update
sudo apt install -y \
  curl ca-certificates gnupg2 lsb-release locales software-properties-common \
  git wget unzip build-essential cmake pkg-config \
  python3-pip python3-venv python3-dev

log "Locale en_US.UTF-8..."
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

if [[ "${FORCE_ROS_INSTALL}" == "1" || "${ROS_INSTALLED}" == "0" || "${GAZEBO_INSTALLED}" == "0" ]]; then
  log "Enable universe (ROS precisa)..."
  sudo add-apt-repository -y universe

  log "Configurando fonte ROS 2 (sem GitHub)..."
  ROS_REPO_PRESENT=0
  if grep -Rqs "packages\.ros\.org/ros2" /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null; then
    ROS_REPO_PRESENT=1
  fi

  if [[ "${ROS_REPO_PRESENT}" -eq 1 ]]; then
    log "Repo ROS já configurado."
    sudo apt update
    sudo apt install -y ros2-apt-source || true
  else
    warn "Repo ROS não encontrado. Tentando instalar ros2-apt-source via apt..."
    sudo apt update
    if ! sudo apt install -y ros2-apt-source; then
      cat <<EOF

[error] Não consegui instalar ros2-apt-source via apt.
Configure o repositório ROS manualmente conforme a doc:
https://docs.ros.org/en/${ROS_DISTRO}/Installation/Ubuntu-Install-Debs.html

Depois rode este script novamente.
EOF
      exit 1
    fi
  fi

  log "apt update..."
  sudo apt update

  if [[ "${SKIP_UPGRADE}" == "0" ]]; then
    log "apt upgrade..."
    sudo apt upgrade -y
  else
    warn "SKIP_UPGRADE=1 (pulando upgrade)."
  fi

  if [[ "${FORCE_ROS_INSTALL}" == "1" || "${ROS_INSTALLED}" == "0" ]]; then
    log "Instalando ROS 2 ${ROS_DISTRO} + tools..."
    sudo apt install -y "ros-${ROS_DISTRO}-desktop" \
      ros-dev-tools python3-colcon-common-extensions python3-rosdep python3-argcomplete \
      "ros-${ROS_DISTRO}-xacro" "ros-${ROS_DISTRO}-robot-state-publisher" "ros-${ROS_DISTRO}-tf2-ros" \
      "ros-${ROS_DISTRO}-teleop-twist-keyboard" "ros-${ROS_DISTRO}-rqt-robot-steering"
  else
    log "ROS 2 já instalado. Pulando instalação ROS."
  fi

  if [[ "${FORCE_ROS_INSTALL}" == "1" || "${GAZEBO_INSTALLED}" == "0" ]]; then
    log "Instalando Gazebo Classic + gazebo_ros_pkgs..."
    sudo apt install -y "ros-${ROS_DISTRO}-gazebo-ros-pkgs" gazebo
  else
    log "Gazebo + gazebo_ros_pkgs já instalados. Pulando instalação Gazebo."
  fi

  log "rosdep init/update..."
  if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init
  else
    warn "rosdep já inicializado."
  fi
  rosdep update
else
  log "ROS + Gazebo detectados. Pulando tudo de ROS/Gazebo. (Só venv RL)"
fi

log "Evitar quebrar colcon: garantir setuptools<80 no user-site (não afeta a venv)..."
python3 -m pip install --user "setuptools==58.2.0" "wheel<0.45" || true

if [[ "${FORCE_VENV_RECREATE}" == "1" && -d "${VENV_DIR}" ]]; then
  warn "FORCE_VENV_RECREATE=1: removendo venv existente em ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

log "Criando/atualizando venv: ${VENV_DIR}"
mkdir -p "$(dirname "${VENV_DIR}")"
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
export PIP_CONFIG_FILE="/dev/null"

log "Upgrade pip tooling na venv..."
python -m pip install --no-cache-dir -U pip setuptools wheel

log "Instalando torch primeiro (evita resolver cuda/dep de formas estranhas)..."
if [[ "${TORCH_CPU_ONLY}" == "1" ]]; then
  log "Torch CPU-only (download menor)..."
  python -m pip install --no-cache-dir -U --index-url https://download.pytorch.org/whl/cpu torch
else
  log "Torch padrão (pode baixar libs CUDA e ser grande)..."
  python -m pip install --no-cache-dir -U torch
fi

log "Instalando deps RL..."
python -m pip install --no-cache-dir -U \
  numpy matplotlib pandas rich tensorboard pyyaml \
  gymnasium stable-baselines3 catkin_pkg tqdm pygame

log "Teste rápido imports..."
python - <<'PY'
import sys
import torch
import gymnasium
import stable_baselines3
import rich
import tensorboard
import yaml
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("Gymnasium:", gymnasium.__version__)
print("SB3:", stable_baselines3.__version__)
print("TensorBoard:", tensorboard.__version__)
PY


log "OK."
log "Para RL:  source '${VENV_DIR}/bin/activate'"
log "Para ROS: source /opt/ros/${ROS_DISTRO}/setup.bash"

log "Atualizando ~/.bashrc com ROS/Gazebo + alias da venv..."

BASHRC="$HOME/.bashrc"
BLOCK_START="# >>> robot_navigation env >>>"
BLOCK_END="# <<< robot_navigation env <<<"

if ! grep -Fq "${BLOCK_START}" "${BASHRC}"; then
  cat >> "${BASHRC}" <<EOF

${BLOCK_START}
# ROS 2
source /opt/ros/${ROS_DISTRO}/setup.bash

# Gazebo Classic
if [ -f /usr/share/gazebo/setup.bash ]; then
  source /usr/share/gazebo/setup.bash
elif [ -f /usr/share/gazebo/setup.sh ]; then
  source /usr/share/gazebo/setup.sh
fi

# Activate RL venv

alias robot-env='source ${VENV_DIR}/bin/activate'
${BLOCK_END}
EOF
else
  warn "Bloco já existe no ~/.bashrc (pulando)."
fi