#!/usr/bin/env bash
set -e

# === CONFIG ===
SWEEP_ID="test_trebi/full_sweep"
MACHINES=(
  erinys03
  erinys04
  erinys05
  erinys06
)
DOCKER_USER="ghcr.io/uncovsky"
IMAGE_NAME="trebi"
TAG="latest"
APP_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"

# GPUs per machine (adjust as needed)
GPUS=(0 1 2 3)

if [ -z "$1" ]; then
  echo "Usage: $0 <GHCR_PAT>"
  exit 1
fi
GHCR_PAT="$1"

for HOST in "${MACHINES[@]}"; do
  echo "🚀 Deploying to $HOST"

  ssh "$HOST" bash -c "'
    set -e

    echo $GHCR_PAT | docker login ghcr.io -u uncovsky --password-stdin

    cd ~/trebi

    make pull
    make down || true
    make up &

    sleep 5  # wait for container to start

    for GPU in ${GPUS[@]}; do
      docker exec -d --env CUDA_VISIBLE_DEVICES=\$GPU trebi \
        wandb agent $SWEEP_ID &
    done
    wait
  '"
done
