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

# Check GHCR PAT argument
if [ -z "$1" ]; then
  echo "Usage: $0 <GHCR_PAT>"
  exit 1
fi

GHCR_PAT="$1"
SWEEP_IP="$2"

for HOST in "${MACHINES[@]}"; do
  echo "🚀 Deploying to $HOST"

  ssh "$HOST" bash -s << EOF
set -e

if [ ! -d /var/data/xuncovsk_trebi ]; then
    mkdir /var/data/xuncovsk_trebi
fi

cd /var/data/xuncovsk_trebi

if [ ! -d /var/data/xuncovsk_trebi/trebi ]; then
  git clone git@github.com:uncovsky/trebi.git
  cd trebi
else
  cd trebi
  git pull
fi

# Login to GHCR
echo "$GHCR_PAT" | docker login ghcr.io -u uncovsky --password-stdin


# Pull image and start container
make pull
make down || true
make up

# Start two W&B agents inside container
docker exec -d trebi wandb agent $SWEEP_ID &
docker exec -d trebi wandb agent $SWEEP_ID &

EOF

done
