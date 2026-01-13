#!/usr/bin/env bash
set -e

# test edit
SWEEP_ID="test_trebi/full_sweep"
MACHINES=(
  erinys03
  erinys04
  erinys05
  erinys06
)

for HOST in "${MACHINES[@]}"; do
  echo "Deploying to $HOST"

  ssh "$HOST" bash -c "'
    set -e

    cd ~/trebi

    make up

    echo \"Starting\"
      docker exec -d --env CUDA_VISIBLE_DEVICES=\$GPU trebi \
	wandb agent $SWEEP_ID &
    done
    wait
  '"
done
