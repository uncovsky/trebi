import os
from pathlib import Path


# Important: set the dataset path before importing dsrl
data_path = Path("/data")
os.environ["DSRL_DATASET_DIR"] = str(data_path)

import gymnasium as gym
import dsrl

def main():
    print(f"Downloading DSRL datasets to: {data_path}")

    # TODO: add other tested envs here, notably METADRIVE
    dataset_names =  [
        "OfflineSwimmerVelocityGymnasium-v1",
        "OfflineHopperVelocityGymnasium-v1",
        "OfflineHalfCheetahVelocityGymnasium-v1",
        "OfflineCarButton1Gymnasium-v0",
        "OfflineCarButton2Gymnasium-v0",
        "OfflineCarPush1Gymnasium-v0",
        "OfflineCarPush2Gymnasium-v0",
        "OfflineCarGoal1Gymnasium-v0",
        "OfflineCarGoal2Gymnasium-v0",
        "OfflineCarCircle-v0",
        "OfflineAntCircle-v0",
        "OfflineDroneCircle-v0",
        "OfflineBallCircle-v0"
    ]

    print(f"Found {len(dataset_names)} datasets")

    for name in dataset_names:
        env = gym.make(name)
        dataset = env.get_dataset()
    print('done')

if __name__ == "__main__":
    main()
