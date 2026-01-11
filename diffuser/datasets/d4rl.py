import os
import collections
import numpy as np
import gym
import pdb
import h5py
from tqdm import tqdm

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- DSRL PORT --------------------------------#
#-----------------------------------------------------------------------------#
def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


# Stolen from DSRL, use to map to files from env names
DATASET_URLS = {
    # Safety Gymnasium - velocity
    "OfflineSwimmerVelocityGymnasium-v1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetySwimmerVelocityGymnasium-v1-200-1686.hdf5",
    "OfflineHopperVelocityGymnasium-v1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyHopperVelocityGymnasium-v1-250-2240.hdf5",
    "OfflineHalfCheetahVelocityGymnasium-v1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyHalfCheetahVelocityGymnasium-v1-250-2495.hdf5",
    "OfflineAntVelocityGymnasium-v1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyAntVelocityGymnasium-v1-250-2249.hdf5",
    "OfflineWalker2dVelocityGymnasium-v1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyWalker2dVelocityGymnasium-v1-300-2729.hdf5",

    # Safety Gymnasium - Car
    "OfflineCarButton1Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarButton1Gymnasium-v0-250-2656.hdf5",
    "OfflineCarButton2Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarButton2Gymnasium-v0-300-3755.hdf5",
    "OfflineCarPush1Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarPush1Gymnasium-v0-200-2871.hdf5",
    "OfflineCarPush2Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarPush2Gymnasium-v0-250-4407.hdf5",
    "OfflineCarGoal1Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarGoal1Gymnasium-v0-120-1671.hdf5",
    "OfflineCarGoal2Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarGoal2Gymnasium-v0-200-4105.hdf5",
    "OfflineCarCircle-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarCircle-v0-100-1450.hdf5",

    # Safety Gymnasium - Point / Drone / Ball
    "OfflineAntCircle-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyAntCircle-v0-200-5728.hdf5",
    "OfflineDroneCircle-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyDroneCircle-v0-100-1923.hdf5",
    "OfflineBallCircle-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyBallCircle-v0-80-886.hdf5",
    "OfflinePointButton1Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointButton1Gymnasium-v0-200-2268.hdf5",
    "OfflinePointButton2Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointButton2Gymnasium-v0-250-3288.hdf5",
    "OfflinePointCircle1Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointCircle1Gymnasium-v0-200-1098.hdf5",
    "OfflinePointCircle2Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointCircle2Gymnasium-v0-300-895.hdf5",
    "OfflinePointGoal1Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointGoal1Gymnasium-v0-100-2022.hdf5",
    "OfflinePointGoal2Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointGoal2Gymnasium-v0-200-3442.hdf5",
    "OfflinePointPush1Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointPush1Gymnasium-v0-150-2379.hdf5",
    "OfflinePointPush2Gymnasium-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointPush2Gymnasium-v0-200-3242.hdf5",

    # Safe MetaDrive
    "OfflineMetaDrive-easysparse-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-easysparse-v0-85-1000.hdf5",
    "OfflineMetaDrive-easymean-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-easymean-v0-85-1000.hdf5",
    "OfflineMetaDrive-easydense-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-easydense-v0-85-1000.hdf5",
    "OfflineMetaDrive-mediumsparse-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-mediumsparse-v0-50-1000.hdf5",
    "OfflineMetaDrive-mediummean-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-mediummean-v0-50-1000.hdf5",
    "OfflineMetaDrive-mediumdense-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-mediumdense-v0-50-1000.hdf5",
    "OfflineMetaDrive-hardsparse-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-hardsparse-v0-85-1000.hdf5",
    "OfflineMetaDrive-hardmean-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-hardmean-v0-85-1000.hdf5",
    "OfflineMetaDrive-harddense-v0": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-harddense-v0-85-1000.hdf5"
}

def get_dsrl_dataset(name : str = None):
    
    if name not in DATASET_URLS:
        raise ValueError(f"Dataset name {name} not found in DSRL dataset urls.")

    h5path = DATASET_URLS[name].split('/')[-1]
    
    # WARN: hardcoded data directory, works, but need to keep in mind
    h5path = os.path.join(os.environ.get("DSRL_DATASET_DIR", "/data/"), h5path)

    if not os.path.exists(h5path):
        print("Please run download_datasets.py before trying to load DSRL datasets.")
        raise FileNotFoundError(f"Dataset file {h5path} not found")
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in [
        'observations', 'next_observations', 'actions', 'rewards', 'costs',
        'terminals', 'timeouts'
    ]:
        assert key in data_dict, 'Dataset is missing key %s' % key
    N_samples = data_dict['observations'].shape[0]
    data_dict["observations"] = data_dict["observations"].astype("float32")
    data_dict["actions"] = data_dict["actions"].astype("float32")
    data_dict["next_observations"] = data_dict["next_observations"].astype("float32")
    data_dict["rewards"] = data_dict["rewards"].astype("float32")
    data_dict["costs"] = data_dict["costs"].astype("float32")
    return data_dict

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


def get_dataset(env):

    # Fix Loading of other OSRL envs here
    if 'Safety' in env.spec.id:
        import torch
        dataset = torch.load(f'./dataset/{env.spec.id}_medium-replay.pkl')
        return dataset

    dataset = env.get_dataset()

    if 'antmaze' in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    return dataset

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """

    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
