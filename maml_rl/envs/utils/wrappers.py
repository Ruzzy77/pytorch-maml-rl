from gym.envs.registration import load
# from gym.wrappers.time_limit import TimeLimit
from gym.wrappers import TimeLimit

# from gymnasium.envs.registration import load_env_creator
# from gymnasium.wrappers.time_limit import TimeLimit


from maml_rl.envs.utils.normalized_env import NormalizedActionWrapper


def mujoco_wrapper(entry_point, **kwargs):
    normalization_scale = kwargs.pop("normalization_scale", 1.0)
    max_episode_steps = kwargs.pop("max_episode_steps", 200)

    # Load the environment from its entry point
    env_cls = load(entry_point)
    # env_cls = load_env_creator(entry_point)

    env = env_cls(**kwargs)

    # Normalization wrapper
    env = NormalizedActionWrapper(env, scale=normalization_scale)

    # Time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env
