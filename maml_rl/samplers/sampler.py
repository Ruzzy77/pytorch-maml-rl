import gym
# import gymnasium as gym

# def make_env(env_name, env_kwargs={}, seed=None):
#     def _make_env():
#         env = gym.make(env_name, **env_kwargs)
#         if hasattr(env, "seed"):
#             env.seed(seed)
#         return env

#     return _make_env


class make_env:
    def __init__(self, env_name, env_kwargs={}, seed=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.seed = seed

    def __call__(self):
        env = gym.make(self.env_name, **self.env_kwargs)
        if hasattr(env, "seed"):
            env.seed(self.seed)
        return env


class Sampler:
    def __init__(self, env_name, env_kwargs, batch_size, policy, seed=None, env=None, render_mode="rgb_array"):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.batch_size = batch_size
        self.policy = policy
        self.seed = seed

        if env is None:
            env = gym.make(env_name, **env_kwargs, render_mode=render_mode)
        self.env = env
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        self.env.close()
        self.closed = False

    def sample_async(self, *args, **kwargs):
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        return self.sample_async(*args, **kwargs)
