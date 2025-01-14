import numpy as np

from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
# from gym.envs.mujoco.half_cheetah import HalfCheetahEnv as HalfCheetahEnv_

# from gymnasium.envs.mujoco.half_cheetah import HalfCheetahEnv as HalfCheetahEnv_


class HalfCheetahEnv(HalfCheetahEnv_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_mode = kwargs.get("render_mode", "rgb_array")

    def _get_obs(self):
        return (
            np.concatenate(
                [
                    self.sim.data.qpos.flat[1:],
                    self.sim.data.qvel.flat,
                    self.get_body_com("torso").flat,
                ]
            )
            .astype(np.float32)
            .flatten()
        )

    def viewer_setup(self):
        camera_id = self.model.camera_name2id("track")

        if self.viewer is None:
            raise ValueError("Viewer must be initialized before camera can be adjusted")

        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id  # type: ignore
        self.viewer.cam.distance = self.model.stat.extent * 0.35  # type: ignore
        # Hide the overlay
        self.viewer._hide_overlay = True  # pylint: disable=protected-access

    def render(self, mode="human"):
        _viewer = self._get_viewer(mode=mode)
        if _viewer is None:
            raise ValueError("Viewer is not initialized")

        if mode == "rgb_array":
            _viewer.render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = _viewer.read_pixels(width, height, depth=False)
            return data
        elif mode == "human":
            _viewer.render()
            return _viewer._gui_lock  # pylint: disable=protected-access
        else:
            raise ValueError(f"Mode {mode} is not valid for rendering")


class HalfCheetahVelEnv(HalfCheetahEnv):
    """Half-cheetah environment with target velocity, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a penalty equal to the
    difference between its current velocity and the target velocity. The tasks
    are generated by sampling the target velocities from the uniform
    distribution on [0, 2].

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(self, task={}, low=0.0, high=2.0, **kwargs):
        self._task = task
        self.low = low
        self.high = high

        self._goal_vel = task.get("velocity", 0.0)

        super().__init__(**kwargs)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "task": self._task,
        }

        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks):
        velocities = self.np_random.uniform(self.low, self.high, size=(num_tasks,))
        tasks = [{"velocity": velocity} for velocity in velocities]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_vel = task["velocity"]


class HalfCheetahDirEnv(HalfCheetahEnv):
    """Half-cheetah environment with target direction, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand_direc.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a reward equal to its
    velocity in the target direction. The tasks are generated by sampling the
    target directions from a Bernoulli distribution on {-1, 1} with parameter
    0.5 (-1: backward, +1: forward).

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(self, task={}):
        self._task = task
        self._goal_dir = task.get("direction", 1)
        super().__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._goal_dir * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "task": self._task,
        }

        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks):
        directions = 2 * self.np_random.binomial(1, p=0.5, size=(num_tasks,)) - 1
        tasks = [{"direction": direction} for direction in directions]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_dir = task["direction"]
