import json
import os

import gym

# import gymnasium as gym

import numpy as np

import torch
import yaml
from tqdm import trange
import warnings

# from gymnasium.experimental.wrappers import RecordVideoV0
# from gym.wrappers.record_video import RecordVideo

# import maml_rl.envs
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.metalearners import MAMLTRPO
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_input_size, get_policy_for_env
from maml_rl.utils.reinforcement_learning import get_returns

# Add DLL path for MuJoCo
os.add_dll_directory(os.path.join(os.environ["USERPROFILE"], ".mujoco/mujoco210/bin"))

# Disable gym logger UserWarning (Bounding box float32 precision)
warnings.filterwarnings("ignore", category=UserWarning)
# Disable mujoco env upgrade to v4 deprecated warning
warnings.filterwarnings("ignore", category=DeprecationWarning)


# MAMLTRPO 대신 TRPO를 사용할지 여부 (True: TRPO, False: MAMLTRPO)
VANILLA_TRPO = True


def main(args):
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, "policy.th")
        config_filename = os.path.join(args.output_folder, "config.json")

        with open(config_filename, "w") as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # env.seed(args.seed)

    env = gym.make(config["env-name"], **config.get("env-kwargs", {}))
    env.close()

    # Record video of the agent's trajectory with all the tasks
    # env = gym.make(config["env-name"], **config.get("env-kwargs", {}), render_mode="rgb_array")
    # env = RecordVideoV0(  # pylint: disable=redefined-variable-type
    #     env,  # type: ignore
    #     video_folder=args.output_folder,
    #     video_length=0,  # Record whole episode
    #     episode_trigger=lambda x: x % 1 == 0,  # Record every episode
    #     name_prefix=f"{config['env-name']}-",
    #     disable_logger=False,
    # )

    # Policy
    policy = get_policy_for_env(
        env, hidden_sizes=config["hidden-sizes"], nonlinearity=config["nonlinearity"]
    )
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    print(f"num_workers: {args.num_workers}")
    print(f"fast_batch_size: {config['fast-batch-size']}")
    print(f"meta_batch_size: {config['meta-batch-size']}")
    print(f"num_steps: {config['num-steps']}")
    print(f"num_batches: {config['num-batches']}")

    # Sampler
    sampler = MultiTaskSampler(
        config["env-name"],
        env_kwargs=config.get("env-kwargs", {}),
        batch_size=config["fast-batch-size"],
        policy=policy,
        baseline=baseline,
        env=env,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    # region MAML version
    metalearner = MAMLTRPO(
        policy,
        fast_lr=config["fast-lr"],
        first_order=config["first-order"],
        device=args.device,
    )

    num_iterations = 0
    for batch in trange(config["num-batches"]):
        tasks = sampler.sample_tasks(num_tasks=config["meta-batch-size"])
        futures = sampler.sample_async(
            tasks,
            num_steps=config["num-steps"],
            fast_lr=config["fast-lr"],
            gamma=config["gamma"],
            gae_lambda=config["gae-lambda"],
            device=args.device,
        )

        logs = metalearner.step(
            *futures,
            max_kl=config["max-kl"],
            cg_iters=config["cg-iters"],
            cg_damping=config["cg-damping"],
            ls_max_steps=config["ls-max-steps"],
            ls_backtrack_ratio=config["ls-backtrack-ratio"],
            vanilla_trpo=VANILLA_TRPO,
        )

        # Test the performance of the agent with respect to the first task
        # sampled (for benchmarking purposes)
        # env.reset()
        # env.render()
        # _action = env.action_space.sample()
        # obs, reward, done, trunc, info = env.step(_action)
        # print(f"obs: {obs}, reward: {reward}, done: {done}, trunc: {trunc}, info: {info}")

        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        logs.update(
            tasks=tasks,
            num_iterations=num_iterations,
            train_returns=get_returns(train_episodes[0]),
            valid_returns=get_returns(valid_episodes),
        )

        # Print logs
        _log_interval = 1  # 몇번의 배치마다 로그를 출력할지 (1이면 매번 출력)
        if batch % _log_interval == 0:
            print("=================================================")
            print(f"* Batch {batch + 1} of {config['num-batches']}")
            print(f"* Num tasks: {len(tasks)}")
            print(f"* Step {int(logs['num_iterations'] / len(tasks))}")

            best_loss_before = np.min(logs["loss_before"])
            best_kl_before = np.min(logs["kl_before"])
            print(f"\tLoss before (best): \t{best_loss_before:.6f}")
            print(f"\tKL before (best): \t{best_kl_before:.6f}")

            best_loss_after = np.min(logs["loss_after"])
            best_kl_after = np.min(logs["kl_after"])
            print(f"\tLoss after (best): \t{best_loss_after:.6f}")
            print(f"\tKL after (best): \t{best_kl_after:.6f}")

            best_train_reward = np.max(logs["train_returns"])
            best_valid_reward = np.max(logs["valid_returns"])
            print(f"\tTrain reward (best): \t{best_train_reward:.2f}")
            print(f"\tValid reward (best): \t{best_valid_reward:.2f}")
            print("=================================================")

        # Save policy
        if args.output_folder is not None:
            with open(policy_filename, "wb") as f:  # type: ignore
                torch.save(policy.state_dict(), f)

    # endregion

    env.close()

    # region TRPO version (stable-baselines3) (not implemented)

    # from sb3_contrib import TRPO

    # TODO MultiTaskSampler에 맞게 TRPO 모델 학습을 변형시켜야 함
    # trpolearner = TRPO(
    #     policy="MlpPolicy",       # policy,  # 필수 파라미터
    #     env=env,        # 필수 파라미터
    #     learning_rate=config["fast-lr"],    # default to 0.001
    #     # first_order=, MAML의 컴퓨팅속도를 위한 1차 근사 계산을 위한 것으로 TRPO는 해당없음
    #     device=args.device,

    #     use_sde=False,
    #     target_kl=config["max-kl"],                                     # 0.01,
    #     cg_max_steps=config["cg-iters"],                                # 15,
    #     cg_damping=config["cg-damping"],                                # 0.1,
    #     line_search_max_iter=config["ls-max-steps"],                    # 10,
    #     line_search_shrinking_factor=config["ls-backtrack-ratio"],      # 0.5,
    # )

    # TODO 문제 : futures의 결과를 trpolearner에 전달해줘야하는데 전달할 파라미터가 없는듯?
    # 가능한 해결방법
    # 1. TRPO를 상속받아 futures 받을수있도록 구현하던가
    # 2. maml_trpo.py를 복제해서 TRPO로 바꾸는 방법

    # trained_model = trpolearner.learn(    # step 말고 learn으로 바꿔야함
    #     total_timesteps=config["num-steps"],
    #     # total_timesteps (int) – The total number of samples (env steps) to train on
    #     progress_bar=True,
    # )

    # endregion


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(
        description="Reinforcement learning with " "Model-Agnostic Meta-Learning (MAML) - Train"
    )

    parser.add_argument("--config", type=str, required=True, help="path to the configuration file.")

    # Miscellaneous
    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument("--output-folder", type=str, help="name of the output folder")
    misc.add_argument("--seed", type=int, default=None, help="random seed")
    misc.add_argument(
        "--num-workers",
        type=int,
        default=mp.cpu_count() - 1,
        help=f"number of workers for trajectories sampling (default: {mp.cpu_count() - 1})",
    )
    misc.add_argument(
        "--use-cuda",
        action="store_true",
        help="use cuda (default: false, use cpu). WARNING: Full support for cuda "
        "is not guaranteed. Using CPU is encouraged.",
    )

    args = parser.parse_args()
    args.device = "cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu"

    main(args)
