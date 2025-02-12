"""Tests for `imitation.algorithms.density_baselines`."""

from typing import Sequence

import numpy as np
import pytest
import stable_baselines3
from stable_baselines3.common import policies

from imitation.algorithms.density import DensityAlgorithm, DensityType
from imitation.data import rollout, types
from imitation.data.types import TrajectoryWithRew
from imitation.policies.base import RandomPolicy

parametrize_density_stationary = pytest.mark.parametrize(
    "density_type,is_stationary",
    [(density_type, True) for density_type in DensityType]
    + [(DensityType.STATE_DENSITY, False)],
)


def score_trajectories(
    trajectories: Sequence[types.Trajectory],
    density_reward: DensityAlgorithm,
):
    # score trajectories under given reward function w/o discount
    returns = []
    for traj in trajectories:
        dones = np.zeros(len(traj), dtype=bool)
        dones[-1] = True
        steps = np.arange(0, len(traj.acts))
        rewards = density_reward(traj.obs[:-1], traj.acts, traj.obs[1:], dones, steps)
        ret = np.sum(rewards)
        returns.append(ret)
    return np.mean(returns)


# test on Pendulum rather than Cartpole because I don't handle episodes that
# terminate early yet (see issue #40)
@parametrize_density_stationary
def test_density_reward(
    density_type,
    is_stationary,
    pendulum_venv,
    pendulum_expert_trajectories: Sequence[TrajectoryWithRew],
):
    # use only a subset of trajectories
    expert_trajectories_all = pendulum_expert_trajectories[:8]
    n_experts = len(expert_trajectories_all)
    expert_trajectories_train = expert_trajectories_all[: n_experts // 2]
    reward_fn = DensityAlgorithm(
        demonstrations=expert_trajectories_train,
        density_type=density_type,
        kernel="gaussian",
        venv=pendulum_venv,
        is_stationary=is_stationary,
        kernel_bandwidth=0.2,
        standardise_inputs=True,
    )
    reward_fn.train()

    # check that expert policy does better than a random policy under our reward
    # function
    random_policy = RandomPolicy(
        pendulum_venv.observation_space,
        pendulum_venv.action_space,
    )
    sample_until = rollout.make_min_episodes(15)
    random_trajectories = rollout.generate_trajectories(
        random_policy,
        pendulum_venv,
        sample_until=sample_until,
    )
    expert_trajectories_test = expert_trajectories_all[n_experts // 2 :]
    random_score = score_trajectories(random_trajectories, reward_fn)
    expert_score = score_trajectories(expert_trajectories_test, reward_fn)
    assert expert_score > random_score


@pytest.mark.expensive
def test_density_trainer_smoke(
    pendulum_venv,
    pendulum_expert_trajectories: Sequence[TrajectoryWithRew],
):
    # tests whether density trainer runs, not whether it's good
    # (it's actually really poor)
    rollouts = pendulum_expert_trajectories[:2]
    rl_algo = stable_baselines3.PPO(policies.ActorCriticPolicy, pendulum_venv)
    density_trainer = DensityAlgorithm(
        demonstrations=rollouts,
        venv=pendulum_venv,
        rl_algo=rl_algo,
    )
    density_trainer.train()
    density_trainer.train_policy(n_timesteps=2)
    density_trainer.test_policy(n_trajectories=2)
