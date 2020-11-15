"""
Unit tests for dqn.py
"""

import torch
import gym
import numpy as np
from spinup.algos_from_scratch.dqn import epsilon_greedy_policy_fn


def test_epsilon_greedy_policy_fn():
    env = gym.make("CartPole-v1")
    assert type(env.action_space) is gym.spaces.discrete.Discrete
    assert type(env.action_space.sample()) is int  # 1 number, not an array of numbers
    n = env.action_space.n

    # Set RNG seeds
    torch.manual_seed(0)
    np.random.seed(0)
    env.seed(0)
    env.action_space.seed(0)  # for some reason, gym's action_space random seed is separate from global env seed

    def q1(obs):
        return torch.sum(obs) + torch.arange(1, n + 1)

    def q2(obs):
        return -(torch.sum(obs) + torch.arange(1, n + 1))

    obs = torch.as_tensor([-0.061, -0.758, 0.057, 1.155], dtype=torch.float32)

    pi1 = epsilon_greedy_policy_fn(q1, env.action_space, epsilon=0.0)
    assert pi1(obs) == 1

    pi2 = epsilon_greedy_policy_fn(q2, env.action_space, epsilon=0.0)
    assert pi2(obs) == 0

    pi3 = epsilon_greedy_policy_fn(q1, env.action_space, epsilon=0.5)
    arr = np.zeros(1000)
    for i in range(arr.shape[0]):
        arr[i] = int(pi3(obs))
    assert np.sum(arr) == 778  # random seed makes this deterministic; as expected, it's close to 75%

    pi4 = epsilon_greedy_policy_fn(q1, env.action_space, epsilon=1.0)
    arr = np.zeros(1000)
    for i in range(arr.shape[0]):
        arr[i] = int(pi4(obs))
    assert np.sum(arr) == 479  # random seed makes this deterministic; as expected, it's close to 50%

