"""
Unit tests for dqn.py
"""

import torch
import gym
from spinup.algos_from_scratch.dqn import epsilon_greedy_policy_fn


def test_epsilon_greedy_policy_fn():
    env = gym.make("CartPole-v1")
    assert type(env.action_space) is gym.spaces.discrete.Discrete
    assert type(env.action_space.sample()) is int  # 1 number, not an array of numbers
    n = env.action_space.n

    def q1(obs):
        return torch.sum(obs) + torch.arange(1, n + 1)

    def q2(obs):
        return -(torch.sum(obs) + torch.arange(1, n + 1))

    obs = torch.as_tensor([-0.061, -0.758, 0.057, 1.155], dtype=torch.float32)

    pi1 = epsilon_greedy_policy_fn(q1, env.action_space, epsilon=0.0)
    assert pi1(obs) == 1

    pi2 = epsilon_greedy_policy_fn(q2, env.action_space, epsilon=0.0)
    assert pi2(obs) == 0

    # TODO: test randomness of epsilon using mock obj; if time permits


