"""
Purpose: implement DQN (Deep Q-Networks), reproducing the original paper by Minh et al. Note that a lot of the code
here is based from OpenAI's SU.
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import argparse

from spinup.utils.logx import EpochLogger
from spinup.algos.pytorch.ddpg.core import mlp
from spinup.algos.pytorch.ddpg.ddpg import ReplayBuffer


def epsilon_greedy_policy_fn(q, action_space, epsilon):
    """
    Call this to create the policy function.
    assumes type(action_space) is gym.spaces.discrete.Discrete
    assumes type(action_space.sample()) is int  # 1 number, not an array of numbers
    """
    def epsilon_greedy_policy(obs):
        # Assume batch_size == 1. Need to generalize to allow batch_size > 1
        assert obs.dim() == 1

        if float(torch.rand(1)) > epsilon:
            return q(obs).argmax()
        else:
            return torch.tensor(action_space.sample())

    return epsilon_greedy_policy


class MLPQFunction(nn.Module):
    # some code that may or may not be helpful when I get around to coding Q
    # # TODO: generalize to other spaces; consider converting to numpy, e.g. to use np.repeat / np.tile
    # assert type(action_space) is gym.spaces.discrete.Discrete
    # assert type(action_space.sample()) is int  # 1 number, not an array of numbers
    # num_actions = action_space.n
    # possible_actions = torch.arange(num_actions)
    # Assume obs shape is either (batch_size, obs_dim) or (obs_dim), to extract batch size
    # assert obs.dim() in [1, 2]
    # if obs.dim() == 2:
    #     batch_size = obs.shape[0]
    # else:
    #     batch_size = 1
    #
    # if float(torch.rand(1)) > epsilon:
    #     act_input = possible_actions.repeat_interleave(batch_size).reshape((batch_size * num_actions, 1))
    #
    #     if batch_size == 1:
    #         obs = obs.reshape((1, -1))  # makes it a row vector
    #     obs_input = obs.repeat((num_actions, 1))
    #
    #     # act_input and obs_input are wrangled to a shape s.t. they will concatenate properly when fed to Q-function
    #     # example, if batch size is 2, possible actions are [0,1,2], and obs is [[10, 20, 30],
    #     #                                                                        [40, 50, 60]]
    #     # then:
    #     # act_input is: [[0],
    #     #                [0],
    #     #                [1],
    #     #                [1],
    #     #                [2],
    #     #                [2]]
    #     # obs_input is: [[10, 20, 30],
    #     #                [40, 50, 60],
    #     #                [10, 20, 30],
    #     #                [40, 50, 60],
    #     #                [10, 20, 30],
    #     #                [40, 50, 60]]
    #
    #     q_vals = q(obs_input)
    #     q_vals = q_vals.reshape((batch_size, -1))
    #
    #     # # q_vals should now be like:
    #     # #                action 0      action 1      action 2
    #     # # batch 0      [[10,           20,           30,
    #     # # batch 1        11,           21,           31]]
    #
    #     assert batch_size == 1
    #     arg = q_vals.flatten().argmax()
    #     return int(possible_actions[arg])
    #
    # else:
    #     assert batch_size == 1
    #     return action_space.sample()

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        # TODO: include action_limit to cap pi?
        # act_limit = action_space.high[0]

        # build policy and value functions

        # self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)





    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()



def dqn(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """
    DQN (Deep Q-Networks). Reproduce the original paper from Minh et al.
    """




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    dqn(lambda : gym.make(args.env), actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)










