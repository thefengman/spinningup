"""
This will be an implementation of the REINFORCE policy gradient algorithm (RPG), the most basic (no-frills) policy
gradient method. Note that it will not have the improvements used by Spinning Up's VPG, which are (1) estimating V,
for evaluating advantage, and (2) using return-to-go rather than the full return. We use the version of REINFORCE
discussed in UC Berkeley's CS 285 (Deep RL) course.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import gym
from torch.utils.tensorboard import SummaryWriter

# MLP code is taken from SU
from spinup.algos.pytorch.vpg.core import MLPCategoricalActor
from spinup.utils.logx import EpochLogger


def rpg(env_fn=(lambda: gym.make("CartPole-v1")), batch_size=100, hidden_sizes = (32, 32),
        pi_lr=0.0003, logger_kwargs=dict()):
    """

    """
    max_traj_length = 500
    activation = nn.Tanh
    num_epochs = 100

    env = env_fn()
    # Assume obs space is a Box
    obs_dim = env.observation_space.shape[0]
    # Assume action space is Discrete (categorical), which is why we evaulate .n (rather than .shape[0])
    act_dim = env.action_space.n
    pi = MLPCategoricalActor(obs_dim, act_dim, hidden_sizes, activation)
    pi_optimizer = Adam(pi.parameters(), lr=pi_lr)

    logger = EpochLogger(**logger_kwargs)
    logger.setup_pytorch_saver(pi)

    for ep in range(num_epochs):
        print(f"Epoch num: {ep}")

        batch_rewards = torch.zeros(batch_size)
        batch_log_prob = torch.zeros(batch_size)
        for i in range(batch_size):
            # for each trajectory
            o = env.reset()
            d = False  # bool for "done"

            # buffers for o, a, r; contains all values for one trajectory
            # assumes cartpole dimensions
            buffer_o = np.zeros((max_traj_length, obs_dim))
            buffer_a = np.zeros(max_traj_length)
            buffer_r = np.zeros(max_traj_length)
            ptr = 0  # pointer to the position in the buffer

            while not d:
                o = torch.as_tensor(o, dtype=torch.float32)
                a = pi._distribution(o).sample()  # sample Categorical policy to get an action
                o2, r, d, _ = env.step(a.numpy())
                o2 = torch.as_tensor(o2, dtype=torch.float32)

                buffer_o[ptr] = o.numpy()
                buffer_a[ptr] = a
                buffer_r[ptr] = r

                o = o2
                ptr += 1
                if ptr >= max_traj_length:
                    break

            batch_rewards[i] = buffer_r[:ptr].sum()
            log_probs = pi._log_prob_from_distribution(pi._distribution(torch.as_tensor(buffer_o[:ptr], dtype=torch.float32)),
                                                       torch.as_tensor(buffer_a[:ptr], dtype=torch.float32))
            batch_log_prob[i] = log_probs.sum()

        pi_optimizer.zero_grad()
        loss = -1 * (batch_log_prob * batch_rewards).mean()
        loss.backward()
        pi_optimizer.step()

        writer.add_scalar("pi loss", float(loss), ep)
        writer.add_scalar("avg return", float(batch_rewards.mean()), ep)
        if ep % 10 == 0:
            logger.save_state({'env': env}, None)  # also saves pi


    print("Done training the agent.")
    logger.save_state({'env': env}, None)  # also saves pi


if __name__ == '__main__':
    global writer

    lr_list = [0.0003, 0.001, 0.003, 0.01, 0.03]
    for lr in lr_list:
        logger_kwargs = {"output_dir": f"logged_data/logged_data_pi_lr_{lr}",
                         "exp_name": f"pi_lr_{lr}"}

        writer = SummaryWriter(log_dir=f"runs/sweep_pi_lr_{lr}", flush_secs=30)
        rpg(batch_size=100, pi_lr=lr, logger_kwargs=logger_kwargs)
        writer.close()