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

def rpg(env_fn=(lambda: gym.make("CartPole-v1"))):
    """

    """
    batch_size = 1000
    pi_lr = 0.00003
    hidden_sizes = (32, 32)
    activation = nn.Tanh
    num_epochs = 100

    env = env_fn()
    # Assume obs space is a Box
    obs_dim = env.observation_space.shape[0]
    # Assume action space is Discrete (categorical), which is why we evaulate .n (rather than .shape[0])
    act_dim = env.action_space.n
    pi = MLPCategoricalActor(obs_dim, act_dim, hidden_sizes, activation)
    pi_optimizer = Adam(pi.parameters(), lr=pi_lr)


    for ep in range(num_epochs):
        print(f"Epoch num: {ep}")

        batch_rewards = torch.zeros(batch_size)
        batch_log_prob = torch.zeros(batch_size)
        for i in range(batch_size):
            # for each trajectory
            o = env.reset()
            sum_log_prob = 0
            sum_rewards = 0
            d = False  # bool for "done"

            # get sum_log_prob and sum_rewards
            while not d:
                o = torch.as_tensor(o, dtype=torch.float32)
                a = pi._distribution(o).sample()  # sample Categorical policy to get an action
                o2, r, d, _ = env.step(a.numpy())
                o2 = torch.as_tensor(o2, dtype=torch.float32)

                log_prob = pi._log_prob_from_distribution(pi._distribution(o), a)
                sum_log_prob += log_prob

                sum_rewards += r

                o = o2
            batch_rewards[i] = sum_rewards
            batch_log_prob[i] = sum_log_prob

        loss = -1 * (batch_log_prob * batch_rewards).mean()

        pi_optimizer.zero_grad()
        loss.backward()
        pi_optimizer.step()

        print(f"pi loss is: {loss}")
        writer.add_scalar("pi loss", float(loss), ep)
        writer.add_scalar("avg return", float(batch_rewards.mean()), ep)

    print("Done training the agent.")



if __name__ == '__main__':
    global writer
    writer = SummaryWriter(flush_secs=30)
    rpg()