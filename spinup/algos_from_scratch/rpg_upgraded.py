"""
This will be an implementation of the an upgraded version of RPG (REINFORCE policy
gradient), which includes 2 improvements. The 2 improvements are: (1) use rewards-to-go rather than full total
rewards, and (2) use a reward baseline, in particular, just use the average reward of trajectories

As of 2020/11/11, I have only implemented the 2nd improvment. In order to implement the 1st improvement, a couple of
modifications would need to be done to the structure of the sums, which can be done at a future point if desired.
For now, I've verified with avg return plots that rpg_upgraded converges faster (avg return of 500 for cartpole at
~40 epochs) and is more stable (fewer random dips in avg return) than rpg (avg return of 500 at ~60 epochs).
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import gym
from torch.utils.tensorboard import SummaryWriter

# some code taken from SU for convenience
from spinup.algos.pytorch.vpg.core import MLPCategoricalActor
from spinup.utils.logx import EpochLogger


def rpg_upgraded(env_fn=(lambda: gym.make("CartPole-v1")), max_traj_length=500, batch_size=100, num_epochs=100,
        hidden_sizes=(32, 32), activation=nn.Tanh, pi_lr=0.0003, logger_kwargs=dict(), writer_kwargs=dict()):
    """
    Assumes env is CartPole-v1; if want to cleanup later, just make the dimensions general to any env
    """
    env = env_fn()
    # Assume obs space is a Box
    obs_dim = env.observation_space.shape[0]
    # Assume action space is Discrete (categorical), which is why we evaluate .n (rather than .shape[0])
    act_dim = env.action_space.n
    pi = MLPCategoricalActor(obs_dim, act_dim, hidden_sizes, activation)
    pi_optimizer = Adam(pi.parameters(), lr=pi_lr)

    logger = EpochLogger(**logger_kwargs)
    logger.setup_pytorch_saver(pi)
    writer = SummaryWriter(**writer_kwargs)

    for ep in range(num_epochs):
        print(f"Epoch num: {ep}")

        # batch arrays; contains relevant data for batch of trajectories
        batch_rewards = torch.zeros(batch_size)
        batch_log_prob = torch.zeros(batch_size)
        for i in range(batch_size):
            o = env.reset()
            d = False  # bool for "done"

            # buffers for o, a, r; contains all values for one trajectory
            # assumes cartpole dimensions
            buffer_o = np.zeros((max_traj_length, obs_dim))
            buffer_a = np.zeros(max_traj_length)
            buffer_r = np.zeros(max_traj_length)
            ptr = 0  # pointer to the position in the buffer

            # take data for one entire trajectory
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

            # save traj data into batch arrays
            batch_rewards[i] = buffer_r[:ptr].sum()
            log_probs = pi._log_prob_from_distribution(pi._distribution(torch.as_tensor(buffer_o[:ptr],
                                                                                        dtype=torch.float32)),
                                                       torch.as_tensor(buffer_a[:ptr], dtype=torch.float32))
            batch_log_prob[i] = log_probs.sum()

        # run one step of gradient descent optimizer
        pi_optimizer.zero_grad()
        loss = -1 * (batch_log_prob * (batch_rewards - batch_rewards.mean())).mean()
        loss.backward()
        pi_optimizer.step()

        # logging
        writer.add_scalar("pi loss", float(loss), ep)
        writer.add_scalar("avg return", float(batch_rewards.mean()), ep)
        if ep % 10 == 0:
            logger.save_state({'env': env}, None)  # also saves pi

    print("Done training the agent.")
    logger.save_state({'env': env}, None)  # also saves pi
    writer.close()


if __name__ == '__main__':
    lr = 0.003
    tag = "rpg_upgraded_test_only_upgraded_baseline"
    logger_kwargs = {"output_dir": f"logged_data/{tag}_{lr}",
                     "exp_name": f"{tag}_{lr}"}
    writer_kwargs = {"log_dir": f"runs/{tag}_{lr}",
                     "flush_secs": 30}

    rpg_upgraded(batch_size=100, pi_lr=lr, logger_kwargs=logger_kwargs, writer_kwargs=writer_kwargs)

