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
import copy
import time

from spinup.utils.logx import EpochLogger
import spinup.algos.pytorch.ddpg.core as core


class EpsilonGreedy(nn.Module):
    def __init__(self, q, action_space, epsilon):
        super().__init__()

        """
        assumes type(action_space) is gym.spaces.discrete.Discrete
        assumes type(action_space.sample()) is int  # 1 number, not an array of numbers
        Returns a numpy array (usually of one element)
        """
        self.q = q
        self.action_space = action_space
        self.epsilon = epsilon

    def forward(self, obs):
        # Assume batch_size == 1. Need to generalize to allow batch_size > 1?
        assert obs.dim() == 1

        if float(torch.rand(1)) > self.epsilon:
            return self.q(obs).argmax().numpy()
        else:
            return np.array(self.action_space.sample())


class QFunction(nn.Module):
    # some code that may or may not be helpful when I get around to coding Q
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
        # Emulating DQN paper, use `act_dim` number of output nodes
        # TODO: for cartpole test, use a fully connected NN (i.e. MLP), but need to change for Atari games
        self.q = core.mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        return self.q(obs)


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU, epsilon=0.0):
        super().__init__()
        # TODO: generalize to different envs; currently its set for cartpole
        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # build policy and action-value functions
        self.q = QFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.pi = EpsilonGreedy(self.q, action_space, epsilon)

    def act(self, obs):
        return self.pi(obs)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=int)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])

        batch_tensors = {}
        for k, v in batch.items():
            if k == "act":
                batch_tensors[k] = torch.as_tensor(v, dtype=torch.int64)
            else:
                batch_tensors[k] = torch.as_tensor(v, dtype=torch.float32)
        return batch_tensors


def preprocess_obs(obs):
    """
    input is np array, output is also np array
    """
    # TODO: implement preprocessing step; note: needs to include last 4 images
    return obs


def compute_loss_q(batch_data, ac, q_targ, gamma):
    # unpack batch data; "_b" means "batch"
    op_b, a_b, r_b, o2p_b, d_b = batch_data['obs'], batch_data['act'], batch_data['rew'], batch_data['obs2'], \
                                 batch_data['done']
    # assumes `a` is a 1D integer array of size batch_size
    max_q_targ_vals = q_targ(o2p_b).max(dim=-1)[0]  # max of Q over acts; now 1D
    q_vals = ac.q(op_b)[np.arange(a_b.shape[0]), a_b]

    loss = ((r_b + gamma * max_q_targ_vals * (1 - d_b) - q_vals) ** 2).mean()

    return loss


def run_test_episode(test_env, ac):
    """
    run a test episode, returning the episode return and episode length (in steps)
    """
    total_r = 0
    total_steps = 0
    o = test_env.reset()
    op = preprocess_obs(o)
    d = False
    while not d:
        a = ac.act(torch.as_tensor(op, dtype=torch.float32))
        o2, r, d, _ = test_env.step(a)
        o2p = preprocess_obs(o2)
        op = o2p
        total_r += r
        total_steps += 1
    return total_r, total_steps


def dqn(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(), seed=0, steps_per_epoch=5000, epochs=100,
        replay_size=int(1e5), batch_size=100, gamma=0.99, q_lr=1e-4, start_steps=10000,
        update_after=1000, update_targ_every=50, num_test_episodes=10,
        max_ep_len=1000, epsilon=0.01, epsilon_decay=0.99995, logger_kwargs=dict(), writer_kwargs=dict(), save_freq=1):
    """
    DQN (Deep Q-Networks). Reproduce the original paper from Minh et al.
    """
    # Instantiate env
    env = env_fn()
    test_env = env_fn()
    # TODO: might have to assert discrete, or otherwise take only first index of shape or so
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Set up actor (pi) & critic (Q), and data buffer
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    q_targ = copy.deepcopy(ac.q)
    for p in q_targ.parameters():
        p.requires_grad = False
    q_optimizer = torch.optim.Adam(ac.q.parameters(), lr=q_lr)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set RNG seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

    # Set up logging
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    logger.setup_pytorch_saver(ac)
    writer = SummaryWriter(**writer_kwargs)

    start_time = time.time()
    total_steps = epochs * steps_per_epoch
    o = env.reset()
    op = preprocess_obs(o)  # "op" = "observation_preprocessed"
    ep_return = 0  # episode return, counter
    ep_length = 0  # episode length, counter
    for step in range(total_steps):
        # Take an env step, then store data in replay buffer
        if step > start_steps:
            ac.pi.epsilon = max(epsilon, epsilon_decay**step)
            a = ac.act(torch.as_tensor(op, dtype=torch.float32))
        else:
            a = env.action_space.sample()
        o2, r, d, _ = env.step(a)
        o2p = preprocess_obs(o2)
        replay_buffer.store(op, a, r, o2p, d)

        # TODO: does DQN paper say to do 1 GD update with mean of minibatch, or many 1-data-point updates?
        # Sample a random batch from replay buffer and perform one GD step
        q_optimizer.zero_grad()
        batch_data = replay_buffer.sample_batch(batch_size)
        loss_q = compute_loss_q(batch_data, ac, q_targ, gamma)
        loss_q.backward()
        q_optimizer.step()

        # Update target network every so often
        if (step % update_targ_every == 0) and (step >= update_after):
            q_targ = copy.deepcopy(ac.q)
            for p in q_targ.parameters():
                p.requires_grad = False

        # Keep track of episode return and length (for logging purposes)
        ep_return += r
        ep_length += 1

        # If episode done, reset env
        if d:
            o = env.reset()
            op = preprocess_obs(o)
            logger.store(EpRet=ep_return, EpLen=ep_length)
            ep_return = 0
            ep_length = 0
        else:
            op = o2p

        # TODO: confirm: no need for test set if test agent & env are same as training agent & env (e.g. would need
        #  test set if algo added noise to training but not test
        # If epoch end, then do a test to see average return thus far
        if step % steps_per_epoch == steps_per_epoch - 1:
            for ep_i in range(num_test_episodes):
                # turn off epsilon exploration:
                old_epsilon = ac.pi.epsilon
                ac.pi.epsilon = 0

                test_ep_return, test_ep_length = run_test_episode(test_env, ac)
                logger.store(TestEpRet=test_ep_return, TestEpLen=test_ep_length)

                # turn it back on
                ac.pi.epsilon = old_epsilon

        # If epoch end, save models and show logged data
        if step % steps_per_epoch == steps_per_epoch - 1:
            epoch_i = int(step // steps_per_epoch)

            writer.add_scalar("EpRet_mean", logger.get_stats("EpRet")[0], epoch_i)  # first item in `get_stats` is mean
            writer.add_scalar("EpRet_std", logger.get_stats("EpRet")[1], epoch_i)  # 2nd item in `get_stats` is std
            writer.add_scalar("TestEpRet_mean", logger.get_stats("TestEpRet")[0], epoch_i)
            writer.add_scalar("TestEpRet_std", logger.get_stats("TestEpRet")[1], epoch_i)
            writer.add_scalar("epsilon", ac.pi.epsilon, epoch_i)

            logger.save_state({'env': env}, None)  # saves both ac and env
            logger.log_tabular("Epoch", epoch_i)
            logger.log_tabular("EpRet", with_min_and_max=True)
            logger.log_tabular("EpLen", average_only=True)
            logger.log_tabular("TestEpRet", with_min_and_max=True)
            logger.log_tabular("TestEpLen", average_only=True)
            logger.log_tabular("TimeFromStart", time.time() - start_time)
            logger.dump_tabular()

    # Save model at end
    logger.save_state({'env': env}, None)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=0.02)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='dqn')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    tag = "dqn_cartpoletest_exp1_eps0.01_epsdecay0.99995"
    writer_kwargs = {"log_dir": f"runs/{tag}",
                     "flush_secs": 30}

    dqn(lambda : gym.make(args.env), actor_critic=ActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, epsilon=args.epsilon),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, writer_kwargs=writer_kwargs)










