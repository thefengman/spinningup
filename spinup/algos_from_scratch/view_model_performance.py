


import torch
import torch.nn as nn
from spinup.algos.pytorch.vpg.core import MLPCategoricalActor

# pi = MLPCategoricalActor(1,1,(1,), nn.Tanh)
# pi.load_state_dict(torch.load("logged_data/logged_data_pi_lr_0.003/pyt_save/model.pt"))

pi = torch.load("logged_data/logged_data_pi_lr_0.003/pyt_save/model.pt")




import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        observation = torch.as_tensor(observation, dtype=torch.float32)
        action = pi._distribution(observation).sample().numpy()
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
env.close()