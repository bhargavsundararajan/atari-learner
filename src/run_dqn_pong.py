from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer
import pickle as pkl

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 2000000
batch_size = 32
gamma = 0.99

replay_initial = 20000
replay_buffer = ReplayBuffer(200000)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
if USE_CUDA:
    model = model.cuda()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()


for frame_idx in range(1, num_frames + 1):

    if frame_idx > 0 and frame_idx % 20000 == 0:
        print("{}% Done..".format(frame_idx/20000))
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)
    #print(action)
    next_state, reward, done, _ = env.step(action)
    #if frame_idx % 10000 == 0:
       # print(np.shape(next_state))
    #print(reward)
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        #print(episode_reward)
        episode_reward = 0

    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(model, batch_size, gamma, replay_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.cpu().numpy())

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses)))
        print('Last-10 average reward: %f' % np.mean(all_rewards[-10:]))

pkl.dump(losses, open("loss.pkl","wb"))
pkl.dump(all_rewards, open("reward.pkl","wb"))
torch.save(model.state_dict(),'model_trained.pt') 
