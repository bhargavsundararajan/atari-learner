from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from dqn_eval import QLearner, compute_td_loss, ReplayBuffer
import pickle as pkl

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 100000
batch_size = 32
gamma = 0.99

replay_initial = 20000
replay_buffer = ReplayBuffer(200000)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
model.load_state_dict(torch.load('model_trained.pt', map_location='cpu'))
'''
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
'''
optimizer = optim.Adam(model.parameters(), lr=0.00001)
if USE_CUDA:
    model = model.cuda()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


state = env.reset()
rewards_initial = []
rewards_middle = []
rewards_final = []

states_initial = []
states_middle = []
states_final = []

actions_initial = []
actions_middle = []
actions_final = []

dones_initial = []
dones_middle = []
dones_final = []

hid_outs_initial = []
hid_outs_middle = []
hid_outs_final = []
counter = 0
episode_length = 0

all_rewards = []
episode_reward = 0
states = []
actions = []
dones = []
hid_outs = []
for frame_idx in range(1, num_frames + 1):
    #epsilon = epsilon_by_frame(frame_idx)
    #print("loop")
    states.append(state)
    hid_out, action = model.act(state)
    hid_outs.append(hid_out)
    actions.append(action)
    #print(action)
    next_state, reward, done, _ = env.step(action)
    episode_reward += reward
    all_rewards.append(episode_reward)
    dones.append(done)
    #print(hid)
    #if frame_idx % 10000 == 0:
       # print(np.shape(next_state))
    #print(reward)
    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    #episode_reward += reward
    episode_length += 1
    if done:
        print("Done")
        third = int(episode_length/3)
        two_third = int(episode_length*2/3)
        counter += 1
        if counter == 3:
            break
        state = env.reset()
        rewards_initial += all_rewards[:third]
        rewards_middle += all_rewards[third:two_third]
        rewards_final += all_rewards[two_third:]

        states_initial += states[:third]
        states_middle += states[third:two_third]
        states_final += states[two_third:]

        actions_initial += actions[:third]
        actions_middle += actions[third:two_third]
        actions_final += actions[two_third:]

        dones_initial += dones[:third]
        dones_middle += dones[third:two_third]
        dones_final += dones[two_third:]

        hid_outs_initial += hid_outs[:third]
        hid_outs_middle += hid_outs[third:two_third]
        hid_outs_final += hid_outs[two_third:]

        all_rewards = []
        episode_reward = 0
        states = []
        actions = []
        dones = []
        hid_outs = []
        episode_length = 0
        #print(episode_reward)
        episode_reward = 0

#print(all_rewards)
hid_outs_initial = torch.stack(hid_outs_initial, dim =0)
hid_outs_middle = torch.stack(hid_outs_middle, dim =0)
hid_outs_final = torch.stack(hid_outs_final, dim =0)


plt.figure(figsize=(16,10))
def plot_graph(hid_outs, actions_plot):
    hid_feat_cols = ['node_'+str(i) for i in range(hid_outs.shape[1])]
    df = pd.DataFrame(hid_outs, columns = hid_feat_cols)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[hid_feat_cols].values)
    #tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    #tsne = MDS(n_components=2)
    #tsne = LocallyLinearEmbedding(n_components=2)
    #tsne = CCA(n_components=2)
    #tsne_pca_results = tsne.fit_transform(pca_result)

    new_df = pd.DataFrame(pca_result[:,0], columns = ['tsne-one'])
    new_df['tsne-two'] = pca_result[:,1]
    new_df['action'] = np.array(actions_plot)
    #print(np.unique(new_df['action']))
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue = "action",
        palette=sns.color_palette(n_colors= 6),
        data=new_df,
        legend="full",
        alpha=0.3
    )
    #print("test")
    plt.show()

plot_graph(hid_outs_initial, actions_initial)
plot_graph(hid_outs_middle, actions_middle)
plot_graph(hid_outs_final, actions_final)
plt.show()
#print(np.sum(pca.explained_variance_ratio_))
#df['action'] = actions
#df['reward'] = all_rewards
#df['state'] = states
#df['done'] = dones

#print(len(df.columns))
'''
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
'''
