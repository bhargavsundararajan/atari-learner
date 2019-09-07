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
counter = 0
episode_length = 0

all_rewards = []
episode_reward = 0
states = []
actions = []
dones = []
hid_outs = []
times = []
choose = []
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
    #replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    #episode_reward += reward
    episode_length += 1
    if done:
        print("Done")
        third = int(episode_length/3)
        two_third = int(episode_length*2/3)
        counter += 1
        state = env.reset()
        times += [0 for i in range(third)]
        choose += [0 for i in range(0,5)]
        choose += [1 for i in range(5, third)]
        times += [1 for i in range(third, two_third)]
        choose += [0 for i in range(third, third + 5)]
        choose += [1 for i in range(third+5, two_third)]
        times += [2 for i in range(two_third,episode_length)]
        choose += [0 for i in range(two_third, two_third + 5)]
        choose += [1 for i in range(two_third + 5, episode_length)]
        if len(times) != len(states):
            print("Danger")
        episode_length = 0
        #print(episode_reward)
        episode_reward = 0
        if counter == 1:
            break

#print(all_rewards)
hid_outs = torch.stack(hid_outs, dim =0)


def plot_graph(hid_outs, actions_plot, times_plot, rewards_plot, choose_plot):
    plt.figure(figsize=(16,10))
    hid_feat_cols = ['node_'+str(i) for i in range(hid_outs.shape[1])]

    df = pd.DataFrame(hid_outs, columns = hid_feat_cols)
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(df[hid_feat_cols].values)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)


    tsne_pca_results = tsne.fit_transform(pca_result)

    new_df = pd.DataFrame(tsne_pca_results[:,0], columns = ['tsne-one'])
    new_df['tsne-two'] = pca_result[:,1]
    new_df['action'] = np.array(actions_plot)
    new_df['times'] = times_plot
    new_df['reward'] = rewards_plot
    new_df['choose']  = choose_plot
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    #print(np.unique(new_df['action']))
    #new_df = new_df.sample(n = 500, random_state = 1)
    #new_df = new_df.loc[new_df['action'] == 1]
    sns_plot = sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue = "action",
        style = 'times',
        size  = 'choose',
        palette=sns.color_palette("bright", 6),
        data=new_df,
        legend="full",
        s = 100
    )
    fig = sns_plot.get_figure()
    fig.savefig("model_voz.png")
    #print("test")
    plt.show()

plot_graph(hid_outs, actions, times, all_rewards, choose)
plt.savefig("model_viz.png")
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
