from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
USE_CUDA = torch.cuda.is_available()

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state):
        state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
        ######## YOUR CODE HERE! ########
        # TODO: Given state, you should write code to get the Q value and chosen action
        # Complete the R.H.S. of the following 2 lines and uncomment them
        y = self.features(state)
        y = y.view(y.size(0), -1)
        hid_out = self.fc[:-2](y)
        q_value = self.forward(state)
        action = q_value.argmax()
        #print(action)
        ######## YOUR CODE HERE! ########
        return hid_out.squeeze(),action

def compute_td_loss(model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)), requires_grad = True)
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    #print(action.shape)
    action = action.unsqueeze(1)
    #print(action.shape)
    q_current = model.forward(state).gather(1, action).squeeze(-1)
    q_next = model.forward(next_state).max(1)[0]
    #print(np.shape(q_current))
    #y_value = []
    #q_current_action = []
    for i in range(batch_size):
        if done[i]:
            q_next[i] = 0
        #print(action[i])
    y_value = q_next*gamma + reward
    #print(np.shape(q_current_action))
    #print(q_current_action)
    y_value = y_value.detach()
    loss = nn.MSELoss()(y_value, q_current)
    #print(loss)

    '''
    total_loss = 0.0
    for i in range(batch_size):
        if done[i]:
            y_value = reward[i].cpu().numpy()
        else:
            temp_state = next_state[i]
            q_next = model.forward(temp_state.unsqueeze(0))
            y_value = reward[i].cpu().numpy() + gamma*(q_next.argmax().cpu().detach().numpy())
        #print(np.shape(action))
        temp_state = state[i]
        q_current = model.forward(temp_state.unsqueeze(0))
        #print(np.shape(q_current))
        total_loss = total_loss + np.square(y_value - q_current[0][action[i]].cpu().detach().numpy())

    loss_temp = total_loss/batch_size
    ######## YOUR CODE HERE! ########
    # TODO: Implement the Temporal Difference Loss
    # loss =replay_buffer
    ######## YOUR CODE HERE! ########
    loss = Variable(torch.FloatTensor(np.array(loss_temp)), requires_grad = True)
    '''
    #loss = Variable(loss, requires_grad = True)
    #print(loss)
    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        ######## YOUR CODE HERE! ########
        # TODO: Randomly sampling data with specific batch size from the buffer
        # Hint: you may use the python library "random".
        # If you are not familiar with the "deque" python library, please google it.
        ######## YOUR CODE HERE! ########
        sample_batch = random.sample(self.buffer,batch_size)
        state = [sb[0] for sb in sample_batch]
        action = [sb[1] for sb in sample_batch]
        reward = [sb[2] for sb in sample_batch]
        next_state = [sb[3] for sb in sample_batch]
        done = [sb[4] for sb in sample_batch]
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
