'''
A very simple Actor-Critic model implementation.
Credit:
https://github.com/tsmatz/reinforcement-learning-tutorials/blob/master/03-actor-critic.ipynb
'''



import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(5, hidden_dim)
        self.mu = nn.Linear(hidden_dim, 2)  # Mean for lina and anga
        self.log_std = nn.Linear(hidden_dim, 2)  # Log standard deviation for lina and anga

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        mu = self.mu(outs)
        log_std = self.log_std(outs)
        std = torch.exp(log_std).clamp(min=1e-8)  # Ensure standard deviation is positive
        return mu, std

class ValueNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(5, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        value = self.output(outs)
        return value

actor_func = ActorNet().to(device)
value_func = ValueNet().to(device)



MAX_X = 100
MAX_Y = 60
iter_no = 0
MAX_ITER = 1000

MAX_LINV = 1
MAX_ANGV = np.pi/3
MAX_LINA = 1
MAX_ANGA = 1
cur_s = np.zeros(5) # x, y, theta, linv, angv
goal = np.array([40., 20.])
dt = 0.1


gamma = 0.99

def reset():
    global cur_s
    global goal
    global iter_no
    global dt

    cur_s[0] = np.random.uniform(0, MAX_X)
    cur_s[1] = np.random.uniform(0, MAX_Y)
    cur_s[2] = np.random.uniform(0, 2*np.pi)
    cur_s[3] = 0
    cur_s[4] = 0
    iter_no = 0
    return cur_s

def step(a):
    lina, anga = a
    global cur_s
    global goal
    global iter_no
    global dt
    x, y, ang, linv, angv = cur_s

    ib = (x >= 0) and (x < MAX_X) and (y >= 0) and (y < MAX_Y)
    OOB_PENALTY = -100

    linv += dt * lina
    angv += dt * anga
    linv = np.clip(linv, -MAX_LINV, MAX_LINV)
    angv = np.clip(angv, -MAX_ANGV, MAX_ANGV)

    old_dist = -np.sqrt((goal[0]-x)**2 + (goal[1]-y)**2)

    x += dt * linv * math.cos(ang)
    y += dt * linv * math.sin(ang)
    ang += dt * angv

    new_dist = -np.sqrt((goal[0]-x)**2 + (goal[1]-y)**2) if ib else OOB_PENALTY
    rwrd = old_dist - new_dist

    cur_s[:] = [x,y,ang,linv,angv]
    term = (not ib) or (iter_no >= MAX_ITER)
    trunc = term
    iter_no += 1
    return cur_s, rwrd, term, trunc, None




# pick up action with above distribution policy_pi
def pick_sample(s):
    with torch.no_grad():
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
        # Get mean and std from state
        mu, std = actor_func(s_batch)
        # Sample from the Gaussian distribution
        dist = torch.distributions.Normal(mu, std)

        a = dist.sample()
        # Clip the actions to valid ranges for lina and anga
        a = torch.clamp(a, -1, 1)  # Assuming lina and anga range between -1 and 1
        return a.cpu().numpy()[0]

reward_records = []
opt1 = torch.optim.AdamW(value_func.parameters(), lr=0.001)
opt2 = torch.optim.AdamW(actor_func.parameters(), lr=0.001)
for i in range(500):
    #
    # Run episode till done
    #
    done = False
    states = []
    actions = []
    rewards = []
    s = reset()
    while not done:
        states.append(list(s.copy()))
        a = pick_sample(s)
        s, r, term, trunc, _ = step(a)
        done = term or trunc
        actions.append(a)
        rewards.append(r)

    if i % 100 == 0:
        plt.scatter([s[0] for s in states],[s[1] for s in states])
        plt.show()

    #
    # Get cumulative rewards
    #
    cum_rewards = np.zeros_like(rewards)
    reward_len = len(rewards)
    for j in reversed(range(reward_len)):
        cum_rewards[j] = rewards[j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0)

    #
    # Train (optimize parameters)
    #

    # Optimize value loss (Critic)
    opt1.zero_grad()
    states = torch.tensor(states, dtype=torch.float).to(device)
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
    values = value_func(states)
    values = values.squeeze(dim=1)
    vf_loss = F.mse_loss(
        values,
        cum_rewards,
        reduction="none")
    vf_loss.sum().backward()
    opt1.step()

    # Optimize policy loss (Actor)
    with torch.no_grad():
        values = value_func(states)
    opt2.zero_grad()
    actions = torch.tensor(actions, dtype=torch.float64).to(device)
    advantages = cum_rewards - values


    mu, std = actor_func(states)
    dist = torch.distributions.Normal(mu, std)
    log_probs = dist.log_prob(actions.float())  # Ensure actions is float32 if needed
    # If actions is multi-dimensional, sum over the action dimension:
    log_probs = log_probs.sum(dim=-1)

    pi_loss = -log_probs * advantages
    pi_loss.sum().backward()
    opt2.step()

    # Output total rewards in episode (max 500)
    print("Run episode{} with rewards {}".format(i, sum(rewards)), end="\r")
    reward_records.append(sum(rewards))

    # stop if reward mean > 475.0
    if np.average(reward_records[-50:]) > 475.0:
        break

print("\nDone")




# Generate recent 50 interval average
average_reward = []
for idx in range(len(reward_records)):
    avg_list = np.empty(shape=(1,), dtype=int)
    if idx < 50:
        avg_list = reward_records[:idx+1]
    else:
        avg_list = reward_records[idx-49:idx+1]
    average_reward.append(np.average(avg_list))
plt.plot(reward_records)
plt.plot(average_reward)
plt.show()
