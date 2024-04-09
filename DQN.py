import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import random
from collections import deque
import numpy as np

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 4)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
    
class DQL:
    learning_rate = 0.001
    discount_factor_g = 0.5
    network_sync_rate = 10
    replay_memory_size = 1000
    mini_batch_size = 32

    loss_fn = nn.MSELoss()
    optimizer = None

    ACTIONS = ['N','L','M','R'] 

    def train(self,episodes,render=False):
        env = gym.make("LunarLander-v2",render_mode= 'human' if render else None)
        num_states = env.observation_space.shape[0]
        num_actions = 4

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN()
        target_dqn = DQN()

        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        rewards_per_episode = np.zeros(episodes)

        epsilon_history = []

        step_count = 0

        for i in tqdm(range(episodes)):
            state = env.reset()[0]
            terminated = False
            truncated = False

            while(not terminated and not truncated):
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(torch.Tensor(state)).argmax().item()

                new_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state,action,new_state,reward,terminated))
                state = new_state

                step_count += 1
                rewards_per_episode[i] = reward

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon -1/episodes,0)
                    epsilon_history.append(epsilon)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

        env.close()
        torch.save(policy_dqn.state_dict(),"lunar_lander.pt")

        plt.figure(1)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode)

                
        torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig('frozen_lake_dql.png')

    def optimize(self,mini_batch,policy_dqn,target_dqn):
        num_states = 8

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(reward + target_dqn(torch.Tensor(new_state)).max())

            current_q = policy_dqn(torch.Tensor(state))
            current_q_list.append(current_q)

            target_q = target_dqn(torch.Tensor(state))
            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes):
        env = gym.make("LunarLander-v2",render= 'human')
        num_states = 8
        num_actions = 4

        policy_dqn = DQN()
        policy_dqn.load_state_dict(torch.load("lunar_lander.pt"))
        policy_dqn.eval()

        print('Policy (trianed):')

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False

            while(not terminated and not truncated):
                with torch.no_grad():
                    action = policy_dqn(torch.Tensor(state)).argmax().item()
                state,reward,terminated,truncated,_ = env.step(action)
        env.close()

if __name__ == "__main__":
    lander = DQL()
    lander.train(1000)
    lander.test(10)


