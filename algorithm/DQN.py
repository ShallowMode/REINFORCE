import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import ReplayMemory
from collections import namedtuple, deque
from torch.distributions import Categorical

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.SELU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64, out_dim)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        pdparam = self.layer3(x)
        return pdparam
    
    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x)
        pd = Categorical(logits=pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        return action.item()
    


def Blotzman_policy(pi, state, temperature):
    x = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
    Q_values = pi.forward(x)  # Ensure Q_values has the right shape

    # Check Q_values shape and ensure it's a tensor
    if isinstance(Q_values, torch.Tensor):
        scaled_Q = Q_values / temperature

        # Ensure scaled_Q has the correct dimensions for Categorical
        if scaled_Q.dim() == 1:
            scaled_Q = scaled_Q.unsqueeze(0)  # Add batch dimension if missing
        
        pd = Categorical(logits=scaled_Q)
        action = pd.sample()
        return action.item()
    else:
        raise TypeError("Q_values should be a torch.Tensor")

def main():
    env = gym.make('CartPole-v1')
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim)
    optimizer = optim.Adam(pi.parameters(), lr = 0.001)

    Max_steps = 500
    temperature = 1.0
    temperature_min = 0.01
    temperature_decay = 0.995
    max_size = 30000
    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    memory = ReplayMemory(max_size, Transition)
    Batch = 32
    gamma = 0.99

    for m in range(Max_steps):
        state, _ = env.reset()
        done = False
        total_reward = 0
        loss = None

        for i in range(max_size):
            action = Blotzman_policy(pi, state, temperature)
            next_state, reward, done, _, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state

        #print(f"Number of samples in memory: {len(memory)}")
        transitions = memory.sample(Batch)
        #print(f"Example of the memory : {transitions}")
        #print(f"Len of transitions: {len(transitions)}")

        for b in range(Batch):
            data = transitions[b]
            state = torch.tensor(data.state, dtype = torch.float32).unsqueeze(0)
            action = torch.tensor([data.action], dtype = torch.int64)
            reward = torch.tensor([data.reward], dtype=torch.float32)  # 보상을 텐서로 변환
            next_state = torch.tensor(data.next_state, dtype=torch.float32).unsqueeze(0)  # 다음 상태 텐서 변환
            done = torch.tensor([data.done], dtype=torch.float32)

            current_q_value = pi(state).gather(1, action.unsqueeze(1)).squeeze(1)
            
            
            with torch.no_grad():
                next_q_value = pi(next_state).max(1)[0]
            
            next_q_value = next_q_value * (1 - done)

            target_q_value = reward + gamma * next_q_value

            loss = torch.nn.functional.mse_loss(current_q_value, target_q_value)

            # 신경망 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        temperature = max(temperature_min, temperature * temperature_decay)

        if loss is not None:
            print(f"에피소드 {m+1}, Total reward: {total_reward}, loss: {loss.item()}")

if __name__ == '__main__':
    main()

            
                




