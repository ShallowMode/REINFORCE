import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import OnpolicyBatchReplay
from torch.distributions import Categorical

memory_spec = 200

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

def epsilon_greedy(env, state, epsilon, pi):
    if epsilon > np.random.rand():
        action = np.random.choice(range(env.action_space.n))
        return action, None
    else:
        state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        pdparam = pi(state_tensor)
        pd = Categorical(logits=pdparam)
        action = pd.probs.argmax().item()
        return action, None

def train(pi, optimizer, memory, batch_size, gamma):
    # 메모리에서 데이터를 배치 크기만큼 학습
    batch = memory.sample()  # 메모리에서 배치 샘플링
    
    states = torch.tensor(batch['states'], dtype=torch.float32)
    actions = torch.tensor(batch['actions'], dtype=torch.int64)
    rewards = torch.tensor(batch['rewards'], dtype=torch.float32)
    next_states = torch.tensor(batch['next_states'], dtype=torch.float32)
    dones = torch.tensor(batch['dones'], dtype=torch.float32)
    
    # 현재 상태에 대한 정책 신경망 출력
    state_values = pi(states)
    # 다음 상태에 대한 정책 신경망 출력
    with torch.no_grad():  # 다음 상태의 값은 학습하지 않도록
        next_state_values = pi(next_states)
    
    # actions을 사용하여 next_state_values에서 적절한 Q-value 선택
    next_state_values = torch.gather(next_state_values, 1, actions.unsqueeze(1)).squeeze(1)
    
    # TD-Target 및 TD-Error 계산
    td_target = rewards + gamma * next_state_values * (1 - dones)
    
    # 디버깅: TD-Target과 TD-Error의 형태와 내용 출력
    print(f'td_target shape: {td_target.shape}')
    
    state_values = state_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    td_error = td_target - state_values
    
    # 손실 함수 (TD-Error의 제곱합)
    loss = td_error.pow(2).mean()
    
    # 네트워크 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 디버깅: 손실 값 출력
    print(f'Loss: {loss.item()}')
    
    return loss


def main():
    env = gym.make('CartPole-v1')
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim)
    optimizer = optim.RMSprop(pi.parameters(), lr=0.01)
    memory = OnpolicyBatchReplay()
    
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 32
    gamma = 0.99
    training_frequency = 300
    scores, episodes = [], []

    for epi in range(500):
        state, _ = env.reset()
        done = False
        total_reward = 0
        loss = None

        while not done:
            action, next_action = epsilon_greedy(env, state, epsilon, pi)
            next_state, reward, done, _, _ = env.step(action)
            memory.add_experience(state, action, reward, next_state, done, next_action)
            total_reward += reward
            state = next_state

            if memory.size >= training_frequency:
                loss = train(pi, optimizer, memory, batch_size, gamma)  # 수정된 부분
                memory.reset()
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        scores.append(total_reward)
        episodes.append(epi)
        plt.plot(episodes, scores, 'b')
        plt.savefig("./save_graph/cartpole_SARSA.png")

        if loss is not None:
            print(f"Episode {epi} - Total Reward: {total_reward}, Loss: {loss.item()}, Epsilon: {epsilon}")
        else:
            print(f"Episode {epi} - Total Reward: {total_reward}, Epsilon: {epsilon}")

if __name__ == '__main__':
    main()
