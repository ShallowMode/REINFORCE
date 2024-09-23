import gym
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.distributions import Categorical

gamma = 0.99

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64,out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()
    
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    
    def forward(self, x):
        pdparam = self.model(x)
        return pdparam
    
    # 행동 생성 방법
    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x)
        #print(f"pdparam: {pdparam}")
        pd = Categorical(logits = pdparam)   # 확률 분포
        #print(f"pd: {pd}")
        action = pd.sample()                 # 확률 분포를 통한 행동 정책 pi(a|s)
        #print(f"action: {action}")
        log_prob = pd.log_prob(action)       # pi(a|s)의 로그확률
        #print(f"log_prob: {log_prob}")   
        self.log_probs.append(log_prob)      # 훈련을 위해 저장   
        return action.item()
    
def train(pi, optimizer):
    # REINFORCE 알고리즘의 내부 경사 상승 루프
    T = len(pi.rewards)
    rets = np.empty(T, dtype = np.float32)   # 이득
    future_ret = 0.0
    # 이득을 효율적으로 계산
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def main():
    scores, episodes = [], []
    env = gym.make('CartPole-v1')
    in_dim = env.observation_space.shape[0]       # 4
    out_dim = env.action_space.n            # 2
    pi = Pi(in_dim, out_dim)
    optimizer = optim.Adam(pi.parameters(), lr = 0.002)
    for epi in range(500):
        state, _ = env.reset()
        for t in range(200):
            action = pi.act(state)
            #print(env.step(action))
            next_state, reward, done, _, _= env.step(action)
            pi.rewards.append(reward)
            state = next_state
            env.render()
            if done:
                break
        loss = train(pi, optimizer)
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        scores.append(total_reward)
        episodes.append(epi)
        plt.plot(episodes, scores, 'b')
        plt.savefig("./save_graph/cartpole_Reinforce.png")
        pi.onpolicy_reset()                 # 활성정책이므로 훈련 이후 메모리 삭제
        print(f"Episode {epi}, loss: {loss}, \
              total_reward: {total_reward}, solved: {solved}")

if __name__ == '__main__':
    main()