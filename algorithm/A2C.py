import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.distributions import Categorical

EPISODES = 500

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        #self.actor_updater = self.actor_optimizer()
        #self.critic_updater = self.critic_optimizer()

        if self.load_model:
            self.actor.load_weight("./save_model/cartpole_actor_trained.h5")
            self.critic.load_weight("./save_model/cartpole_critic_trained.h5")

    def build_actor(self):
        actor = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size),
            nn.Softmax(dim=-1)
        )

        for layer in actor:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity = 'relu')
        return actor
    
    def build_critic(self):
        critic = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.value_size)
        )

        for layer in critic:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        return critic
    
    def act(self, state):
        state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
        pdparam = self.actor(state)
        pd = Categorical(logits=pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        #print(f"action : {action}")
        #print(f"action.itme : {action.item()}")
        return action.item()
    
    def train_model(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        done = torch.tensor(done, dtype=torch.float32)

        value = self.critic(state).squeeze()
        next_value = self.critic(next_state).squeeze()

        if done:
            next_value = torch.tensor(0.0)
            advantage = reward - value
            target = reward
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value
        
        policy = self.actor(state)
        action_prob = policy[0, action.item()]
        #print(f"action_prob: {action_prob}")
        cross_entropy = torch.log(action_prob) * advantage
        actor_loss = -cross_entropy

        target = target.unsqueeze(0)
        critic_loss = nn.MSELoss()(self.critic(state).squeeze(), target)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph = True)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state, _ = env.reset()

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done or score == 499 else -100

            agent.train_model(state, action,reward, next_state, done)

            score += reward
            state = next_state

            if done:
                score = score if score ==500.0 else score + 100
                scores.append(score)
                episodes.append(e)
                plt.plot(episodes, scores, 'b')
                plt.savefig("./save_graph/cartpole_a2c.png")
                print(f'episode: {e} score: {score}')

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    torch.save(agent.actor.state_dict(), "./save_model/cartpole_actor.pth")
                    torch.save(agent.critic.state_dict(), "./save_model/cartpole_critic.pth")
                    break