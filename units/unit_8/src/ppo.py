import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions.categorical import Categorical

def layer_init(layer, std=1.4142, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        
        print(envs.single_observation_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(envs.single_observation_space.shape).prod(), 24)),
            nn.Tanh(),
            layer_init(nn.Linear(24, 24)),
            nn.Tanh(),
            layer_init(nn.Linear(24, 1), std=1.)
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(envs.single_observation_space.shape).prod(), 24)),
            nn.Tanh(),
            layer_init(nn.Linear(24, 24)),
            nn.Tanh(),
            layer_init(nn.Linear(24, envs.single_action_space.n), std=0.01)
        )
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
