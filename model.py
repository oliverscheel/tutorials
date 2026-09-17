import torch
from torch import nn
from torch.distributions import Categorical


def init_layer(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class SimpleActorPolicy(nn.Module):
    """Small actor network for REINFORCE-style training."""

    def __init__(self, observation_dim=8, hidden_dim=64, action_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation):
        return self.network(observation)

    def sample_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32)
        distribution = Categorical(logits=self(observation))
        action = distribution.sample()
        return int(action.item()), distribution.log_prob(action)


class ActorCriticPolicy(nn.Module):
    """Shared actor-critic policy used by baseline RL and PPO."""

    def __init__(self, observation_dim=8, hidden_dim=128, action_dim=4):
        super().__init__()

        self.encoder = nn.Sequential(
            init_layer(nn.Linear(observation_dim, hidden_dim), std=2**0.5),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, hidden_dim), std=2**0.5),
            nn.Tanh(),
        )

        self.actor = nn.Sequential(
            init_layer(nn.Linear(hidden_dim, 64), std=2**0.5),
            nn.Tanh(),
            init_layer(nn.Linear(64, action_dim), std=0.01),
        )

        self.critic = nn.Sequential(
            init_layer(nn.Linear(hidden_dim, 64), std=2**0.5),
            nn.Tanh(),
            init_layer(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, observation):
        latent = self.encoder(observation)
        policy_logits = self.actor(latent)
        value = self.critic(latent)
        return policy_logits, value

    def sample_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32)
        policy_logits, value = self(observation)
        distribution = Categorical(logits=policy_logits)
        action = distribution.sample()
        return int(action.item()), distribution.log_prob(action), value

    def evaluate(self, observations, actions):
        logits, values = self(observations)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions.long())
        entropy = distribution.entropy()
        return log_probs, values.squeeze(-1), entropy
