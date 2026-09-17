import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions import Categorical


class CartPolePolicy(nn.Module):
    """A small actor network for CartPole."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.Tanh(),
            nn.Linear(64, 4),
        )

    def forward(self, observation):
        return self.network(observation)

    def sample_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32)
        distribution = Categorical(logits=self(observation))
        action = distribution.sample()
        return int(action.item()), distribution.log_prob(action)


env = gym.make("LunarLander-v3")
policy = CartPolePolicy()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

NUM_EPOCHS = 10000
BATCH_SIZE = 1024
epoch_rewards = []

def plot_rewards(epoch_rewards):
    plt.plot(range(len(epoch_rewards)), epoch_rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("CartPole Reward per Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_per_epoch.png", dpi=150)
    plt.close()

for epoch in range(NUM_EPOCHS):
    batch_log_probs = []
    batch_returns = []

    num_steps = 0
    episode_rewards = []

    while num_steps < BATCH_SIZE:
        observation, _ = env.reset()

        log_probs = []
        rewards = []

        terminated = truncated = False

        while not (terminated or truncated):
            action, log_prob = policy.sample_action(observation)

            observation, reward, terminated, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            num_steps += 1

        # compute reward-to-go
        returns = []
        G = 0.0

        for reward in reversed(rewards):
            G = reward + 0.99 * G
            returns.append(G)

        returns.reverse()

        batch_log_probs.extend(log_probs)
        batch_returns.extend(returns)

        episode_rewards.append(sum(rewards))

    returns = torch.tensor(batch_returns, dtype=torch.float32)
    log_probs = torch.stack(batch_log_probs)

    loss = -(log_probs * returns).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_rewards.append(sum(episode_rewards) / len(episode_rewards))

    if epoch % (NUM_EPOCHS // 100) == 0:
        print(f"Epoch {epoch}, loss: {loss.item()}")
        print(f"Episode reward: {sum(episode_rewards) / len(episode_rewards)}")
        plot_rewards(epoch_rewards)

env.close()