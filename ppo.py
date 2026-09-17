import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions import Categorical

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2

def init_layer(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, 0.0)
    return layer

class CartPolePolicy(nn.Module):
    """A small actor network for CartPole."""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            init_layer(nn.Linear(8, 128), std=2**0.5),
            nn.Tanh(),
            init_layer(nn.Linear(128, 128), std=2**0.5),
            nn.Tanh(),
        )

        self.actor = nn.Sequential(
            init_layer(nn.Linear(128, 64), std=2**0.5),
            nn.Tanh(),
            init_layer(nn.Linear(64, 4), std=0.01),
        )

        self.critic = nn.Sequential(
            init_layer(nn.Linear(128, 64), std=2**0.5),
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

    def get_log_prob(self, observation, actions):
        policy_logits, _ = self(observation)
        log_probs = Categorical(logits=policy_logits).log_prob(actions)
        return log_probs


env = gym.make("LunarLander-v3")
policy = CartPolePolicy()

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

NUM_EPOCHS = 10000
BATCH_SIZE = 4096
k = 1

epoch_rewards = []
epoch_explained_variances = []

def compute_explained_variance(predictions, targets):
    target_variance = torch.var(targets)
    if target_variance == 0:
        return 0.0
    return (1 - torch.var(targets - predictions) / target_variance).item()


def plot_rewards(epoch_rewards, explained_variances):
    epochs = range(len(epoch_rewards))
    figure, reward_axis = plt.subplots()
    reward_axis.plot(epochs, epoch_rewards, label="Reward")
    reward_axis.set_xlabel("Epoch")
    reward_axis.set_ylabel("Reward")
    reward_axis.set_title("LunarLander Reward and Explained Variance")
    reward_axis.grid(True, alpha=0.3)

    variance_axis = reward_axis.twinx()
    variance_axis.plot(
        epochs,
        explained_variances,
        color="tab:orange",
        label="Explained variance",
    )
    variance_axis.set_ylabel("Explained variance")
    variance_axis.set_ylim(-1, 1)

    lines = reward_axis.lines + variance_axis.lines
    reward_axis.legend(lines, [line.get_label() for line in lines])
    figure.tight_layout()
    figure.savefig("reward_per_epoch_ppo.png", dpi=150)
    plt.close(figure)

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute GAE for one complete episode.

    rewards[t] = r_t
    values[t]  = V(s_t)
    """

    advantages = torch.zeros(len(rewards), dtype=torch.float32)

    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1].detach()

        delta = (
            rewards[t]
            + gamma * next_value
            - values[t].detach()
        )

        gae = delta + gamma * lam * gae
        advantages[t] = gae

    return advantages

def evaluate(self, observations, actions):
    logits, values = self(observations)

    distribution = Categorical(logits=logits)

    log_probs = distribution.log_prob(actions)
    entropy = distribution.entropy()

    return log_probs, values.squeeze(-1), entropy

for epoch in range(NUM_EPOCHS):
    batch_log_probs = []
    batch_values = []
    batch_value_targets = []
    batch_advantages = []
    batch_actions = []
    batch_observations = []

    num_steps = 0
    episode_rewards = []

    while num_steps < BATCH_SIZE:
        observation, _ = env.reset()

        log_probs = []
        rewards = []
        values = []
        actions = []
        observations = []

        terminated = truncated = False

        while not (terminated or truncated):
            observations.append(torch.from_numpy(observation))

            action, log_prob, value = policy.sample_action(observation)

            observation, reward, terminated, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.squeeze())
            actions.append(torch.Tensor(action)) # TODO?

            num_steps += 1

        advantages = compute_gae(
            rewards=rewards,
            values=values,
            gamma=GAMMA,
            lam=GAE_LAMBDA,
        )

        values_tensor = torch.stack(values)
        value_targets = values_tensor.detach() + advantages
    
        batch_log_probs.extend(log_probs)
        batch_advantages.extend(advantages)
        batch_values.extend(values)
        batch_value_targets.extend(value_targets)

        batch_actions.extend(actions)
        batch_observations.extend(observations)

        episode_rewards.append(sum(rewards))

    log_probs = torch.stack(batch_log_probs).detach() # TODO?
    advantages = torch.stack(batch_advantages)
    values = torch.stack(batch_values)
    value_targets = torch.stack(batch_value_targets)
    explained_variance = compute_explained_variance(values.detach(), value_targets)

    observations = torch.stack(batch_observations)

    actions = torch.stack(batch_actions)

    advantages = (
        advantages - advantages.mean()
    ) / (
        advantages.std() + 1e-8
    )

    bs = BATCH_SIZE // k
    for i in range(k):
        mb_log_probs = log_probs[bs * i : bs * (i + 1)]
        mb_values = values[bs * i : bs * (i + 1)]
        mb_value_targets = value_targets[bs * i : bs * (i + 1)]
        mb_advantages = advantages[bs * i : bs * (i + 1)]

        mb_actions = actions[bs * i : bs * (i + 1)]
        mb_observations = observations[bs * i : bs * (i + 1)]

        new_log_probs, new_values, entropy = policy.evaluate(
            mb_observations,
            mb_actions,
        )

        ratio = torch.exp(new_log_probs - mb_log_probs)

        surrogate1 = ratio * mb_advantages

        surrogate2 = (
            torch.clamp(
                ratio,
                1.0 - CLIP_EPS,
                1.0 + CLIP_EPS,
            )
            * mb_advantages
        )

        policy_loss = -torch.min(
            surrogate1,
            surrogate2,
        ).mean()

        value_loss = ((new_values - value_targets) ** 2).mean()

        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

    epoch_rewards.append(sum(episode_rewards) / len(episode_rewards))
    epoch_explained_variances.append(explained_variance)

    if epoch % (NUM_EPOCHS // 100) == 0:
        print(f"Epoch {epoch}, policy loss: {policy_loss.item():.3f}, value loss: {value_loss.item():.3f}")
        print(f"Episode reward: {sum(episode_rewards) / len(episode_rewards)}")
        print(f"Explained variance: {explained_variance:.3f}")
        plot_rewards(epoch_rewards, epoch_explained_variances)

env.close()