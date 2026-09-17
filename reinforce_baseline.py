import gymnasium as gym
import torch

from common import compute_explained_variance, compute_gae, plot_training_curve
from model import ActorCriticPolicy

GAMMA = 0.99
GAE_LAMBDA = 0.95
NUM_EPOCHS = 10000
BATCH_SIZE = 4096

env = gym.make("LunarLander-v3")
policy = ActorCriticPolicy()
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

epoch_rewards = []
epoch_explained_variances = []

for epoch in range(NUM_EPOCHS):
    batch_log_probs = []
    batch_values = []
    batch_value_targets = []
    batch_advantages = []

    num_steps = 0
    episode_rewards = []

    while num_steps < BATCH_SIZE:
        observation, _ = env.reset()

        log_probs = []
        rewards = []
        values = []

        terminated = truncated = False

        while not (terminated or truncated):
            action, log_prob, value = policy.sample_action(observation)

            observation, reward, terminated, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.squeeze())
            num_steps += 1

        if terminated:
            last_value = 0.0
        else:
            # truncated
            with torch.no_grad():
                final_observation = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                )
                _, last_value = policy(final_observation)
                last_value = last_value.squeeze()

        advantages = compute_gae(
            rewards=rewards,
            values=values,
            gamma=GAMMA,
            lam=GAE_LAMBDA,
            last_value=last_value,
        )

        values_tensor = torch.stack(values)
        value_targets = values_tensor.detach() + advantages

        batch_log_probs.extend(log_probs)
        batch_advantages.extend(advantages)
        batch_values.extend(values)
        batch_value_targets.extend(value_targets)

        episode_rewards.append(sum(rewards))

    log_probs = torch.stack(batch_log_probs)
    advantages = torch.stack(batch_advantages)
    values = torch.stack(batch_values)
    value_targets = torch.stack(batch_value_targets)
    explained_variance = compute_explained_variance(
        values.detach(),
        value_targets,
    )

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = ((values - value_targets.detach()) ** 2).mean()

    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()

    epoch_rewards.append(sum(episode_rewards) / len(episode_rewards))
    epoch_explained_variances.append(explained_variance)

    if epoch % (NUM_EPOCHS // 100) == 0:
        print(
            "Epoch "
            f"{epoch}, policy loss: {policy_loss.item():.3f}, "
            f"value loss: {value_loss.item():.3f}"
        )
        print(f"Episode reward: {sum(episode_rewards) / len(episode_rewards)}")
        print(f"Explained variance: {explained_variance:.3f}")
        plot_training_curve(
            epoch_rewards,
            epoch_explained_variances,
            save_path="plots/reinforce_baseline.png",
        )

env.close()
