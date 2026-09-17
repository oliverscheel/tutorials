import gymnasium as gym
import torch

from common import compute_explained_variance, compute_gae, plot_training_curve
from model import ActorCriticPolicy

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 4
MINIBATCH_SIZE = 512
ENTROPY_COEF = 0.01


env = gym.make("LunarLander-v3")
policy = ActorCriticPolicy()

optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

NUM_EPOCHS = 10000
BATCH_SIZE = 4096

epoch_rewards = []
epoch_explained_variances = []

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
            observations.append(torch.as_tensor(observation, dtype=torch.float32))

            action, log_prob, value = policy.sample_action(observation)

            observation, reward, terminated, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.squeeze())
            actions.append(torch.tensor(action, dtype=torch.long))

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

        batch_actions.extend(actions)
        batch_observations.extend(observations)

        episode_rewards.append(sum(rewards))

    old_log_probs = torch.stack(batch_log_probs).detach()
    advantages = torch.stack(batch_advantages).detach()
    values = torch.stack(batch_values)
    value_targets = torch.stack(batch_value_targets).detach()
    explained_variance = compute_explained_variance(values.detach(), value_targets)

    observations = torch.stack(batch_observations)
    actions = torch.stack(batch_actions)

    advantages = (
        advantages - advantages.mean()
    ) / (
        advantages.std() + 1e-8
    )

    N = len(actions)

    for ppo_epoch in range(PPO_EPOCHS):
        indices = torch.randperm(N)

        for start in range(0, N, MINIBATCH_SIZE):
            mb_indices = indices[start:start + MINIBATCH_SIZE]

            mb_old_log_probs = old_log_probs[mb_indices]
            mb_value_targets = value_targets[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_actions = actions[mb_indices]
            mb_observations = observations[mb_indices]

            new_log_probs, new_values, entropy = policy.evaluate(
                mb_observations,
                mb_actions,
            )

            ratio = torch.exp(
                new_log_probs - mb_old_log_probs
            )

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

            value_loss = (
                (new_values - mb_value_targets) ** 2
            ).mean()

            entropy_loss = entropy.mean()
            loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), 0.5
            )
            optimizer.step()

    epoch_rewards.append(sum(episode_rewards) / len(episode_rewards))
    epoch_explained_variances.append(explained_variance)

    if epoch % (NUM_EPOCHS // 100) == 0:
        print(f"Epoch {epoch}, policy loss: {policy_loss.item():.3f}, value loss: {value_loss.item():.3f}, entropy_loss: {entropy_loss.item():.3f}")
        print(f"Episode reward: {sum(episode_rewards) / len(episode_rewards)}")
        print(f"Explained variance: {explained_variance:.3f}")
        plot_training_curve(epoch_rewards, epoch_explained_variances, save_path="plots/ppo.png")

env.close()