import matplotlib.pyplot as plt
import torch


def compute_explained_variance(predictions, targets):
    target_variance = torch.var(targets)
    if torch.isclose(target_variance, torch.tensor(0.0)):
        return 0.0
    return (1 - torch.var(targets - predictions) / target_variance).item()


def compute_gae(
    rewards,
    values,
    last_value,
    gamma=0.99,
    lam=0.95,
):
    """
    Compute GAE for one trajectory.

    rewards[t] = r_t
    values[t] = V(s_t)

    last_value:
        0                    if genuinely terminated
        V(s_{T+1})           if truncated
    """
    advantages = torch.zeros(
        len(rewards),
        dtype=torch.float32,
    )

    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1].detach()

        delta = rewards[t] + gamma * next_value - values[t].detach()

        gae = delta + gamma * lam * gae
        advantages[t] = gae

    return advantages


def plot_training_curve(
    epoch_rewards, explained_variances=None, save_path="reward_per_epoch.png"
):
    epochs = range(len(epoch_rewards))
    figure, reward_axis = plt.subplots()
    reward_axis.plot(epochs, epoch_rewards, label="Reward")
    reward_axis.set_xlabel("Epoch")
    reward_axis.set_ylabel("Reward")
    reward_axis.set_title("Training reward")
    reward_axis.grid(True, alpha=0.3)

    if explained_variances is not None:
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
    else:
        reward_axis.legend()

    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)
