# Used to conglomerate Training Rewards for charts

import os
import numpy as np
import matplotlib.pyplot as plt

# increase all font sizes
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18
})

# color_map = {
#     "1e-4": "tab:blue",
#     "5e-4": "tab:orange",
#     "5e-4_t500": "tab:green",
#     "5e-5": "tab:red"
# }

data_directory = "Training Rewards Directory"

window = 100

# color_map = {
#     "01D": "tab:blue",
#     "01ND": "tab:orange",
#     "02D": "tab:green"
# }

def moving_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# For Exploration Noise
# def parse_label(prefix):
#     """
#     Convert filename prefix into readable label.
#
#     01D  -> exploration noise = 0.1 with decay
#     01ND -> exploration noise = 0.1 without decay
#     02D  -> exploration noise = 0.2 with decay
#     """
#
#     if prefix == "01D":
#         return "Noise=0.1 (decay)"
#     elif prefix == "01ND":
#         return "Noise=0.1 (no decay)"
#     elif prefix == "02D":
#         return "Noise=0.2 (decay)"
#     else:
#         return prefix

# For Learning rate
# def parse_label(prefix):
#     """
#     Convert filename prefix into readable label.
#     Examples:
#     1e-4        -> LR = 1e-4 (baseline)
#     5e-4        -> LR = 5e-4
#     5e-5        -> LR = 5e-5
#     5e-4_t500   -> LR = 5e-4, max steps = 500
#     """
#
#     if "_t" in prefix:
#         lr, step = prefix.split("_t")
#         return f"LR={lr}, max steps={step}"
#     else:
#         if prefix == "1e-4":
#             return f"LR={prefix} (baseline)"
#         return f"LR={prefix}"

# For Episode length
color_map = {
    "1000": "tab:blue",   # baseline
    "500": "tab:orange",
    "250": "tab:green"
}

def parse_label(prefix):
    """
    Convert filename prefix into readable label.

    250  -> max steps = 250
    500  -> max steps = 500
    1000 -> max steps = 1000 (baseline)
    """

    if prefix == "1000":
        return "Max steps = 1000 (baseline)"
    else:
        return f"Max steps = {prefix}"

plt.figure(figsize=(10,6))

for file in sorted(os.listdir(data_directory)):

    if file.endswith("_training_rewards.txt"):

        filepath = os.path.join(data_directory, file)

        prefix = file.replace("_training_rewards.txt", "")
        label = parse_label(prefix)

        rewards = []

        with open(filepath, "r") as f:
            lines = f.readlines()

            # skip header
            for line in lines[1:]:
                parts = line.strip().split()

                if len(parts) == 2:
                    reward = float(parts[1])
                    rewards.append(reward)

        rewards = np.array(rewards)

        if len(rewards) < window:
            continue

        avg_rewards = moving_average(rewards, window)
        episodes = np.arange(len(avg_rewards))

        plt.plot(
            episodes,
            avg_rewards,
            label=label,
            color=color_map.get(prefix)   # <-- only fix added
        )


plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Effect of Episode Length on TD3 Training Performance")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("file name", dpi=300)

plt.show()