# Used to conglomerate Evaluation Rewards for charts


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
    "legend.fontsize": 16
})

data_directory = "Testing Rewards Directory"

# color_map = {
#     "01D": "tab:blue",
#     "01ND": "tab:orange",
#     "02D": "tab:green"
# }

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


# color_map = {
#     "1e-4": "tab:blue",
#     "5e-4": "tab:orange",
#     "5e-4_t500": "tab:green",
#     "5e-5": "tab:red"
# }
#
#
# def parse_label(prefix):
#     """
#     Convert filename prefix into readable label.
#
#     Examples
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

    if file.endswith("_evaluation_rewards.txt"):

        filepath = os.path.join(data_directory, file)

        prefix = file.replace("_evaluation_rewards.txt", "")
        label = parse_label(prefix)

        rewards = []

        with open(filepath, "r") as f:

            lines = f.readlines()

            # Skip header
            for line in lines[1:]:

                parts = line.strip().split()

                if len(parts) == 2:
                    reward = float(parts[1])
                    rewards.append(reward)

        rewards = np.array(rewards)

        episodes = np.arange(1, len(rewards) + 1)

        plt.plot(
            episodes,
            rewards,
            marker='o',
            label=label,
            color=color_map.get(prefix)
        )


plt.xlabel("Evaluation Episode")
plt.ylabel("Reward")
plt.title("Evaluation Performance for Different Episode Lengths")
# Legend moved to bottom to not be in way of data
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.grid(True)

plt.tight_layout()

plt.savefig("FileName", dpi=300, bbox_inches="tight")

plt.show()