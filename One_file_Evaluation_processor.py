# Use to generate a chart for a single models training data


import numpy as np
import matplotlib.pyplot as plt


# Matplotlib settings (large fonts)
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18
})

episodes = []
rewards = []

with open("data Path", "r") as f:
    # skip header
    next(f)
    for line in f:
        ep, reward = line.strip().split()
        episodes.append(int(ep))
        rewards.append(float(reward))

episodes = np.array(episodes)
rewards = np.array(rewards)

plt.figure(figsize=(12,6))

plt.plot(
    episodes,
    rewards,
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=5,
    label="Evaluation Reward"
)

plt.xlabel("Evaluation Episode")
plt.ylabel("Reward")
plt.title("Evaluation Performance Over 100 Episodes")

plt.grid(True)
plt.legend()

plt.tight_layout()

# Save high resolution figure
plt.savefig("filename.png", dpi=300)

plt.show()