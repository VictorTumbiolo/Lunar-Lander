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

with open("Data path", "r") as f:
    # skip header line
    next(f)
    for line in f:
        ep, reward = line.strip().split()
        episodes.append(int(ep))
        rewards.append(float(reward))

episodes = np.array(episodes)
rewards = np.array(rewards)


# 100-episode moving average
window = 100
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

# episodes for moving average (shifted because of window)
ma_episodes = episodes[window-1:]

plt.figure(figsize=(12,6))

plt.plot(episodes, rewards, alpha=0.4, label="Episode Reward")
plt.plot(ma_episodes, moving_avg, linewidth=3, label="100-Episode Moving Average")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Performance")

plt.legend()

plt.grid(True)
plt.tight_layout()

# save high resolution
plt.savefig("file_name.png", dpi=300)

plt.show()