# Used to train the model, and log its data
# Logs are commented out to avoid generating files on the graders device
# IMPORTS
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from TD3Agent import TD3Agent
from replay_buffer import ReplayBuffer

# Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Params
state_dim = 8
action_dim = 2
max_action = 1.0

train_episodes = 1500
batch_size = 256


exploration_noise = 0.1
noise_min = 0.05
noise_decay = 0.9991 # Decay rate per episode

# Env setup
env = gym.make("LunarLanderContinuous-v3",
    continuous=True,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5)




# TRAINING (no render)

directory_prefix = "destination directory"

agent = TD3Agent(state_dim, action_dim, max_action, device=device)
replay_buffer = ReplayBuffer(state_dim, action_dim, device=device)

print("Starting training...")

rewards_history = []


# log_file = open(f"{directory_prefix}/training_rewards.txt", "w") # FILE NAME
# log_file.write("episode reward\n")


best_avg_reward = -np.inf  # Track the highest 100-episode average
save_path = f"{directory_prefix}/best_lander_actor.pth"

for episode in range(train_episodes):

    state, _ = env.reset()
    episode_reward = 0


    for t in range(500):

        # Warmup
        if replay_buffer.size < 10000:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)
            action = (
                # Gaussian Noise
                action + np.random.normal(0, exploration_noise, size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode_reward += reward  # <-- REQUIRED

        replay_buffer.add(state, action, reward, next_state, float(done))

        state = next_state


        if replay_buffer.size > 10000:
            agent.train(replay_buffer, batch_size)


        if done:
            break

    # Store the episode reward
    rewards_history.append(episode_reward)
    # log_file.write(f"{episode + 1} {episode_reward}\n")

    #  Calculate rolling average of last 100 episodes
    if len(rewards_history) >= 100:
        current_avg = np.mean(rewards_history[-100:])

        # If this is the best keep
        if current_avg > best_avg_reward:
            best_avg_reward = current_avg
            torch.save(agent.actor.state_dict(), save_path)
            # print(f"--> New Best Average: {best_avg_reward:.2f} (Model Saved) Episode number {episode + 1}")

    exploration_noise = max(noise_min, exploration_noise * noise_decay)
    print(f"Episode {episode+1} | Reward: {episode_reward:.2f}")

env.close()
# log_file.close()

print("Training complete.")
print("Training complete.")

# Save the weights

# -------------------------
# VISUAL EVALUATION PHASE
# -------------------------

plt.figure(figsize=(10,5))
plt.plot(rewards_history, label="Episode Reward", alpha=0.3)
# Calculate moving average for the plot
if len(rewards_history) >= 100:
    moving_avg = [np.mean(rewards_history[max(0, i-100):i+1]) for i in range(len(rewards_history))]
    plt.plot(moving_avg, label="100-Ep Moving Avg", color='red')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Lunar Lander TD3 Training Progress")
plt.legend()
# plt.savefig(f"{directory_prefix}/training_rewards.png")
plt.show()

eval_env = gym.make(
    "LunarLanderContinuous-v3",
    continuous=True,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
    render_mode="human"
)

# Loads the just trained model and displays performance in the env
print(f"Loading best model with peak average: {best_avg_reward:.2f}")
agent.actor.load_state_dict(torch.load(save_path))
agent.actor.eval() # Switch to eval mode (no dropout/batchnorm updates)
print("Starting visual evaluation...")

while True:

    state, _ = eval_env.reset()
    done = False
    episode_reward = 0

    while not done:

        # Deterministic action (NO exploration noise)
        action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated

        state = next_state
        episode_reward += reward

        # force rendering
        eval_env.render()

    print(f"Evaluation Reward: {episode_reward:.2f}")