# Used to evaluate the trained polocies over 100 episodes.
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from actor import Actor


def evaluate(model_path, episodes=100, render=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("LunarLanderContinuous-v3",
                   continuous=True,
                   gravity=-10.0,
                   enable_wind=False,
                   wind_power=15.0,
                   turbulence_power=1.5)

    actor = Actor(state_dim=8, action_dim=2, max_action=1.0).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()

    rewards = []

    # open log file
    log_file = open("Directory for Evaluation Rewards", "w")
    log_file.write("episode reward\n")

    episode_steps = 1000
    for ep in range(episodes):

        state, _ = env.reset()
        done = False
        ep_reward = 0

        for t in range(episode_steps):

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy()[0]

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            ep_reward += reward

            if terminated or truncated:
                break


        rewards.append(ep_reward)


        print(f"Episode {ep+1:3d} | Reward: {ep_reward:.2f}")

        log_file.write(f"{ep+1} {ep_reward}\n")
        log_file.flush()

    log_file.close()

    final_avg = np.mean(rewards)
    print("\nFinal Average:", final_avg)

    env.close()


    # Plot results

    plt.figure(figsize=(10,5))

    plt.plot(rewards, label="Episode Reward", alpha=0.6)


    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Lunar Lander Evaluation Rewards")
    plt.legend()

    plt.savefig("Directory")
    plt.show()


if __name__ == "__main__":
    evaluate("model path", episodes=100, render=False)