# Used for viewing models in the enviornment

import torch
import gymnasium as gym
from actor import Actor


# MODEL_PATH = "HP_Steps_Per_Episode/500/500s_best_lander_actor.pth"
# MODEL_PATH = "Baseline/1000s_best_lander_actor.pth"
MODEL_PATH = "HP_Steps_Per_Episode/250/250s_best_lander_actor.pth"

T_MAX = 500


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = gym.make(
    "LunarLanderContinuous-v3",
    render_mode="human",
    continuous=True,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5
)


actor = Actor(state_dim=8, action_dim=2, max_action=1.0).to(device)
actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
actor.eval()


print("Model loaded.")
print("Starting infinite evaluation loop...\n")


episode = 1

while True:

    state, _ = env.reset()
    episode_reward = 0

    for t in range(T_MAX):

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action = actor(state_tensor).cpu().numpy()[0]

        next_state, reward, terminated, truncated, _ = env.step(action)

        state = next_state
        episode_reward += reward

        if terminated or truncated:
            break

    print(f"Episode {episode} | Reward: {episode_reward:.2f}")

    episode += 1