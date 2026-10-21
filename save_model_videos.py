# Records episodes of each model in the environment and saves them as mp4 videos
# Requires: pip install imageio imageio-ffmpeg

import os
import torch
import gymnasium as gym
import imageio
from actor import Actor


MODEL_PATHS = [
    "HP_Steps_Per_Episode/500/500s_best_lander_actor.pth",
    "Baseline/1000s_best_lander_actor.pth",
    "HP_Steps_Per_Episode/250/250s_best_lander_actor.pth",
]

T_MAX = 500
NUM_EPISODES = 25
FPS = 50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for MODEL_PATH in MODEL_PATHS:

    env = gym.make(
        "LunarLanderContinuous-v3",
        render_mode="rgb_array",
        continuous=True,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5
    )

    actor = Actor(state_dim=8, action_dim=2, max_action=1.0).to(device)
    actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    actor.eval()

    model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    video_path = f"{model_name}_Playback.mp4"

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Recording {NUM_EPISODES} episodes to {video_path}...\n")

    with imageio.get_writer(video_path, fps=FPS) as writer:

        for episode in range(1, NUM_EPISODES + 1):

            state, _ = env.reset()
            episode_reward = 0

            writer.append_data(env.render())

            for t in range(T_MAX):

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                with torch.no_grad():
                    action = actor(state_tensor).cpu().numpy()[0]

                next_state, reward, terminated, truncated, _ = env.step(action)

                writer.append_data(env.render())

                state = next_state
                episode_reward += reward

                if terminated or truncated:
                    break

            print(f"Episode {episode} | Reward: {episode_reward:.2f}")

    env.close()
    print(f"\nSaved {video_path}\n")
