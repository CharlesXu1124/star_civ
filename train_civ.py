import pygame
import random
import math
import numpy as np
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env import GalaxySimEnv

# --- Constants ---
# Screen dimensions (optional for rendering)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
RENDER_FPS = 15 # Lower FPS for rendering during evaluation

# Galaxy properties
NUM_STARS = 150 # Reduced for potentially faster simulation steps
GALAXY_PADDING = 50

# Star properties
STAR_RADIUS = 3
UNCLAIMED_COLOR = (100, 100, 100)
DESTROYED_STAR_COLOR = (255, 0, 0)
RESOURCE_INDICATOR_COLOR = (255, 255, 0)
RESOURCE_INDICATOR_RADIUS = 1
STAR_RESOURCE_PROBABILITY = 0.15
MIN_STAR_RESOURCE = 50
MAX_STAR_RESOURCE = 200
STAR_MAX_VELOCITY = 0.03

# --- Visualization ---
DASH_LENGTH = 4
GAP_LENGTH = 3

# --- Helper Function for Dashed Lines ---
def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=DASH_LENGTH, gap_length=GAP_LENGTH):
    """Draws a dashed line on a surface."""
    x1, y1 = start_pos
    x2, y2 = end_pos
    dl = dash_length
    gl = gap_length

    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0: return # Avoid division by zero

    dashes = int(dist / (dl + gl))
    if dashes == 0: # If too short for a full dash+gap, draw solid
         pygame.draw.line(surf, color, start_pos, end_pos, width)
         return

    # Calculate unit vector
    ux = dx / dist
    uy = dy / dist

    # Draw dashes
    for i in range(dashes):
        start = (x1 + ux * (dl + gl) * i, y1 + uy * (dl + gl) * i)
        end = (x1 + ux * ((dl + gl) * i + dl), y1 + uy * ((dl + gl) * i + dl))
        pygame.draw.line(surf, color, start, end, width)

    # Draw the last partial dash if needed
    remaining_dist = dist - dashes * (dl + gl)
    if remaining_dist > 0:
        start = (x1 + ux * (dl + gl) * dashes, y1 + uy * (dl + gl) * dashes)
        # Draw only up to the end point or the dash length, whichever is smaller
        final_dash_len = min(dl, remaining_dist)
        end = (start[0] + ux * final_dash_len, start[1] + uy * final_dash_len)
        # Ensure the final segment doesn't overshoot the end_pos due to float precision
        final_end_x = min(end[0], x2) if dx > 0 else max(end[0], x2)
        final_end_y = min(end[1], y2) if dy > 0 else max(end[1], y2)
        # Crude check to prevent overshooting on near-vertical/horizontal lines
        if abs(final_end_x - x1) > abs(dx) or abs(final_end_y - y1) > abs(dy):
             final_end_x, final_end_y = x2, y2

        pygame.draw.line(surf, color, start, (final_end_x, final_end_y), width)


# --- Example Usage (Requires stable-baselines3) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="whether to run training")
    args = parser.parse_args()

    print("Creating the environment...")
    # Create the environment with human rendering enabled
    env = GalaxySimEnv(render_mode='human')
    print("Environment created. Running random actions for a few steps...")

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    # Run for a limited number of steps with random actions
    if not args.train:
        max_random_steps = 10000 # Increased steps to see more action
        while step_count < max_random_steps and not terminated and not truncated:
            action = env.action_space.sample() # Choose a random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            # Render the environment state (if render_mode='human')
            env.render()
            # time.sleep(0.01) # Optional small delay
            if terminated or truncated:
                print(f"Episode finished after {step_count} steps. Terminated: {terminated}, Truncated: {truncated}")
                print(f"Total Reward: {total_reward}")
                # obs, info = env.reset() # Reset for next episode if needed
                # terminated = False
                # truncated = False
                # total_reward = 0
                # step_count = 0

        print("Finished random action example.")
        env.close()

    else:
        try:
            # Instantiate the env
            # For training, typically don't render: render_mode=None
            # Use make_vec_env for parallel environments (optional but faster)
            # vec_env = make_vec_env(lambda: GalaxySimEnv(render_mode=None), n_envs=4)
            train_env = GalaxySimEnv(render_mode=None)

            # Define the model
            model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./galaxy_ppo_tensorboard/")

            # Train the model
            print("Starting PPO training (this will take a long time)...")
            # Increase total_timesteps significantly for actual training (e.g., 1e6 or more)
            model.learn(total_timesteps=10000, log_interval=10) # Train for a small number of steps as demo
            model.save("ppo_galaxy_sim")
            print("Training finished and model saved.")

        except ImportError:
            print("\nStable Baselines3 not found. Skipping training example.")
            print("Install it using: pip install stable-baselines3[extra]")
        except Exception as e:
            print(f"\nAn error occurred during the training/evaluation placeholder: {e}")
