# import os
# import time
# import torch
# import numpy as np
# import pygame
# import matplotlib.pyplot as plt
# from DQN_agent import DQNAgent
# from paths.circular_road import VehicleEnv

# def plot_test_results(episode_rewards, save_dir="test_results"):
#     os.makedirs(save_dir, exist_ok=True)
#     plt.figure(figsize=(10, 5))
#     plt.plot(episode_rewards, 'b-', label='Episode Reward')
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#     plt.title('Test Rewards Over Time')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     plt.savefig(os.path.join(save_dir, f'test_results_{timestamp}.png'))
#     plt.close()

# def test_model(model_path, num_episodes=5, force_cpu=False):
#     device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     env = VehicleEnv()
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
#     agent = DQNAgent(state_size, action_size)
#     agent.epsilon = 0.0  # No exploration during testing
#     print(f"Loading model from: {model_path}")
#     agent.load_model(model_path)
#     agent.policy_net = agent.policy_net.to(device)
#     agent.target_net = agent.target_net.to(device)
#     episode_rewards = []
#     try:
#         for episode in range(num_episodes):
#             state, _ = env.reset()
#             total_reward = 0
#             done = False
#             print(f"\nStarting Episode {episode + 1}")
#             while not done:
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
#                 with torch.no_grad():
#                     action = agent.policy_net(state_tensor).argmax().item()
#                 next_state, reward, done, _, _ = env.step(action)
#                 total_reward += reward
#                 state = next_state
#                 env.render()
#                 time.sleep(0.5)  # Slow down visualization
#                 print(f"\rStep Reward: {reward:.2f}, Total Reward: {total_reward:.2f}", end="")
#             print(f"\nEpisode {episode + 1} finished. Total Reward: {total_reward:.2f}")
#             episode_rewards.append(total_reward)
#             time.sleep(1)
#         plot_test_results(episode_rewards)
#     except KeyboardInterrupt:
#         print("\nTesting interrupted")
#     finally:
#         pygame.quit()

# if __name__ == "__main__":
#     model_path = input("Enter the path to your trained model file: ")
#     num_episodes = int(input("Enter number of episodes to test (default=5): ") or "5")
#     force_cpu = input("Force CPU usage? (y/n, default=n): ").lower().startswith('y')
#     test_model(model_path, num_episodes, force_cpu)

import os
import time
import torch
import numpy as np
import pygame
import matplotlib.pyplot as plt
from DQN_agent import DQNAgent
from paths.circular_road import VehicleEnv
# from paths.square_road import VehicleEnv
# from paths.S_path_road import VehicleEnv
from paths.square_with_curved_corner import VehicleEnv

def plot_test_results(episode_rewards, save_dir="test_results"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, 'b-', label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Test Rewards Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(save_dir, f'test_results_{timestamp}.png'))
    plt.close()


def test_model(model_path, num_episodes=5, force_cpu=False):
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VehicleEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.epsilon = 0.0  # No exploration during testing
    agent.load_model(model_path)
    agent.policy_net = agent.policy_net.to(device)
    agent.target_net = agent.target_net.to(device)

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        a = b = c = d = e = f = 0  # Initialize counters

        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.policy_net(state_tensor).argmax().item()
            next_state, reward, done, _, _ = env.step(action)

            total_reward += reward

            # Counting reward cases
            if reward > 0:
                a += 1  # positive reward
            elif -5 < reward <= 0:
                b += 1  # slight path deviation negative reward
            elif reward <= -5:
                c += 1  # major path deviation negative reward

            # Obstacles related rewards
            closest_obstacle_dist = min(
                np.sqrt((obs[0] - env.vehicle_pos[0])**2 + (obs[1] - env.vehicle_pos[1])**2)
                for obs in env.obstacles)

            if closest_obstacle_dist < (env.vehicle_size + env.obstacle_size) / 2:
                d += 1  # hit obstacle
            elif (env.vehicle_size + env.obstacle_size) / 2 <= closest_obstacle_dist <= env.obstacle_size * 2:
                e += 1  # crosses obstacle closely
            elif closest_obstacle_dist > env.obstacle_size * 2:
                f += 1  # completely avoids obstacle

            state = next_state
            env.render()
            time.sleep(0.0001)

        episode_rewards.append(total_reward)

        print(f"\nEpisode {episode + 1} finished. Total Reward: {total_reward:.2f}")
        print(f"Reward summary:")
        print(f"Positive rewards given: {a}")
        print(f"Slight path deviation penalties: {b}")
        print(f"Major path deviation penalties: {c}")
        print(f"Obstacle hit penalties: {d}")
        print(f"Obstacle closely crossed events: {e}")
        print(f"Obstacle completely avoided events: {f}")

        time.sleep(0.0005)

    plot_test_results(episode_rewards)


if __name__ == "__main__":
    model_path = input("Enter the path to your trained model file: ")
    num_episodes = int(input("Enter number of episodes to test (default=5): ") or "5")
    force_cpu = input("Force CPU usage? (y/n, default=n): ").lower().startswith('y')
    test_model(model_path, num_episodes, force_cpu)
