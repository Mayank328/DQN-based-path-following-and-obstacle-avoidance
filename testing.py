# import gymnasium as gym
# import numpy as np
# import pygame
# import torch
# import time
# from Vehicle_environment import VehicleEnv
# from DQN_agent import DQNAgent

# def test_model(model_path, num_episodes=5):
#     """
#     Test a trained model
#     Args:
#         model_path: Path to the saved model file
#         num_episodes: Number of episodes to test
#     """
#     # Initialize environment and agent
#     env = VehicleEnv()
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
    
#     # Create agent and load trained weights
#     agent = DQNAgent(state_size, action_size)
#     agent.epsilon = 0.0  # No exploration during testing
    
#     try:
#         # Load the model
#         agent.load_model(model_path)
        
#         for episode in range(num_episodes):
#             state, _ = env.reset()
#             total_reward = 0
#             steps = 0
#             done = False
            
#             print(f"\nStarting Episode {episode + 1}")
            
#             while not done:
#                 # Get action from trained agent
#                 action = agent.act(state)
                
#                 # Take action in environment
#                 next_state, reward, done, _, _ = env.step(action)
                
#                 total_reward += reward
#                 steps += 1
#                 state = next_state
                
#                 # Render the environment
#                 env.render()
                
#                 # Add delay to make visualization watchable
                
#                 # Print real-time metrics
#                 print(f"\rStep: {steps}, Current Reward: {reward:.2f}, "
#                       f"Total Reward: {total_reward:.2f}", end="")
                
#                 if done:
#                     print(f"\nEpisode {episode + 1} finished after {steps} steps")
#                     print(f"Total Reward: {total_reward:.2f}")
#                     time.sleep(1)  # Pause briefly between episodes
    
#     except KeyboardInterrupt:
#         print("\nTesting interrupted by user")
    
#     finally:
#         pygame.quit()

# if __name__ == "__main__":
#     # Get model path from user
#     model_path = input("Enter the path to your trained model file: ")
#     num_episodes = int(input("Enter number of episodes to test (default=5): ") or "5")
    
#     print(f"\nTesting model from: {model_path}")
#     print(f"Running {num_episodes} test episodes")
    
#     test_model(model_path, num_episodes)
import gymnasium as gym
import numpy as np
import pygame
import torch
import time
# from paths.Vehicle_environment_circular import VehicleEnv
# from paths.Vehicle_environment_non_circular import VehicleEnvNonCircular as VehicleEnv
from paths.Vehicle_environment_S import VehicleEnvSmoothPath as VehicleEnv
from DQN_agent import DQNAgent


def test_model(model_path, num_episodes=5):
    """Test the trained DQN agent on the VehicleEnv."""
    env = VehicleEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize agent
    agent = DQNAgent(state_size, action_size)
    agent.epsilon = 0.0  # Disable exploration during testing
    agent.load_model(model_path)

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        print(f"\nStarting Episode {episode + 1}")

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward

            env.render()
            time.sleep(0.0005)  # Slow down for visualization

            print(f"\rStep Reward: {reward:.2f}, Total Reward: {total_reward:.2f}", end="")

        print(f"\nEpisode {episode + 1} finished. Total Reward: {total_reward:.2f}")
        time.sleep(1)  # Pause between episodes

    env.close()


if __name__ == "__main__":
    model_path = input("Enter the path to your trained model file: ")
    num_episodes = int(input("Enter number of episodes to test (default=5): ") or 5)
    test_model(model_path, num_episodes)
