# import gymnasium as gym
# import numpy as np
# import pygame
# import torch
# import time
# from paths.Vehicle_environment_circular import VehicleEnv
# # from paths.Vehicle_environment_non_circular import VehicleEnvNonCircular as VehicleEnv
# # from paths.Vehicle_environment_S import VehicleEnvSmoothPath as VehicleEnv
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
# import gymnasium as gym
# import numpy as np
# import pygame
# import torch
# import time
# from paths.Vehicle_environment_circular import VehicleEnv
# # from paths.Vehicle_environment_non_circular import VehicleEnvNonCircular as VehicleEnv
# # from paths.Vehicle_environment_S import VehicleEnvSmoothPath as VehicleEnv
# from DQN_agent import DQNAgent


# def test_model(model_path, num_episodes=5):
#     """Test the trained DQN agent on the VehicleEnv."""
#     env = VehicleEnv()
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n

#     # Initialize agent
#     agent = DQNAgent(state_size, action_size)
#     agent.epsilon = 0.0  # Disable exploration during testing
#     agent.load_model(model_path)

#     for episode in range(num_episodes):
#         state, _ = env.reset()
#         total_reward = 0
#         done = False

#         print(f"\nStarting Episode {episode + 1}")

#         while not done:
#             action = agent.act(state)
#             next_state, reward, done, _, _ = env.step(action)
#             state = next_state
#             total_reward += reward

#             env.render()
#             time.sleep(0.0005)  # Slow down for visualization

#             print(f"\rStep Reward: {reward:.2f}, Total Reward: {total_reward:.2f}", end="")

#         print(f"\nEpisode {episode + 1} finished. Total Reward: {total_reward:.2f}")
#         time.sleep(1)  # Pause between episodes

#     env.close()


# if __name__ == "__main__":
#     model_path = input("Enter the path to your trained model file: ")
#     num_episodes = int(input("Enter number of episodes to test (default=5): ") or 5)
#     test_model(model_path, num_episodes)

# import gymnasium as gym
# import numpy as np
# import pygame
# import torch
# import time
# from paths.Vehicle_environment_circular import VehicleEnv
# # from paths.Vehicle_environment_non_circular import VehicleEnvNonCircular as VehicleEnv
# # from paths.Vehicle_environment_S import VehicleEnvSmoothPath as VehicleEnv
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
#                 time.sleep(0.05)
                
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

# import gymnasium as gym
# import numpy as np
# import pygame
# import torch
# import time
# from paths.Vehicle_environment_circular import VehicleEnv
# # from paths.Vehicle_environment_non_circular import VehicleEnvNonCircular as VehicleEnv
# # from paths.Vehicle_environment_S import VehicleEnvSmoothPath as VehicleEnv
# from DQN_agent import DQNAgent

# def test_model(model_path, num_episodes=5):
#     """
#     Test a trained model using GPU acceleration
#     Args:
#         model_path: Path to the saved model file
#         num_episodes: Number of episodes to test
#     """
#     # Verify CUDA availability and set up device
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA is not available. Please check your GPU setup.")
    
#     device = torch.device("cuda")
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
#     print(f"CUDA Device Count: {torch.cuda.device_count()}")
    
#     # Initialize environment and agent
#     env = VehicleEnv()
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
    
#     # Create agent and explicitly move to GPU
#     agent = DQNAgent(state_size, action_size)
#     agent.policy_net = agent.policy_net.to(device)
#     agent.target_net = agent.target_net.to(device)
#     agent.epsilon = 0.0  # No exploration during testing
    
#     try:
#         # Load the model with GPU mapping
#         print(f"Loading model from: {model_path}")
#         checkpoint = torch.load(model_path, map_location=device)
#         agent.policy_net.load_state_dict(checkpoint)
#         agent.target_net.load_state_dict(checkpoint)
#         print("Model loaded successfully on GPU")
        
#         # Performance metrics
#         episode_rewards = []
#         episode_steps = []
        
#         for episode in range(num_episodes):
#             state, _ = env.reset()
#             total_reward = 0
#             steps = 0
#             done = False
            
#             print(f"\nStarting Episode {episode + 1}")
#             episode_start_time = time.time()
            
#             while not done:
#                 # Convert state to tensor and move to GPU
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
#                 # Get action using GPU-accelerated forward pass
#                 with torch.no_grad():
#                     action = agent.policy_net(state_tensor).argmax().item()
                
#                 # Take action in environment
#                 next_state, reward, done, _, _ = env.step(action)
                
#                 total_reward += reward
#                 steps += 1
#                 state = next_state
                
#                 # Render the environment
#                 env.render()
                
#                 # Control visualization speed
#                 time.sleep(0.05)
                
#                 # Print real-time metrics with GPU memory usage
#                 gpu_memory_used = torch.cuda.memory_allocated(device) / 1024**2  # Convert to MB
#                 print(f"\rStep: {steps}, Current Reward: {reward:.2f}, "
#                       f"Total Reward: {total_reward:.2f}, "
#                       f"GPU Memory Used: {gpu_memory_used:.2f}MB", end="")
                
#                 if done:
#                     episode_time = time.time() - episode_start_time
#                     episode_rewards.append(total_reward)
#                     episode_steps.append(steps)
                    
#                     print(f"\nEpisode {episode + 1} Summary:")
#                     print(f"Steps: {steps}")
#                     print(f"Total Reward: {total_reward:.2f}")
#                     print(f"Episode Time: {episode_time:.2f}s")
#                     print(f"Average Step Time: {episode_time/steps:.4f}s")
#                     time.sleep(1)  # Pause briefly between episodes
        
#         # Print final statistics
#         print("\nTesting Summary:")
#         print(f"Average Reward: {np.mean(episode_rewards):.2f}")
#         print(f"Average Steps per Episode: {np.mean(episode_steps):.2f}")
#         print(f"Best Episode Reward: {max(episode_rewards):.2f}")
#         print(f"Worst Episode Reward: {min(episode_rewards):.2f}")
    
#     except KeyboardInterrupt:
#         print("\nTesting interrupted by user")
#     except Exception as e:
#         print(f"\nError during testing: {str(e)}")
#     finally:
#         # Clean up GPU memory
#         torch.cuda.empty_cache()
#         pygame.quit()

# if __name__ == "__main__":
#     try:
#         # Get model path from user
#         model_path = input("Enter the path to your trained model file: ")
#         num_episodes = int(input("Enter number of episodes to test (default=5): ") or "5")
        
#         print("\nInitializing GPU testing...")
#         test_model(model_path, num_episodes)
    
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         print("Please ensure your GPU and CUDA are properly configured.")

# import gymnasium as gym
# import numpy as np
# import pygame
# import torch
# import time
# from paths.Vehicle_environment_circular import VehicleEnv
# # from paths.Vehicle_environment_non_circular import VehicleEnvNonCircular as VehicleEnv
# # from paths.Vehicle_environment_S import VehicleEnvSmoothPath as VehicleEnv
# from DQN_agent import DQNAgent


# def test_model(model_path, num_episodes=5):
#     """
#     Test a trained model using GPU acceleration
#     Args:
#         model_path: Path to the saved model file
#         num_episodes: Number of episodes to test
#     """
#     # Verify CUDA availability and set up device
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA is not available. Please check your GPU setup.")
    
#     device = torch.device("cuda")
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
#     print(f"CUDA Device Count: {torch.cuda.device_count()}")
    
#     # Initialize environment and agent
#     env = VehicleEnv()
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
    
#     # Create agent and explicitly move to GPU
#     agent = DQNAgent(state_size, action_size)
#     agent.policy_net = agent.policy_net.to(device)
#     agent.target_net = agent.target_net.to(device)
#     agent.epsilon = 0.0  # No exploration during testing
    
#     try:
#         # Load the model with GPU mapping
#         print(f"Loading model from: {model_path}")
#         checkpoint = torch.load(model_path, map_location=device)
#         agent.policy_net.load_state_dict(checkpoint)
#         agent.target_net.load_state_dict(checkpoint)
#         print("Model loaded successfully on GPU")
        
#         # Performance metrics
#         episode_rewards = []
#         episode_steps = []
        
#         for episode in range(num_episodes):
#             state, _ = env.reset()
#             total_reward = 0
#             steps = 0
#             done = False
            
#             print(f"\nStarting Episode {episode + 1}")
#             episode_start_time = time.time()
            
#             while not done:
#                 # Convert state to tensor and move to GPU
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
#                 # Get action using GPU-accelerated forward pass
#                 with torch.no_grad():
#                     action = agent.policy_net(state_tensor).argmax().item()
                
#                 # Take action in environment
#                 next_state, reward, done, _, _ = env.step(action)
                
#                 total_reward += reward
#                 steps += 1
#                 state = next_state
                
#                 # Render the environment
#                 env.render()
                
#                 # Control visualization speed
#                 time.sleep(0.05)
                
#                 # Print real-time metrics with GPU memory usage
#                 gpu_memory_used = torch.cuda.memory_allocated(device) / 1024**2  # Convert to MB
#                 print(f"\rStep: {steps}, Current Reward: {reward:.2f}, "
#                       f"Total Reward: {total_reward:.2f}, "
#                       f"GPU Memory Used: {gpu_memory_used:.2f}MB", end="")
                
#                 if done:
#                     episode_time = time.time() - episode_start_time
#                     episode_rewards.append(total_reward)
#                     episode_steps.append(steps)
                    
#                     print(f"\nEpisode {episode + 1} Summary:")
#                     print(f"Steps: {steps}")
#                     print(f"Total Reward: {total_reward:.2f}")
#                     print(f"Episode Time: {episode_time:.2f}s")
#                     print(f"Average Step Time: {episode_time/steps:.4f}s")
#                     time.sleep(1)  # Pause briefly between episodes
        
#         # Print final statistics
#         print("\nTesting Summary:")
#         print(f"Average Reward: {np.mean(episode_rewards):.2f}")
#         print(f"Average Steps per Episode: {np.mean(episode_steps):.2f}")
#         print(f"Best Episode Reward: {max(episode_rewards):.2f}")
#         print(f"Worst Episode Reward: {min(episode_rewards):.2f}")
    
#     except KeyboardInterrupt:
#         print("\nTesting interrupted by user")
#     except Exception as e:
#         print(f"\nError during testing: {str(e)}")
#     finally:
#         # Clean up GPU memory
#         torch.cuda.empty_cache()
#         pygame.quit()

# if __name__ == "__main__":
#     try:
#         # Get model path from user
#         model_path = input("Enter the path to your trained model file: ")
#         num_episodes = int(input("Enter number of episodes to test (default=5): ") or "5")
        
#         print("\nInitializing GPU testing...")
#         test_model(model_path, num_episodes)
    
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         print("Please ensure your GPU and CUDA are properly configured.")



###FORCED CPU#####

# import gymnasium as gym
# import numpy as np
# import pygame
# import torch
# import time
# from paths.Vehicle_environment_circular import VehicleEnv
# from DQN_agent import DQNAgent

# def test_model(model_path, num_episodes=5, force_cpu=False):
#     """
#     Test a trained model with flexible device selection
#     Args:
#         model_path: Path to the saved model file
#         num_episodes: Number of episodes to test
#         force_cpu: Force CPU usage even if GPU is available
#     """
#     # Device selection
#     if force_cpu:
#         device = torch.device("cpu")
#         print("Forcing CPU usage")
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if device.type == "cuda":
#             print(f"Using GPU: {torch.cuda.get_device_name(0)}")
#         else:
#             print("Using CPU (GPU not available)")
    
#     # Initialize environment and agent
#     env = VehicleEnv()
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
    
#     # Create agent
#     agent = DQNAgent(state_size, action_size)
#     agent.epsilon = 0.0  # No exploration during testing
    
#     try:
#         # Load the model with appropriate device mapping
#         print(f"Loading model from: {model_path}")
#         checkpoint = torch.load(model_path, map_location=device)
#         agent.policy_net.load_state_dict(checkpoint)
#         agent.target_net.load_state_dict(checkpoint)
        
#         # Move networks to selected device
#         agent.policy_net = agent.policy_net.to(device)
#         agent.target_net = agent.target_net.to(device)
#         print(f"Model loaded successfully on {device.type.upper()}")
        
#         # Performance metrics
#         episode_rewards = []
#         episode_steps = []
        
#         for episode in range(num_episodes):
#             state, _ = env.reset()
#             total_reward = 0
#             steps = 0
#             done = False
            
#             print(f"\nStarting Episode {episode + 1}")
#             episode_start_time = time.time()
            
#             while not done:
#                 # Convert state to tensor and move to appropriate device
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
#                 # Get action
#                 with torch.no_grad():
#                     action = agent.policy_net(state_tensor).argmax().item()
                
#                 # Take action in environment
#                 next_state, reward, done, _, _ = env.step(action)
                
#                 total_reward += reward
#                 steps += 1
#                 state = next_state
                
#                 # Render the environment
#                 env.render()
                
#                 # Control visualization speed
#                 time.sleep(0.05)
                
#                 # Print real-time metrics
#                 if device.type == "cuda":
#                     memory_used = torch.cuda.memory_allocated(device) / 1024**2  # MB
#                     memory_info = f", Memory Used: {memory_used:.2f}MB"
#                 else:
#                     memory_info = ""
                    
#                 print(f"\rStep: {steps}, Current Reward: {reward:.2f}, "
#                       f"Total Reward: {total_reward:.2f}{memory_info}", end="")
                
#                 if done:
#                     episode_time = time.time() - episode_start_time
#                     episode_rewards.append(total_reward)
#                     episode_steps.append(steps)
                    
#                     print(f"\nEpisode {episode + 1} Summary:")
#                     print(f"Steps: {steps}")
#                     print(f"Total Reward: {total_reward:.2f}")
#                     print(f"Episode Time: {episode_time:.2f}s")
#                     print(f"Average Step Time: {episode_time/steps:.4f}s")
#                     time.sleep(1)
        
#         # Print final statistics
#         print("\nTesting Summary:")
#         print(f"Average Reward: {np.mean(episode_rewards):.2f}")
#         print(f"Average Steps per Episode: {np.mean(episode_steps):.2f}")
#         print(f"Best Episode Reward: {max(episode_rewards):.2f}")
#         print(f"Worst Episode Reward: {min(episode_rewards):.2f}")
    
#     except KeyboardInterrupt:
#         print("\nTesting interrupted by user")
#     except Exception as e:
#         print(f"\nError during testing: {str(e)}")
#     finally:
#         if device.type == "cuda":
#             torch.cuda.empty_cache()
#         pygame.quit()

# if __name__ == "__main__":
#     try:
#         # Get model path and device preference from user
#         model_path = input("Enter the path to your trained model file: ")
#         num_episodes = int(input("Enter number of episodes to test (default=5): ") or "5")
#         force_cpu = input("Force CPU usage? (y/n, default=n): ").lower().startswith('y')
        
#         print("\nInitializing testing...")
#         test_model(model_path, num_episodes, force_cpu)
    
#     except Exception as e:
#         print(f"Error: {str(e)}")

###FORCED CPU + REWARD PLOTTING###

import gymnasium as gym
import numpy as np
import pygame
import torch
import time
import matplotlib.pyplot as plt
from paths.Vehicle_environment_circular import VehicleEnv
# from paths.Vehicle_environment_square import VehicleEnvSquarePath as VehicleEnv
from DQN_agent import DQNAgent
import os

def plot_test_results(episode_rewards, episode_steps, save_dir="test_results"):
    """
    Plot and save test results
    Args:
        episode_rewards: List of rewards per episode
        episode_steps: List of steps per episode
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Episode Rewards
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, 'b-', label='Episode Reward')
    plt.plot(episode_rewards, 'ro', markersize=8)  # Add points for each episode
    plt.axhline(y=np.mean(episode_rewards), color='g', linestyle='--', 
                label=f'Mean Reward: {np.mean(episode_rewards):.2f}')
    plt.fill_between(range(len(episode_rewards)), 
                     np.min(episode_rewards), np.max(episode_rewards), 
                     alpha=0.2, color='blue')
    plt.title('Test Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Episode Steps
    plt.subplot(2, 1, 2)
    plt.plot(episode_steps, 'r-', label='Episode Steps')
    plt.plot(episode_steps, 'bo', markersize=8)  # Add points for each episode
    plt.axhline(y=np.mean(episode_steps), color='g', linestyle='--', 
                label=f'Mean Steps: {np.mean(episode_steps):.2f}')
    plt.fill_between(range(len(episode_steps)), 
                     np.min(episode_steps), np.max(episode_steps), 
                     alpha=0.2, color='red')
    plt.title('Test Episode Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save plots
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(save_dir, f'test_results_{timestamp}.png'))
    plt.close()
    
    # Create a summary statistics plot
    plt.figure(figsize=(10, 6))
    
    # Box plot for rewards
    plt.boxplot([episode_rewards], labels=['Rewards'])
    plt.title('Reward Distribution Across Test Episodes')
    plt.grid(True, alpha=0.3)
    
    # Save statistics plot
    plt.savefig(os.path.join(save_dir, f'test_statistics_{timestamp}.png'))
    plt.close()
    
    return timestamp

def test_model(model_path, num_episodes=5, force_cpu=False):
    """
    Test a trained model with flexible device selection and plot results
    Args:
        model_path: Path to the saved model file
        num_episodes: Number of episodes to test
        force_cpu: Force CPU usage even if GPU is available
    """
    # Device selection
    if force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU (GPU not available)")
    
    # Initialize environment and agent
    env = VehicleEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    agent.epsilon = 0.0  # No exploration during testing
    
    # Store metrics for plotting
    episode_rewards = []
    episode_steps = []
    step_rewards = []  # Store rewards at each step
    
    try:
        # Load the model with appropriate device mapping
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        agent.policy_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(checkpoint)
        
        # Move networks to selected device
        agent.policy_net = agent.policy_net.to(device)
        agent.target_net = agent.target_net.to(device)
        print(f"Model loaded successfully on {device.type.upper()}")
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            episode_step_rewards = []  # Store rewards for this episode
            done = False
            
            print(f"\nStarting Episode {episode + 1}")
            episode_start_time = time.time()
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    action = agent.policy_net(state_tensor).argmax().item()
                
                next_state, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                steps += 1
                if steps > 1000:
                    done = True
                episode_step_rewards.append(reward)
                state = next_state
                
                env.render()
                time.sleep(0.0005)
                
                if device.type == "cuda":
                    memory_used = torch.cuda.memory_allocated(device) / 1024**2
                    memory_info = f", Memory Used: {memory_used:.2f}MB"
                else:
                    memory_info = ""
                    
                print(f"\rStep: {steps}, Current Reward: {reward:.2f}, "
                      f"Total Reward: {total_reward:.2f}{memory_info}", end="")
                
                if done:
                    episode_time = time.time() - episode_start_time
                    episode_rewards.append(total_reward)
                    episode_steps.append(steps)
                    step_rewards.extend(episode_step_rewards)
                    
                    print(f"\nEpisode {episode + 1} Summary:")
                    print(f"Steps: {steps}")
                    print(f"Total Reward: {total_reward:.2f}")
                    print(f"Episode Time: {episode_time:.2f}s")
                    print(f"Average Step Time: {episode_time/steps:.4f}s")
                    time.sleep(1)
        
        # Plot results
        timestamp = plot_test_results(episode_rewards, episode_steps)
        print(f"\nPlots saved with timestamp: {timestamp}")
        
        # Print final statistics
        print("\nTesting Summary:")
        print(f"Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"Average Steps per Episode: {np.mean(episode_steps):.2f}")
        print(f"Best Episode Reward: {max(episode_rewards):.2f}")
        print(f"Worst Episode Reward: {min(episode_rewards):.2f}")
    
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        if len(episode_rewards) > 0:
            plot_test_results(episode_rewards, episode_steps)
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pygame.quit()

if __name__ == "__main__":
    try:
        model_path = input("Enter the path to your trained model file: ")
        num_episodes = int(input("Enter number of episodes to test (default=5): ") or "5")
        force_cpu = input("Force CPU usage? (y/n, default=n): ").lower().startswith('y')
        
        print("\nInitializing testing...")
        test_model(model_path, num_episodes, force_cpu)
    
    except Exception as e:
        print(f"Error: {str(e)}")

#### FORCED CPU + REWARD PLOTTING + FIXED STARTING POSITION ####
# import gymnasium as gym
# import numpy as np
# import pygame
# import torch
# import time
# import matplotlib.pyplot as plt
# # from paths.Vehicle_environment_square import VehicleEnvSquarePath as VehicleEnv
# from paths.Vehicle_environment_circular import VehicleEnv

# from DQN_agent import DQNAgent
# import os

# def plot_test_results(episode_rewards, episode_steps, save_dir="test_results"):
#     """
#     Plot and save test results
#     Args:
#         episode_rewards: List of rewards per episode
#         episode_steps: List of steps per episode
#         save_dir: Directory to save plots
#     """
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Create figure with subplots
#     plt.figure(figsize=(15, 10))
    
#     # Plot 1: Episode Rewards
#     plt.subplot(2, 1, 1)
#     plt.plot(episode_rewards, 'b-', label='Episode Reward')
#     plt.plot(episode_rewards, 'ro', markersize=8)  # Add points for each episode
#     plt.axhline(y=np.mean(episode_rewards), color='g', linestyle='--', 
#                 label=f'Mean Reward: {np.mean(episode_rewards):.2f}')
#     plt.fill_between(range(len(episode_rewards)), 
#                      np.min(episode_rewards), np.max(episode_rewards), 
#                      alpha=0.2, color='blue')
#     plt.title('Test Episode Rewards')
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     # Plot 2: Episode Steps
#     plt.subplot(2, 1, 2)
#     plt.plot(episode_steps, 'r-', label='Episode Steps')
#     plt.plot(episode_steps, 'bo', markersize=8)  # Add points for each episode
#     plt.axhline(y=np.mean(episode_steps), color='g', linestyle='--', 
#                 label=f'Mean Steps: {np.mean(episode_steps):.2f}')
#     plt.fill_between(range(len(episode_steps)), 
#                      np.min(episode_steps), np.max(episode_steps), 
#                      alpha=0.2, color='red')
#     plt.title('Test Episode Steps')
#     plt.xlabel('Episode')
#     plt.ylabel('Steps')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     plt.tight_layout()
    
#     # Save plots
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     plt.savefig(os.path.join(save_dir, f'test_results_{timestamp}.png'))
#     plt.close()
    
#     # Create a summary statistics plot
#     plt.figure(figsize=(10, 6))
#     plt.boxplot([episode_rewards], labels=['Rewards'])
#     plt.title('Reward Distribution Across Test Episodes')
#     plt.grid(True, alpha=0.3)
#     plt.savefig(os.path.join(save_dir, f'test_statistics_{timestamp}.png'))
#     plt.close()
    
#     return timestamp

# class FixedStartVehicleEnv(VehicleEnv):
#     def reset(self, seed=None):
#         """Override reset to ensure fixed starting position"""
#         super().reset(seed=seed)
        
#         # Calculate the fixed starting position
#         offset = (self.window_size - self.square_size) // 2
        
#         # Set fixed position and orientation
#         self.vehicle_pos = [offset, offset]  # Top-left corner of the path
#         self.vehicle_angle = 0  # Facing right
        
#         # Generate fixed obstacles
#         self._generate_obstacles()
        
#         return self._get_observation(), {}

# def test_model(model_path, num_episodes=5, force_cpu=False):
#     """
#     Test a trained model with consistent starting position
#     Args:
#         model_path: Path to the saved model file
#         num_episodes: Number of episodes to test
#         force_cpu: Force CPU usage even if GPU is available
#     """
#     # Device selection
#     if force_cpu:
#         device = torch.device("cpu")
#         print("Forcing CPU usage")
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    
#     # Initialize environment with fixed starting position
#     env = FixedStartVehicleEnv()
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
    
#     # Create agent
#     agent = DQNAgent(state_size, action_size)
#     agent.epsilon = 0.0  # No exploration during testing
    
#     # Store metrics
#     episode_rewards = []
#     episode_steps = []
#     step_rewards = []
    
#     try:
#         # Load the model
#         print(f"Loading model from: {model_path}")
#         checkpoint = torch.load(model_path, map_location=device)
#         agent.policy_net.load_state_dict(checkpoint)
#         agent.target_net.load_state_dict(checkpoint)
        
#         # Move networks to device
#         agent.policy_net = agent.policy_net.to(device)
#         agent.target_net = agent.target_net.to(device)
#         print(f"Model loaded successfully on {device.type.upper()}")
        
#         for episode in range(num_episodes):
#             state, _ = env.reset()  # Will use fixed starting position
#             total_reward = 0
#             steps = 0
#             episode_step_rewards = []
#             done = False
            
#             print(f"\nStarting Episode {episode + 1}")
#             episode_start_time = time.time()
            
#             while not done:
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
#                 with torch.no_grad():
#                     action = agent.policy_net(state_tensor).argmax().item()
                
#                 next_state, reward, done, _, _ = env.step(action)
                
#                 total_reward += reward
#                 steps += 1
#                 episode_step_rewards.append(reward)
#                 state = next_state
                
#                 env.render()
#                 time.sleep(0.05)  # Delay for visualization
                
#                 # Print progress with memory info if using GPU
#                 memory_info = f", Memory Used: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB" if device.type == "cuda" else ""
#                 print(f"\rStep: {steps}, Current Reward: {reward:.2f}, Total Reward: {total_reward:.2f}{memory_info}", end="")
                
#                 if done:
#                     episode_time = time.time() - episode_start_time
#                     episode_rewards.append(total_reward)
#                     episode_steps.append(steps)
#                     step_rewards.extend(episode_step_rewards)
                    
#                     print(f"\nEpisode {episode + 1} Summary:")
#                     print(f"Steps: {steps}")
#                     print(f"Total Reward: {total_reward:.2f}")
#                     print(f"Episode Time: {episode_time:.2f}s")
#                     print(f"Average Step Time: {episode_time/steps:.4f}s")
#                     time.sleep(1)
        
#         # Plot and save results
#         timestamp = plot_test_results(episode_rewards, episode_steps)
#         print(f"\nPlots saved with timestamp: {timestamp}")
        
#         # Print final statistics
#         print("\nTesting Summary:")
#         print(f"Average Reward: {np.mean(episode_rewards):.2f}")
#         print(f"Average Steps per Episode: {np.mean(episode_steps):.2f}")
#         print(f"Best Episode Reward: {max(episode_rewards):.2f}")
#         print(f"Worst Episode Reward: {min(episode_rewards):.2f}")
    
#     except KeyboardInterrupt:
#         print("\nTesting interrupted by user")
#         if len(episode_rewards) > 0:
#             plot_test_results(episode_rewards, episode_steps)
#     except Exception as e:
#         print(f"\nError during testing: {str(e)}")
#     finally:
#         if device.type == "cuda":
#             torch.cuda.empty_cache()
#         pygame.quit()

# if __name__ == "__main__":
#     try:
#         model_path = input("Enter the path to your trained model file: ")
#         num_episodes = int(input("Enter number of episodes to test (default=5): ") or "5")
#         force_cpu = input("Force CPU usage? (y/n, default=n): ").lower().startswith('y')
        
#         print("\nInitializing testing...")
#         test_model(model_path, num_episodes, force_cpu)
    
#     except Exception as e:
#         print(f"Error: {str(e)}")