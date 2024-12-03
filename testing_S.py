import gymnasium as gym
import numpy as np
import pygame
import torch
import time
import matplotlib.pyplot as plt
from paths.Vehicle_environment_S_obs_in_path import VehicleEnvSmoothPath as VehicleEnv
from DQN_agent import DQNAgent
import os

class FixedStartVehicleEnv(VehicleEnv):
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Fixed start position
        self.vehicle_pos = list(self.path_points[0])
        dx = self.path_points[1][0] - self.path_points[0][0]
        dy = self.path_points[1][1] - self.path_points[0][1]
        self.vehicle_angle = np.arctan2(dy, dx)
        
        # Generate path obstacles
        self.path_obstacles = self._generate_path_obstacles()
        self.random_obstacles = []
        
        # Add random obstacles
        for _ in range(self.num_random_obstacles):
            attempts = 0
            while attempts < 100:
                pos = [
                    np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size),
                    np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size)
                ]
                if self._is_safe_obstacle_position(pos):
                    self.random_obstacles.append(pos)
                    break
                attempts += 1
        
        self.obstacles = self.path_obstacles + self.random_obstacles
        self.current_segment = 0
        
        return self._get_observation(), {}
    
    def _is_safe_obstacle_position(self, pos):
        _, dist_to_path, _ = self._get_closest_path_point(pos)
        if dist_to_path < self.obstacle_size * 2:
            return False
            
        for obs in self.path_obstacles + self.random_obstacles:
            dist = np.sqrt((pos[0] - obs[0])**2 + (pos[1] - obs[1])**2)
            if dist < self.obstacle_size * 3:
                return False
        return True

def plot_test_results(episode_rewards, episode_steps, save_dir="test_results"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(15, 10))
    
    # Rewards plot
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, 'b-', label='Episode Reward')
    plt.plot(episode_rewards, 'ro', markersize=8)
    plt.axhline(y=np.mean(episode_rewards), color='g', linestyle='--', 
                label=f'Mean: {np.mean(episode_rewards):.2f}')
    plt.fill_between(range(len(episode_rewards)), 
                     np.min(episode_rewards), np.max(episode_rewards), 
                     alpha=0.2, color='blue')
    plt.title('Test Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Steps plot
    plt.subplot(2, 1, 2)
    plt.plot(episode_steps, 'r-', label='Episode Steps')
    plt.plot(episode_steps, 'bo', markersize=8)
    plt.axhline(y=np.mean(episode_steps), color='g', linestyle='--', 
                label=f'Mean: {np.mean(episode_steps):.2f}')
    plt.fill_between(range(len(episode_steps)), 
                     np.min(episode_steps), np.max(episode_steps), 
                     alpha=0.2, color='red')
    plt.title('Test Episode Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(save_dir, f'test_results_{timestamp}.png'))
    plt.close()
    
    # Statistics plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([episode_rewards], labels=['Rewards'])
    plt.title('Reward Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'test_statistics_{timestamp}.png'))
    plt.close()
    
    return timestamp

def test_model(model_path, num_episodes=5, force_cpu=False):
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    
    env = FixedStartVehicleEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    agent.epsilon = 0.0
    
    episode_rewards = []
    episode_steps = []
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        agent.policy_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(checkpoint)
        
        agent.policy_net = agent.policy_net.to(device)
        agent.target_net = agent.target_net.to(device)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            print(f"\nStarting Episode {episode + 1}")
            start_time = time.time()
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = agent.policy_net(state_tensor).argmax().item()
                
                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward
                steps += 1
                state = next_state
                
                env.render()
                time.sleep(0.0005)
                
                memory_info = f", Memory Used: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB" if device.type == "cuda" else ""
                print(f"\rStep: {steps}, Reward: {reward:.2f}, Total: {total_reward:.2f}{memory_info}", end="")
                
                if done or steps >= 1000:
                    episode_time = time.time() - start_time
                    episode_rewards.append(total_reward)
                    episode_steps.append(steps)
                    
                    print(f"\nEpisode {episode + 1} Summary:")
                    print(f"Steps: {steps}")
                    print(f"Total Reward: {total_reward:.2f}")
                    print(f"Time: {episode_time:.2f}s")
                    time.sleep(1)
                    break
        
        timestamp = plot_test_results(episode_rewards, episode_steps)
        print(f"\nPlots saved with timestamp: {timestamp}")
        
        print("\nTesting Summary:")
        print(f"Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"Average Steps: {np.mean(episode_steps):.2f}")
        print(f"Best Reward: {max(episode_rewards):.2f}")
        print(f"Worst Reward: {min(episode_rewards):.2f}")
    
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        if episode_rewards:
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
        test_model(model_path, num_episodes, force_cpu)
    except Exception as e:
        print(f"Error: {str(e)}")