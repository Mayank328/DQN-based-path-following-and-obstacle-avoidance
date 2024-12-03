import os
import pygame
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from paths.Vehicle_environment_circular import VehicleEnv
from DQN_agent import DQNAgent

class FixedStartEnvBase(VehicleEnv):
    def __init__(self):
        super().__init__()
        self.start_position = None
        self.start_angle = None
        
        # Set fixed start position for circular path
        start_angle = 0  # Start at rightmost point
        self.start_position = [
            self.path_center[0] + self.path_radius * np.cos(start_angle),
            self.path_center[1] + self.path_radius * np.sin(start_angle)
        ]
        self.start_angle = start_angle + np.pi/2  # Face tangent to circle
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Override with fixed position and angle
        self.vehicle_pos = list(self.start_position)
        self.vehicle_angle = self.start_angle
        
        # Generate obstacles after setting position
        self.obstacles = self._generate_path_obstacles()
        
        return self._get_observation(), {}

def plot_rewards(rewards, window_size=100):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    
    if len(rewards) >= window_size:
        running_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), running_avg, 
                label=f'Running Average ({window_size} episodes)', 
                color='red', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plots_dir = "training_plots"
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'reward_plot.png'))
    plt.close()

def train(render=True):
    models_dir = "trained_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize environment with fixed start position
    env = FixedStartEnvBase()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    episodes = 1000
    max_steps = 800
    target_update_frequency = 20
    save_frequency = 500
    
    all_rewards = []
    best_reward = float('-inf')
    
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                
                total_reward += reward
                state = next_state
                
                if render:
                    env.render()
                
                if done:
                    break
            
            all_rewards.append(total_reward)
            
            if episode % target_update_frequency == 0:
                agent.update_target_network()
            
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save_model(os.path.join(models_dir, 'vehicle_model_best.pth'))
            
            if episode % save_frequency == 0:
                save_path = os.path.join(models_dir, f'vehicle_model_episode_{episode}.pth')
                agent.save_model(save_path)
                plot_rewards(all_rewards)
                print(f"\nModel saved to: {save_path}")
            
            print(f"Episode: {episode + 1}/{episodes}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Best Reward: {best_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print("------------------------")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        final_save_path = os.path.join(models_dir, 'vehicle_model_final.pth')
        agent.save_model(final_save_path)
        plot_rewards(all_rewards)
        print(f"\nFinal model saved to: {final_save_path}")
        pygame.quit()

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    train()