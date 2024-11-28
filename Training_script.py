import os
import pygame
# import time
import numpy as np
import matplotlib.pyplot as plt
# from paths.Vehicle_environment_circular import VehicleEnv
# from paths.Vehicle_environment_non_circular import VehicleEnvNonCircular as VehicleEnv
# from paths.Vehicle_environment_S import VehicleEnvSmoothPath as VehicleEnv
from paths.Vehicle_environment_S_obs_in_path import VehicleEnvSmoothPath as VehicleEnv
from DQN_agent import DQNAgent


def plot_rewards(rewards, window_size=100):
    """Plot the rewards and running average"""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    
    # Calculate and plot running average
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
    
    # Save the plot
    plots_dir = "training_plots"
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'reward_plot.png'))
    plt.close()

def train(render=True):
    # Create necessary directories
    models_dir = "trained_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = VehicleEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    episodes = 1000
    max_steps = 1000
    target_update_frequency = 15
    save_frequency = 200
    
    # Lists to store metrics
    all_rewards = []
    best_reward = float('-inf')
    
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            episode_steps = []  # Store step-wise rewards
            
            for step in range(max_steps):
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                
                total_reward += reward
                episode_steps.append(reward)
                state = next_state
                
                if render:
                    env.render()
                
                if done:
                    break
            
            # Store episode metrics
            all_rewards.append(total_reward)
            
            # Update target network periodically
            if episode % target_update_frequency == 0:
                agent.update_target_network()
            
            # Save best model
            if total_reward > best_reward:
                best_reward = total_reward
                best_model_path = os.path.join(models_dir, 'vehicle_model_best.pth')
                agent.save_model(best_model_path)
            
            # Save model periodically
            if episode % save_frequency == 0:
                save_path = os.path.join(models_dir, f'vehicle_model_episode_{episode}.pth')
                agent.save_model(save_path)
                # Plot current rewards
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
        # Save final model and plot
        final_save_path = os.path.join(models_dir, 'vehicle_model_final.pth')
        agent.save_model(final_save_path)
        plot_rewards(all_rewards)
        print(f"\nFinal model saved to: {final_save_path}")
        pygame.quit()

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    train()