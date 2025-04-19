import os
import pygame
import numpy as np
import matplotlib.pyplot as plt
from paths.circular_road import VehicleEnv
from DQN_agent import DQNAgent

def plot_combined(rewards, mse_list, window_size=100, save_path="training_plots/combined_plot.png"):
    """
    Plot rewards (top) and MSE (bottom) in one figure with a running average.
    
    Args:
        rewards (list or array): Episode rewards.
        mse_list (list or array): Per-episode MSE values.
        window_size (int): Window size for the running average.
        save_path (str): Where to save the combined figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 8))

    # ----- Top Subplot: Rewards -----
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    if len(rewards) >= window_size:
        running_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards)),
                 running_avg, color='red', linewidth=2,
                 label=f'Running Average ({window_size} episodes)')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.title('Training Rewards Over Time', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ----- Bottom Subplot: MSE -----
    plt.subplot(2, 1, 2)
    plt.plot(mse_list, label='MSE', alpha=0.6)
    if len(mse_list) >= window_size:
        running_avg_mse = np.convolve(mse_list, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(mse_list)),
                 running_avg_mse, color='red', linewidth=2,
                 label=f'MSE Running Average ({window_size} episodes)')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title('Path Deviation MSE Over Time', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(render=True):
    """
    Main training function that:
      - Creates the environment
      - Initializes the DQN agent
      - Collects rewards and MSE over episodes
      - Saves models periodically
      - Plots combined reward and MSE curves
    """
    # Create necessary directories
    models_dir = "trained_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = VehicleEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    episodes = 2000
    max_steps = 1000
    target_update_frequency = 15
    save_frequency = 500
    
    # Lists to store metrics
    all_rewards = []
    all_mse = []
    best_reward = float('-inf')
    
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            mse_values = []  # Collect MSE values for this episode
            
            for step in range(max_steps):
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                
                # Capture the MSE loss from replay (if available)
                loss_val = agent.replay()
                if loss_val is not None:
                    mse_values.append(loss_val)
                
                total_reward += reward
                state = next_state
                
                if render:
                    env.render()
                
                if done:
                    break
            
            # Compute average MSE for this episode
            avg_mse = np.mean(mse_values) if mse_values else 0
            all_mse.append(avg_mse)
            all_rewards.append(total_reward)
            
            # Update target network periodically
            if episode % target_update_frequency == 0:
                agent.update_target_network()
            
            # Save best model
            if total_reward > best_reward:
                best_reward = total_reward
                best_model_path = os.path.join(models_dir, 'vehicle_model_best.pth')
                agent.save_model(best_model_path)
            
            # Save model and plot periodically
            if episode % save_frequency == 0:
                save_path = os.path.join(models_dir, f'vehicle_model_episode_{episode}.pth')
                agent.save_model(save_path)
                
                # Plot both curves in a single figure
                plot_combined(all_rewards, all_mse, window_size=100,
                              save_path="training_plots/combined_plot.png")
                print(f"\nModel saved to: {save_path}")
            
            print(f"Episode: {episode + 1}/{episodes}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Average MSE: {avg_mse:.4f}")
            print(f"Best Reward: {best_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print("------------------------")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Save final model and plot
        final_save_path = os.path.join(models_dir, 'vehicle_model_final.pth')
        agent.save_model(final_save_path)
        
        # Plot both curves in a single figure
        plot_combined(all_rewards, all_mse, window_size=100,
                      save_path="training_plots/combined_plot.png")
        
        print(f"\nFinal model saved to: {final_save_path}")
        pygame.quit()

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    train()
