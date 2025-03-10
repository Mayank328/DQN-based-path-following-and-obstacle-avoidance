import os
import numpy as np
import matplotlib.pyplot as plt
import pygame
from DQN_agent_improved import DQNAgent
from paths.circular_road import VehicleEnv as VehicleEnv

def plot_rewards(rewards, window_size=100):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    if len(rewards) >= window_size:
        running_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), running_avg, label=f'Running Average ({window_size} episodes)', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs("training_plots", exist_ok=True)
    plt.savefig("training_plots/reward_plot.png")
    plt.close()

def train(render=True):
    models_dir = "trained_models"
    os.makedirs(models_dir, exist_ok=True)
    env = VehicleEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 5000  # Increased training duration
    max_steps = 1000
    target_update_frequency = 10
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
                agent.save_model(os.path.join(models_dir, f'vehicle_model_episode_{episode}.pth'))
                plot_rewards(all_rewards)
                print(f"\nModel saved at episode {episode}")
            print(f"Episode {episode + 1}/{episodes} - Reward: {total_reward:.2f} - Best: {best_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    finally:
        agent.save_model(os.path.join(models_dir, 'vehicle_model_final.pth'))
        plot_rewards(all_rewards)
        pygame.quit()

if __name__ == "__main__":
    train()
