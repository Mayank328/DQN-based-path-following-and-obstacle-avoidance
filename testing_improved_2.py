import torch
import time
import matplotlib.pyplot as plt
from DQN_agent import DQNAgent
from circular_road import VehicleEnv
import os


def plot_test_results(episode_rewards, episode_steps, save_dir="test_results"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, 'b-', label='Episode Reward')
    plt.title('Test Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episode_steps, 'r-', label='Episode Steps')
    plt.title('Test Episode Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    plt.legend()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(save_dir, f'test_results_{timestamp}.png'))
    plt.close()


def test_model(model_path, num_episodes=10, render=True):
    env = VehicleEnv(normalize_observations=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load_model(model_path)

    episode_rewards = []
    episode_steps = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state

            total_reward += reward
            steps += 1

            if render:
                env.render()
                time.sleep(0.01)

            if done or steps >= 1000:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        print(f"Test Episode {ep+1}/{num_episodes} - Reward: {total_reward:.2f}, Steps: {steps}")

    plot_test_results(episode_rewards, episode_steps)


if __name__ == "__main__":
    model_path = input("Enter model path: ")
    test_model(model_path, num_episodes=10, render=True)

