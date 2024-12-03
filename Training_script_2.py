import matplotlib.pyplot as plt
import pygame
from Simulation_environment import Environment
from DQN_Agent3 import DQNAgent

def train_agent():
    env = Environment()
    agent = DQNAgent(state_dim=4, action_dim=4)
    episodes = 500
    max_steps = 200
    
    rewards = []  # List to store total rewards for each episode

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            env.render()
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
        agent.update_target_network()
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
    
    pygame.quit()
    
    # Plot the total rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    train_agent()
