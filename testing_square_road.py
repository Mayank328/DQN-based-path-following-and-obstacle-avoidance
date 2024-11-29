import os
import pygame
import numpy as np
import torch
from paths.Vehicle_environment_square import VehicleEnvSquarePath as VehicleEnv
from DQN_agent import DQNAgent

def evaluate_episode(env, agent, render=True, max_steps=1000):
    """Run a single evaluation episode"""
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < max_steps:
        # Use the model to select action (no exploration)
        agent.epsilon = 0  # Ensure no random actions during testing
        action = agent.act(state)
        
        # Take action in environment
        next_state, reward, done, _, _ = env.step(action)
        
        # Update state and metrics
        state = next_state
        total_reward += reward
        steps += 1
        
        # Render if requested
        if render:
            env.render()
            pygame.time.wait(50)  # Add delay to make visualization easier to follow
    
    return total_reward, steps

def run_evaluation(model_path, num_episodes=10, render=True):
    """Evaluate a trained model over multiple episodes"""
    # Initialize environment and agent
    env = VehicleEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Load trained model
    try:
        agent.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Evaluation metrics
    rewards = []
    steps_list = []
    success_count = 0
    
    try:
        for episode in range(num_episodes):
            print(f"\nRunning evaluation episode {episode + 1}/{num_episodes}")
            
            # Run episode
            total_reward, steps = evaluate_episode(env, agent, render)
            rewards.append(total_reward)
            steps_list.append(steps)
            
            # Count successful episodes (you may need to adjust this criterion)
            if total_reward > 0:  # Define your success criterion
                success_count += 1
            
            # Print episode results
            print(f"Episode Reward: {total_reward:.2f}")
            print(f"Episode Steps: {steps}")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    
    finally:
        # Print summary statistics
        print("\nEvaluation Summary:")
        print(f"Number of episodes: {num_episodes}")
        print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Average steps: {np.mean(steps_list):.2f} ± {np.std(steps_list):.2f}")
        print(f"Success rate: {success_count/num_episodes*100:.2f}%")
        
        # Close pygame window
        pygame.quit()

def main():
    # Specify the path to your trained model
    model_path = "trained_models/vehicle_model_best.pth"  # Adjust path as needed
    
    # Set evaluation parameters
    num_episodes = 10  # Number of episodes to evaluate
    render = True      # Whether to render the environment
    
    # Run evaluation
    run_evaluation(model_path, num_episodes, render)

if __name__ == "__main__":
    main()