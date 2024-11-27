# import gymnasium as gym
# import numpy as np
# import pygame
# from gymnasium import spaces
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import math
# from collections import deque
# import random

# class VehicleEnv(gym.Env):
#     def __init__(self):
#         super(VehicleEnv, self).__init__()
        
#         # Window size
#         self.window_size = 800
#         self.display = None
        
#         # Vehicle properties
#         self.vehicle_size = 20
#         self.vehicle_speed = 5
        
#         # Path properties (making a simple circular path)
#         self.path_radius = 300
#         self.path_center = (self.window_size // 2, self.window_size // 2)
        
#         # Obstacle properties
#         self.obstacles = []
#         self.num_obstacles = 3
#         self.obstacle_size = 30
        
#         # Action space: [steer_left, no_action, steer_right]
#         self.action_space = spaces.Discrete(3)
        
#         # Observation space: [vehicle_x, vehicle_y, vehicle_angle, 
#         #                    closest_obstacle_x, closest_obstacle_y,
#         #                    distance_to_path]
#         self.observation_space = spaces.Box(
#             low=np.array([0, 0, -np.pi, 0, 0, 0]),
#             high=np.array([self.window_size, self.window_size, np.pi, 
#                           self.window_size, self.window_size, self.window_size]),
#             dtype=np.float32
#         )
        
#         # Initial state
#         self.vehicle_pos = None
#         self.vehicle_angle = None
        
#     def reset(self, seed=None):
#         super().reset(seed=seed)
#         # Initialize vehicle position on the path
#         angle = np.random.random() * 2 * np.pi
#         self.vehicle_pos = [
#             self.path_center[0] + self.path_radius * np.cos(angle),
#             self.path_center[1] + self.path_radius * np.sin(angle)
#         ]
#         self.vehicle_angle = angle
        
#         # Generate random obstacles
#         self.obstacles = []
#         for _ in range(self.num_obstacles):
#             while True:
#                 pos = [
#                     np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size),
#                     np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size)
#                 ]
#                 # Ensure obstacles are not too close to the path
#                 dist_to_center = np.sqrt((pos[0] - self.path_center[0])**2 + 
#                                       (pos[1] - self.path_center[1])**2)
#                 if abs(dist_to_center - self.path_radius) > self.obstacle_size * 2:
#                     self.obstacles.append(pos)
#                     break
        
#         return self._get_observation(), {}
    
#     def _get_observation(self):
#         # Find closest obstacle
#         closest_obstacle = min(self.obstacles, 
#                              key=lambda obs: np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
#                                                    (obs[1] - self.vehicle_pos[1])**2))
        
#         # Calculate distance to path
#         dist_to_center = np.sqrt((self.vehicle_pos[0] - self.path_center[0])**2 + 
#                                 (self.vehicle_pos[1] - self.path_center[1])**2)
#         dist_to_path = abs(dist_to_center - self.path_radius)
        
#         return np.array([
#             self.vehicle_pos[0],
#             self.vehicle_pos[1],
#             self.vehicle_angle,
#             closest_obstacle[0],
#             closest_obstacle[1],
#             dist_to_path
#         ], dtype=np.float32)
    
#     def step(self, action):
#         # Update vehicle angle based on action
#         if action == 0:  # steer left
#             self.vehicle_angle -= 0.1
#         elif action == 2:  # steer right
#             self.vehicle_angle += 0.1
        
#         # Update vehicle position
#         self.vehicle_pos[0] += self.vehicle_speed * np.cos(self.vehicle_angle)
#         self.vehicle_pos[1] += self.vehicle_speed * np.sin(self.vehicle_angle)
        
#         # Calculate reward
#         reward = 0
#         done = False
        
#         # Penalty for being far from path
#         dist_to_center = np.sqrt((self.vehicle_pos[0] - self.path_center[0])**2 + 
#                                 (self.vehicle_pos[1] - self.path_center[1])**2)
#         path_deviation = abs(dist_to_center - self.path_radius)
#         reward -= path_deviation * 0.01
        
#         # Penalty for hitting obstacles
#         for obs in self.obstacles:
#             dist_to_obs = np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
#                                  (obs[1] - self.vehicle_pos[1])**2)
#             if dist_to_obs < (self.vehicle_size + self.obstacle_size) / 2:
#                 reward -= 100
#                 done = True
        
#         # Check if vehicle is out of bounds
#         if (self.vehicle_pos[0] < 0 or self.vehicle_pos[0] > self.window_size or
#             self.vehicle_pos[1] < 0 or self.vehicle_pos[1] > self.window_size):
#             reward -= 100
#             done = True
        
#         return self._get_observation(), reward, done, False, {}
    
#     def render(self):
#         if self.display is None:
#             pygame.init()
#             self.display = pygame.display.set_mode((self.window_size, self.window_size))
        
#         self.display.fill((255, 255, 255))
        
#         # Draw path
#         pygame.draw.circle(self.display, (200, 200, 200), self.path_center, 
#                          self.path_radius, 2)
        
#         # Draw obstacles
#         for obs in self.obstacles:
#             pygame.draw.circle(self.display, (255, 0, 0), 
#                              (int(obs[0]), int(obs[1])), 
#                              self.obstacle_size)
        
#         # Draw vehicle
#         pygame.draw.circle(self.display, (0, 0, 255), 
#                          (int(self.vehicle_pos[0]), int(self.vehicle_pos[1])), 
#                          self.vehicle_size)
        
#         pygame.display.flip()
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class VehicleEnv(gym.Env):
    def __init__(self):
        super(VehicleEnv, self).__init__()
        
        # Window size
        self.window_size = 800
        self.display = None
        
        # Vehicle properties
        self.vehicle_size = 20
        self.vehicle_speed = 5
        self.max_steering_angle = 0.1  # Maximum steering angle per step
        
        # Path properties
        self.path_radius = 300
        self.path_center = (self.window_size // 2, self.window_size // 2)
        
        # Obstacle properties
        self.obstacles = []
        self.num_obstacles = 7
        self.obstacle_size = 30
        
        # Action space: [steer_left, no_action, steer_right]
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [vehicle_x, vehicle_y, vehicle_angle, 
        #                    closest_obstacle_x, closest_obstacle_y,
        #                    distance_to_path, angle_to_target]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, 0, 0, -self.window_size, -np.pi]),
            high=np.array([self.window_size, self.window_size, np.pi, 
                          self.window_size, self.window_size, self.window_size, np.pi]),
            dtype=np.float32
        )
        
        # Initial state
        self.vehicle_pos = None
        self.vehicle_angle = None
        self.target_angle = None
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialize vehicle position slightly off the path
        angle = np.random.random() * 2 * np.pi
        offset = np.random.uniform(-20, 20)  # Start slightly off the path
        self.vehicle_pos = [
            self.path_center[0] + (self.path_radius + offset) * np.cos(angle),
            self.path_center[1] + (self.path_radius + offset) * np.sin(angle)
        ]
        # Initialize vehicle angle tangent to the circle
        self.vehicle_angle = angle + np.pi/2  # Tangent angle
        self.target_angle = self.vehicle_angle
        
        # Generate random obstacles
        self.obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                pos = [
                    np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size),
                    np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size)
                ]
                dist_to_center = np.sqrt((pos[0] - self.path_center[0])**2 + 
                                      (pos[1] - self.path_center[1])**2)
                if abs(dist_to_center - self.path_radius) > self.obstacle_size * 2:
                    self.obstacles.append(pos)
                    break
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Find closest obstacle
        closest_obstacle = min(self.obstacles, 
                             key=lambda obs: np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
                                                   (obs[1] - self.vehicle_pos[1])**2))
        
        # Calculate distance and angle to path
        dx = self.vehicle_pos[0] - self.path_center[0]
        dy = self.vehicle_pos[1] - self.path_center[1]
        dist_to_center = np.sqrt(dx**2 + dy**2)
        dist_to_path = dist_to_center - self.path_radius
        
        # Calculate ideal tangent angle at current position
        ideal_angle = np.arctan2(dy, dx) + np.pi/2
        angle_diff = self._normalize_angle(ideal_angle - self.vehicle_angle)
        
        return np.array([
            self.vehicle_pos[0],
            self.vehicle_pos[1],
            self.vehicle_angle,
            closest_obstacle[0],
            closest_obstacle[1],
            dist_to_path,
            angle_diff
        ], dtype=np.float32)
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def step(self, action):
        # Update vehicle angle based on action
        steering = 0
        if action == 0:  # steer left
            steering = -self.max_steering_angle
        elif action == 2:  # steer right
            steering = self.max_steering_angle
        
        self.vehicle_angle += steering
        
        # Update vehicle position
        self.vehicle_pos[0] += self.vehicle_speed * np.cos(self.vehicle_angle)
        self.vehicle_pos[1] += self.vehicle_speed * np.sin(self.vehicle_angle)
        
        # Calculate reward
        reward = 0
        done = False
        
        # Reward for staying on path
        dx = self.vehicle_pos[0] - self.path_center[0]
        dy = self.vehicle_pos[1] - self.path_center[1]
        dist_to_center = np.sqrt(dx**2 + dy**2)
        path_deviation = abs(dist_to_center - self.path_radius)
        
        # Calculate ideal tangent angle at current position
        ideal_angle = np.arctan2(dy, dx) + np.pi/2
        angle_diff = abs(self._normalize_angle(ideal_angle - self.vehicle_angle))
        
        # Combined reward considering both position and angle
        path_reward = np.exp(-0.01 * path_deviation)  # Exponential decay for path deviation
        angle_reward = np.exp(-1.0 * angle_diff)  # Exponential decay for angle difference
        reward = path_reward + angle_reward
        
        # Penalties
        if path_deviation > 30:  # Stronger penalty for being far from path
            reward -= 5
        
        # Obstacle collision
        for obs in self.obstacles:
            dist_to_obs = np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
                                 (obs[1] - self.vehicle_pos[1])**2)
            if dist_to_obs < (self.vehicle_size + self.obstacle_size) / 2:
                reward -= 30
                done = True
        
        # Out of bounds
        if (self.vehicle_pos[0] < 0 or self.vehicle_pos[0] > self.window_size or
            self.vehicle_pos[1] < 0 or self.vehicle_pos[1] > self.window_size):
            reward -= 20
            done = True
        
        return self._get_observation(), reward, done, False, {}
    
    def render(self):
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.window_size, self.window_size))
        
        self.display.fill((255, 255, 255))
        
        # Draw path
        pygame.draw.circle(self.display, (200, 200, 200), self.path_center, 
                         self.path_radius, 2)
        
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.circle(self.display, (255, 0, 0), 
                             (int(obs[0]), int(obs[1])), 
                             self.obstacle_size)
        
        # Draw vehicle
        pygame.draw.circle(self.display, (0, 0, 255), 
                         (int(self.vehicle_pos[0]), int(self.vehicle_pos[1])), 
                         self.vehicle_size)
        
        # Draw vehicle direction indicator
        end_pos = (int(self.vehicle_pos[0] + self.vehicle_size * np.cos(self.vehicle_angle)),
                  int(self.vehicle_pos[1] + self.vehicle_size * np.sin(self.vehicle_angle)))
        pygame.draw.line(self.display, (0, 255, 0), 
                        (int(self.vehicle_pos[0]), int(self.vehicle_pos[1])), 
                        end_pos, 2)
        
        pygame.display.flip()