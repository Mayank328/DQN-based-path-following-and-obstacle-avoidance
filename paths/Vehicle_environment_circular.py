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
        self.max_steering_angle = 0.1
        
        # Path properties
        self.path_radius = 300
        self.path_center = (self.window_size // 2, self.window_size // 2)
        
        # Obstacle properties
        self.obstacles = []
        self.num_random_obstacles = 3  # Reduced random obstacles
        self.num_path_obstacles = 4    # New: Obstacles on the path
        self.obstacle_size = 30
        
        # Action space: [steer_left, no_action, steer_right]
        self.action_space = spaces.Discrete(3)
        
        # Observation space
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
        
    def _generate_path_obstacles(self):
        """Generate obstacles along the circular path"""
        path_obstacles = []
        # Place obstacles at specific angles along the path
        angles = [np.pi/4, np.pi/2, np.pi, 3*np.pi/2]  # Specific positions along the circle
        
        for angle in angles:
            # Place obstacle slightly inside or outside the path
            offset = np.random.choice([-20, 20])  # Randomly place inside or outside
            pos = [
                self.path_center[0] + (self.path_radius + offset) * np.cos(angle),
                self.path_center[1] + (self.path_radius + offset) * np.sin(angle)
            ]
            path_obstacles.append(pos)
        
        return path_obstacles
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialize vehicle position slightly off the path
        angle = np.random.random() * 2 * np.pi
        offset = np.random.uniform(-20, 20)
        self.vehicle_pos = [
            self.path_center[0] + (self.path_radius + offset) * np.cos(angle),
            self.path_center[1] + (self.path_radius + offset) * np.sin(angle)
        ]
        self.vehicle_angle = angle + np.pi/2
        self.target_angle = self.vehicle_angle
        
        # Generate path obstacles
        self.obstacles = self._generate_path_obstacles()
        
        # Add random obstacles
        for _ in range(self.num_random_obstacles):
            while True:
                pos = [
                    np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size),
                    np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size)
                ]
                dist_to_center = np.sqrt((pos[0] - self.path_center[0])**2 + 
                                      (pos[1] - self.path_center[1])**2)
                
                # Check if position is far enough from existing obstacles
                too_close = False
                for obs in self.obstacles:
                    if np.sqrt((pos[0] - obs[0])**2 + (pos[1] - obs[1])**2) < self.obstacle_size * 3:
                        too_close = True
                        break
                
                if not too_close and abs(dist_to_center - self.path_radius) > self.obstacle_size * 2:
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
        
        # Calculate ideal tangent angle
        ideal_angle = np.arctan2(dy, dx) + np.pi/2
        angle_diff = abs(self._normalize_angle(ideal_angle - self.vehicle_angle))
        
        # Enhanced reward structure
        path_reward = np.exp(-0.01 * path_deviation)
        angle_reward = np.exp(-1.0 * angle_diff)
        obstacle_avoidance_reward = 0
        
        # Add obstacle avoidance reward
        closest_obstacle_dist = min(np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
                                          (obs[1] - self.vehicle_pos[1])**2) 
                                  for obs in self.obstacles)
        if closest_obstacle_dist > self.obstacle_size * 2:
            obstacle_avoidance_reward = 0.5  # Reward for keeping safe distance
        
        reward = path_reward + angle_reward + obstacle_avoidance_reward
        
        # Penalties
        if path_deviation > 50:
            reward -= 5
        
        # Enhanced obstacle collision penalty
        for obs in self.obstacles:
            dist_to_obs = np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
                                 (obs[1] - self.vehicle_pos[1])**2)
            if dist_to_obs < (self.vehicle_size + self.obstacle_size) / 2:
                reward -= 30
                done = True
            elif dist_to_obs < self.obstacle_size * 2:
                reward -= (2 * self.obstacle_size - dist_to_obs) * 0.1  # Gradual penalty for getting too close
        
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
        
        # Draw obstacles with different colors for path and random obstacles
        for i, obs in enumerate(self.obstacles):
            color = (255, 0, 0) if i < self.num_path_obstacles else (255, 165, 0)  # Red for path obstacles, orange for random
            pygame.draw.circle(self.display, color, 
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