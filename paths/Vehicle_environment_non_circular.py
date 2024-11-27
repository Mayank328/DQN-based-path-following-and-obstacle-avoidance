import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class VehicleEnvNonCircular(gym.Env):
    def __init__(self):
        super(VehicleEnvNonCircular, self).__init__()
        
        # Window size
        self.window_size = 800
        self.display = None
        
        # Vehicle properties
        self.vehicle_size = 20
        self.vehicle_speed = 5
        self.max_steering_angle = 0.1
        
        # Path properties
        self.path_points = self._generate_complex_path()
        self.path_segments = len(self.path_points) - 1
        self.current_segment = 0
        
        # Obstacle properties
        self.obstacles = []
        self.num_obstacles = 5
        self.obstacle_size = 30
        
        # Action space: [steer_left, no_action, steer_right]
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [vehicle_x, vehicle_y, vehicle_angle, 
        #                    closest_obstacle_x, closest_obstacle_y,
        #                    distance_to_path, angle_to_target,
        #                    next_waypoint_x, next_waypoint_y]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, 0, 0, -self.window_size, -np.pi, 0, 0]),
            high=np.array([self.window_size, self.window_size, np.pi, 
                          self.window_size, self.window_size, self.window_size, np.pi,
                          self.window_size, self.window_size]),
            dtype=np.float32
        )
        
        # Initial state
        self.vehicle_pos = None
        self.vehicle_angle = None
        
    def _generate_complex_path(self):
        """Generate a complex path with different segments"""
        points = []
        
        # Start point
        points.append((200, 600))
        
        # Right straight section
        points.append((600, 600))
        
        # Upper right curve
        for angle in np.linspace(3*np.pi/2, 0, 20):
            x = 600 + 100 * np.cos(angle)
            y = 500 + 100 * np.sin(angle)
            points.append((x, y))
        
        # Upper straight section
        points.append((700, 400))
        points.append((700, 200))
        
        # Upper left curve
        for angle in np.linspace(0, np.pi/2, 20):
            x = 600 + 100 * np.cos(angle)
            y = 200 + 100 * np.sin(angle)
            points.append((x, y))
        
        # Left straight section
        points.append((600, 300))
        points.append((200, 300))
        
        # Lower left curve
        for angle in np.linspace(np.pi/2, np.pi, 20):
            x = 200 + 100 * np.cos(angle)
            y = 400 + 100 * np.sin(angle)
            points.append((x, y))
        
        return points
    
    def _get_closest_path_point(self, position):
        """Get the closest point on the path and its segment"""
        min_dist = float('inf')
        closest_point = None
        segment_idx = 0
        
        for i in range(len(self.path_points) - 1):
            p1 = np.array(self.path_points[i])
            p2 = np.array(self.path_points[i + 1])
            pos = np.array(position)
            
            # Get closest point on line segment
            segment = p2 - p1
            length_sq = np.sum(segment**2)
            if length_sq == 0:
                continue
                
            t = max(0, min(1, np.dot(pos - p1, segment) / length_sq))
            projection = p1 + t * segment
            
            dist = np.linalg.norm(pos - projection)
            if dist < min_dist:
                min_dist = dist
                closest_point = projection
                segment_idx = i
        
        return closest_point, min_dist, segment_idx
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialize vehicle at the start of the path
        self.vehicle_pos = [self.path_points[0][0], self.path_points[0][1]]
        
        # Calculate initial angle based on first path segment
        dx = self.path_points[1][0] - self.path_points[0][0]
        dy = self.path_points[1][1] - self.path_points[0][1]
        self.vehicle_angle = np.arctan2(dy, dx)
        
        # Generate obstacles away from the path
        self.obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                pos = [
                    np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size),
                    np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size)
                ]
                
                # Check distance from path
                _, dist_to_path, _ = self._get_closest_path_point(pos)
                if dist_to_path > self.obstacle_size * 3:
                    self.obstacles.append(pos)
                    break
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Find closest obstacle
        closest_obstacle = min(self.obstacles, 
                             key=lambda obs: np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
                                                   (obs[1] - self.vehicle_pos[1])**2))
        
        # Get path information
        closest_point, dist_to_path, segment_idx = self._get_closest_path_point(self.vehicle_pos)
        
        # Get next waypoint
        next_point_idx = min(segment_idx + 2, len(self.path_points) - 1)
        next_waypoint = self.path_points[next_point_idx]
        
        # Calculate angle to next waypoint
        dx = next_waypoint[0] - self.vehicle_pos[0]
        dy = next_waypoint[1] - self.vehicle_pos[1]
        target_angle = np.arctan2(dy, dx)
        angle_diff = self._normalize_angle(target_angle - self.vehicle_angle)
        
        return np.array([
            self.vehicle_pos[0],
            self.vehicle_pos[1],
            self.vehicle_angle,
            closest_obstacle[0],
            closest_obstacle[1],
            dist_to_path,
            angle_diff,
            next_waypoint[0],
            next_waypoint[1]
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
        
        # Get path information
        closest_point, dist_to_path, segment_idx = self._get_closest_path_point(self.vehicle_pos)
        
        # Calculate angle to path
        next_point_idx = min(segment_idx + 1, len(self.path_points) - 1)
        next_point = self.path_points[next_point_idx]
        target_angle = np.arctan2(next_point[1] - self.vehicle_pos[1],
                                 next_point[0] - self.vehicle_pos[0])
        angle_diff = abs(self._normalize_angle(target_angle - self.vehicle_angle))
        
        # Reward structure
        path_reward = np.exp(-0.01 * dist_to_path)
        angle_reward = np.exp(-1.0 * angle_diff)
        reward = path_reward + angle_reward
        
        # Penalties
        if dist_to_path > 50:
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
        
        # Progress reward
        if segment_idx > self.current_segment:
            reward += 10  # Reward for reaching new segment
            self.current_segment = segment_idx
        
        # Episode completion
        if segment_idx >= len(self.path_points) - 2:
            reward += 50  # Bonus for completing the path
            done = True
        
        return self._get_observation(), reward, done, False, {}
    
    def render(self):
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.window_size, self.window_size))
        
        self.display.fill((255, 255, 255))
        
        # Draw path
        for i in range(len(self.path_points) - 1):
            pygame.draw.line(self.display, (200, 200, 200),
                           (int(self.path_points[i][0]), int(self.path_points[i][1])),
                           (int(self.path_points[i+1][0]), int(self.path_points[i+1][1])), 2)
        
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