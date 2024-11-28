import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class VehicleEnvSmoothPath(gym.Env):
    def __init__(self):
        super(VehicleEnvSmoothPath, self).__init__()
        self.window_size = 800
        self.display = None
        self.vehicle_size = 20
        self.vehicle_speed = 5
        self.max_steering_angle = 0.1
        
        # Path creation
        self.path_points = self._generate_smooth_s_path()
        self.path_segments = len(self.path_points) - 1
        self.current_segment = 0
        
        # Obstacles
        self.path_obstacles = []
        self.random_obstacles = []
        self.num_path_obstacles = 12
        self.num_random_obstacles = 4
        self.obstacle_size = 30
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, 0, 0, -self.window_size, -np.pi, 0, 0]),
            high=np.array([self.window_size, self.window_size, np.pi, 
                          self.window_size, self.window_size, self.window_size, np.pi,
                          self.window_size, self.window_size]),
            dtype=np.float32
        )
        
        self.vehicle_pos = None
        self.vehicle_angle = None
    
    def _generate_smooth_s_path(self):
        """Generate a smooth S-shaped path"""
        points = []
        w, h = self.window_size, self.window_size
        
        # Control points for Bezier curves
        controls = [
            [(100, 100), (200, 100), (300, 100)],  # First horizontal
            [(300, 100), (550, 100), (550, 300)],  # First curve
            [(550, 300), (550, 500), (300, 500)],  # First vertical
            [(300, 500), (50, 500), (50, 700)],    # Second curve
            [(50, 700), (50, 750), (500, 750)]     # Final horizontal
        ]
        
        # Generate points along the path
        for curve in controls:
            for t in np.linspace(0, 1, 20):
                if len(curve) == 3:  # Quadratic Bezier
                    x = (1-t)**2 * curve[0][0] + 2*(1-t)*t * curve[1][0] + t**2 * curve[2][0]
                    y = (1-t)**2 * curve[0][1] + 2*(1-t)*t * curve[1][1] + t**2 * curve[2][1]
                    points.append((x, y))
        
        return points
    
    def _generate_path_obstacles(self):
        """Generate obstacles along the S path"""
        obstacles = []
        key_indices = [20, 40, 60, 80]  # Points along curves
        
        for idx in key_indices:
            if idx < len(self.path_points):
                base_point = self.path_points[idx]
                next_point = self.path_points[min(idx + 1, len(self.path_points) - 1)]
                
                # Calculate normal vector
                dx = next_point[0] - base_point[0]
                dy = next_point[1] - base_point[1]
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    normal = (-dy/length, dx/length)
                    
                    # Place obstacles on both sides
                    offset = 40
                    obstacles.append([
                        base_point[0] + offset * normal[0],
                        base_point[1] + offset * normal[1]
                    ])
                    obstacles.append([
                        base_point[0] - offset * normal[0],
                        base_point[1] - offset * normal[1]
                    ])
        
        return obstacles

    # [Previous methods remain the same: _get_closest_path_point, reset, _is_safe_obstacle_position, 
    # _get_observation, _normalize_angle, step, render]
    
    def _get_closest_path_point(self, position):
        min_dist = float('inf')
        closest_point = None
        segment_idx = 0
        
        for i in range(len(self.path_points) - 1):
            p1 = np.array(self.path_points[i])
            p2 = np.array(self.path_points[i + 1])
            pos = np.array(position)
            
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
        self.vehicle_pos = list(self.path_points[0])
        dx = self.path_points[1][0] - self.path_points[0][0]
        dy = self.path_points[1][1] - self.path_points[0][1]
        self.vehicle_angle = np.arctan2(dy, dx)
        
        self.path_obstacles = self._generate_path_obstacles()
        self.random_obstacles = []
        
        for _ in range(self.num_random_obstacles):
            while True:
                pos = [
                    np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size),
                    np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size)
                ]
                if self._is_safe_obstacle_position(pos):
                    self.random_obstacles.append(pos)
                    break
        
        self.obstacles = self.path_obstacles + self.random_obstacles
        self.current_segment = 0
        return self._get_observation(), {}

    def _is_safe_obstacle_position(self, pos):
        _, dist_to_path, _ = self._get_closest_path_point(pos)
        if dist_to_path < self.obstacle_size * 2:
            return False
            
        for obs in self.path_obstacles + self.random_obstacles:
            dist = np.sqrt((pos[0] - obs[0])**2 + (pos[1] - obs[1])**2)
            if dist < self.obstacle_size * 3:
                return False
        
        return True

    def _get_observation(self):
        closest_obstacle = min(self.obstacles, 
                             key=lambda obs: np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
                                                   (obs[1] - self.vehicle_pos[1])**2))
        
        closest_point, dist_to_path, segment_idx = self._get_closest_path_point(self.vehicle_pos)
        next_point_idx = min(segment_idx + 2, len(self.path_points) - 1)
        next_waypoint = self.path_points[next_point_idx]
        
        dx = next_waypoint[0] - self.vehicle_pos[0]
        dy = next_waypoint[1] - self.vehicle_pos[1]
        target_angle = np.arctan2(dy, dx)
        angle_diff = self._normalize_angle(target_angle - self.vehicle_angle)
        
        return np.array([
            self.vehicle_pos[0], self.vehicle_pos[1], self.vehicle_angle,
            closest_obstacle[0], closest_obstacle[1],
            dist_to_path, angle_diff,
            next_waypoint[0], next_waypoint[1]
        ], dtype=np.float32)
    
    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    

    def step(self, action):
        steering = (-self.max_steering_angle if action == 0 
                   else self.max_steering_angle if action == 2 
                   else 0)
        self.vehicle_angle += steering
        
        self.vehicle_pos[0] += self.vehicle_speed * np.cos(self.vehicle_angle)
        self.vehicle_pos[1] += self.vehicle_speed * np.sin(self.vehicle_angle)
        
        closest_point, dist_to_path, segment_idx = self._get_closest_path_point(self.vehicle_pos)
        next_point = self.path_points[min(segment_idx + 1, len(self.path_points) - 1)]
        target_angle = np.arctan2(next_point[1] - self.vehicle_pos[1],
                                 next_point[0] - self.vehicle_pos[0])
        angle_diff = abs(self._normalize_angle(target_angle - self.vehicle_angle))
        
        reward = (np.exp(-0.01 * dist_to_path) + 
                 np.exp(-1.0 * angle_diff) +
                 0.1 * (segment_idx - self.current_segment))
        
        done = False
        
        if dist_to_path > 50:
            reward -= 3.0
        
        for obs in self.obstacles:
            dist_to_obs = np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
                                 (obs[1] - self.vehicle_pos[1])**2)
            if dist_to_obs < (self.vehicle_size + self.obstacle_size) / 2:
                reward -= 20
                done = True
            elif dist_to_obs < self.obstacle_size * 2:
                reward -= (2 * self.obstacle_size - dist_to_obs) * 0.1
        
        if (self.vehicle_pos[0] < 0 or self.vehicle_pos[0] > self.window_size or
            self.vehicle_pos[1] < 0 or self.vehicle_pos[1] > self.window_size):
            reward -= 20
            done = True
        
        if segment_idx >= len(self.path_points) - 2:
            reward += 100
            done = True
        
        self.current_segment = segment_idx
        return self._get_observation(), reward, done, False, {}

    def render(self):
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.window_size, self.window_size))
        
        self.display.fill((255, 255, 255))
        
        # Draw path
        points = [(int(p[0]), int(p[1])) for p in self.path_points]
        if len(points) > 1:
            pygame.draw.lines(self.display, (200, 200, 200), False, points, 3)
        
        # Draw path obstacles
        for obs in self.path_obstacles:
            pygame.draw.circle(self.display, (255, 0, 0),
                             (int(obs[0]), int(obs[1])),
                             self.obstacle_size)
        
        # Draw random obstacles
        for obs in self.random_obstacles:
            pygame.draw.circle(self.display, (255, 165, 0),
                             (int(obs[0]), int(obs[1])),
                             self.obstacle_size)
        
        # Draw vehicle
        pygame.draw.circle(self.display, (0, 0, 255),
                         (int(self.vehicle_pos[0]), int(self.vehicle_pos[1])),
                         self.vehicle_size)
        
        # Direction indicator
        end_pos = (int(self.vehicle_pos[0] + self.vehicle_size * np.cos(self.vehicle_angle)),
                  int(self.vehicle_pos[1] + self.vehicle_size * np.sin(self.vehicle_angle)))
        pygame.draw.line(self.display, (0, 255, 0),
                        (int(self.vehicle_pos[0]), int(self.vehicle_pos[1])),
                        end_pos, 2)
        
        pygame.display.flip()
