import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class VehicleEnvSmoothPath(gym.Env):
    def __init__(self):
        super(VehicleEnvSmoothPath, self).__init__()
        # Basic settings
        self.window_size = 800
        self.display = None
        self.vehicle_size = 20
        self.vehicle_speed = 5
        self.max_steering_angle = 0.1
        
        # Road properties
        self.road_width = 60
        self.line_width = 2
        self.dash_length = 20
        self.dash_gap = 20
        
        # Colors
        self.ROAD_COLOR = (80, 80, 80)      # Dark gray
        self.GRASS_COLOR = (34, 139, 34)    # Green
        self.BORDER_COLOR = (255, 255, 255)  # White
        self.MARKING_COLOR = (255, 240, 0)   # Yellow
        
        # Path and obstacles
        self.path_points = self._generate_smooth_s_path()
        self.path_segments = len(self.path_points) - 1
        self.current_segment = 0
        self.fixed_obstacle_positions = self._generate_fixed_obstacles()
        self.path_obstacles = []
        self.random_obstacles = []
        self.num_path_obstacles = 4
        self.num_random_obstacles = 3
        self.obstacle_size = 15
        
        # Spaces
        self.action_space = spaces.Discrete(3)
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
        self.obstacles = []
    
    def _generate_smooth_s_path(self):
        """Generate a smooth S-shaped path using Bezier curves"""
        points = []
        
        # Control points for Bezier curves
        controls = [
            [(100, 100), (200, 100), (300, 100)],  # First horizontal
            [(300, 100), (550, 100), (550, 300)],  # First curve
            [(550, 300), (550, 500), (300, 500)],  # Middle section
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
	
    def _generate_fixed_obstacles(self):
        """Generate fixed obstacle positions along the path"""
        fixed_positions = [
            (350, 150),  # First curve entry
            (500, 200),  # Middle of first curve
            (450, 400),  # Between curves
            (200, 500),  # Second curve
            (250, 700)   # Final stretch
        ]
        return fixed_positions

    def _generate_path_obstacles(self):
        """Generate obstacles along the path with some randomization"""
        obstacles = []
        
        # Add fixed obstacles with slight position variation
        for base_pos in self.fixed_obstacle_positions:
            variation = 20  # pixels
            pos = [
                base_pos[0] + np.random.randint(-variation, variation),
                base_pos[1] + np.random.randint(-variation, variation)
            ]
            if self._is_safe_obstacle_position(pos):
                obstacles.append(pos)
        
        return obstacles

    def _get_closest_path_point(self, position):
        """Find the closest point on the path to the given position"""
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

    def _is_safe_obstacle_position(self, pos):
        """Check if an obstacle position is safe"""
        _, dist_to_path, _ = self._get_closest_path_point(pos)
        
        # Ensure obstacles aren't too far from the path
        if dist_to_path > self.road_width * 1.5:
            return False
            
        # Ensure obstacles aren't too close to the path center
        if dist_to_path < self.road_width * 0.3:
            return False
            
        # Check distance from other obstacles
        for obs in self.path_obstacles + self.random_obstacles:
            dist = np.sqrt((pos[0] - obs[0])**2 + (pos[1] - obs[1])**2)
            if dist < self.obstacle_size * 3:
                return False
        
        return True

    def reset(self, seed=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        self.vehicle_pos = list(self.path_points[0])
        dx = self.path_points[1][0] - self.path_points[0][0]
        dy = self.path_points[1][1] - self.path_points[0][1]
        self.vehicle_angle = np.arctan2(dy, dx)
        
        # Generate path obstacles
        self.path_obstacles = self._generate_path_obstacles()
        
        # Generate random obstacles
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

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle

    def _get_observation(self):
        """Get current observation of the environment"""
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
    def _get_parallel_points(self, p1, p2, distance):
    
    # Get vector from p1 to p2
     dx = p2[0] - p1[0]
     dy = p2[1] - p1[1]
    
    # Calculate length
     length = np.sqrt(dx*dx + dy*dy)
    
     if length == 0:
        return p1, p2
        
    # Normalize the vector
     dx = dx / length
     dy = dy / length
    
    # Calculate perpendicular vector (rotate 90 degrees)
     pdx = -dy
     pdy = dx
    
    # Calculate parallel points
     p1_parallel = (
        p1[0] + pdx * distance,
        p1[1] + pdy * distance
    )
     p2_parallel = (
        p2[0] + pdx * distance,
        p2[1] + pdy * distance
    )
    
     return p1_parallel, p2_parallel

    def step(self, action):
        """Execute one step in the environment"""
        # Apply steering
        steering = (-self.max_steering_angle if action == 0 
                   else self.max_steering_angle if action == 2 
                   else 0)
        self.vehicle_angle += steering
        
        # Update position
        self.vehicle_pos[0] += self.vehicle_speed * np.cos(self.vehicle_angle)
        self.vehicle_pos[1] += self.vehicle_speed * np.sin(self.vehicle_angle)
        
        # Get path information
        closest_point, dist_to_path, segment_idx = self._get_closest_path_point(self.vehicle_pos)
        next_point = self.path_points[min(segment_idx + 1, len(self.path_points) - 1)]
        target_angle = np.arctan2(next_point[1] - self.vehicle_pos[1],
                                 next_point[0] - self.vehicle_pos[0])
        angle_diff = abs(self._normalize_angle(target_angle - self.vehicle_angle))
        
        # Calculate rewards
        path_reward = np.exp(-0.03 * dist_to_path)
        angle_reward = np.exp(-1.0 * angle_diff)
        progress_reward = 0.2 * (segment_idx - self.current_segment)
        
        reward = 2.0 * path_reward + angle_reward + progress_reward
        done = False
        
        # Handle path deviation
        if dist_to_path > 30:
            reward -= (dist_to_path - 30) * 0.2
        
        if dist_to_path > 100:
            reward -= 50
            done = True
        
        # Handle collisions
        for obs in self.obstacles:
            dist_to_obs = np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
                                 (obs[1] - self.vehicle_pos[1])**2)
            if dist_to_obs < (self.vehicle_size + self.obstacle_size) / 2:
                reward -= 30
                done = True
            elif dist_to_obs < self.obstacle_size * 2:
                reward -= (2 * self.obstacle_size - dist_to_obs) * 0.15
        
        # Check bounds
        if (self.vehicle_pos[0] < 0 or self.vehicle_pos[0] > self.window_size or
            self.vehicle_pos[1] < 0 or self.vehicle_pos[1] > self.window_size):
            reward -= 100
            done = True
        
        # Check completion
        if segment_idx >= len(self.path_points) - 2:
            reward += 200
            done = True
        
        self.current_segment = segment_idx
        return self._get_observation(), reward, done, False, {}
    
    def _draw_dashed_line(self, surface, color, start_pos, end_pos, width=2, dash_length=20, gap_length=20):
    
    # Calculate the vector from start to end
     x1, y1 = start_pos
     x2, y2 = end_pos
     dx = x2 - x1
     dy = y2 - y1
    
    # Calculate the total distance
     distance = np.sqrt(dx*dx + dy*dy)
     if distance == 0:
        return
    
    # Normalize the vector
     dx = dx / distance
     dy = dy / distance
    
    # Calculate number of dashes
     dash_count = int(distance / (dash_length + gap_length))
    
    # Draw each dash
     for i in range(dash_count):
        start_x = x1 + (dash_length + gap_length) * i * dx
        start_y = y1 + (dash_length + gap_length) * i * dy
        end_x = start_x + dash_length * dx
        end_y = start_y + dash_length * dy
        
        pygame.draw.line(surface, color,
                        (int(start_x), int(start_y)),
                        (int(end_x), int(end_y)), width)

    def render(self):
        """Render the environment"""
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.window_size, self.window_size))
        
        # Fill background with grass color
        self.display.fill(self.GRASS_COLOR)
        
        # Create surface for road
        road_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        
        # Draw road segments
        for i in range(len(self.path_points) - 1):
            p1 = self.path_points[i]
            p2 = self.path_points[i + 1]
            
            # Get parallel points for road edges
            outer_left = self._get_parallel_points(p1, p2, self.road_width/2)
            outer_right = self._get_parallel_points(p1, p2, -self.road_width/2)
            
            # Draw road segment
            points = [
                (int(outer_left[0][0]), int(outer_left[0][1])),
                (int(outer_left[1][0]), int(outer_left[1][1])),
                (int(outer_right[1][0]), int(outer_right[1][1])),
                (int(outer_right[0][0]), int(outer_right[0][1]))
            ]
            pygame.draw.polygon(road_surface, self.ROAD_COLOR, points)
            
            # Draw road borders
            pygame.draw.line(self.display, self.BORDER_COLOR,
                           (int(outer_left[0][0]), int(outer_left[0][1])),
                           (int(outer_left[1][0]), int(outer_left[1][1])), 3)
            pygame.draw.line(self.display, self.BORDER_COLOR,
                           (int(outer_right[0][0]), int(outer_right[0][1])),
                           (int(outer_right[1][0]), int(outer_right[1][1])), 3)
            
            # Draw center dashed line
            self._draw_dashed_line(self.display, self.MARKING_COLOR,
                                 (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])),
                                 2, self.dash_length, self.dash_gap)
        
        # Blend road surface
        self.display.blit(road_surface, (0, 0))
        
        # Draw path obstacles (fixed + random variations)
        for obs in self.path_obstacles:
            pygame.draw.circle(self.display, (255, 0, 0),
                             (int(obs[0]), int(obs[1])),
                             self.obstacle_size)
        
        # Draw random obstacles
        for obs in self.random_obstacles:
            pygame.draw.circle(self.display, (255, 165, 0),
                             (int(obs[0]), int(obs[1])),
                             self.obstacle_size)
        
        # Draw vehicle (car-like shape)
        car_center = (int(self.vehicle_pos[0]), int(self.vehicle_pos[1]))
        
        # Draw vehicle body (as a triangle to indicate direction)
        car_points = [
            (car_center[0] + self.vehicle_size * np.cos(self.vehicle_angle),
             car_center[1] + self.vehicle_size * np.sin(self.vehicle_angle)),
            (car_center[0] + self.vehicle_size * np.cos(self.vehicle_angle + 2.6),
             car_center[1] + self.vehicle_size * np.sin(self.vehicle_angle + 2.6)),
            (car_center[0] + self.vehicle_size * np.cos(self.vehicle_angle - 2.6),
             car_center[1] + self.vehicle_size * np.sin(self.vehicle_angle - 2.6))
        ]
        pygame.draw.polygon(self.display, (0, 0, 255), car_points)
        
        # Draw headlight
        headlight_pos = (int(car_center[0] + self.vehicle_size * 0.8 * np.cos(self.vehicle_angle)),
                        int(car_center[1] + self.vehicle_size * 0.8 * np.sin(self.vehicle_angle)))
        pygame.draw.circle(self.display, (255, 255, 0), headlight_pos, 3)
        
        pygame.display.flip()