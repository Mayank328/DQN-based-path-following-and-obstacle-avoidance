# import gymnasium as gym
# import numpy as np
# import pygame
# from gymnasium import spaces

# class VehicleEnvSquarePath(gym.Env):
#     def __init__(self):
#         super().__init__()  # Call parent class's __init__
        
#         # Set basic attributes first
#         self.window_size = 800  # Set this before using it
#         self.vehicle_size = 20
#         self.vehicle_speed = 5
#         self.max_steering_angle = 0.1
#         self.display = None
        
#         # Path creation - now with uniform square settings
#         self.square_size = 600  # Size of the square path
#         self.path_points = self._generate_square_path()
#         self.path_segments = len(self.path_points) - 1
#         self.current_segment = 0
        
#         # Obstacles
#         self.path_obstacles = []
#         self.random_obstacles = []
#         self.num_path_obstacles = 4  # One obstacle per side of the square
#         self.num_random_obstacles = 4
#         self.obstacle_size = 20
        
#         # Action and observation spaces
#         self.action_space = spaces.Discrete(3)
#         self.observation_space = spaces.Box(
#             low=np.array([0, 0, -np.pi, 0, 0, -self.window_size, -np.pi, 0, 0, 0]),
#             high=np.array([self.window_size, self.window_size, np.pi, 
#                           self.window_size, self.window_size, self.window_size, np.pi,
#                           self.window_size, self.window_size, self.window_size]),
#             dtype=np.float32
#         )
        
#         self.vehicle_pos = None
#         self.vehicle_angle = None

#     def _generate_square_path(self):
#         """Generate a perfectly uniform square path"""
#         # Calculate the offset to center the square in the window
#         offset = (self.window_size - self.square_size) // 2
        
#         # Generate square corners in clockwise order, starting from top-left
#         return [
#             (offset, offset),                          # Top-left corner
#             (offset + self.square_size, offset),       # Top-right corner
#             (offset + self.square_size, offset + self.square_size),  # Bottom-right corner
#             (offset, offset + self.square_size),       # Bottom-left corner
#             (offset, offset)                           # Back to start to close the square
#         ]

#     def _generate_path_obstacles(self):
#         """Generate obstacles along the square path with uniform spacing"""
#         obstacles = []
#         for i in range(self.num_path_obstacles):
#             p1 = np.array(self.path_points[i])
#             p2 = np.array(self.path_points[i + 1])
            
#             # Calculate midpoint exactly
#             mid_point = (p1 + p2) / 2
            
#             # Add a small consistent offset perpendicular to the path
#             if i == 0:  # Top edge
#                 mid_point = mid_point + np.array([0, 30])
#             elif i == 1:  # Right edge
#                 mid_point = mid_point + np.array([-30, 0])
#             elif i == 2:  # Bottom edge
#                 mid_point = mid_point + np.array([0, -30])
#             else:  # Left edge
#                 mid_point = mid_point + np.array([30, 0])
            
#             obstacles.append(mid_point)
#         return obstacles

#     def _get_closest_path_point(self, position):
#         min_dist = float('inf')
#         closest_point = None
#         segment_idx = 0
        
#         for i in range(len(self.path_points) - 1):
#             p1 = np.array(self.path_points[i])
#             p2 = np.array(self.path_points[i + 1])
#             pos = np.array(position)
            
#             segment = p2 - p1
#             length_sq = np.sum(segment**2)
#             if length_sq == 0:
#                 continue
                
#             t = max(0, min(1, np.dot(pos - p1, segment) / length_sq))
#             projection = p1 + t * segment
            
#             dist = np.linalg.norm(pos - projection)
#             if dist < min_dist:
#                 min_dist = dist
#                 closest_point = projection
#                 segment_idx = i
        
#         return closest_point, min_dist, segment_idx

#     def _get_nearest_obstacles(self, k=3):
#         """Get the k nearest obstacles and their distances"""
#         distances = []
#         for obs in self.obstacles:
#             dist = np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
#                           (obs[1] - self.vehicle_pos[1])**2)
#             distances.append((dist, obs))
        
#         distances.sort(key=lambda x: x[0])
#         return distances[:k]

#     def _normalize_angle(self, angle):
#         while angle > np.pi: angle -= 2 * np.pi
#         while angle < -np.pi: angle += 2 * np.pi
#         return angle

#     def _get_observation(self):
#         # Get distances and positions of nearest obstacle
#         nearest_obstacles = self._get_nearest_obstacles(k=1)
#         closest_obstacle = nearest_obstacles[0][1]
#         closest_dist = nearest_obstacles[0][0]
        
#         closest_point, dist_to_path, segment_idx = self._get_closest_path_point(self.vehicle_pos)
#         next_point_idx = min(segment_idx + 2, len(self.path_points) - 1)
#         next_waypoint = self.path_points[next_point_idx]
        
#         dx = next_waypoint[0] - self.vehicle_pos[0]
#         dy = next_waypoint[1] - self.vehicle_pos[1]
#         target_angle = np.arctan2(dy, dx)
#         angle_diff = self._normalize_angle(target_angle - self.vehicle_angle)
        
#         return np.array([
#             self.vehicle_pos[0], self.vehicle_pos[1], self.vehicle_angle,
#             closest_obstacle[0], closest_obstacle[1],
#             dist_to_path,
#             angle_diff,
#             next_waypoint[0], next_waypoint[1],
#             closest_dist  # Distance to nearest obstacle
#         ], dtype=np.float32)

#     def step(self, action):
#         # Steering: action 0 = left, 1 = straight, 2 = right
#         steering = (-self.max_steering_angle if action == 0 
#                    else self.max_steering_angle if action == 2 
#                    else 0)
#         self.vehicle_angle += steering
        
#         # Update vehicle position
#         new_pos_x = self.vehicle_pos[0] + self.vehicle_speed * np.cos(self.vehicle_angle)
#         new_pos_y = self.vehicle_pos[1] + self.vehicle_speed * np.sin(self.vehicle_angle)
        
#         # Check if new position would cause collision
#         would_collide = False
#         for obs in self.obstacles:
#             dist_to_obs = np.sqrt((obs[0] - new_pos_x)**2 + (obs[1] - new_pos_y)**2)
#             if dist_to_obs < (self.vehicle_size + self.obstacle_size) / 1.5:
#                 would_collide = True
#                 break
        
#         # Only update position if it wouldn't cause collision
#         if not would_collide:
#             self.vehicle_pos[0] = new_pos_x
#             self.vehicle_pos[1] = new_pos_y
        
#         # Get current state information
#         closest_point, dist_to_path, segment_idx = self._get_closest_path_point(self.vehicle_pos)
#         next_point = self.path_points[min(segment_idx + 1, len(self.path_points) - 1)]
        
#         # Calculate angle to target
#         dx = next_point[0] - self.vehicle_pos[0]
#         dy = next_point[1] - self.vehicle_pos[1]
#         target_angle = np.arctan2(dy, dx)
#         angle_diff = abs(self._normalize_angle(target_angle - self.vehicle_angle))
        
#         # --- Reward Calculation ---
#         reward = 0
        
#         # Base reward for staying on path
#         reward += 2.0 * np.exp(-0.03 * dist_to_path)
        
#         # Penalty for poor orientation
#         reward -= 0.3 * angle_diff
        
#         # Reward for forward progress
#         if segment_idx > self.current_segment:
#             reward += 15.0
#             self.current_segment = segment_idx
        
#         # Penalties for obstacle proximity
#         nearest_obstacles = self._get_nearest_obstacles(k=1)
#         closest_dist = nearest_obstacles[0][0]
        
#         # Progressive penalty that increases as vehicle gets closer to obstacles
#         if closest_dist < self.obstacle_size * 3:
#             obstacle_penalty = (3 * self.obstacle_size - closest_dist) * 0.2
#             reward -= obstacle_penalty
        
#         # Collision detection and penalty
#         if would_collide:
#             reward -= 30
#             return self._get_observation(), reward, True, False, {}
        
#         # Boundary checking
#         if (self.vehicle_pos[0] < 0 or self.vehicle_pos[0] > self.window_size or
#             self.vehicle_pos[1] < 0 or self.vehicle_pos[1] > self.window_size):
#             reward -= 30
#             return self._get_observation(), reward, True, False, {}
        
#         # Success condition
#         done = False
#         if segment_idx >= len(self.path_points) - 2:
#             reward += 100.0
#             done = True
        
#         return self._get_observation(), reward, done, False, {}

#     def reset(self, seed=None):
#         super().reset(seed=seed)
#         self.vehicle_pos = list(self.path_points[0])
#         dx = self.path_points[1][0] - self.path_points[0][0]
#         dy = self.path_points[1][1] - self.path_points[0][1]
#         self.vehicle_angle = np.arctan2(dy, dx)
        
#         self.path_obstacles = self._generate_path_obstacles()
#         self.random_obstacles = []
        
#         # Generate random obstacles away from the start position
#         start_pos = np.array(self.vehicle_pos)
#         for _ in range(self.num_random_obstacles):
#             attempts = 0
#             while attempts < 100:
#                 pos = [
#                     np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size),
#                     np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size)
#                 ]
                
#                 # Check distance from start position
#                 if np.linalg.norm(start_pos - np.array(pos)) < 100:
#                     attempts += 1
#                     continue
                    
#                 if self._is_safe_obstacle_position(pos):
#                     self.random_obstacles.append(pos)
#                     break
#                 attempts += 1
        
#         self.obstacles = self.path_obstacles + self.random_obstacles
#         self.current_segment = 0
#         return self._get_observation(), {}

#     def _is_safe_obstacle_position(self, pos):
#         _, dist_to_path, _ = self._get_closest_path_point(pos)
#         if dist_to_path < self.obstacle_size * 2:
#             return False
            
#         for obs in self.path_obstacles + self.random_obstacles:
#             dist = np.sqrt((pos[0] - obs[0])**2 + (pos[1] - obs[1])**2)
#             if dist < self.obstacle_size * 3:
#                 return False
        
#         return True

#     def render(self):
#         if self.display is None:
#             pygame.init()
#             self.display = pygame.display.set_mode((self.window_size, self.window_size))
        
#         self.display.fill((255, 255, 255))  # White background
        
#         # Draw path - thicker gray line for better visibility
#         points = [(int(p[0]), int(p[1])) for p in self.path_points]
#         if len(points) > 1:
#             # Draw main path
#             pygame.draw.lines(self.display, (200, 200, 200), False, points, 80)
            
#             # Draw path borders for better visibility
#             pygame.draw.lines(self.display, (180, 180, 180), False, points, 82)
        
#         # Draw path obstacles as red circles
#         for obs in self.path_obstacles:
#             pygame.draw.circle(
#                 self.display, (255, 0, 0),
#                 (int(obs[0]), int(obs[1])), self.obstacle_size // 2
#             )
        
#         # Draw random obstacles as orange circles
#         for obs in self.random_obstacles:
#             pygame.draw.circle(
#                 self.display, (255, 165, 0),
#                 (int(obs[0]), int(obs[1])), self.obstacle_size // 2
#             )
        
#         # Draw vehicle as a triangle
#         x, y = self.vehicle_pos
#         size = self.vehicle_size
#         angle = self.vehicle_angle
#         points = [
#             (x + size * np.cos(angle), y + size * np.sin(angle)),  # Tip
#             (x - size * np.cos(angle + np.pi/4), y - size * np.sin(angle + np.pi/4)),  # Left
#             (x - size * np.cos(angle - np.pi/4), y - size * np.sin(angle - np.pi/4))   # Right
#         ]
#         pygame.draw.polygon(self.display, (0, 0, 255), points)
        
#         pygame.display.flip()

### SQUARE ROADS USING PYGAME####

# import pygame
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces

# class VehicleEnvSquarePath(gym.Env):
#     def __init__(self):
#         super().__init__()
#         # Window and display settings
#         self.window_size = 800
#         self.display = None
        
#         # Road parameters
#         self.road_width = 80
#         self.square_size = 600  # Size of the square road's centerline
#         self.outer_color = (100, 100, 100)  # Dark gray for road
#         self.inner_color = (200, 200, 200)  # Light gray for inner area
#         self.line_color = (255, 255, 255)   # White for road markings
        
#         # Vehicle parameters
#         self.vehicle_size = 20
#         self.vehicle_speed = 5
#         self.max_steering_angle = 0.15
#         self.vehicle_pos = None
#         self.vehicle_angle = None
        
#         # Obstacles
#         self.obstacles = []
#         self.num_obstacles = 5
#         self.obstacle_size = 20
        
#         # Gym spaces
#         self.action_space = spaces.Discrete(3)  # Left, Straight, Right
#         self.observation_space = spaces.Box(
#             low=np.array([0, 0, -np.pi, 0, 0, 0, -np.pi, 0, 0, 0]),
#             high=np.array([self.window_size, self.window_size, np.pi, 
#                           self.window_size, self.window_size, self.window_size, np.pi,
#                           self.window_size, self.window_size, self.window_size]),
#             dtype=np.float32
#         )

#     def _generate_road_points(self):
#         """Generate points for the square road"""
#         offset = (self.window_size - self.square_size) // 2
        
#         # Outer square points
#         outer_offset = self.road_width // 2
#         self.outer_points = [
#             (offset - outer_offset, offset - outer_offset),
#             (offset + self.square_size + outer_offset, offset - outer_offset),
#             (offset + self.square_size + outer_offset, offset + self.square_size + outer_offset),
#             (offset - outer_offset, offset + self.square_size + outer_offset)
#         ]
        
#         # Inner square points
#         inner_offset = self.road_width // 2
#         self.inner_points = [
#             (offset + inner_offset, offset + inner_offset),
#             (offset + self.square_size - inner_offset, offset + inner_offset),
#             (offset + self.square_size - inner_offset, offset + self.square_size - inner_offset),
#             (offset + inner_offset, offset + self.square_size - inner_offset)
#         ]
        
#         # Center line points for path following
#         self.center_points = [
#             (offset, offset),
#             (offset + self.square_size, offset),
#             (offset + self.square_size, offset + self.square_size),
#             (offset, offset + self.square_size),
#             (offset, offset)  # Close the loop
#         ]
# def _generate_obstacles(self):
#     """Generate fixed obstacles along the road"""
#     self.obstacles = []
#     offset = (self.window_size - self.square_size) // 2
    
#     # Fixed obstacle positions - one in middle of each side and at corners
#     fixed_positions = [
#         # Top side
#         (offset + self.square_size//4, offset),  # Quarter way
#         (offset + 3*self.square_size//4, offset),  # Three quarters way
        
#         # Right side
#         (offset + self.square_size, offset + self.square_size//4),
#         (offset + self.square_size, offset + 3*self.square_size//4),
        
#         # Bottom side
#         (offset + self.square_size//4, offset + self.square_size),
#         (offset + 3*self.square_size//4, offset + self.square_size),
        
#         # Left side
#         (offset, offset + self.square_size//4),
#         (offset, offset + 3*self.square_size//4)
#     ]
    
#     # Add slight offset from the center line to avoid direct path blocking
#     road_offsets = [
#         (0, self.road_width//4),  # Top side obstacles
#         (0, -self.road_width//4),
#         (-self.road_width//4, 0),  # Right side obstacles
#         (self.road_width//4, 0),
#         (0, -self.road_width//4),  # Bottom side obstacles
#         (0, self.road_width//4),
#         (self.road_width//4, 0),  # Left side obstacles
#         (-self.road_width//4, 0)
#     ]
    
#     # Add each obstacle with its offset
#     for (x, y), (offset_x, offset_y) in zip(fixed_positions, road_offsets):
#         self.obstacles.append((x + offset_x, y + offset_y))
	
#     # def _generate_obstacles(self):
#     #     """Generate obstacles along the road"""
#     #     self.obstacles = []
#     #     offset = (self.window_size - self.square_size) // 2
        
#     #     # Generate obstacles along the road
#     #     for _ in range(self.num_obstacles):
#     #         # Randomly choose a side of the square
#     #         side = np.random.randint(0, 4)
#     #         if side == 0:  # Top
#     #             x = np.random.randint(offset, offset + self.square_size)
#     #             y = offset + np.random.randint(-self.road_width//3, self.road_width//3)
#     #         elif side == 1:  # Right
#     #             x = offset + self.square_size + np.random.randint(-self.road_width//3, self.road_width//3)
#     #             y = np.random.randint(offset, offset + self.square_size)
#     #         elif side == 2:  # Bottom
#     #             x = np.random.randint(offset, offset + self.square_size)
#     #             y = offset + self.square_size + np.random.randint(-self.road_width//3, self.road_width//3)
#     #         else:  # Left
#     #             x = offset + np.random.randint(-self.road_width//3, self.road_width//3)
#     #             y = np.random.randint(offset, offset + self.square_size)
            
#     #         self.obstacles.append((x, y))

#     def reset(self, seed=None):
#         super().reset(seed=seed)
        
#         # Generate road layout
#         self._generate_road_points()
        
#         # Initialize vehicle at start position
#         offset = (self.window_size - self.square_size) // 2
#         self.vehicle_pos = [offset, offset]  # Start at top-left corner
#         self.vehicle_angle = 0  # Facing right initially
        
#         # Generate new obstacles
#         self._generate_obstacles()
        
#         return self._get_observation(), {}

#     def _get_observation(self):
#         """Get the current state observation"""
#         # Find closest obstacle
#         closest_obs = min(self.obstacles, 
#                          key=lambda obs: np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
#                                                (obs[1] - self.vehicle_pos[1])**2))
#         closest_dist = np.sqrt((closest_obs[0] - self.vehicle_pos[0])**2 + 
#                              (closest_obs[1] - self.vehicle_pos[1])**2)
        
#         # Find closest point on path
#         closest_center, dist_to_center, segment_idx = self._get_closest_path_point(self.vehicle_pos)
#         next_point_idx = min(segment_idx + 1, len(self.center_points) - 1)
#         next_point = self.center_points[next_point_idx]
        
#         # Calculate angle to next point
#         dx = next_point[0] - self.vehicle_pos[0]
#         dy = next_point[1] - self.vehicle_pos[1]
#         target_angle = np.arctan2(dy, dx)
#         angle_diff = self._normalize_angle(target_angle - self.vehicle_angle)
        
#         return np.array([
#             self.vehicle_pos[0], self.vehicle_pos[1], self.vehicle_angle,
#             closest_obs[0], closest_obs[1],
#             closest_dist,
#             angle_diff,
#             next_point[0], next_point[1],
#             dist_to_center
#         ], dtype=np.float32)

#     def render(self):
#         if self.display is None:
#             pygame.init()
#             self.display = pygame.display.set_mode((self.window_size, self.window_size))
        
#         # Fill background
#         self.display.fill((50, 50, 50))  # Dark background
        
#         # Draw outer road
#         pygame.draw.polygon(self.display, self.outer_color, self.outer_points)
        
#         # Draw inner area
#         pygame.draw.polygon(self.display, self.inner_color, self.inner_points)
        
#         # Draw road markings (dashed lines)
#         for i in range(len(self.center_points) - 1):
#             p1 = self.center_points[i]
#             p2 = self.center_points[i + 1]
            
#             # Draw dashed line
#             dash_length = 20
#             total_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
#             num_dashes = int(total_length / (dash_length * 2))
            
#             for j in range(num_dashes):
#                 start_fraction = j * 2 * dash_length / total_length
#                 end_fraction = (j * 2 + 1) * dash_length / total_length
                
#                 start_x = p1[0] + (p2[0] - p1[0]) * start_fraction
#                 start_y = p1[1] + (p2[1] - p1[1]) * start_fraction
#                 end_x = p1[0] + (p2[0] - p1[0]) * end_fraction
#                 end_y = p1[1] + (p2[1] - p1[1]) * end_fraction
                
#                 pygame.draw.line(self.display, self.line_color, 
#                                (int(start_x), int(start_y)), 
#                                (int(end_x), int(end_y)), 2)
        
#         # Draw obstacles
#         for obs in self.obstacles:
#             pygame.draw.circle(self.display, (255, 0, 0), 
#                              (int(obs[0]), int(obs[1])), 
#                              self.obstacle_size // 2)
        
#         # Draw vehicle
#         x, y = self.vehicle_pos
#         size = self.vehicle_size
#         angle = self.vehicle_angle
#         points = [
#             (x + size * np.cos(angle), y + size * np.sin(angle)),  # Front
#             (x - size * np.cos(angle + np.pi/4), y - size * np.sin(angle + np.pi/4)),  # Back left
#             (x - size * np.cos(angle - np.pi/4), y - size * np.sin(angle - np.pi/4))   # Back right
#         ]
#         pygame.draw.polygon(self.display, (0, 0, 255), points)
        
#         pygame.display.flip()

#     def _get_closest_path_point(self, position):
#         """Find the closest point on the center path"""
#         min_dist = float('inf')
#         closest_point = None
#         segment_idx = 0
        
#         for i in range(len(self.center_points) - 1):
#             p1 = np.array(self.center_points[i])
#             p2 = np.array(self.center_points[i + 1])
#             pos = np.array(position)
            
#             segment = p2 - p1
#             length_sq = np.sum(segment**2)
#             if length_sq == 0:
#                 continue
                
#             t = max(0, min(1, np.dot(pos - p1, segment) / length_sq))
#             projection = p1 + t * segment
            
#             dist = np.linalg.norm(pos - projection)
#             if dist < min_dist:
#                 min_dist = dist
#                 closest_point = projection
#                 segment_idx = i
        
#         return closest_point, min_dist, segment_idx

#     def _normalize_angle(self, angle):
#         """Normalize angle to [-pi, pi]"""
#         while angle > np.pi:
#             angle -= 2 * np.pi
#         while angle < -np.pi:
#             angle += 2 * np.pi
#         return angle

#     # def step(self, action):
#     #     # Steering: 0 = left, 1 = straight, 2 = right
#     #     steering = (-self.max_steering_angle if action == 0 
#     #                else self.max_steering_angle if action == 2 
#     #                else 0)
#     #     self.vehicle_angle += steering
        
#     #     # Update vehicle position
#     #     new_pos = [
#     #         self.vehicle_pos[0] + self.vehicle_speed * np.cos(self.vehicle_angle),
#     #         self.vehicle_pos[1] + self.vehicle_speed * np.sin(self.vehicle_angle)
#     #     ]
        
#     #     # Check for collisions with obstacles
#     #     collision = False
#     #     for obs in self.obstacles:
#     #         dist = np.sqrt((new_pos[0] - obs[0])**2 + (new_pos[1] - obs[1])**2)
#     #         if dist < (self.vehicle_size + self.obstacle_size) / 2:
#     #             collision = True
#     #             break
        
#     #     # Check if vehicle is on the road
#     #     offset = (self.window_size - self.square_size) // 2
#     #     on_road = True
#     #     x, y = new_pos
        
#     #     # Simple road boundary check
#     #     if (x < offset - self.road_width//2 or 
#     #         x > offset + self.square_size + self.road_width//2 or
#     #         y < offset - self.road_width//2 or
#     #         y > offset + self.square_size + self.road_width//2):
#     #         on_road = False
        
#     #     # Update position if no collision and on road
#     #     if not collision and on_road:
#     #         self.vehicle_pos = new_pos
        
#     #     # Calculate reward
#     #     reward = 1.0  # Base reward for staying alive
        
#     #     if collision:
#     #         reward = -50
#     #         done = True
#     #     elif not on_road:
#     #         reward = -50
#     #         done = True
#     #     else:
#     #         # Reward for following the path
#     #         _, dist_to_center, _ = self._get_closest_path_point(self.vehicle_pos)
#     #         reward += np.exp(-0.05 * dist_to_center)  # Higher reward for staying close to center
#     #         done = False
        
#     #     return self._get_observation(), reward, done, False, {}
#     def step(self, action):
#         # Steering: 0 = left, 1 = straight, 2 = right
#         steering = (-self.max_steering_angle if action == 0 
#                    else self.max_steering_angle if action == 2 
#                    else 0)
#         self.vehicle_angle += steering
        
#         # Update vehicle position
#         new_pos = [
#             self.vehicle_pos[0] + self.vehicle_speed * np.cos(self.vehicle_angle),
#             self.vehicle_pos[1] + self.vehicle_speed * np.sin(self.vehicle_angle)
#         ]
        
#         # Check for collisions with obstacles
#         collision = False
#         for obs in self.obstacles:
#             dist = np.sqrt((new_pos[0] - obs[0])**2 + (new_pos[1] - obs[1])**2)
#             if dist < (self.vehicle_size + self.obstacle_size) / 2:
#                 collision = True
#                 break
        
#         # Check road boundaries more precisely
#         offset = (self.window_size - self.square_size) // 2
#         x, y = new_pos
        
#         # Calculate distances to inner and outer boundaries
#         outer_boundary_violation = False
#         inner_boundary_violation = False
        
#         # Outer boundary check
#         if (x < offset - self.road_width/2 or 
#             x > offset + self.square_size + self.road_width/2 or
#             y < offset - self.road_width/2 or
#             y > offset + self.square_size + self.road_width/2):
#             outer_boundary_violation = True
            
#         # Inner boundary check
#         if (x > offset + self.road_width/2 and 
#             x < offset + self.square_size - self.road_width/2 and
#             y > offset + self.road_width/2 and
#             y < offset + self.square_size - self.road_width/2):
#             inner_boundary_violation = True
            
#         # Initialize reward
#         reward = 0
#         done = False
        
#         # Collision penalty
#         if collision:
#             reward -= 50  # Heavy penalty for collision
#             done = True
        
#         # Boundary violation penalties
#         if outer_boundary_violation:
#             reward -= 20  # Penalty for crossing outer boundary
#             done = True
#         elif inner_boundary_violation:
#             reward -= 20  # Penalty for crossing inner boundary
#             done = True
#         else:
#             # Only update position if no violations occurred
#             self.vehicle_pos = new_pos
            
#             # Calculate reward for good behavior
#             _, dist_to_center, _ = self._get_closest_path_point(self.vehicle_pos)
            
#             # Base reward for staying alive and on track
#             reward += 2.0
            
#             # Reward for staying close to the center line
#             center_line_reward = np.exp(-0.05 * dist_to_center)
#             reward += 3.0 * center_line_reward
            
#             # Small penalty for being far from center (but still on road)
#             if dist_to_center > self.road_width/3:
#                 reward -= 0.5 * (dist_to_center / self.road_width)
            
#             # Check if near boundary (warning zone)
#             near_boundary = (dist_to_center > self.road_width/2 - self.vehicle_size)
#             if near_boundary:
#                 reward -= 1.0  # Small penalty for being close to boundaries
        
#         return self._get_observation(), reward, done, False, {}

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class VehicleEnvSquarePath(gym.Env):
    def __init__(self):
        super().__init__()
        # Window and display settings
        self.window_size = 800
        self.display = None
        
        # Road parameters
        self.road_width = 80
        self.square_size = 600  # Size of the square road's centerline
        self.outer_color = (100, 100, 100)  # Dark gray for road
        self.inner_color = (200, 200, 200)  # Light gray for inner area
        self.line_color = (255, 255, 255)   # White for road markings
        
        # Vehicle parameters
        self.vehicle_size = 20
        self.vehicle_speed = 5
        self.max_steering_angle = 0.1
        self.vehicle_pos = None
        self.vehicle_angle = None
        
        # Initialize obstacles list
        self.obstacles = []
        self.obstacle_size = 20
        
        # Gym spaces
        self.action_space = spaces.Discrete(3)  # Left, Straight, Right
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, 0, 0, 0, -np.pi, 0, 0, 0]),
            high=np.array([self.window_size, self.window_size, np.pi, 
                          self.window_size, self.window_size, self.window_size, np.pi,
                          self.window_size, self.window_size, self.window_size]),
            dtype=np.float32
        )

    def _generate_road_points(self):
        """Generate points for the square road"""
        offset = (self.window_size - self.square_size) // 2
        
        # Outer square points
        outer_offset = self.road_width // 2
        self.outer_points = [
            (offset - outer_offset, offset - outer_offset),
            (offset + self.square_size + outer_offset, offset - outer_offset),
            (offset + self.square_size + outer_offset, offset + self.square_size + outer_offset),
            (offset - outer_offset, offset + self.square_size + outer_offset)
        ]
        
        # Inner square points
        inner_offset = self.road_width // 2
        self.inner_points = [
            (offset + inner_offset, offset + inner_offset),
            (offset + self.square_size - inner_offset, offset + inner_offset),
            (offset + self.square_size - inner_offset, offset + self.square_size - inner_offset),
            (offset + inner_offset, offset + self.square_size - inner_offset)
        ]
        
        # Center line points for path following
        self.center_points = [
            (offset, offset),
            (offset + self.square_size, offset),
            (offset + self.square_size, offset + self.square_size),
            (offset, offset + self.square_size),
            (offset, offset)  # Close the loop
        ]

    def _generate_obstacles(self):
        """Generate fixed obstacles along the road"""
        offset = (self.window_size - self.square_size) // 2
        
        # Fixed obstacle positions
        fixed_positions = [
            # Top side
            (offset + self.square_size//4, offset),
            (offset + 3*self.square_size//4, offset),
            
            # Right side
            (offset + self.square_size, offset + self.square_size//4),
            (offset + self.square_size, offset + 3*self.square_size//4),
            
            # Bottom side
            (offset + self.square_size//4, offset + self.square_size),
            (offset + 3*self.square_size//4, offset + self.square_size),
            
            # Left side
            (offset, offset + self.square_size//4),
            (offset, offset + 3*self.square_size//4)
        ]
        
        # Offsets from center line
        road_offsets = [
            (0, self.road_width//4),  # Top side
            (0, -self.road_width//4),
            (-self.road_width//4, 0),  # Right side
            (self.road_width//4, 0),
            (0, -self.road_width//4),  # Bottom side
            (0, self.road_width//4),
            (self.road_width//4, 0),  # Left side
            (-self.road_width//4, 0)
        ]
        
        # Create obstacles with offsets
        self.obstacles = [
            (x + offset_x, y + offset_y) 
            for (x, y), (offset_x, offset_y) in zip(fixed_positions, road_offsets)
        ]

    def _get_observation(self):
        """Get current state observation"""
        # Find closest obstacle
        closest_obs = min(self.obstacles, 
                         key=lambda obs: np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
                                               (obs[1] - self.vehicle_pos[1])**2))
        closest_dist = np.sqrt((closest_obs[0] - self.vehicle_pos[0])**2 + 
                             (closest_obs[1] - self.vehicle_pos[1])**2)
        
        # Find closest point on path
        closest_center, dist_to_center, segment_idx = self._get_closest_path_point(self.vehicle_pos)
        next_point_idx = min(segment_idx + 1, len(self.center_points) - 1)
        next_point = self.center_points[next_point_idx]
        
        # Calculate angle to next point
        dx = next_point[0] - self.vehicle_pos[0]
        dy = next_point[1] - self.vehicle_pos[1]
        target_angle = np.arctan2(dy, dx)
        angle_diff = self._normalize_angle(target_angle - self.vehicle_angle)
        
        return np.array([
            self.vehicle_pos[0], self.vehicle_pos[1], self.vehicle_angle,
            closest_obs[0], closest_obs[1],
            closest_dist,
            angle_diff,
            next_point[0], next_point[1],
            dist_to_center
        ], dtype=np.float32)

    def _get_closest_path_point(self, position):
        """Find closest point on the center path"""
        min_dist = float('inf')
        closest_point = None
        segment_idx = 0
        
        for i in range(len(self.center_points) - 1):
            p1 = np.array(self.center_points[i])
            p2 = np.array(self.center_points[i + 1])
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

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def reset(self, seed=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Generate road layout
        self._generate_road_points()
        
        # Initialize vehicle at start position with some offset from the wall
        offset = (self.window_size - self.square_size) // 2
	
    	# Position vehicle exactly on the top-left corner point of the path
    	# with slight adjustment to be on the path
        self.vehicle_pos = [offset, offset]

        self.vehicle_angle = 0  # Facing right initially
        
        # Generate fixed obstacles
        self._generate_obstacles()
        
        return self._get_observation(), {}

    def step(self, action):
        """Execute one environment step"""
        # Steering: 0 = left, 1 = straight, 2 = right
        steering = (-self.max_steering_angle if action == 0 
                   else self.max_steering_angle if action == 2 
                   else 0)
        self.vehicle_angle += steering
        
        # Update vehicle position
        new_pos = [
            self.vehicle_pos[0] + self.vehicle_speed * np.cos(self.vehicle_angle),
            self.vehicle_pos[1] + self.vehicle_speed * np.sin(self.vehicle_angle)
        ]
        
        # Check for collisions with obstacles
        collision = False
        for obs in self.obstacles:
            dist = np.sqrt((new_pos[0] - obs[0])**2 + (new_pos[1] - obs[1])**2)
            if dist < (self.vehicle_size + self.obstacle_size) / 2:
                collision = True
                break
        
        # Check road boundaries
        offset = (self.window_size - self.square_size) // 2
        x, y = new_pos
        
        # Boundary violations
        outer_violation = (x < offset - self.road_width/2 or 
                         x > offset + self.square_size + self.road_width/2 or
                         y < offset - self.road_width/2 or
                         y > offset + self.square_size + self.road_width/2)
        
        inner_violation = (x > offset + self.road_width/2 and 
                         x < offset + self.square_size - self.road_width/2 and
                         y > offset + self.road_width/2 and
                         y < offset + self.square_size - self.road_width/2)
        
        # Calculate reward and check terminal conditions
        reward = 0
        done = False
        
        if collision:
            reward = -50  # Collision penalty
            done = True
        elif outer_violation or inner_violation:
            reward = -30  # Boundary violation penalty
            done = True
        else:
            # Update position if no violations
            self.vehicle_pos = new_pos
            
            # Get distance to center line
            _, dist_to_center, segment_idx = self._get_closest_path_point(self.vehicle_pos)
            
            # Base reward for staying on track
            reward += 1.0
            
            # Reward for staying close to center line
            center_reward = np.exp(-0.05 * dist_to_center)
            reward += 2.0 * center_reward
            
            # Penalty for being far from center
            if dist_to_center > self.road_width/3:
                reward -= 0.5 * (dist_to_center / self.road_width)
            
            # Penalty for being near boundaries
            if dist_to_center > self.road_width/2 - self.vehicle_size:
                reward -= 1.0

        return self._get_observation(), reward, done, False, {}

    def render(self):
        """Render the environment"""
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.window_size, self.window_size))
        
        # Fill background
        self.display.fill((50, 50, 50))
        
        # Draw road
        pygame.draw.polygon(self.display, self.outer_color, self.outer_points)
        pygame.draw.polygon(self.display, self.inner_color, self.inner_points)
        
        # Draw center line (dashed)
        for i in range(len(self.center_points) - 1):
            p1 = self.center_points[i]
            p2 = self.center_points[i + 1]
            
            # Create dashed line
            dash_length = 20
            total_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            num_dashes = int(total_length / (dash_length * 2))
            
            for j in range(num_dashes):
                start_frac = j * 2 * dash_length / total_length
                end_frac = (j * 2 + 1) * dash_length / total_length
                
                start_x = p1[0] + (p2[0] - p1[0]) * start_frac
                start_y = p1[1] + (p2[1] - p1[1]) * start_frac
                end_x = p1[0] + (p2[0] - p1[0]) * end_frac
                end_y = p1[1] + (p2[1] - p1[1]) * end_frac
                
                pygame.draw.line(self.display, self.line_color, 
                               (int(start_x), int(start_y)), 
                               (int(end_x), int(end_y)), 2)
        
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.circle(self.display, (255, 0, 0), 
                             (int(obs[0]), int(obs[1])), 
                             self.obstacle_size // 2)
        
        # Draw vehicle
        x, y = self.vehicle_pos
        size = self.vehicle_size
        angle = self.vehicle_angle
        points = [
            (x + size * np.cos(angle), y + size * np.sin(angle)),  # Front
            (x - size * np.cos(angle + np.pi/4), y - size * np.sin(angle + np.pi/4)),  # Back left
            (x - size * np.cos(angle - np.pi/4), y - size * np.sin(angle - np.pi/4))   # Back right
        ]
        pygame.draw.polygon(self.display, (0, 0, 255), points)
        
        pygame.display.flip()