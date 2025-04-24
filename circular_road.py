#//////ORIGINAL CIRCULAR ENVIRONMENT//////

# import gymnasium as gym
# import numpy as np
# import pygame
# from gymnasium import spaces

# class VehicleEnv(gym.Env):
#     def __init__(self):
#         super(VehicleEnv, self).__init__()
        
#         # Window size
#         self.window_size = 800
#         self.display = None
        
#         # Vehicle properties
#         self.vehicle_size = 20
#         self.vehicle_speed = 5
#         self.max_steering_angle = 0.1
        
#         # Path properties
#         self.path_radius = 300
#         self.path_center = (self.window_size // 2, self.window_size // 2)
#         self.road_width = 60  # Width of the road
#         self.lane_marking_length = 20  # Length of each dash in the lane marking
#         self.lane_marking_gap = 20  # Gap between dashes
        
#         # Obstacle properties
#         self.obstacles = []
#         self.num_random_obstacles = 3
#         self.num_path_obstacles = 4
#         self.obstacle_size = 30
        
#         # Action space: [steer_left, no_action, steer_right]
#         self.action_space = spaces.Discrete(3)
        
#         # Observation space
#         self.observation_space = spaces.Box(
#             low=np.array([0, 0, -np.pi, 0, 0, -self.window_size, -np.pi]),
#             high=np.array([self.window_size, self.window_size, np.pi, 
#                           self.window_size, self.window_size, self.window_size, np.pi]),
#             dtype=np.float32
#         )
        
#         # Colors
#         self.ROAD_COLOR = (80, 80, 80)  # Dark gray for road
#         self.BORDER_COLOR = (255, 255, 255)  # White for road borders
#         self.MARKING_COLOR = (255, 240, 0)  # Yellow for center line
#         self.GRASS_COLOR = (34, 139, 34)  # Green for surrounding area
        
#         # Initial state
#         self.vehicle_pos = None
#         self.vehicle_angle = None
#         self.target_angle = None

#     def _generate_path_obstacles(self):
#         """Generate obstacles along the circular path"""
#         path_obstacles = []
#         # Place obstacles at specific angles along the path
#         angles = [np.pi/4, np.pi/2, np.pi, 3*np.pi/2]  # Specific positions along the circle
        
#         for angle in angles:
#             # Place obstacle slightly inside or outside the path
#             offset = np.random.choice([-20, 20])  # Randomly place inside or outside
#             pos = [
#                 self.path_center[0] + (self.path_radius + offset) * np.cos(angle),
#                 self.path_center[1] + (self.path_radius + offset) * np.sin(angle)
#             ]
#             path_obstacles.append(pos)
        
#         return path_obstacles

#     def reset(self, seed=None):
#         super().reset(seed=seed)
#         # Initialize vehicle position slightly off the path
#         angle = np.random.random() * 2 * np.pi
#         offset = np.random.uniform(-20, 20)
#         self.vehicle_pos = [
#             self.path_center[0] + (self.path_radius + offset) * np.cos(angle),
#             self.path_center[1] + (self.path_radius + offset) * np.sin(angle)
#         ]
#         self.vehicle_angle = angle + np.pi/2
#         self.target_angle = self.vehicle_angle
        
#         # Generate path obstacles
#         self.obstacles = self._generate_path_obstacles()
        
#         # Add random obstacles
#         for _ in range(self.num_random_obstacles):
#             while True:
#                 pos = [
#                     np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size),
#                     np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size)
#                 ]
#                 dist_to_center = np.sqrt((pos[0] - self.path_center[0])**2 + 
#                                       (pos[1] - self.path_center[1])**2)
                
#                 # Check if position is far enough from existing obstacles
#                 too_close = False
#                 for obs in self.obstacles:
#                     if np.sqrt((pos[0] - obs[0])**2 + (pos[1] - obs[1])**2) < self.obstacle_size * 3:
#                         too_close = True
#                         break
                
#                 if not too_close and abs(dist_to_center - self.path_radius) > self.obstacle_size * 2:
#                     self.obstacles.append(pos)
#                     break
        
#         return self._get_observation(), {}

#     def _get_observation(self):
#         # Find closest obstacle
#         closest_obstacle = min(self.obstacles, 
#                              key=lambda obs: np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
#                                                    (obs[1] - self.vehicle_pos[1])**2))
        
#         # Calculate distance and angle to path
#         dx = self.vehicle_pos[0] - self.path_center[0]
#         dy = self.vehicle_pos[1] - self.path_center[1]
#         dist_to_center = np.sqrt(dx**2 + dy**2)
#         dist_to_path = dist_to_center - self.path_radius
        
#         # Calculate ideal tangent angle at current position
#         ideal_angle = np.arctan2(dy, dx) + np.pi/2
#         angle_diff = self._normalize_angle(ideal_angle - self.vehicle_angle)
        
#         return np.array([
#             self.vehicle_pos[0],
#             self.vehicle_pos[1],
#             self.vehicle_angle,
#             closest_obstacle[0],
#             closest_obstacle[1],
#             dist_to_path,
#             angle_diff
#         ], dtype=np.float32)

#     def _normalize_angle(self, angle):
#         while angle > np.pi:
#             angle -= 2 * np.pi
#         while angle < -np.pi:
#             angle += 2 * np.pi
#         return angle

#     def step(self, action):
#         # Update vehicle angle based on action
#         steering = 0
#         if action == 0:  # steer left
#             steering = -self.max_steering_angle
#         elif action == 2:  # steer right
#             steering = self.max_steering_angle
        
#         self.vehicle_angle += steering
        
#         # Update vehicle position
#         self.vehicle_pos[0] += self.vehicle_speed * np.cos(self.vehicle_angle)
#         self.vehicle_pos[1] += self.vehicle_speed * np.sin(self.vehicle_angle)
        
#         # Calculate reward
#         reward = 0
#         done = False
        
#         # Reward for staying on path
#         dx = self.vehicle_pos[0] - self.path_center[0]
#         dy = self.vehicle_pos[1] - self.path_center[1]
#         dist_to_center = np.sqrt(dx**2 + dy**2)
#         path_deviation = abs(dist_to_center - self.path_radius)
        
#         # Calculate ideal tangent angle
#         ideal_angle = np.arctan2(dy, dx) + np.pi/2
#         angle_diff = abs(self._normalize_angle(ideal_angle - self.vehicle_angle))
        
#         # Enhanced reward structure
#         path_reward = np.exp(-0.01 * path_deviation)
#         angle_reward = np.exp(-1.0 * angle_diff)
#         obstacle_avoidance_reward = 0
        
#         # Add obstacle avoidance reward
#         closest_obstacle_dist = min(np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
#                                           (obs[1] - self.vehicle_pos[1])**2) 
#                                   for obs in self.obstacles)
#         if closest_obstacle_dist > self.obstacle_size * 2:
#             obstacle_avoidance_reward = 0.5  # Reward for keeping safe distance
        
#         reward = path_reward + angle_reward + obstacle_avoidance_reward
        
#         # Penalties
#         if path_deviation > 50:
#             reward -= 5
        
#         # Enhanced obstacle collision penalty
#         for obs in self.obstacles:
#             dist_to_obs = np.sqrt((obs[0] - self.vehicle_pos[0])**2 + 
#                                  (obs[1] - self.vehicle_pos[1])**2)
#             if dist_to_obs < (self.vehicle_size + self.obstacle_size) / 2:
#                 reward -= 30
#                 done = True
#             elif dist_to_obs < self.obstacle_size * 2:
#                 reward -= (2 * self.obstacle_size - dist_to_obs) * 0.1  # Gradual penalty for getting too close
        
#         # Out of bounds
#         if (self.vehicle_pos[0] < 0 or self.vehicle_pos[0] > self.window_size or
#             self.vehicle_pos[1] < 0 or self.vehicle_pos[1] > self.window_size):
#             reward -= 20
#             done = True
        
#         return self._get_observation(), reward, done, False, {}

#     def _draw_dashed_circle(self, surface, color, center, radius, width=2, dash_length=20, gap_length=20):
#         """Draw a dashed circular line"""
#         circumference = 2 * np.pi * radius
#         n_dashes = int(circumference / (dash_length + gap_length))
#         angle_per_dash = 2 * np.pi / n_dashes
        
#         for i in range(n_dashes):
#             start_angle = i * angle_per_dash
#             end_angle = start_angle + (angle_per_dash * dash_length/(dash_length + gap_length))
            
#             # Calculate points for the arc
#             points = []
#             for angle in np.linspace(start_angle, end_angle, 20):
#                 x = center[0] + radius * np.cos(angle)
#                 y = center[1] + radius * np.sin(angle)
#                 points.append((int(x), int(y)))
            
#             # Draw the dash
#             if len(points) > 1:
#                 pygame.draw.lines(surface, color, False, points, width)

#     def render(self):
#         if self.display is None:
#             pygame.init()
#             self.display = pygame.display.set_mode((self.window_size, self.window_size))
        
#         # Fill background with grass color
#         self.display.fill(self.GRASS_COLOR)
        
#         # Draw outer and inner road borders
#         outer_radius = self.path_radius + self.road_width//2
#         inner_radius = self.path_radius - self.road_width//2
        
#         # Draw road (filled circle with center removed)
#         pygame.draw.circle(self.display, self.ROAD_COLOR, self.path_center, outer_radius)
        
#         # Create center hole for inner grass
#         inner_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
#         pygame.draw.circle(inner_surface, self.GRASS_COLOR, self.path_center, inner_radius)
#         self.display.blit(inner_surface, (0, 0))
        
#         # Draw road borders (solid white lines)
#         pygame.draw.circle(self.display, self.BORDER_COLOR, self.path_center, outer_radius, 3)
#         pygame.draw.circle(self.display, self.BORDER_COLOR, self.path_center, inner_radius, 3)
        
#         # Draw center dashed line
#         self._draw_dashed_circle(
#             self.display, 
#             self.MARKING_COLOR, 
#             self.path_center, 
#             self.path_radius, 
#             width=2, 
#             dash_length=self.lane_marking_length, 
#             gap_length=self.lane_marking_gap
#         )
        
#         # Draw obstacles with different colors for path and random obstacles
#         for i, obs in enumerate(self.obstacles):
#             color = (255, 0, 0) if i < self.num_path_obstacles else (255, 165, 0)
#             pygame.draw.circle(self.display, color, 
#                              (int(obs[0]), int(obs[1])), 
#                              self.obstacle_size)
        
#         # Draw vehicle (car-like shape)
#         car_center = (int(self.vehicle_pos[0]), int(self.vehicle_pos[1]))
        
#         # Draw vehicle body
#         car_points = [
#             (car_center[0] + self.vehicle_size * np.cos(self.vehicle_angle - 0.5),
#              car_center[1] + self.vehicle_size * np.sin(self.vehicle_angle - 0.5)),
#             (car_center[0] + self.vehicle_size * np.cos(self.vehicle_angle + 0.5),
#              car_center[1] + self.vehicle_size * np.sin(self.vehicle_angle + 0.5)),
#             (car_center[0] - self.vehicle_size * np.cos(self.vehicle_angle - 0.5),
#              car_center[1] - self.vehicle_size * np.sin(self.vehicle_angle - 0.5)),
#             (car_center[0] - self.vehicle_size * np.cos(self.vehicle_angle + 0.5),
#              car_center[1] - self.vehicle_size * np.sin(self.vehicle_angle + 0.5))
#         ]
#         pygame.draw.polygon(self.display, (0, 0, 255), car_points)  # Blue vehicle
        
#         # Draw direction indicator (headlights)
#         front_center = (car_center[0] + self.vehicle_size * np.cos(self.vehicle_angle),
#                        car_center[1] + self.vehicle_size * np.sin(self.vehicle_angle))
#         pygame.draw.circle(self.display, (255, 255, 0), 
#                          (int(front_center[0]), int(front_center[1])), 4)
        
#         pygame.display.flip()

#     def close(self):
#         if self.display is not None:
#             pygame.quit()
#             self.display = None






#///////NEW CIRCULAR_ENVIRONMENT WITH LIDAR SENSING OBSTACLES,PATH BASED ON WAYPOINTS,FIXED STARTING POINT OF THE AGENT/////
#///////no of obstacle reduced and increased training speed//////
#/////// Added a waypoint-progress term, Harden the deviation penalty, Flip your obstacle bonus into a smaller penalty /////


import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class VehicleEnv(gym.Env):
    def __init__(self,
                 window_size=800,
                 path_radius=300,
                 road_width=40,
                 num_waypoints=72,
                 num_lidar_beams=8,
                 lidar_max_range=200,
                 obstacle_size=20,
                 num_random_obstacles=2):
        super().__init__()
        # rendering handle
        self.display = None

        # Environment dimensions
        self.window_size = window_size
        self.path_radius = path_radius
        self.road_width = road_width
        self.path_center = (window_size // 2, window_size // 2)

        # Waypoints forming the track
        angles = np.linspace(0, 2*np.pi, num_waypoints, endpoint=False)
        self.waypoints = [
            (self.path_center[0] + path_radius * np.cos(a),
             self.path_center[1] + path_radius * np.sin(a))
            for a in angles
        ]
        # start just after obstacle index
        self.start_idx = 1

        # Sensor parameters
        self.num_lidar_beams = num_lidar_beams
        self.lidar_max_range = lidar_max_range

        # Vehicle dynamics
        self.vehicle_size = 20
        self.vehicle_speed = 10
        self.max_steering_angle = 0.1

        # Reward coefficients
        self.progress_coef = 1.0            # waypoint progress weight
        self.deviation_coef = 0.1           # penalty per pixel deviation beyond threshold
        self.obstacle_coef = 5.0            # obstacle avoidance penalty weight

        # Obstacles: fixed + random
        self.obstacle_size = obstacle_size
        fixed_angles = [0, 2*np.pi/3, 4*np.pi/3]
        self.fixed_obstacles = [
            (self.path_center[0] + path_radius * np.cos(a),
             self.path_center[1] + path_radius * np.sin(a))
            for a in fixed_angles
        ]
        self.num_random_obstacles = num_random_obstacles
        self.random_obstacles = []
        self.obstacles = []

        # Action & observation spaces
        self.action_space = spaces.Discrete(3)
        low = np.array([0, 0, -np.pi, 0, -np.pi] + [0]*num_lidar_beams, dtype=np.float32)
        high = np.array([window_size, window_size, np.pi, path_radius, np.pi]
                        + [lidar_max_range]*num_lidar_beams,
                        dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Colors
        self.ROAD_COLOR = (80, 80, 80)
        self.GRASS_COLOR = (34, 139, 34)
        self.FIXED_OBS_COLOR = (255, 0, 0)
        self.RANDOM_OBS_COLOR = (255, 165, 0)

        # State
        self.vehicle_pos = None
        self.vehicle_angle = None

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Fixed start
        w0 = np.array(self.waypoints[self.start_idx])
        w1 = np.array(self.waypoints[(self.start_idx + 1) % len(self.waypoints)])
        tangent = np.arctan2(w1[1]-w0[1], w1[0]-w0[0])
        self.vehicle_pos = w0.tolist()
        self.vehicle_angle = tangent

        # Obstacles
        self.random_obstacles = []
        for _ in range(self.num_random_obstacles):
            while True:
                x = np.random.uniform(self.obstacle_size, self.window_size - self.obstacle_size)
                y = np.random.uniform(self.obstacle_size, self.window_size - self.obstacle_size)
                if all(np.hypot(x-ox, y-oy) > self.obstacle_size*3
                       for ox,oy in self.fixed_obstacles + self.random_obstacles):
                    self.random_obstacles.append((x,y))
                    break
        self.obstacles = list(self.fixed_obstacles) + self.random_obstacles

        return self._get_observation(), {}

    def _get_observation(self):
        dev, tangent = self._compute_path_error()
        heading_err = self._normalize_angle(tangent - self.vehicle_angle)
        lidar = self._perform_lidar_scan()
        return np.array([*self.vehicle_pos, self.vehicle_angle, dev, heading_err] + lidar, dtype=np.float32)

    def _compute_path_error(self):
        P = np.array(self.vehicle_pos)
        min_dist, best_tan = float('inf'), 0.0
        W = np.array(self.waypoints)
        for i in range(len(W)):
            A, B = W[i], W[(i+1) % len(W)]
            AB = B - A
            t = np.clip(np.dot(P-A, AB) / np.dot(AB,AB), 0, 1)
            proj = A + t*AB
            d = np.linalg.norm(P-proj)
            if d < min_dist:
                min_dist, best_tan = d, np.arctan2(AB[1], AB[0])
        return min_dist, best_tan

    def _perform_lidar_scan(self):
        readings = []
        for rel in np.linspace(-np.pi/2, np.pi/2, self.num_lidar_beams):
            ang = self.vehicle_angle + rel
            dist = self.lidar_max_range
            for d in np.linspace(0, self.lidar_max_range, 100):
                x = self.vehicle_pos[0] + d*np.cos(ang)
                y = self.vehicle_pos[1] + d*np.sin(ang)
                if not (0 <= x <= self.window_size and 0 <= y <= self.window_size):
                    dist = d
                    break
                if any(np.hypot(x-ox, y-oy) < self.obstacle_size/2 for ox,oy in self.obstacles):
                    dist = d
                    break
            readings.append(dist)
        return readings

    def _find_prev_waypoint(self):
        dists = [np.linalg.norm(np.array(w)-np.array(self.vehicle_pos)) for w in self.waypoints]
        idx = int(np.argmin(dists))
        return idx, (idx+1) % len(self.waypoints)

    def step(self, action):
        old_pos = np.array(self.vehicle_pos)
        if action == 0:
            self.vehicle_angle -= self.max_steering_angle
        elif action == 2:
            self.vehicle_angle += self.max_steering_angle
        self.vehicle_pos[0] += self.vehicle_speed * np.cos(self.vehicle_angle)
        self.vehicle_pos[1] += self.vehicle_speed * np.sin(self.vehicle_angle)

        dev, tangent = self._compute_path_error()
        heading_err = abs(self._normalize_angle(tangent - self.vehicle_angle))
        lidar = self._perform_lidar_scan()
        center = lidar[self.num_lidar_beams // 2]

        reward = np.exp(-0.01 * dev) + np.exp(-1.0 * heading_err)
        if center < self.lidar_max_range:
            reward -= self.obstacle_coef * (1 - center / self.lidar_max_range)

        prev_idx, next_idx = self._find_prev_waypoint()
        seg = np.array(self.waypoints[next_idx]) - np.array(self.waypoints[prev_idx])
        move = np.array(self.vehicle_pos) - old_pos
        prog = np.dot(move, seg) / np.linalg.norm(seg) if np.linalg.norm(seg) > 0 else 0
        if prog > 0:
            reward += self.progress_coef * (prog / np.linalg.norm(seg))

        if dev > self.road_width / 2:
            reward -= self.deviation_coef * (dev - self.road_width / 2)

        done = False
        for ox,oy in self.obstacles:
            if np.hypot(self.vehicle_pos[0]-ox, self.vehicle_pos[1]-oy) < (self.vehicle_size + self.obstacle_size)/2:
                reward -= 30
                done = True
                break
        x,y = self.vehicle_pos
        if not (0 <= x <= self.window_size and 0 <= y <= self.window_size):
            reward -= 20
            done = True

        return self._get_observation(), reward, done, False, {}

    def render(self):
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.window_size, self.window_size))
        self.display.fill(self.GRASS_COLOR)
        for i in range(len(self.waypoints)):
            p1 = tuple(map(int, self.waypoints[i]))
            p2 = tuple(map(int, self.waypoints[(i+1) % len(self.waypoints)]))
            pygame.draw.line(self.display, self.ROAD_COLOR, p1, p2, self.obstacle_size)
        for ox,oy in self.fixed_obstacles:
            pygame.draw.circle(self.display, self.FIXED_OBS_COLOR, (int(ox),int(oy)), self.obstacle_size)
        for ox,oy in self.random_obstacles:
            pygame.draw.circle(self.display, self.RANDOM_OBS_COLOR, (int(ox),int(oy)), self.obstacle_size)
        cx,cy = map(int, self.vehicle_pos)
        pts = [
            (cx + self.vehicle_size * np.cos(self.vehicle_angle + ang),
             cy + self.vehicle_size * np.sin(self.vehicle_angle + ang))
            for ang in (-0.5, 0.5, np.pi-0.5, np.pi+0.5)
        ]
        pygame.draw.polygon(self.display, (0,0,255), pts)
        for rel, dist in zip(np.linspace(-np.pi/2, np.pi/2, self.num_lidar_beams), self._perform_lidar_scan()):
            ang = self.vehicle_angle + rel
            ex = cx + dist * np.cos(ang)
            ey = cy + dist * np.sin(ang)
            pygame.draw.line(self.display, (0,255,0), (cx,cy), (int(ex),int(ey)), 1)
        pygame.display.flip()

    def close(self):
        if self.display:
            pygame.quit()
            self.display = None

    def _normalize_angle(self, a):
        return (a + np.pi) % (2*np.pi) - np.pi

