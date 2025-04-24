import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class VehicleEnvOvalTrack(gym.Env):
    """
    An oval track environment with lidar-based obstacle sensing.
    Renders a road-like oval corridor, 2 fixed obstacles, and 3 random obstacles.
    Applies a heavy penalty when deviating beyond the track boundaries.
    """
    def __init__(self):
        super().__init__()
        # Window settings
        self.window_size = 800
        self.display = None

        # Vehicle dynamics
        self.vehicle_size = 20
        self.vehicle_speed = 4.0  # slower speed for learning
        self.max_steering_angle = 0.1  # radians per step

        # Road appearance
        self.road_width = 60
        self.border_width = 2
        self.dash_length = 20
        self.dash_gap = 20
        self.ROAD_COLOR = (80, 80, 80)
        self.BORDER_COLOR = (255, 255, 255)
        self.MARKING_COLOR = (255, 240, 0)
        self.GRASS_COLOR = (34, 139, 34)

        # Colors for obstacles and vehicle
        self.FIXED_OBS_COLOR = (255, 0, 0)
        self.RANDOM_OBS_COLOR = (255, 165, 0)
        self.VEHICLE_COLOR = (0, 0, 255)

        # Lidar settings
        self.num_lidar_beams = 8
        self.lidar_max_range = 200
        self.LIDAR_COLOR = (0, 255, 0)

        # Obstacles
        self.obstacle_size = 15
        self.fixed_obstacles = [(self.window_size*0.5 + self.road_width, self.window_size*0.25),
                                 (self.window_size*0.5 - self.road_width, self.window_size*0.75)]
        self.num_random_obstacles = 3
        self.random_obstacles = []
        self.obstacles = []

        # Track
        self.path_points = self._generate_oval_path(num_points=100)
        self.curr_idx = 0  # track current waypoint index

        # Penalties
        self.deviation_coef = 10.0  # heavy penalty for straying

        # Spaces: [x, y, theta, closest_obs_x, closest_obs_y, dist_to_path, angle_diff, next_wp_x, next_wp_y] + lidar
        low = np.concatenate([np.zeros(9), np.zeros(self.num_lidar_beams)])
        high = np.concatenate([
            np.array([
                self.window_size, self.window_size, np.pi,
                self.window_size, self.window_size,
                self.window_size, self.window_size,
                self.window_size, self.window_size
            ]),
            np.full(self.num_lidar_beams, self.lidar_max_range)
        ])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Initial state
        self.vehicle_pos = [0.0, 0.0]
        self.vehicle_angle = 0.0

    def _generate_oval_path(self, num_points=100):
        cx, cy = self.window_size/2, self.window_size/2
        a, b = self.window_size*0.4, self.window_size*0.3
        thetas = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        return [(cx + a*np.cos(t), cy + b*np.sin(t)) for t in thetas]

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Reset to first waypoint
        self.curr_idx = 0
        self.vehicle_pos = list(self.path_points[0])
        dx, dy = np.subtract(self.path_points[1], self.path_points[0])
        self.vehicle_angle = np.arctan2(dy, dx)

        # Fixed + random obstacles
        self.obstacles = list(self.fixed_obstacles)
        self.random_obstacles = []
        while len(self.random_obstacles) < self.num_random_obstacles:
            x = np.random.uniform(self.obstacle_size, self.window_size - self.obstacle_size)
            y = np.random.uniform(self.obstacle_size, self.window_size - self.obstacle_size)
            if all(np.hypot(x-ox, y-oy) > self.obstacle_size*3 for ox, oy in self.obstacles):
                self.random_obstacles.append((x, y))
                self.obstacles.append((x, y))
        return self._get_observation(), {}

    def _perform_lidar(self):
        readings = []
        for rel in np.linspace(-np.pi/2, np.pi/2, self.num_lidar_beams):
            ang = self.vehicle_angle + rel
            dist = self.lidar_max_range
            for d in np.linspace(0, self.lidar_max_range, 200):
                x = self.vehicle_pos[0] + d * np.cos(ang)
                y = self.vehicle_pos[1] + d * np.sin(ang)
                if not (0 <= x <= self.window_size and 0 <= y <= self.window_size):
                    dist = d; break
                if any(np.hypot(x-ox, y-oy) < self.obstacle_size for ox, oy in self.obstacles):
                    dist = d; break
            readings.append(dist)
        return readings

    def _get_closest_path_point(self, pos):
        best_d = float('inf')
        best_proj = pos
        for i in range(len(self.path_points)-1):
            p1 = np.array(self.path_points[i])
            p2 = np.array(self.path_points[i+1])
            seg = p2 - p1
            L2 = np.dot(seg, seg)
            if L2 == 0: continue
            t = np.clip((np.array(pos)-p1).dot(seg)/L2, 0, 1)
            proj = p1 + t*seg
            d = np.linalg.norm(np.array(pos)-proj)
            if d < best_d:
                best_d, best_proj = d, proj
        return best_proj, best_d

    def _get_observation(self):
        closest_obs = min(self.obstacles,
                          key=lambda o: np.hypot(o[0]-self.vehicle_pos[0], o[1]-self.vehicle_pos[1]))
        proj, dist_to_path = self._get_closest_path_point(self.vehicle_pos)
        target_idx = min(self.curr_idx+1, len(self.path_points)-1)
        nxt = self.path_points[target_idx]
        angle_diff = self._normalize_angle(np.arctan2(nxt[1]-self.vehicle_pos[1], nxt[0]-self.vehicle_pos[0])
                                          - self.vehicle_angle)
        lidar = self._perform_lidar()
        return np.array([
            self.vehicle_pos[0], self.vehicle_pos[1], self.vehicle_angle,
            closest_obs[0], closest_obs[1], dist_to_path, angle_diff,
            nxt[0], nxt[1]
        ] + lidar, dtype=np.float32)

    def step(self, action):
        if action == 0: self.vehicle_angle -= self.max_steering_angle
        elif action == 2: self.vehicle_angle += self.max_steering_angle
        self.vehicle_pos[0] += self.vehicle_speed * np.cos(self.vehicle_angle)
        self.vehicle_pos[1] += self.vehicle_speed * np.sin(self.vehicle_angle)

        # Advance to next waypoint
        target_idx = min(self.curr_idx+1, len(self.path_points)-1)
        tx, ty = self.path_points[target_idx]
        if np.hypot(self.vehicle_pos[0]-tx, self.vehicle_pos[1]-ty) < self.vehicle_size:
            self.curr_idx = target_idx

        obs = self._get_observation()
        reward, done = 0.0, False

        # Deviation penalty
        _, dist_to_path = self._get_closest_path_point(self.vehicle_pos)
        if dist_to_path > self.road_width/2:
            reward -= self.deviation_coef * (dist_to_path - self.road_width/2)

        # Collision penalty
        for ox, oy in self.obstacles:
            if np.hypot(self.vehicle_pos[0]-ox, self.vehicle_pos[1]-oy) < self.obstacle_size:
                reward -= 30; done = True; break

        # Bounds penalty
        x, y = self.vehicle_pos
        if not (0 <= x <= self.window_size and 0 <= y <= self.window_size):
            reward -= 20; done = True

        # Goal check
        if self.curr_idx >= len(self.path_points)-1:
            reward += 50; done = True

        return obs, reward, done, False, {}

    def render(self):
        if self.display is None:
            pygame.init(); self.display = pygame.display.set_mode((self.window_size, self.window_size))
        # Background
        self.display.fill(self.GRASS_COLOR)
        # Road corridor
        pts = [(int(x), int(y)) for x, y in self.path_points]
        road_surf = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        pygame.draw.lines(road_surf, self.ROAD_COLOR, False, pts, self.road_width)
        self.display.blit(road_surf, (0, 0))
        pygame.draw.lines(self.display, self.BORDER_COLOR, False, pts, self.border_width)
        # Center dashes
        for i in range(len(pts)-1):
            start, end = pts[i], pts[i+1]
            dist = np.hypot(end[0]-start[0], end[1]-start[1])
            ux, uy = (end[0]-start[0])/dist, (end[1]-start[1])/dist
            segs = int(dist/(self.dash_length+self.dash_gap))
            for k in range(segs):
                sx = start[0] + (self.dash_length+self.dash_gap)*k*ux
                sy = start[1] + (self.dash_length+self.dash_gap)*k*uy
                ex = sx + self.dash_length*ux; ey = sy + self.dash_length*uy
                pygame.draw.line(self.display, self.MARKING_COLOR, (int(sx), int(sy)), (int(ex), int(ey)), self.border_width)
        # Obstacles
        for ox, oy in self.fixed_obstacles:
            pygame.draw.circle(self.display, self.FIXED_OBS_COLOR, (int(ox), int(oy)), self.obstacle_size)
        for ox, oy in self.random_obstacles:
            pygame.draw.circle(self.display, self.RANDOM_OBS_COLOR, (int(ox), int(oy)), self.obstacle_size)
        # Vehicle
        cx, cy = map(int, self.vehicle_pos)
        car_pts = [(cx + self.vehicle_size*np.cos(self.vehicle_angle+ang), cy + self.vehicle_size*np.sin(self.vehicle_angle+ang)) for ang in (0, 2.6, -2.6)]
        pygame.draw.polygon(self.display, self.VEHICLE_COLOR, car_pts)
        # Lidar
        for rel, dist in zip(np.linspace(-np.pi/2, np.pi/2, self.num_lidar_beams), self._perform_lidar()):
            ang = self.vehicle_angle + rel
            ex = cx + dist*np.cos(ang); ey = cy + dist*np.sin(ang)
            pygame.draw.line(self.display, self.LIDAR_COLOR, (cx, cy), (int(ex), int(ey)), 1)
        pygame.display.flip()

# alias
VehicleEnv = VehicleEnvOvalTrack
