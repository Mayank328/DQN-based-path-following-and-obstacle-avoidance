import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class VehicleEnvZigZag(gym.Env):
    """
    An environment with a zigzag path of curved corners, lidar-based obstacle sensing,
    rendering a road corridor, 2 fixed obstacles, and 3 random obstacles.
    Heavy penalty for deviating off the road.
    """
    def __init__(self):
        super().__init__()
        # Window settings
        self.window_size = 800
        self.display = None

        # Vehicle dynamics
        self.vehicle_size = 20
        self.vehicle_speed = 10
        self.max_steering_angle = 0.1

        # Road appearance
        self.road_width = 60
        self.border_width = 2
        self.dash_length = 20
        self.dash_gap = 20
        self.ROAD_COLOR = (80, 80, 80)
        self.BORDER_COLOR = (255, 255, 255)
        self.MARKING_COLOR = (255, 240, 0)
        self.GRASS_COLOR = (34, 139, 34)

        # Obstacles colors
        self.FIXED_OBS_COLOR = (255, 0, 0)
        self.RANDOM_OBS_COLOR = (255, 165, 0)
        self.VEHICLE_COLOR = (0, 0, 255)

        # Lidar settings
        self.num_lidar_beams = 8
        self.lidar_max_range = 200
        self.LIDAR_COLOR = (0, 255, 0)

        # Obstacles
        self.obstacle_size = 15
        verts = self._zigzag_vertices()
        self.fixed_obstacles = [verts[1], verts[-2]]
        self.num_random_obstacles = 3
        self.random_obstacles = []
        self.obstacles = []

        # Path
        self.path_points = self._generate_zigzag_path()
        self.curr_idx = 0

        # Deviation penalty
        self.deviation_coef = 10.0

        # Action & observation spaces
        low = np.concatenate([np.zeros(9), np.zeros(self.num_lidar_beams)])
        high = np.concatenate([
            np.array([self.window_size, self.window_size, np.pi,
                      self.window_size, self.window_size,
                      self.window_size, self.window_size,
                      self.window_size, self.window_size]),
            np.full(self.num_lidar_beams, self.lidar_max_range)
        ])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.vehicle_pos = [0.0, 0.0]
        self.vehicle_angle = 0.0

    def _zigzag_vertices(self, num_zigs=5):
        # create zigzag control points alternating top/bottom
        xs = np.linspace(self.road_width, self.window_size - self.road_width, num_zigs)
        ys = [self.road_width*2 if i % 2 == 0 else self.window_size - self.road_width*2 for i in range(num_zigs)]
        return list(zip(xs, ys))

    def _generate_zigzag_path(self):
        verts = self._zigzag_vertices()
        # Catmull-Rom spline through verts
        pts = []
        ext = [verts[0]] + verts + [verts[-1]]
        for i in range(len(ext) - 3):
            P0, P1, P2, P3 = map(np.array, ext[i:i+4])
            for t in np.linspace(0, 1, 20):
                # compute powers of t
                t2 = t * t
                t3 = t2 * t
                # Catmull-Rom basis functions
                f1 = -0.5 * t3 + t2 - 0.5 * t
                f2 =  1.5 * t3 - 2.5 * t2 + 1.0
                f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
                f4 =  0.5 * t3 - 0.5 * t2
                pt = P0 * f1 + P1 * f2 + P2 * f3 + P3 * f4
                pts.append((pt[0], pt[1]))
        return pts

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.curr_idx = 0
        self.vehicle_pos = list(self.path_points[0])
        dx, dy = np.subtract(self.path_points[1], self.path_points[0])
        self.vehicle_angle = np.arctan2(dy, dx)
        # obstacles
        self.obstacles = list(self.fixed_obstacles)
        self.random_obstacles = []
        while len(self.random_obstacles) < self.num_random_obstacles:
            x = np.random.uniform(self.obstacle_size, self.window_size - self.obstacle_size)
            y = np.random.uniform(self.obstacle_size, self.window_size - self.obstacle_size)
            if all(np.hypot(x - ox, y - oy) > self.obstacle_size * 3 for ox, oy in self.obstacles):
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
                    dist = d
                    break
                if any(np.hypot(x - ox, y - oy) < self.obstacle_size for ox, oy in self.obstacles):
                    dist = d
                    break
            readings.append(dist)
        return readings

    def _get_closest_path_point(self, pos):
        best_d = float('inf')
        best_proj = pos
        for i in range(len(self.path_points) - 1):
            p1 = np.array(self.path_points[i])
            p2 = np.array(self.path_points[i+1])
            seg = p2 - p1
            L2 = np.dot(seg, seg)
            if L2 == 0:
                continue
            t = np.clip((np.array(pos) - p1).dot(seg) / L2, 0, 1)
            proj = p1 + t * seg
            d = np.linalg.norm(np.array(pos) - proj)
            if d < best_d:
                best_d, best_proj = d, proj
        return best_proj, best_d

    def _get_observation(self):
        closest_obs = min(
            self.obstacles,
            key=lambda o: np.hypot(o[0] - self.vehicle_pos[0], o[1] - self.vehicle_pos[1])
        )
        proj, dist_to_path = self._get_closest_path_point(self.vehicle_pos)
        target_idx = min(self.curr_idx + 1, len(self.path_points) - 1)
        nxt = self.path_points[target_idx]
        angle_diff = self._normalize_angle(
            np.arctan2(nxt[1] - self.vehicle_pos[1], nxt[0] - self.vehicle_pos[0]) - self.vehicle_angle
        )
        lidar = self._perform_lidar()
        return np.array([
            self.vehicle_pos[0], self.vehicle_pos[1], self.vehicle_angle,
            closest_obs[0], closest_obs[1], dist_to_path, angle_diff,
            nxt[0], nxt[1]
        ] + lidar, dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.vehicle_angle -= self.max_steering_angle
        elif action == 2:
            self.vehicle_angle += self.max_steering_angle
        self.vehicle_pos[0] += self.vehicle_speed * np.cos(self.vehicle_angle)
        self.vehicle_pos[1] += self.vehicle_speed * np.sin(self.vehicle_angle)

        # Advance waypoint if reached
        target_idx = min(self.curr_idx + 1, len(self.path_points) - 1)
        tx, ty = self.path_points[target_idx]
        if np.hypot(self.vehicle_pos[0] - tx, self.vehicle_pos[1] - ty) < self.vehicle_size:
            self.curr_idx = target_idx

        obs = self._get_observation()
        reward, done = 0.0, False

        # Deviation penalty
        _, dist_to_path = self._get_closest_path_point(self.vehicle_pos)
        if dist_to_path > self.road_width / 2:
            reward -= self.deviation_coef * (dist_to_path - self.road_width / 2)

        # Collision penalty
        for ox, oy in self.obstacles:
            if np.hypot(self.vehicle_pos[0] - ox, self.vehicle_pos[1] - oy) < self.obstacle_size:
                reward -= 30
                done = True
                break

        # Bounds penalty
        x, y = self.vehicle_pos
        if not (0 <= x <= self.window_size and 0 <= y <= self.window_size):
            reward -= 20
            done = True

        # Goal check
        if self.curr_idx >= len(self.path_points) - 1:
            reward += 50
            done = True

        return obs, reward, done, False, {}

    def render(self):
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.window_size, self.window_size))
        # Background
        self.display.fill(self.GRASS_COLOR)

        # Draw road corridor fill only (no white border lines)
        pts = [(int(x), int(y)) for x, y in self.path_points]
        road_surf = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        pygame.draw.lines(road_surf, self.ROAD_COLOR, False, pts, self.road_width)
        self.display.blit(road_surf, (0, 0))

        # Draw dashed centerline markings
        for i in range(len(pts) - 1):
            start, end = pts[i], pts[i + 1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            dist = np.hypot(dx, dy)
            ux, uy = dx / dist, dy / dist
            segs = int(dist / (self.dash_length + self.dash_gap))
            for k in range(segs):
                sx = start[0] + (self.dash_length + self.dash_gap) * k * ux
                sy = start[1] + (self.dash_length + self.dash_gap) * k * uy
                ex = sx + self.dash_length * ux
                ey = sy + self.dash_length * uy
                pygame.draw.line(
                    self.display,
                    self.MARKING_COLOR,
                    (int(sx), int(sy)),
                    (int(ex), int(ey)),
                    self.border_width
                )

        # Draw original zigzag vertices as waypoints
        for vx, vy in self._zigzag_vertices():
            pygame.draw.circle(
                self.display,
                self.BORDER_COLOR,
                (int(vx), int(vy)),
                5
            )

        # Draw obstacles
        for ox, oy in self.fixed_obstacles:
            pygame.draw.circle(
                self.display,
                self.FIXED_OBS_COLOR,
                (int(ox), int(oy)),
                self.obstacle_size
            )
        for ox, oy in self.random_obstacles:
            pygame.draw.circle(
                self.display,
                self.RANDOM_OBS_COLOR,
                (int(ox), int(oy)),
                self.obstacle_size
            )

        # Draw vehicle
        cx, cy = map(int, self.vehicle_pos)
        car_pts = [
            (cx + self.vehicle_size * np.cos(self.vehicle_angle + ang),
             cy + self.vehicle_size * np.sin(self.vehicle_angle + ang))
            for ang in (0, 2.6, -2.6)
        ]
        pygame.draw.polygon(self.display, self.VEHICLE_COLOR, car_pts)

        # Draw lidar beams
        for rel, dist in zip(
            np.linspace(-np.pi/2, np.pi/2, self.num_lidar_beams),
            self._perform_lidar()
        ):
            ang = self.vehicle_angle + rel
            ex = cx + dist * np.cos(ang)
            ey = cy + dist * np.sin(ang)
            pygame.draw.line(
                self.display,
                self.LIDAR_COLOR,
                (cx, cy),
                (int(ex), int(ey)),
                1
            )
        pygame.display.flip()

# alias
VehicleEnv = VehicleEnvZigZag

