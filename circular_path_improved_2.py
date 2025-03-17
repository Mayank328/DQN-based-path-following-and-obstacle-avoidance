import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class VehicleEnv(gym.Env):
    def __init__(self, normalize_observations=False):
        super(VehicleEnv, self).__init__()
        self.window_size = 800
        self.display = None

        self.vehicle_size = 20
        self.vehicle_speed = 5
        self.max_steering_angle = 0.1

        self.path_radius = 300
        self.path_center = (self.window_size // 2, self.window_size // 2)
        self.road_width = 60

        self.num_random_obstacles = 3
        self.num_path_obstacles = 4
        self.obstacle_size = 30

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([-1]*7),
            high=np.array([1]*7),
            dtype=np.float32
        )

        self.normalize_observations = normalize_observations
        self.vehicle_pos = None
        self.vehicle_angle = None
        self.target_angle = None

    def reset(self, seed=None):
        super().reset(seed=seed)

        angle = np.random.random() * 2 * np.pi
        offset = np.random.uniform(-20, 20)

        self.vehicle_pos = [
            self.path_center[0] + (self.path_radius + offset) * np.cos(angle),
            self.path_center[1] + (self.path_radius + offset) * np.sin(angle)
        ]
        self.vehicle_angle = angle + np.pi / 2
        self.target_angle = self.vehicle_angle

        self.obstacles = self._generate_path_obstacles()
        self._add_random_obstacles()

        return self._get_observation(), {}

    def _generate_path_obstacles(self):
        angles = [np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]
        path_obstacles = []

        for angle in angles:
            offset = np.random.choice([-20, 20])
            pos = [
                self.path_center[0] + (self.path_radius + offset) * np.cos(angle),
                self.path_center[1] + (self.path_radius + offset) * np.sin(angle)
            ]
            path_obstacles.append(pos)

        return path_obstacles

    def _add_random_obstacles(self):
        for _ in range(self.num_random_obstacles):
            while True:
                pos = [
                    np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size),
                    np.random.randint(self.obstacle_size, self.window_size - self.obstacle_size)
                ]
                dist_to_center = np.linalg.norm(np.array(pos) - np.array(self.path_center))
                too_close = any(np.linalg.norm(np.array(pos) - np.array(obs)) < self.obstacle_size * 3
                                for obs in self.obstacles)
                if not too_close and abs(dist_to_center - self.path_radius) > self.obstacle_size * 2:
                    self.obstacles.append(pos)
                    break

    def _get_observation(self):
        closest_obstacle = min(self.obstacles,
                               key=lambda obs: np.linalg.norm(np.array(obs) - np.array(self.vehicle_pos)))

        dx = self.vehicle_pos[0] - self.path_center[0]
        dy = self.vehicle_pos[1] - self.path_center[1]
        dist_to_center = np.linalg.norm([dx, dy])
        dist_to_path = dist_to_center - self.path_radius

        ideal_angle = np.arctan2(dy, dx) + np.pi / 2
        angle_diff = self._normalize_angle(ideal_angle - self.vehicle_angle)

        obs = np.array([
            self.vehicle_pos[0],
            self.vehicle_pos[1],
            self.vehicle_angle,
            closest_obstacle[0],
            closest_obstacle[1],
            dist_to_path,
            angle_diff
        ], dtype=np.float32)

        if self.normalize_observations:
            obs = obs / self.window_size

        return obs

    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def step(self, action):
        steering = 0
        if action == 0:
            steering = -self.max_steering_angle
        elif action == 2:
            steering = self.max_steering_angle

        self.vehicle_angle += steering

        self.vehicle_pos[0] += self.vehicle_speed * np.cos(self.vehicle_angle)
        self.vehicle_pos[1] += self.vehicle_speed * np.sin(self.vehicle_angle)

        dx = self.vehicle_pos[0] - self.path_center[0]
        dy = self.vehicle_pos[1] - self.path_center[1]
        dist_to_center = np.linalg.norm([dx, dy])
        path_deviation = abs(dist_to_center - self.path_radius)

        ideal_angle = np.arctan2(dy, dx) + np.pi / 2
        angle_diff = abs(self._normalize_angle(ideal_angle - self.vehicle_angle))

        path_reward = np.exp(-0.02 * path_deviation)+3
        angle_reward = np.exp(-2.0 * angle_diff)+1

        closest_obstacle_dist = min(np.linalg.norm(np.array(obs) - np.array(self.vehicle_pos))
                                    for obs in self.obstacles)
        obstacle_avoidance_reward = 1.0 if closest_obstacle_dist > self.obstacle_size * 2 + 5 else 0.0

        reward = path_reward + angle_reward + obstacle_avoidance_reward

        done = False

        if path_deviation > 50:
            reward -= 5
        if closest_obstacle_dist < (self.vehicle_size + self.obstacle_size) / 2:
            reward -= 30
            done = True

        if (self.vehicle_pos[0] < 0 or self.vehicle_pos[0] > self.window_size or
                self.vehicle_pos[1] < 0 or self.vehicle_pos[1] > self.window_size):
            reward -= 20
            done = True

        return self._get_observation(), reward, done, False, {}

    def render(self):
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.window_size, self.window_size))

        self.display.fill((34, 139, 34))  # Grass

        outer_radius = self.path_radius + self.road_width // 2
        inner_radius = self.path_radius - self.road_width // 2

        pygame.draw.circle(self.display, (80, 80, 80), self.path_center, outer_radius)
        pygame.draw.circle(self.display, (34, 139, 34), self.path_center, inner_radius)

        for obs in self.obstacles:
            pygame.draw.circle(self.display, (255, 0, 0), (int(obs[0]), int(obs[1])), self.obstacle_size)

        x, y = int(self.vehicle_pos[0]), int(self.vehicle_pos[1])
        pygame.draw.circle(self.display, (0, 0, 255), (x, y), self.vehicle_size)
        pygame.display.flip()

    def close(self):
        if self.display is not None:
            pygame.quit()
            self.display = None
