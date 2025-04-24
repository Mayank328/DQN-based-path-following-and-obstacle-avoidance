# import gymnasium as gym
# import numpy as np
# import pygame
# from gymnasium import spaces

# class BaseVehicleEnv(gym.Env):
#     """
#     Base class implementing common logic: lidar, rendering, step, reset placeholder.
#     Subclasses must define self.waypoints and self.start_idx in __init__.
#     """
#     def __init__(self,
#                  window_size=800,
#                  road_width=40,
#                  num_lidar_beams=8,
#                  lidar_max_range=200,
#                  obstacle_size=20,
#                  num_random_obstacles=2):
#         super().__init__()
#         self.display = None
#         self.window_size = window_size
#         self.road_width = road_width
#         self.lidar_max_range = lidar_max_range
#         self.num_lidar_beams = num_lidar_beams
#         self.obstacle_size = obstacle_size
#         self.num_random_obstacles = num_random_obstacles
#         # dynamics
#         self.vehicle_size = 20
#         self.vehicle_speed = 10
#         self.max_steering_angle = 0.1
#         # rewards
#         self.progress_coef = 1.0
#         self.deviation_coef = 0.1
#         self.obstacle_coef = 5.0
#         # obstacles lists
#         self.fixed_obstacles = []
#         self.random_obstacles = []
#         self.obstacles = []
#         # action / obs spaces
#         self.action_space = spaces.Discrete(3)
#         low = np.array([0, 0, -np.pi, 0, -np.pi] + [0]*self.num_lidar_beams, dtype=np.float32)
#         high = np.array([window_size, window_size, np.pi, window_size, np.pi]
#                         + [lidar_max_range]*self.num_lidar_beams, dtype=np.float32)
#         self.observation_space = spaces.Box(low, high, dtype=np.float32)
#         # colors
#         self.ROAD_COLOR = (80,80,80)
#         self.GRASS_COLOR = (34,139,34)
#         self.FIXED_OBS_COLOR = (255,0,0)
#         self.RANDOM_OBS_COLOR = (255,165,0)
#         # state
#         self.vehicle_pos = None
#         self.vehicle_angle = None

#     def _interpolate(self, pts, n):
#         """Uniformly interpolate n points along the closed polyline defined by pts."""
#         res = []
#         m = len(pts)
#         count_per_seg = max(1, n // m)
#         for i in range(m):
#             A = np.array(pts[i])
#             B = np.array(pts[(i+1) % m])
#             for t in np.linspace(0, 1, count_per_seg, endpoint=False):
#                 res.append(tuple((A + t*(B - A)).tolist()))
#         return res

#     def reset(self, seed=None):
#         super().reset(seed=seed)
#         # place vehicle at start waypoint with tangent heading
#         w0 = np.array(self.waypoints[self.start_idx])
#         w1 = np.array(self.waypoints[(self.start_idx + 1) % len(self.waypoints)])
#         self.vehicle_pos = w0.tolist()
#         self.vehicle_angle = np.arctan2(w1[1] - w0[1], w1[0] - w0[0])
#         # generate random obstacles
#         self.random_obstacles = []
#         for _ in range(self.num_random_obstacles):
#             while True:
#                 x = np.random.uniform(self.obstacle_size, self.window_size - self.obstacle_size)
#                 y = np.random.uniform(self.obstacle_size, self.window_size - self.obstacle_size)
#                 if all(np.hypot(x-ox, y-oy) > self.obstacle_size*3
#                        for ox,oy in self.fixed_obstacles + self.random_obstacles):
#                     self.random_obstacles.append((x,y))
#                     break
#         self.obstacles = self.fixed_obstacles + self.random_obstacles
#         return self._get_observation(), {}

#     def _get_observation(self):
#         dev, tan = self._compute_path_error()
#         head_err = abs(self._normalize(tan - self.vehicle_angle))
#         lidar = self._perform_lidar()
#         return np.array([*self.vehicle_pos, self.vehicle_angle, dev, head_err] + lidar,
#                         dtype=np.float32)

#     def _compute_path_error(self):
#         P = np.array(self.vehicle_pos)
#         min_d, best_tan = float('inf'), 0.0
#         for i in range(len(self.waypoints)):
#             A = np.array(self.waypoints[i])
#             B = np.array(self.waypoints[(i+1) % len(self.waypoints)])
#             AB = B - A
#             t = np.clip(np.dot(P - A, AB) / np.dot(AB, AB), 0, 1)
#             proj = A + t*AB
#             d = np.linalg.norm(P - proj)
#             if d < min_d:
#                 min_d, best_tan = d, np.arctan2(AB[1], AB[0])
#         return min_d, best_tan

#     def _perform_lidar(self):
#         rays = []
#         for rel in np.linspace(-np.pi/2, np.pi/2, self.num_lidar_beams):
#             ang = self.vehicle_angle + rel
#             dist = self.lidar_max_range
#             for d in np.linspace(0, self.lidar_max_range, 100):
#                 x = self.vehicle_pos[0] + d*np.cos(ang)
#                 y = self.vehicle_pos[1] + d*np.sin(ang)
#                 if not (0 <= x <= self.window_size and 0 <= y <= self.window_size):
#                     dist = d
#                     break
#                 if any(np.hypot(x-ox, y-oy) < self.obstacle_size/2 for ox,oy in self.obstacles):
#                     dist = d
#                     break
#             rays.append(dist)
#         return rays

#     def _normalize(self, a):
#         return (a + np.pi) % (2*np.pi) - np.pi

#     def step(self, action):
#         old = np.array(self.vehicle_pos)
#         if action == 0:
#             self.vehicle_angle -= self.max_steering_angle
#         elif action == 2:
#             self.vehicle_angle += self.max_steering_angle
#         self.vehicle_pos[0] += self.vehicle_speed * np.cos(self.vehicle_angle)
#         self.vehicle_pos[1] += self.vehicle_speed * np.sin(self.vehicle_angle)
#         dev, tan = self._compute_path_error()
#         head_err = abs(self._normalize(tan - self.vehicle_angle))
#         lidar = self._perform_lidar()
#         cen = lidar[self.num_lidar_beams // 2]
#         reward = np.exp(-0.01 * dev) + np.exp(-1.0 * head_err)
#         if cen < self.lidar_max_range:
#             reward -= self.obstacle_coef * (1 - cen/self.lidar_max_range)
#         dists = [np.linalg.norm(np.array(w) - old) for w in self.waypoints]
#         i = int(np.argmin(dists))
#         j = (i + 1) % len(self.waypoints)
#         seg = np.array(self.waypoints[j]) - np.array(self.waypoints[i])
#         mv = np.array(self.vehicle_pos) - old
#         seg_len = np.linalg.norm(seg)
#         if seg_len > 0:
#             prog = np.dot(mv, seg) / seg_len
#             if prog > 0:
#                 reward += self.progress_coef * (prog/seg_len)
#         if dev > self.road_width / 2:
#             reward -= self.deviation_coef * (dev - self.road_width/2)
#         done = False
#         for ox,oy in self.obstacles:
#             if np.hypot(self.vehicle_pos[0]-ox, self.vehicle_pos[1]-oy) < (self.vehicle_size+self.obstacle_size)/2:
#                 reward -= 30
#                 done = True
#                 break
#         x,y = self.vehicle_pos
#         if not (0 <= x <= self.window_size and 0 <= y <= self.window_size):
#             reward -= 20
#             done = True
#         return self._get_observation(), reward, done, False, {}

#     def render(self):
#         if self.display is None:
#             pygame.init()
#             self.display = pygame.display.set_mode((self.window_size, self.window_size))
#         self.display.fill(self.GRASS_COLOR)
#         # draw closed track
#         pts = [tuple(map(int, w)) for w in self.waypoints]
#         pygame.draw.lines(self.display, self.ROAD_COLOR, True, pts, self.obstacle_size)
#         # draw obstacles
#         for ox,oy in self.fixed_obstacles:
#             pygame.draw.circle(self.display, self.FIXED_OBS_COLOR, (int(ox),int(oy)), self.obstacle_size)
#         for ox,oy in self.random_obstacles:
#             pygame.draw.circle(self.display, self.RANDOM_OBS_COLOR, (int(ox),int(oy)), self.obstacle_size)
#         # draw vehicle
#         cx,cy = map(int, self.vehicle_pos)
#         car_pts = [
#             (cx + self.vehicle_size * np.cos(self.vehicle_angle + ang),
#              cy + self.vehicle_size * np.sin(self.vehicle_angle + ang))
#             for ang in (-0.5, 0.5, np.pi-0.5, np.pi+0.5)
#         ]
#         pygame.draw.polygon(self.display, (0,0,255), car_pts)
#         # draw lidar beams
#         for rel, dist in zip(np.linspace(-np.pi/2, np.pi/2, self.num_lidar_beams), self._perform_lidar()):
#             ang = self.vehicle_angle + rel
#             ex, ey = cx + dist*np.cos(ang), cy + dist*np.sin(ang)
#             pygame.draw.line(self.display, (0,255,0), (cx,cy), (int(ex),int(ey)), 1)
#         pygame.display.flip()

# class SquareTrackEnv(BaseVehicleEnv):
#     """
#     Environment where waypoints form a square around the center.
#     """
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         s = self.window_size * 0.6
#         c = self.window_size / 2
#         half = s / 2
#         corners = [
#             (c-half, c-half),
#             (c+half, c-half),
#             (c+half, c+half),
#             (c-half, c+half),
#         ]
#         # interpolate to 60 points
#         self.waypoints = self._interpolate(corners, 60)
#         self.start_idx = 0
#         # place three fixed obstacles: one at corner 1, two at midpoints of segments 2 & 3
#         fixed_idxs = [15, 37, 52]
#         self.fixed_obstacles = [self.waypoints[i] for i in fixed_idxs]

# class FigureEightEnv(BaseVehicleEnv):
#     """
#     Waypoints form a figure-eight (lemniscate).
#     """
#     def __init__(self, num_points=200, **kwargs):
#         super().__init__(**kwargs)
#         c = self.window_size / 2
#         a = self.window_size * 0.3
#         t = np.linspace(0, 2*np.pi, num_points)
#         x = c + a * np.sin(t) / (1 + np.cos(t)**2)
#         y = c + a * np.sin(t)*np.cos(t) / (1 + np.cos(t)**2)
#         self.waypoints = list(zip(x, y))
#         self.start_idx = 0
#         self.fixed_obstacles = []

# # for compatibility with existing training scripts:
# VehicleEnv = SquareTrackEnv


#////ROAD like environment for asthetics//////


import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class BaseVehicleEnv(gym.Env):
    """
    Base class: implements dynamics, lidar, reward, and generic render.
    Subclasses define self.waypoints and self.start_idx (and may override render).
    """
    def __init__(self,
                 window_size=800,
                 road_width=40,
                 num_lidar_beams=8,
                 lidar_max_range=200,
                 obstacle_size=20,
                 num_random_obstacles=2):
        super().__init__()
        self.display = None
        self.window_size = window_size
        self.road_width = road_width
        self.num_lidar_beams = num_lidar_beams
        self.lidar_max_range = lidar_max_range
        self.obstacle_size = obstacle_size
        self.num_random_obstacles = num_random_obstacles

        # dynamics
        self.vehicle_size = 20
        self.vehicle_speed = 8
        self.max_steering_angle = np.pi / 6  # 30° per step

        # reward coefs
        self.progress_coef  = 1.0
        self.deviation_coef = 0.1
        self.obstacle_coef  = 5.0

        # obstacles
        self.fixed_obstacles  = []
        self.random_obstacles = []
        self.obstacles        = []

        # action/observation spaces
        self.action_space = spaces.Discrete(3)
        low  = np.array([0,0,-np.pi, 0, -np.pi] + [0]*self.num_lidar_beams, dtype=np.float32)
        high = np.array([window_size,window_size,np.pi, window_size, np.pi]
                        + [lidar_max_range]*self.num_lidar_beams, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # colors
        self.ROAD_COLOR      = (80, 80, 80)
        self.GRASS_COLOR     = (34, 139, 34)
        self.BORDER_COLOR    = (255,255,255)
        self.FIXED_OBS_COLOR = (255, 0, 0)
        self.RANDOM_OBS_COLOR= (255,165, 0)

        # state
        self.vehicle_pos   = None
        self.vehicle_angle = None

    def _interpolate(self, pts, n):
        """Uniformly interpolate n points along the closed polygon pts."""
        res = []
        m = len(pts)
        per = max(1, n//m)
        for i in range(m):
            A = np.array(pts[i])
            B = np.array(pts[(i+1)%m])
            for t in np.linspace(0,1,per,endpoint=False):
                res.append(tuple((A + t*(B-A)).tolist()))
        return res

    def reset(self, seed=None):
        super().reset(seed=seed)
        # set start
        w0 = np.array(self.waypoints[self.start_idx])
        w1 = np.array(self.waypoints[(self.start_idx+1)%len(self.waypoints)])
        self.vehicle_pos   = w0.tolist()
        self.vehicle_angle = np.arctan2(w1[1]-w0[1], w1[0]-w0[0])
        # random obstacles
        self.random_obstacles = []
        while len(self.random_obstacles) < self.num_random_obstacles:
            x = np.random.uniform(self.obstacle_size, self.window_size-self.obstacle_size)
            y = np.random.uniform(self.obstacle_size, self.window_size-self.obstacle_size)
            if all(np.hypot(x-ox,y-oy) > self.obstacle_size*3
                   for ox,oy in self.fixed_obstacles + self.random_obstacles):
                self.random_obstacles.append((x,y))
        self.obstacles = self.fixed_obstacles + self.random_obstacles
        return self._get_observation(), {}

    def _get_observation(self):
        dev, tan = self._compute_path_error()
        head_err = abs(self._normalize(tan - self.vehicle_angle))
        lidar    = self._perform_lidar()
        return np.array([*self.vehicle_pos, self.vehicle_angle, dev, head_err] + lidar,
                        dtype=np.float32)

    def _compute_path_error(self):
        P = np.array(self.vehicle_pos)
        min_d, best_tan = float('inf'), 0.0
        for i in range(len(self.waypoints)):
            A = np.array(self.waypoints[i])
            B = np.array(self.waypoints[(i+1)%len(self.waypoints)])
            AB = B - A
            t  = np.clip(np.dot(P-A,AB)/np.dot(AB,AB), 0,1)
            proj = A + t*AB
            d = np.linalg.norm(P-proj)
            if d < min_d:
                min_d, best_tan = d, np.arctan2(AB[1],AB[0])
        return min_d, best_tan

    def _perform_lidar(self):
        rays=[]
        for rel in np.linspace(-np.pi/2, np.pi/2, self.num_lidar_beams):
            ang = self.vehicle_angle + rel
            dist = self.lidar_max_range
            for d in np.linspace(0, self.lidar_max_range, 100):
                x = self.vehicle_pos[0] + d*np.cos(ang)
                y = self.vehicle_pos[1] + d*np.sin(ang)
                if not(0<=x<=self.window_size and 0<=y<=self.window_size):
                    dist = d; break
                if any(np.hypot(x-ox,y-oy)<self.obstacle_size/2 for ox,oy in self.obstacles):
                    dist = d; break
            rays.append(dist)
        return rays

    def _normalize(self,a):
        return (a+np.pi)%(2*np.pi) - np.pi

    def step(self, action):
        old = np.array(self.vehicle_pos)
        if action==0: self.vehicle_angle -= self.max_steering_angle
        elif action==2: self.vehicle_angle += self.max_steering_angle
        self.vehicle_pos[0] += self.vehicle_speed*np.cos(self.vehicle_angle)
        self.vehicle_pos[1] += self.vehicle_speed*np.sin(self.vehicle_angle)

        dev, tan = self._compute_path_error()
        head_err= abs(self._normalize(tan-self.vehicle_angle))
        lidar   = self._perform_lidar()
        cen     = lidar[self.num_lidar_beams//2]

        # base
        reward = np.exp(-0.01*dev) + np.exp(-1.0*head_err)
        # obstacle penalty
        if cen<self.lidar_max_range:
            reward -= self.obstacle_coef*(1-cen/self.lidar_max_range)
        # progress
        dists = [np.linalg.norm(np.array(w)-old) for w in self.waypoints]
        i = int(np.argmin(dists)); j=(i+1)%len(self.waypoints)
        seg = np.array(self.waypoints[j]) - np.array(self.waypoints[i])
        mv  = np.array(self.vehicle_pos) - old
        seg_len = np.linalg.norm(seg)
        if seg_len>0:
            prog = np.dot(mv,seg)/seg_len
            if prog>0: reward += self.progress_coef*(prog/seg_len)
        # hard dev penalty
        if dev>self.road_width/2:
            reward -= self.deviation_coef*(dev-self.road_width/2)

        done=False
        for ox,oy in self.obstacles:
            if np.hypot(self.vehicle_pos[0]-ox,self.vehicle_pos[1]-oy) < (self.vehicle_size+self.obstacle_size)/2:
                reward-=30; done=True; break
        x,y = self.vehicle_pos
        if not(0<=x<=self.window_size and 0<=y<=self.window_size):
            reward-=20; done=True

        return self._get_observation(), reward, done, False, {}

    def render(self):
        # fallback generic render: centerline
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.window_size,self.window_size))
        self.display.fill(self.GRASS_COLOR)
        pts = [tuple(map(int,w)) for w in self.waypoints]
        pygame.draw.lines(self.display, self.ROAD_COLOR, True, pts, self.obstacle_size)
        for ox,oy in self.fixed_obstacles:
            pygame.draw.circle(self.display,self.FIXED_OBS_COLOR,(int(ox),int(oy)),self.obstacle_size)
        for ox,oy in self.random_obstacles:
            pygame.draw.circle(self.display,self.RANDOM_OBS_COLOR,(int(ox),int(oy)),self.obstacle_size)
        cx,cy=map(int,self.vehicle_pos)
        car = [(cx+self.vehicle_size*np.cos(self.vehicle_angle+ang),
                cy+self.vehicle_size*np.sin(self.vehicle_angle+ang))
               for ang in (-0.5,0.5,np.pi-0.5,np.pi+0.5)]
        pygame.draw.polygon(self.display,(0,0,255),car)
        for rel,dist in zip(np.linspace(-np.pi/2,np.pi/2,self.num_lidar_beams),self._perform_lidar()):
            ang=self.vehicle_angle+rel
            ex,ey = cx+dist*np.cos(ang), cy+dist*np.sin(ang)
            pygame.draw.line(self.display,(0,255,0),(cx,cy),(int(ex),int(ey)),1)
        pygame.display.flip()

class SquareTrackEnv(BaseVehicleEnv):
    """
    Sharp-cornered square road, drawn as filled band for aesthetics.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # override speed/steering
        self.vehicle_speed      = 8
        self.max_steering_angle = np.pi/6

        # square corners
        s = self.window_size * 0.6
        c = self.window_size / 2
        half = s/2
        self.corners = [
            (c-half, c-half),
            (c+half, c-half),
            (c+half, c+half),
            (c-half, c+half),
        ]
        # centerline waypoints
        self.waypoints = self._interpolate(self.corners, 60)
        self.start_idx = 0
        # three fixed obstacles on sides
        fixed_idxs = [15, 37, 52]
        self.fixed_obstacles = [self.waypoints[i] for i in fixed_idxs]

    def render(self):
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.window_size,self.window_size))

        # draw grass
        self.display.fill(self.GRASS_COLOR)

        # compute filled road band
        half = self.road_width/2
        outer = [
            (self.corners[0][0]-half, self.corners[0][1]-half),
            (self.corners[1][0]+half, self.corners[1][1]-half),
            (self.corners[2][0]+half, self.corners[2][1]+half),
            (self.corners[3][0]-half, self.corners[3][1]+half),
        ]
        inner = [
            (self.corners[0][0]+half, self.corners[0][1]+half),
            (self.corners[1][0]-half, self.corners[1][1]+half),
            (self.corners[2][0]-half, self.corners[2][1]-half),
            (self.corners[3][0]+half, self.corners[3][1]-half),
        ]

        # draw road band
        pygame.draw.polygon(self.display, self.ROAD_COLOR, outer)
        pygame.draw.polygon(self.display, self.GRASS_COLOR, inner)

        # draw borders
        pygame.draw.lines(self.display, self.BORDER_COLOR, True, outer, 2)
        pygame.draw.lines(self.display, self.BORDER_COLOR, True, inner, 2)

        # draw obstacles
        for ox,oy in self.fixed_obstacles:
            pygame.draw.circle(self.display,self.FIXED_OBS_COLOR,(int(ox),int(oy)),self.obstacle_size)
        for ox,oy in self.random_obstacles:
            pygame.draw.circle(self.display,self.RANDOM_OBS_COLOR,(int(ox),int(oy)),self.obstacle_size)

        # draw vehicle
        cx,cy = map(int,self.vehicle_pos)
        car = [(cx+self.vehicle_size*np.cos(self.vehicle_angle+ang),
                cy+self.vehicle_size*np.sin(self.vehicle_angle+ang))
               for ang in (-0.5,0.5,np.pi-0.5,np.pi+0.5)]
        pygame.draw.polygon(self.display,(0,0,255),car)

        # draw lidar beams
        for rel,dist in zip(np.linspace(-np.pi/2,np.pi/2,self.num_lidar_beams),self._perform_lidar()):
            ang = self.vehicle_angle + rel
            ex,ey = cx + dist*np.cos(ang), cy + dist*np.sin(ang)
            pygame.draw.line(self.display,(0,255,0),(cx,cy),(int(ex),int(ey)),1)

        pygame.display.flip()

# alias for consistency
VehicleEnv = SquareTrackEnv
