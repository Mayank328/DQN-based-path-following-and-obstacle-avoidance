#FULL SQUARE TRACK 

# import gymnasium as gym
# import numpy as np
# import pygame
# from gymnasium import spaces

# class VehicleEnvSquareTrack(gym.Env):
#     """
#     A square track with smoothly curved corners (superellipse), lidar-based obstacle sensing, 
#     3 fixed obstacles and 2 random obstacles, heavy deviation penalty.
#     """
#     def __init__(self,
#                  window_size=800,
#                  half_side=300,
#                  road_width=40,
#                  num_waypoints=72,
#                  num_lidar_beams=8,
#                  lidar_max_range=200,
#                  obstacle_size=20,
#                  num_random_obstacles=2):
#         super().__init__()
#         # rendering handle
#         self.display = None

#         # dimensions
#         self.window_size = window_size
#         self.half_side = half_side
#         self.road_width = road_width
#         self.path_center = (window_size//2, window_size//2)

#         # generate track waypoints (superellipse for smooth corners)
#         self.waypoints = self._generate_square_path(num_waypoints)
#         # start offset so not on fixed obstacle
#         self.start_idx = num_waypoints // 4

#         # lidar
#         self.num_lidar_beams = num_lidar_beams
#         self.lidar_max_range = lidar_max_range

#         # vehicle
#         self.vehicle_size = 20
#         self.vehicle_speed = 10
#         self.max_steering_angle = 0.1

#         # reward weights
#         self.progress_coef  = 1.0
#         self.deviation_coef = 0.1
#         self.obstacle_coef  = 5.0

#         # obstacles: pick 3 fixed evenly spaced
#         idxs = [0, num_waypoints//3, 2*num_waypoints//3]
#         self.fixed_obstacles = [self.waypoints[i] for i in idxs]
#         self.num_random_obstacles = num_random_obstacles
#         self.random_obstacles = []
#         self.obstacles = []
#         self.obstacle_size = obstacle_size

#         # spaces
#         self.action_space = spaces.Discrete(3)
#         low = np.array([0,0,-np.pi, 0, -np.pi] + [0]*num_lidar_beams, dtype=np.float32)
#         high = np.array([window_size,window_size,np.pi, window_size, np.pi]
#                         + [lidar_max_range]*num_lidar_beams, dtype=np.float32)
#         self.observation_space = spaces.Box(low, high, dtype=np.float32)

#         # colors & markings
#         self.GRASS_COLOR   = (34,139,34)
#         self.ROAD_COLOR    = (80,80,80)
#         self.MARKING_COLOR = (255,240,0)
#         self.border_width  = 2
#         self.dash_length   = 20
#         self.dash_gap      = 20

#         self.FIXED_OBS_COLOR  = (255,0,0)
#         self.RANDOM_OBS_COLOR = (255,165,0)
#         self.LIDAR_COLOR      = (0,255,0)

#         # state
#         self.vehicle_pos   = None
#         self.vehicle_angle = None

#     def _generate_square_path(self, num):
#         cx, cy = self.path_center
#         a = b = self.half_side
#         n = 4.0  # superellipse exponent
#         pts = []
#         for theta in np.linspace(0, 2*np.pi, num, endpoint=False):
#             c = np.cos(theta)
#             s = np.sin(theta)
#             x = cx + a * np.sign(c) * (abs(c)**(2/n))
#             y = cy + b * np.sign(s) * (abs(s)**(2/n))
#             pts.append((x, y))
#         return pts

#     def reset(self, seed=None):
#         super().reset(seed=seed)
#         # initialize at start waypoint
#         w0 = np.array(self.waypoints[self.start_idx])
#         w1 = np.array(self.waypoints[(self.start_idx+1)%len(self.waypoints)])
#         self.vehicle_pos   = w0.tolist()
#         self.vehicle_angle = np.arctan2(w1[1]-w0[1], w1[0]-w0[0])

#         # generate random obstacles
#         self.random_obstacles = []
#         while len(self.random_obstacles) < self.num_random_obstacles:
#             x = np.random.uniform(self.obstacle_size, self.window_size-self.obstacle_size)
#             y = np.random.uniform(self.obstacle_size, self.window_size-self.obstacle_size)
#             if all(np.hypot(x-ox,y-oy) > self.obstacle_size*3
#                    for ox,oy in self.fixed_obstacles + self.random_obstacles):
#                 self.random_obstacles.append((x, y))
#         self.obstacles = self.fixed_obstacles + self.random_obstacles
#         return self._get_observation(), {}

#     def _get_observation(self):
#         dev, tangent = self._compute_path_error()
#         heading_err = abs(self._normalize_angle(tangent - self.vehicle_angle))
#         lidar_readings = self._perform_lidar_scan()
#         return np.array([
#             self.vehicle_pos[0], self.vehicle_pos[1], self.vehicle_angle,
#             dev, heading_err
#         ] + lidar_readings, dtype=np.float32)

#     def _compute_path_error(self):
#         P = np.array(self.vehicle_pos)
#         best_d, best_tan = float('inf'), 0.0
#         for i in range(len(self.waypoints)):
#             A = np.array(self.waypoints[i]); B = np.array(self.waypoints[(i+1)%len(self.waypoints)])
#             AB = B - A
#             t = np.clip(np.dot(P-A,AB)/np.dot(AB,AB),0,1)
#             proj = A + t*AB
#             d = np.linalg.norm(P-proj)
#             if d < best_d:
#                 best_d, best_tan = d, np.arctan2(AB[1],AB[0])
#         return best_d, best_tan

#     def _perform_lidar_scan(self):
#         readings = []
#         for rel in np.linspace(-np.pi/2,np.pi/2,self.num_lidar_beams):
#             ang = self.vehicle_angle + rel
#             dist = self.lidar_max_range
#             for d in np.linspace(0, self.lidar_max_range, 100):
#                 x = self.vehicle_pos[0] + d*np.cos(ang)
#                 y = self.vehicle_pos[1] + d*np.sin(ang)
#                 if not (0<=x<=self.window_size and 0<=y<=self.window_size):
#                     dist = d; break
#                 if any(np.hypot(x-ox,y-oy) < self.obstacle_size/2 for ox,oy in self.obstacles):
#                     dist = d; break
#             readings.append(dist)
#         return readings

#     def _find_prev_waypoint(self):
#         dists = [np.linalg.norm(np.array(w)-np.array(self.vehicle_pos)) for w in self.waypoints]
#         idx = int(np.argmin(dists))
#         return idx, (idx+1) % len(self.waypoints)

#     def step(self, action):
#         old = np.array(self.vehicle_pos)
#         if action == 0:
#             self.vehicle_angle -= self.max_steering_angle
#         elif action == 2:
#             self.vehicle_angle += self.max_steering_angle
#         self.vehicle_pos[0] += self.vehicle_speed * np.cos(self.vehicle_angle)
#         self.vehicle_pos[1] += self.vehicle_speed * np.sin(self.vehicle_angle)

#         dev, tangent = self._compute_path_error()
#         head_err = abs(self._normalize_angle(tangent - self.vehicle_angle))
#         lidar = self._perform_lidar_scan()
#         center = lidar[self.num_lidar_beams//2]

#         reward = np.exp(-0.01*dev) + np.exp(-1.0*head_err)
#         if center < self.lidar_max_range:
#             reward -= self.obstacle_coef * (1 - center/self.lidar_max_range)

#         prev_i, next_i = self._find_prev_waypoint()
#         seg = np.array(self.waypoints[next_i]) - np.array(self.waypoints[prev_i])
#         mv = np.array(self.vehicle_pos) - old
#         prog = np.dot(mv,seg)/np.linalg.norm(seg) if np.linalg.norm(seg)>0 else 0
#         if prog>0:
#             reward += self.progress_coef * (prog/np.linalg.norm(seg))

#         if dev > self.road_width/2:
#             reward -= self.deviation_coef * (dev - self.road_width/2)

#         done = False
#         for ox,oy in self.obstacles:
#             if np.hypot(self.vehicle_pos[0]-ox,self.vehicle_pos[1]-oy)<(self.vehicle_size+self.obstacle_size)/2:
#                 reward -= 30; done=True; break
#         x,y = self.vehicle_pos
#         if not (0<=x<=self.window_size and 0<=y<=self.window_size):
#             reward -= 20; done=True
#         return self._get_observation(), reward, done, False, {}

#     def render(self):
#         if self.display is None:
#             pygame.init()
#             self.display = pygame.display.set_mode((self.window_size, self.window_size))
#         self.display.fill(self.GRASS_COLOR)
#         # build road polygon from offset normals
#         left_pts, right_pts = [], []
#         for i,(px,py) in enumerate(self.waypoints):
#             if i < len(self.waypoints)-1:
#                 dx = self.waypoints[i+1][0] - px
#                 dy = self.waypoints[i+1][1] - py
#             else:
#                 dx = px - self.waypoints[i-1][0]
#                 dy = py - self.waypoints[i-1][1]
#             norm = np.hypot(dx,dy)
#             tx,ty = (dx/norm, dy/norm) if norm>0 else (1,0)
#             nx,ny = -ty, tx
#             left_pts.append((px + nx*self.road_width/2, py + ny*self.road_width/2))
#             right_pts.append((px - nx*self.road_width/2, py - ny*self.road_width/2))
#         road_poly = left_pts + right_pts[::-1]
#         pygame.draw.polygon(self.display, self.ROAD_COLOR, road_poly)
#         # center dashes
#         for i in range(len(self.waypoints)-1):
#             start = self.waypoints[i]; end = self.waypoints[i+1]
#             dx,dy = end[0]-start[0], end[1]-start[1]
#             dist = np.hypot(dx,dy)
#             ux,uy = dx/dist, dy/dist
#             segs = int(dist/(self.dash_length + self.dash_gap))
#             for k in range(segs):
#                 sx = start[0] + (self.dash_length + self.dash_gap)*k*ux
#                 sy = start[1] + (self.dash_length + self.dash_gap)*k*uy
#                 ex = sx + self.dash_length*ux; ey = sy + self.dash_length*uy
#                 pygame.draw.line(self.display, self.MARKING_COLOR, (int(sx),int(sy)), (int(ex),int(ey)), self.border_width)
#         # obstacles
#         for ox,oy in self.fixed_obstacles:
#             pygame.draw.circle(self.display, self.FIXED_OBS_COLOR, (int(ox),int(oy)), self.obstacle_size)
#         for ox,oy in self.random_obstacles:
#             pygame.draw.circle(self.display, self.RANDOM_OBS_COLOR, (int(ox),int(oy)), self.obstacle_size)
#         # vehicle
#         cx,cy = map(int, self.vehicle_pos)
#         car_pts = [(cx + self.vehicle_size*np.cos(self.vehicle_angle+ang), cy + self.vehicle_size*np.sin(self.vehicle_angle+ang)) for ang in (-0.5,0.5,np.pi-0.5,np.pi+0.5)]
#         pygame.draw.polygon(self.display, (0,0,255), car_pts)
#         # lidar
#         for rel,dist in zip(np.linspace(-np.pi/2,np.pi/2,self.num_lidar_beams), self._perform_lidar_scan()):
#             ang = self.vehicle_angle + rel
#             ex = cx + dist*np.cos(ang); ey = cy + dist*np.sin(ang)
#             pygame.draw.line(self.display, self.LIDAR_COLOR, (cx,cy), (int(ex),int(ey)), 1)
#         pygame.display.flip()

#     def _normalize_angle(self, a):
#         return (a + np.pi) % (2*np.pi) - np.pi

#     def close(self):
#         if self.display:
#             pygame.quit()
#             self.display = None

# # alias
# VehicleEnv = VehicleEnvSquareTrack


import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class VehicleEnvHalfSquareTrack(gym.Env):
    """
    A half-square track with smooth curved corners (superellipse) from start to end endpoint,
    lidar-based obstacle sensing, 2 fixed obstacles and 2 random obstacles, step limit per episode.
    """
    def __init__(self,
                 window_size=800,
                 half_side=300,
                 road_width=40,
                 num_waypoints=72,
                 num_lidar_beams=8,
                 lidar_max_range=200,
                 obstacle_size=20,
                 num_random_obstacles=2,
                 max_steps=500):
        super().__init__()
        # display
        self.display = None
        # dimensions
        self.window_size = window_size
        self.half_side = half_side
        self.road_width = road_width
        # shift center upward so half-track lies mid-window
        self.path_center = (window_size//2, window_size//2 - half_side//2)
        # full square superellipse path
        full_pts = self._generate_superellipse(num_waypoints)
        # half-track: first half of full_pts
        half_len = num_waypoints // 2 + 1
        self.waypoints = full_pts[:half_len]
        self.start_idx = 0
        # sensors
        self.num_lidar_beams = num_lidar_beams
        self.lidar_max_range = lidar_max_range
        # vehicle
        self.vehicle_size = 20
        self.vehicle_speed = 6.0         # reduced speed for better control
        self.max_steering_angle = 0.1  # increased turning angle
        # rewards
        self.progress_coef = 1.0
        self.deviation_coef = 0.1
        self.obstacle_coef = 5.0
                # obstacle size for both fixed and random
        self.obstacle_size = obstacle_size
        # obstacles: 2 fixed offset from path and 2 random
        idxs = [half_len // 3, 2 * half_len // 3]
        self.fixed_obstacles = []
        for i in idxs:
            p = np.array(self.waypoints[i])
            j = i+1 if i+1 < len(self.waypoints) else i-1
            q = np.array(self.waypoints[j])
            seg = q - p
            norm = np.linalg.norm(seg)
            t_vec = seg / norm if norm>0 else np.array([1.0,0.0])
            ox, oy = -t_vec[1], t_vec[0]
            offset = self.road_width/2 + self.obstacle_size/2 + 5
            pos = (p[0] + ox*offset, p[1] + oy*offset)
            self.fixed_obstacles.append(pos)
        self.num_random_obstacles = num_random_obstacles
        self.random_obstacles = []
        self.obstacles = []
        self.obstacle_size = obstacle_size  # reduced obstacle size
        # step limit
        self.max_steps = max_steps
        self.step_count = 0
        # action & observation spaces
        self.action_space = spaces.Discrete(3)
        low = np.array([0,0,-np.pi, 0, -np.pi] + [0]*num_lidar_beams, dtype=np.float32)
        high = np.array([window_size,window_size,np.pi, window_size, np.pi]
                        + [lidar_max_range]*num_lidar_beams, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # colors & markings
        self.GRASS_COLOR   = (34,139,34)
        self.ROAD_COLOR    = (80,80,80)
        self.MARKING_COLOR = (255,240,0)
        self.border_width  = 2
        self.dash_length   = 20
        self.dash_gap      = 20
        self.FIXED_OBS_COLOR  = (255,0,0)
        self.RANDOM_OBS_COLOR = (255,165,0)
        self.LIDAR_COLOR      = (0,255,0)
        # state
        self.vehicle_pos   = None
        self.vehicle_angle = None

    def _generate_superellipse(self, num):
        cx, cy = self.path_center
        a = b = self.half_side
        n = 4.0  # exponent for smooth corners
        pts = []
        for theta in np.linspace(0, 2*np.pi, num, endpoint=False):
            c, s = np.cos(theta), np.sin(theta)
            x = cx + a * np.sign(c) * (abs(c)**(2/n))
            y = cy + b * np.sign(s) * (abs(s)**(2/n))
            pts.append((x, y))
        return pts

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.step_count = 0
        # initialize at start waypoint
        w0 = np.array(self.waypoints[self.start_idx])
        w1 = np.array(self.waypoints[self.start_idx+1])
        self.vehicle_pos = w0.tolist()
        self.vehicle_angle = np.arctan2(w1[1]-w0[1], w1[0]-w0[0])
        # generate random obstacles avoiding vehicle
        self.random_obstacles = []
        vx, vy = self.vehicle_pos
        while len(self.random_obstacles) < self.num_random_obstacles:
            x = np.random.uniform(self.obstacle_size, self.window_size-self.obstacle_size)
            y = np.random.uniform(self.obstacle_size, self.window_size-self.obstacle_size)
            if (all(np.hypot(x-ox, y-oy) > self.obstacle_size*3 for ox,oy in self.fixed_obstacles + self.random_obstacles)
                and np.hypot(x-vx, y-vy) > self.obstacle_size*3):
                self.random_obstacles.append((x, y))
        self.obstacles = self.fixed_obstacles + self.random_obstacles
        return self._get_observation(), {}

    def _get_observation(self):
        dev, tangent = self._compute_path_error()
        head_err = abs(self._normalize_angle(tangent - self.vehicle_angle))
        lidar = self._perform_lidar_scan()
        return np.array([
            self.vehicle_pos[0], self.vehicle_pos[1], self.vehicle_angle,
            dev, head_err
        ] + lidar, dtype=np.float32)

    def _compute_path_error(self):
        P = np.array(self.vehicle_pos)
        best_d, best_tan = float('inf'), 0.0
        for i in range(len(self.waypoints)-1):
            A = np.array(self.waypoints[i]); B = np.array(self.waypoints[i+1])
            AB = B - A; L2 = np.dot(AB,AB)
            if L2==0: continue
            t = np.clip(np.dot(P-A,AB)/L2, 0,1)
            proj = A + t*AB
            d = np.linalg.norm(P-proj)
            if d<best_d:
                best_d, best_tan = d, np.arctan2(AB[1],AB[0])
        return best_d, best_tan

    def _perform_lidar_scan(self):
        readings=[]
        for rel in np.linspace(-np.pi/2, np.pi/2, self.num_lidar_beams):
            ang = self.vehicle_angle+rel; dist=self.lidar_max_range
            for d in np.linspace(0, self.lidar_max_range, 100):
                x = self.vehicle_pos[0]+d*np.cos(ang)
                y = self.vehicle_pos[1]+d*np.sin(ang)
                if not (0<=x<=self.window_size and 0<=y<=self.window_size): dist=d; break
                if any(np.hypot(x-ox,y-oy)<self.obstacle_size/2 for ox,oy in self.obstacles): dist=d; break
            readings.append(dist)
        return readings

    def step(self, action):
        self.step_count += 1
        old = np.array(self.vehicle_pos)
        if action==0: self.vehicle_angle-=self.max_steering_angle
        elif action==2: self.vehicle_angle+=self.max_steering_angle
        self.vehicle_pos[0]+=self.vehicle_speed*np.cos(self.vehicle_angle)
        self.vehicle_pos[1]+=self.vehicle_speed*np.sin(self.vehicle_angle)
        obs = self._get_observation()
        dev, tang = obs[3], obs[4]; lidar=obs[5:]; cen=lidar[self.num_lidar_beams//2]
        reward = np.exp(-0.01*dev)+np.exp(-1.0*tang)
        if cen<self.lidar_max_range: reward-=self.obstacle_coef*(1-cen/self.lidar_max_range)
        if dev>self.road_width/2: reward-=self.deviation_coef*(dev-self.road_width/2)
        # endpoint
        if np.linalg.norm(np.array(self.vehicle_pos)-np.array(self.waypoints[-1]))<self.vehicle_size:
            reward+=50; done=True
        elif self.step_count>=self.max_steps:
            done=True
        else: done=False
        for ox,oy in self.obstacles:
            if np.hypot(self.vehicle_pos[0]-ox,self.vehicle_pos[1]-oy)<(self.vehicle_size+self.obstacle_size)/2:
                reward-=30; done=True; break
        return obs, reward, done, False, {}

    def render(self):
        if self.display is None: pygame.init(); self.display=pygame.display.set_mode((self.window_size,self.window_size))
        self.display.fill(self.GRASS_COLOR)
        left,right=[],[]
        for i,(px,py) in enumerate(self.waypoints):
            if i<len(self.waypoints)-1: dx=self.waypoints[i+1][0]-px; dy=self.waypoints[i+1][1]-py
            else: dx=px-self.waypoints[i-1][0]; dy=py-self.waypoints[i-1][1]
            norm=np.hypot(dx,dy); tx,ty=dx/norm,dy/norm; ox,oy=-ty,tx
            left.append((px+ox*self.road_width/2,py+oy*self.road_width/2))
            right.append((px-ox*self.road_width/2,py-oy*self.road_width/2))
        poly=left+right[::-1]; pygame.draw.polygon(self.display,self.ROAD_COLOR,poly)
        for ox,oy in self.fixed_obstacles: pygame.draw.circle(self.display,self.FIXED_OBS_COLOR,(int(ox),int(oy)),self.obstacle_size)
        for ox,oy in self.random_obstacles: pygame.draw.circle(self.display,self.RANDOM_OBS_COLOR,(int(ox),int(oy)),self.obstacle_size)
        cx,cy=map(int,self.vehicle_pos)
        car=[(cx+self.vehicle_size*np.cos(self.vehicle_angle+ang),cy+self.vehicle_size*np.sin(self.vehicle_angle+ang)) for ang in(-0.5,0.5,np.pi-0.5,np.pi+0.5)]
        pygame.draw.polygon(self.display,(0,0,255),car)
        for rel,dist in zip(np.linspace(-np.pi/2,np.pi/2,self.num_lidar_beams),self._perform_lidar_scan()): ang=self.vehicle_angle+rel; ex=cx+dist*np.cos(ang); ey=cy+dist*np.sin(ang); pygame.draw.line(self.display,self.LIDAR_COLOR,(cx,cy),(int(ex),int(ey)),1)
        pygame.display.flip()

    def _normalize_angle(self,a): return (a+np.pi)%(2*np.pi)-np.pi
    def close(self):
        if self.display: pygame.quit(); self.display=None

# alias
VehicleEnv = VehicleEnvHalfSquareTrack
