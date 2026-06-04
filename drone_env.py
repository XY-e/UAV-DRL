import airsim
import numpy as np
import random
import time
import sys
import os
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.action_clipping import clip_action
from gymnasium import spaces

class DroneEnv:
    DEFAULT_VEHICLE_NAME = "Drone1"
    BASE_MAP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "map")
    
    EXE_PATHS = {
        "T1_OpenUrbanGrid": os.path.join(BASE_MAP_PATH, "T1_OpenUrbanGrid", "WindowsNoEditor", "Blocks.exe"),
        "T2_MediumCityBlocks": os.path.join(BASE_MAP_PATH, "T2_MediumCityBlocks", "WindowsNoEditor", "Blocks.exe"),
        "T3_Wind": os.path.join(BASE_MAP_PATH, "T3_Wind", "WindowsNoEditor", "Blocks.exe")
    }

    def __init__(self, visualize_target=True, ip="127.0.0.1", port=41451, vehicle_name=None):
        self.vehicle_name = vehicle_name or self.DEFAULT_VEHICLE_NAME
        self.rpc_port = int(port)
        self.airsim_ip = ip or "127.0.0.1"
        self.client = None

        self.visualize_target = visualize_target
        self.curriculum_progress = 1.0
        self.fixed_target = np.array([8.0, 0.0, -5.0], dtype=np.float32)
        self.fixed_target_until_progress = 0.0

        self.scenario_id = "SCN-URB"
        self.scenario_name = "AirSim Urban Environment"
        self.current_map = ""
        self._set_boundary_limits()

        self.ACTIVE_PHASE = 9  
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_z = -5.0

        self.previous_distance = None
        self.current_step = 0
        self.current_episode = 0
        self.current_phase = "train"
        self.max_steps = 200 

        self.target = np.array([0.0, 0.0, -5.0])
        self.enable_wind = True
        self.wind_strength = 1.5
        self.wind = np.zeros(3, dtype=np.float32)
        self.goal_threshold = 3.0
        self.max_velocity = 2.0
        self.step_duration = 0.15
        self.visualize_sensor_ray = visualize_target

        self.distance_sensor_names = ["Distance", "DistanceLeft", "DistanceRight", "DistanceBack", "DistanceDown"]
        self.distance_sensor_max_range = 40.0
        self._warned_distance_sensor_error = False
        self._warned_distance_sensor_invalid = False
        self._warned_missing_distance_sensors = set()
        
        self._cached_obstacle_dist = self.distance_sensor_max_range
        self._cached_sensor_readings = {}
 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

        self.hover_counter = 0
        self.continuous_start_x = None
        self.continuous_start_y = None
        self._prev_raw_action = np.zeros(4, dtype=np.float32)
        self.prev_in_intersection = False

    def _set_boundary_limits(self):
        """Update safety boundaries based on the current map."""
        if "T3_Wind" in self.current_map:
            self.x_min, self.x_max = -15.0, 170.0
            self.y_min, self.y_max = -110.0, 110.0
            self.z_min, self.z_max = -14.0, -1.5
            self.safety_distance = 7.0
            print(f"🌐 Boundaries updated for T3_Wind: X: {self.x_min} to {self.x_max}m | Y: {self.y_min} to {self.y_max}m | Z: {self.z_min} to {self.z_max}m")
        elif "T2_MediumCityBlocks" in self.current_map:
            self.x_min, self.x_max = -22.0, 65.0
            self.y_min, self.y_max = -54.0, 54.0
            self.z_min, self.z_max = -12.0, -1.5
            self.safety_distance = 5.0
            print(f"🏙️ Boundaries updated for T2_City: X: {self.x_min} to {self.x_max}m | Y: {self.y_min} to {self.y_max}m | Z: {self.z_min} to {self.z_max}m")
        else:
            self.x_min, self.x_max = -15.0, 42.0
            self.y_min, self.y_max = -11.0, 12.0
            self.z_min, self.z_max = -12.0, -1.5
            self.safety_distance = 5.0
            print(f"⬜ Boundaries set for T1_Grid: X: {self.x_min} to {self.x_max}m | Y: {self.y_min} to {self.y_max}m | Z: {self.z_min} to {self.z_max}m")

    def _is_safe_start_state(self):
        """Validate startup state to reduce immediate OOB/reset episodes."""
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        collision = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        speed = float(np.linalg.norm([vel.x_val, vel.y_val, vel.z_val]))
        
        near_start_xy = abs(pos.x_val - self.start_x) < 2.0 and abs(pos.y_val - self.start_y) < 2.0
        z_in_band = self.z_min <= float(pos.z_val) <= self.z_max
        
        return (not collision) and (not self.is_out_of_bounds()) and near_start_xy and z_in_band and speed < 0.8
    
    def set_curriculum_progress(self, progress):
        """Set normalized training progress in [0, 1] for target curriculum."""
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))
        if self.curriculum_progress < 0.75:
            self.goal_threshold = 2.5
        else:
            self.goal_threshold = 2.0

    def set_episode_context(self, episode, phase="train"):
        """Attach episode metadata for richer runtime logs."""
        self.current_episode = int(max(0, episode))
        self.current_phase = str(phase)

    def _sample_target(self):
        ep = self.current_episode
        dist_min, dist_max, self.max_steps = 10.0, 15.0, 500
        tz = random.uniform(-6.0, -4.0)

        map1_branch = None
        map2_branch = None

        if self.ACTIVE_PHASE == 1:
            tz = random.uniform(-6.0, -4.0)
            if ep <= 80: dist_min, dist_max, self.max_steps = 5.0, 10.0, 350
            elif ep <= 200: dist_min, dist_max, self.max_steps = 10.0, 15.0, 550
            elif ep <= 280: dist_min, dist_max, self.max_steps = 10.0, 15.0, 600
            elif ep <= 400: dist_min, dist_max, self.max_steps = 15.0, 20.0, 800
            else: dist_min, dist_max, self.max_steps = 20.0, 25.0, 1000

        elif self.ACTIVE_PHASE == 2:
            tz = random.uniform(-10.0, -8.0) 
            if ep <= 280: dist_min, dist_max, self.max_steps = 10.0, 20.0, 800
            else: dist_min, dist_max, self.max_steps = 15.0, 25.0, 1000

        elif self.ACTIVE_PHASE == 3:
            tz = random.uniform(-10.0, -8.0) 
            if ep <= 80:
                dist_min, dist_max, self.max_steps = 10.0, 30.0, 1800
            elif ep <= 200:
                dist_min, dist_max = (30.0, 45.0) if random.random() < 0.9 else (15.0, 30.0)
                self.max_steps = 2200
            else:
                dist_min, dist_max = (30.0, 60.0) if random.random() < 0.9 else (15.0, 30.0)
                self.max_steps = 2200

        elif self.ACTIVE_PHASE == 4:
            tz = random.uniform(-6.0, -4.0)
            self.max_steps = 3000
            if ep <= 280:  dist_min, dist_max = 25.0, 45.0
            else: dist_min, dist_max = 35.0, 60.0
        
        elif self.ACTIVE_PHASE == 5:
            tz = random.uniform(-6.0, -4.0)
            dist_min, dist_max, self.max_steps = 35.0, 60.0, 3000
            if ep <= 240:
                map1_branch = random.choices(["front", "back"], weights=[0.4, 0.6])[0]
                map2_branch = random.choices(["left", "right", "front", "back"], weights=[0.1, 0.1, 0.1, 0.7])[0]
            else:
                map1_branch = random.choices(["front", "back"], weights=[0.5, 0.5])[0]
                map2_branch = random.choices(["left", "right", "front", "back"], weights=[0.25, 0.25, 0.25, 0.25])[0]

        elif self.ACTIVE_PHASE == 6:
            tz = random.uniform(-6.0, -4.0)
            dist_min, dist_max, self.max_steps = 10.0, 100.0, 3000
            map1_branch = random.choices(["front", "back"], weights=[0.5, 0.5])[0]
            map2_branch = random.choices(["left", "right", "front", "back"], weights=[0.25, 0.25, 0.25, 0.25])[0]
        
        elif self.ACTIVE_PHASE == 7:
            tz = random.uniform(-6.0, -4.0)
            dist_min, dist_max = 10.0, 150.0  
            self.max_steps = 3000 if ep <= 360 else 4000
            map1_branch = random.choices(["front", "back"], weights=[0.5, 0.5])[0]
            map2_branch = random.choices(["left", "right", "front", "back"], weights=[0.25, 0.25, 0.25, 0.25])[0]
        
        elif self.ACTIVE_PHASE == 8:
            tz = random.uniform(-6.0, -4.0)
            dist_min, dist_max = 10.0, 150.0
            self.max_steps = 3000 if ep <= 280 else 4000
            map1_branch = random.choices(["front", "back"], weights=[0.5, 0.5])[0]
            map2_branch = random.choices(["left", "right", "front", "back"], weights=[0.25, 0.25, 0.25, 0.25])[0]
            
        elif self.ACTIVE_PHASE == 9:
            tz = random.uniform(-10.0, -8.0)
            
            # PART 1: Straight Approach (Ep 1 - 220)
            if ep <= 220:
                block_idx = (ep - 1) // 20
                tx = 90.0 + (block_idx * 3.0)
                ty = 0.0
                self.max_steps = int(1000 + (tx * 25))
                return np.array([tx, ty, tz], dtype=np.float32)
                
            # PART 2: Left & Right Turn Focus (Ep 221 - 1220)
            elif ep <= 1220:
                turn_ep = ep - 220
                if turn_ep <= 400:
                    block_idx = (turn_ep - 1) // 20
                    is_right = (block_idx % 2 == 1)
                    distance_idx = block_idx // 2
                    turn_distance = 3.0 + (distance_idx * 3.0)
                elif turn_ep <= 640:
                    sub_ep = turn_ep - 400
                    block_idx = (sub_ep - 1) // 40
                    is_right = (block_idx % 2 == 1)
                    distance_idx = block_idx // 2
                    turn_distance = 33.0 + (distance_idx * 3.0)
                else:
                    sub_ep = turn_ep - 640
                    block_idx = (sub_ep - 1) // 60
                    is_right = (block_idx % 2 == 1)
                    distance_idx = block_idx // 2
                    turn_distance = 42.0 + (distance_idx * 3.0)
                    
                tx = 120.0
                ty = turn_distance if is_right else -turn_distance
                total_path = 50.0 + turn_distance
                self.max_steps = int(1000 + (total_path * 25))
                return np.array([tx, ty, tz], dtype=np.float32)
                
            # PART 3: Comprehensive Mixed Maps (Ep 1221 - 1620)
            else:
                dist_min, dist_max = 10.0, 200.0
                self.max_steps = 4000
                map1_branch = random.choices(["front", "back"], weights=[0.20, 0.80])[0]
                map2_branch = random.choices(["left", "right", "front", "back"], weights=[0.25, 0.25, 0.25, 0.25])[0]
                
                if "T3_Wind" in self.current_map:
                    choices = ['vr2_back', 'vr2_front', 'hr2_left', 'hr2_right', 'hr1.1_left', 'hr1.1_right']
                    weights = [0.15, 0.15, 0.15, 0.15, 0.20, 0.20]
                    chosen_branch = random.choices(choices, weights=weights)[0]
                    
                    boxes = {
                        "vr2_back": (17.8, 68.0, -3.0, 3.0),
                        "vr2_front": (72.0, 161.8, -3.0, 3.0),
                        "hr2_left": (65.2, 71.2, -102.5, -2.0),
                        "hr2_right": (65.2, 71.2, 2.0, 102.5),
                        "hr1.1_left": (117.0, 123.0, -48.0, -20.0),
                        "hr1.1_right": (117.0, 123.0, 20.0, 48.0)
                    }
                    
                    c = 2.0
                    xmin, xmax, ymin, ymax = boxes[chosen_branch]
                    tx = random.uniform(xmin + c, xmax - c)
                    ty = random.uniform(ymin + c, ymax - c)
                    return np.array([tx, ty, tz], dtype=np.float32)

        c = 2.0  # Safe clearance boundary

        if self.ACTIVE_PHASE >= 4 and "T1_OpenUrbanGrid" in self.current_map:
            w = [0.6, 0.4] if self.ACTIVE_PHASE == 4 else [0.5, 0.5]
            map1_branch = random.choices(["front", "back"], weights=w)[0]
            if map1_branch == "back": dist_min, dist_max = 5.0, min(dist_max, 13.0) 
            elif map1_branch == "front": dist_max = min(dist_max, 40.0)

        elif "T2_MediumCityBlocks" in self.current_map:
            if self.ACTIVE_PHASE == 3: map2_branch = random.choice(["left", "right"])
            elif self.ACTIVE_PHASE >= 4: map2_branch = random.choice(["left", "right", "front", "back"])

            if map2_branch in ["left", "right"]:
                dist_min, dist_max = max(dist_min, 45.0), max(dist_max, 55.0)
                self.max_steps = max(self.max_steps, 2500)

        # 🚀 STREAMLINED MAP 3 BOX SAMPLER (Removes performance bottleneck)
        if "T3_Wind" in self.current_map and self.ACTIVE_PHASE >= 6:
            boxes = {
                "vr2_back": (15.8, 70.0, -5.0, 5.0),
                "vr2_front": (70.0, 163.8, -5.0, 5.0),
                "hr2_left": (63.2, 73.2, -104.5, 0.0),
                "hr2_right": (63.2, 73.2, 0.0, 104.5),
                "hr1_left": (115.0, 125.0, -104.5, -20.0),
                "hr1_right": (115.0, 125.0, 20.0, 104.5),
                "vr1_1": (125.0, 163.8, -59.8, -49.8),
                "vr1_2": (73.0, 115.0, -59.8, -49.8),
                "vr1_3": (15.8, 63.0, -59.8, -49.8),
                "vr3_1": (125.0, 163.8, 49.8, 59.8),
                "vr3_2": (73.0, 115.0, 49.8, 59.8),
                "vr3_3": (15.8, 63.0, 49.8, 59.8)
            }
            
            if self.ACTIVE_PHASE == 6:
                chosen_branch = random.choices(['vr2_back', 'vr2_front', 'hr2_left', 'hr2_right'], [0.25, 0.25, 0.25, 0.25])[0]
            elif self.ACTIVE_PHASE == 7:
                if ep <= 80: chosen_branch = 'vr2_back'
                elif ep <= 160: chosen_branch = 'vr2_front'
                elif ep <= 280: chosen_branch = 'hr2_left'
                elif ep <= 360: chosen_branch = 'hr2_right'
                elif ep <= 600: chosen_branch = 'hr1_left'
                elif ep <= 840: chosen_branch = 'hr1_right'
                else: chosen_branch = random.choices(['vr2_back', 'vr2_front', 'hr2_left', 'hr2_right', 'hr1_left', 'hr1_right'], [0.15, 0.15, 0.15, 0.15, 0.20, 0.20])[0]
            elif self.ACTIVE_PHASE == 8:
                if ep <= 120: chosen_branch = 'hr2_left'
                elif ep <= 280: chosen_branch = 'hr2_right'
                elif ep <= 760: chosen_branch = 'hr1_left'
                elif ep <= 1240: chosen_branch = 'hr1_right'
                else: chosen_branch = random.choices(['vr2_back', 'vr2_front', 'hr2_left', 'hr2_right', 'hr1_left', 'hr1_right'], [0.15, 0.15, 0.15, 0.15, 0.20, 0.20])[0]
            
            xmin, xmax, ymin, ymax = boxes[chosen_branch]
            tx = random.uniform(xmin + c, xmax - c)
            ty = random.uniform(ymin + c, ymax - c)
            return np.array([tx, ty, tz], dtype=np.float32)

        def is_valid_point(tx, ty):
            if "T1_OpenUrbanGrid" in self.current_map:
                if not (-19.3 + c <= tx <= 40.7 - c and -3.0 + c <= ty <= 3.0 - c): return False
                if self.ACTIVE_PHASE >= 2:
                    if map1_branch == "front" and tx < 0.0: return False
                    if map1_branch == "back" and tx > 0.0: return False
                return True
            elif "T2_MediumCityBlocks" in self.current_map:
                in_horizontal = (5.8 + c <= tx <= 15.7 - c) and (-50.0 + c <= ty <= 50.0 - c)
                in_vertical = (-39.2 + c <= tx <= 60.6 - c) and (-4.9 + c <= ty <= 4.9 - c)
                if self.ACTIVE_PHASE == 2:
                    return in_horizontal
                elif self.ACTIVE_PHASE == 3:
                    if map2_branch == "left" and not (in_horizontal and ty <= -8.0): return False
                    if map2_branch == "right" and not (in_horizontal and ty >= 8.0): return False
                    return True
                elif self.ACTIVE_PHASE >= 4:
                    if map2_branch == "left" and not (in_horizontal and ty <= -45.0): return False
                    if map2_branch == "right" and not (in_horizontal and ty >= 45.0): return False
                    if map2_branch == "front" and not (in_vertical and tx >= 15.7 + c): return False
                    if map2_branch == "back" and not (in_vertical and tx <= 5.8 - c): return False
                    if map2_branch == "front":
                        if (16.0 - c <= tx <= 60.0 + c) and (-4.6 - c <= ty <= -2.6 + c): return False 
                        if (16.0 - c <= tx <= 60.0 + c) and (2.6 - c <= ty <= 4.6 + c): return False  
                    return True
            elif "T3_Wind" in self.current_map:
                return True
            return False

        for _ in range(20000):
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(dist_min, dist_max)
            tx = self.start_x + dist * np.cos(angle)
            ty = self.start_y + dist * np.sin(angle)
            if is_valid_point(tx, ty):
                return np.array([tx, ty, tz], dtype=np.float32)
                
        print(f"⚠️ Failed to sample target between {dist_min}m and {dist_max}m. Supplying fallback.")
        return np.array([self.start_x + 5.0, self.start_y, tz], dtype=np.float32)

    def reset(self):
        """Reset AirSim environment and start a new episode."""
        print("Resetting environment...")
        self.client.confirmConnection()
        self.client.reset()
        time.sleep(0.4)
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        self.start_x, self.start_y = 0.0, 0.0
        self.start_z = -5.0

        if self.ACTIVE_PHASE in [2, 3, 9]:
            self.start_z = -9.0
            
        if self.ACTIVE_PHASE == 2 and "T2_MediumCityBlocks" in self.current_map:
            if random.random() < 0.5:
                self.start_x, self.start_y = 10.0, -46.0  
            else:
                self.start_x, self.start_y = -10.0, -46.0 
        
        if self.ACTIVE_PHASE >= 5 and "T3_Wind" in self.current_map:
            if self.continuous_start_x is not None:
                self.start_x = self.continuous_start_x
                self.start_y = self.continuous_start_y
            else:
                self.start_x, self.start_y = 70.0, 0.0

        start_pose = airsim.Pose(airsim.Vector3r(float(self.start_x), float(self.start_y), float(self.start_z)), airsim.to_quaternion(0, 0, 0))
        self.client.simSetVehiclePose(start_pose, True, vehicle_name=self.vehicle_name)
        time.sleep(0.4)
        
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        self.client.moveToPositionAsync(float(self.start_x), float(self.start_y), float(self.start_z), 3, timeout_sec=2, vehicle_name=self.vehicle_name).join()

        self.client.rotateToYawAsync(0.0, timeout_sec=1, vehicle_name=self.vehicle_name).join()
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(0.15)

        for attempt in range(3):
            if self._is_safe_start_state():
                break
            self.client.moveToPositionAsync(self.start_x, self.start_y, self.start_z, 2, timeout_sec=1, vehicle_name=self.vehicle_name).join()
            self.client.rotateToYawAsync(0.0, timeout_sec=1, vehicle_name=self.vehicle_name).join()
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
            time.sleep(0.15)

        self.target = self._sample_target()
        self.hover_counter = 0  
        self.prev_in_intersection = False

        context = f"[{self.current_phase} ep {self.current_episode}] "
        print(f"{context}🎯 New Target spawned at: X={self.target[0]:.2f}, Y={self.target[1]:.2f}, Z={self.target[2]:.2f}")
        self._draw_target_marker()

        self.previous_distance = self.get_distance_to_target()
        self.current_step = 0

        # =========================================================
        # 🌪️ WIND CURRICULUM
        # =========================================================
        ep = self.current_episode
        self.enable_wind = True
        active_strength = 0.0

        if self.ACTIVE_PHASE == 1:
            if ep <= 200: self.enable_wind = False
            else: active_strength = random.uniform(0.3, 0.5)
        elif self.ACTIVE_PHASE == 2:
            if ep <= 280: active_strength = random.uniform(0.3, 0.5)
            else: active_strength = random.uniform(0.5, 0.8)
        elif self.ACTIVE_PHASE == 3:
            if ep <= 200: active_strength = random.uniform(0.3, 0.5)
            else: active_strength = random.uniform(0.5, 0.8)
        elif self.ACTIVE_PHASE == 4:
            rand_w = random.random()
            if rand_w < 0.20: self.enable_wind = False
            elif rand_w < 0.50: active_strength = random.uniform(0.3, 0.5)
            else: active_strength = random.uniform(0.5, 0.8)
        elif self.ACTIVE_PHASE == 5:
            rand_w = random.random()
            if ep <= 120:
                if rand_w < 0.70: self.enable_wind = False
                else: active_strength = random.uniform(0.3, 0.5)
            elif ep <= 240:
                if rand_w < 0.50: active_strength = random.uniform(0.3, 0.5)
                else: active_strength = random.uniform(0.5, 0.8)
            else:
                if rand_w < 0.20: self.enable_wind = False
                elif rand_w < 0.50: active_strength = random.uniform(0.3, 0.5)
                else: active_strength = random.uniform(0.5, 0.8)
        elif self.ACTIVE_PHASE == 6:
            rand_w = random.random()
            if rand_w < 0.20: self.enable_wind = False
            elif rand_w < 0.50: active_strength = random.uniform(0.3, 0.5)
            else: active_strength = random.uniform(0.5, 0.8)
        elif self.ACTIVE_PHASE == 7:
            if ep <= 360 or ep > 840:
                rand_w = random.random()
                if rand_w < 0.15: self.enable_wind = False
                elif rand_w < 0.45: active_strength = random.uniform(0.3, 0.5)
                elif rand_w < 0.80: active_strength = random.uniform(0.5, 0.8)
                else: active_strength = random.uniform(0.8, 1.0)
            elif 361 <= ep <= 440 or 601 <= ep <= 680:
                rand_w = random.random()
                if rand_w < 0.50: self.enable_wind = False
                else: active_strength = random.uniform(0.3, 0.5)
            elif 441 <= ep <= 520 or 681 <= ep <= 760:
                active_strength = random.uniform(0.5, 0.8)
            elif 521 <= ep <= 600 or 761 <= ep <= 840:
                active_strength = random.uniform(0.8, 1.0)
        elif self.ACTIVE_PHASE == 8:
            if ep <= 280 or ep > 1240:
                rand_w = random.random()
                if rand_w < 0.15: self.enable_wind = False
                elif rand_w < 0.45: active_strength = random.uniform(0.3, 0.5)
                elif rand_w < 0.80: active_strength = random.uniform(0.5, 0.8)
                else: active_strength = random.uniform(0.8, 1.0)
            elif 281 <= ep <= 440 or 761 <= ep <= 920:
                rand_w = random.random()
                if rand_w < 0.50: self.enable_wind = False
                else: active_strength = random.uniform(0.3, 0.5)
            elif 441 <= ep <= 600 or 921 <= ep <= 1080:
                active_strength = random.uniform(0.5, 0.8)
            elif 601 <= ep <= 760 or 1081 <= ep <= 1240:
                active_strength = random.uniform(0.8, 1.0)
        elif self.ACTIVE_PHASE == 9:
            if ep <= 220:
                progress = ep / 220.0
            elif ep <= 1220:
                progress = (ep - 220) / 1000.0
            else:
                progress = 1.0 

            if progress < 0.25:
                self.enable_wind = False if random.random() < 0.7 else True
                active_strength = random.uniform(0.1, 0.3)
            elif progress < 0.60:
                active_strength = random.uniform(0.3, 0.6)
            elif progress < 0.90:
                active_strength = random.uniform(0.6, 0.9)
            else:
                rand_w = random.random()
                if rand_w < 0.15: self.enable_wind = False
                elif rand_w < 0.45: active_strength = random.uniform(0.3, 0.5)
                elif rand_w < 0.80: active_strength = random.uniform(0.5, 0.8)
                else: active_strength = random.uniform(0.8, 1.0)

        if self.enable_wind:
            wind_angle = random.uniform(0, 2 * np.pi)
            vx = active_strength * np.cos(wind_angle)
            vy = active_strength * np.sin(wind_angle)
            z_force = active_strength * 0.12
            vz = random.uniform(-z_force, z_force)
            self.wind = np.array([vx, vy, vz], dtype=np.float32)
            print(f"🌪️ Wind: {active_strength:.2f} m/s | Angle: {np.degrees(wind_angle):.1f}°")
        else:
            self.wind = np.zeros(3, dtype=np.float32)

        self._prev_action = np.zeros(4, dtype=np.float32)
        self._prev_raw_action = np.zeros(4, dtype=np.float32)
        return self.get_state()
    
    def _draw_target_marker(self):
        """Plot the navigation target in the AirSim world as a visual circle (hitbox ring)."""
        if not self.visualize_target:
            return
        try:
            self.client.simFlushPersistentMarkers()
            tx, ty, tz = float(self.target[0]), float(self.target[1]), float(self.target[2])
            
            radius = self.goal_threshold
            points = []
            num_segments = 32  
            
            for i in range(num_segments + 1):
                angle = (i / num_segments) * 2 * np.pi
                px = tx + radius * np.cos(angle)
                py = ty + radius * np.sin(angle)
                points.append(airsim.Vector3r(float(px), float(py), float(tz)))
                
            self.client.simPlotLineStrip(
                points, color_rgba=[0.0, 1.0, 0.25, 1.0], thickness=5.0, duration=-1.0, is_persistent=True,
            )
            
            center = airsim.Vector3r(tx, ty, tz)
            self.client.simPlotPoints(
                [center], color_rgba=[0.0, 1.0, 0.25, 0.8], size=10.0, duration=-1.0, is_persistent=True,
            )
        except Exception as exc:
            print(f"(visualize_target) Could not draw marker: {exc}")

    def _get_obstacle_distance(self):
        """Read all distance sensors and return (min_distance, sensor_dict)."""
        sensor_readings = {}
        valid_readings = []
        
        for sensor_name in self.distance_sensor_names:
            try:
                distance_data = self.client.getDistanceSensorData(distance_sensor_name=sensor_name, vehicle_name=self.vehicle_name)
                dist = float(distance_data.distance)
                if not np.isfinite(dist) or dist < 0.0:
                    if not self._warned_distance_sensor_invalid:
                        print(f"⚠️ Distance sensor returned invalid reading; using fallback {self.distance_sensor_max_range:.1f}m.")
                        self._warned_distance_sensor_invalid = True
                    sensor_readings[sensor_name] = self.distance_sensor_max_range
                    continue
                
                clamped = min(dist, self.distance_sensor_max_range)
                sensor_readings[sensor_name] = clamped
                valid_readings.append(clamped)
                
            except Exception as exc:
                if sensor_name not in self._warned_missing_distance_sensors:
                    print(f"⚠️ Distance sensor read failed (name='{sensor_name}'): {exc}.")
                    self._warned_missing_distance_sensors.add(sensor_name)
                self._warned_distance_sensor_error = True
                sensor_readings[sensor_name] = self.distance_sensor_max_range

        min_distance = float(min(valid_readings)) if valid_readings else self.distance_sensor_max_range
        return min_distance, sensor_readings

    def _draw_sensor_rays(self, multirotor_state, sensor_readings):
        """Draw color-coded rays for each distance sensor (forward/left/right/down)."""
        if not self.visualize_sensor_ray or sensor_readings is None: return
        try:
            pos = multirotor_state.kinematics_estimated.position
            orientation = multirotor_state.kinematics_estimated.orientation
            roll, pitch, yaw = airsim.to_eularian_angles(orientation)
            drone_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
            
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            cos_p, sin_p = np.cos(pitch), np.sin(pitch)
            
            forward_world = np.array([cos_p * cos_y, cos_p * sin_y, sin_p], dtype=np.float32)
            back_world = -forward_world
            right_world = np.array([-sin_y, cos_y, 0.0], dtype=np.float32)
            left_world = -right_world
            down_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            
            sensor_directions = {
                "Distance": (forward_world, [0.0, 1.0, 1.0, 0.9]),     
                "DistanceBack": (back_world, [1.0, 0.0, 1.0, 0.9]),   
                "DistanceLeft": (left_world, [1.0, 1.0, 0.0, 0.9]),     
                "DistanceRight": (right_world, [1.0, 0.5, 0.0, 0.9]),   
                "DistanceDown": (down_world, [0.5, 0.0, 1.0, 0.9])      
            }
            
            for sensor_name, reading in sensor_readings.items():
                if sensor_name not in sensor_directions: continue
                dist = float(reading)
                if dist >= (self.distance_sensor_max_range - 1e-3): continue
                
                direction, base_color = sensor_directions[sensor_name]
                color = [1.0, 0.1, 0.1, 0.95] if dist < 2.0 else [1.0, 0.3, 0.0, 0.95] if dist < 4.0 else base_color
                end = drone_pos + (direction * dist)
                
                self.client.simPlotLineStrip(
                    [airsim.Vector3r(float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2])),
                     airsim.Vector3r(float(end[0]), float(end[1]), float(end[2]))],
                    color_rgba=color, thickness=2.5, duration=0.15, is_persistent=False,
                )
        except Exception as exc:
            if self.current_step % 100 == 0:
                print(f"(sensor_rays) Could not draw sensor rays: {exc}")
        
    def get_state(self):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)

        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation
        angular_vel = state.kinematics_estimated.angular_velocity

        roll, pitch, yaw = airsim.to_eularian_angles(orientation)

        dx = self.target[0] - pos.x_val
        dy = self.target[1] - pos.y_val
        
        heading_obs = np.array([
            dx / (np.linalg.norm([dx, dy]) + 1e-6),
            dy / (np.linalg.norm([dx, dy]) + 1e-6)
        ], dtype=np.float32)

        position = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        linear_velocity = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
        orientation_values = np.array([roll, pitch, yaw], dtype=np.float32)
        angular_velocity = np.array([angular_vel.x_val, angular_vel.y_val, angular_vel.z_val], dtype=np.float32)

        goal_delta = (self.target - position).astype(np.float32)

        obstacle_dist, sensor_readings = self._get_obstacle_distance()
        self._draw_sensor_rays(state, sensor_readings)
        
        self._cached_obstacle_dist = obstacle_dist
        self._cached_sensor_readings = sensor_readings
        
        sensor_values = np.array([
            sensor_readings.get("Distance", 20.0),
            sensor_readings.get("DistanceLeft", 20.0),
            sensor_readings.get("DistanceRight", 20.0),
            sensor_readings.get("DistanceBack", 20.0),
            sensor_readings.get("DistanceDown", 20.0)
        ], dtype=np.float32)

        collision = np.array([1.0 if collision_info.has_collided else 0.0], dtype=np.float32)
        wind_state = self.wind.astype(np.float32)

        return np.concatenate([
            position,           # 3
            linear_velocity,    # 3
            orientation_values, # 3
            angular_velocity,   # 3
            sensor_values,      # 5
            goal_delta,         # 3
            wind_state,         # 3
            collision,          # 1
            heading_obs         # 2
        ])                      # Total: 26 elements

    def get_position(self):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def get_distance_to_target(self):
        position = self.get_position()
        return np.linalg.norm(position - self.target)

    def is_out_of_bounds(self):
            pos = self.get_position()
            x, y, z = pos[0], pos[1], pos[2]
            if x < self.x_min or x > self.x_max: return True
            if y < self.y_min or y > self.y_max: return True
            if z < self.z_min or z > self.z_max: return True
            return False

    def has_collision(self):
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        return collision_info.has_collided

    def step(self, action):
        self.current_step += 1
        action_change = float(np.linalg.norm(action - self._prev_raw_action))
        self._prev_raw_action = action.copy()

        action = clip_action(action)
        action = action.astype(np.float32, copy=False)

        vx, vy, vz, yaw_rate = float(action[0]), float(action[1]), float(action[2]), float(action[3])
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        linear_vel = state.kinematics_estimated.linear_velocity
        angular_vel = state.kinematics_estimated.angular_velocity
        orientation = state.kinematics_estimated.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)

        current_speed = np.linalg.norm([linear_vel.x_val, linear_vel.y_val, linear_vel.z_val])
        rotation_speed = np.linalg.norm([angular_vel.x_val, angular_vel.y_val, angular_vel.z_val])

        if abs(vx) < 0.15: vx = 0.0
        if abs(vy) < 0.15: vy = 0.0
        if abs(vz) < 0.12: vz = 0.0

        if abs(vx) > abs(vy): vy = 0.0 
        elif abs(vy) > abs(vx): vx = 0.0 

        current_pos = state.kinematics_estimated.position
        if current_pos.z_val < -3.0:
            if current_speed < 1.5:
                vx *= 0.75
                vy *= 0.80
                vz *= 0.45
            if rotation_speed > 0.4:
                yaw_rate *= 0.5

        tilt_penalty_y = 1.0 - min(abs(roll), 0.6) / 0.6
        vy *= tilt_penalty_y
        tilt_penalty_x = 1.0 - min(abs(pitch), 0.6) / 0.6
        vx *= tilt_penalty_x

        if not hasattr(self, "_prev_action"):
            self._prev_action = np.zeros(4, dtype=np.float32)
        current_action = np.array([vx, vy, vz, yaw_rate], dtype=np.float32)
        alpha = 0.65
        smoothed_action = alpha * self._prev_action + (1.0 - alpha) * current_action
        self._prev_action = smoothed_action.copy()

        vx, vy, vz, yaw_rate = smoothed_action
        vx = float(np.clip(vx, -2.2, 2.2))
        vy = float(np.clip(vy, -2.2, 2.2))
        vz = float(np.clip(vz, -1.5, 1.5))
        yaw_rate = 0.0

        if self.enable_wind:
            self.client.simSetWind(airsim.Vector3r(*self.wind.tolist()))

        self.client.moveByVelocityAsync(
            vx, vy, vz, duration=self.step_duration,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
            vehicle_name=self.vehicle_name,
        ).join()

        multirotor_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = multirotor_state.kinematics_estimated.position  
        vel = multirotor_state.kinematics_estimated.linear_velocity
        angular_velocity = multirotor_state.kinematics_estimated.angular_velocity
        ang_vel_np = np.array([angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val])

        _, sensor_dict = self._get_obstacle_distance()
        next_state = self.get_state()
        extracted_heading = next_state[-2:]
        distance = self.get_distance_to_target()
        collision = self.has_collision()
        out_of_bounds = self.is_out_of_bounds()
        obstacle_dist = self._cached_obstacle_dist

        if self.current_step % 50 == 0:
            print(f"Step: {self.current_step:3} | Dist: {distance:5.2f}m | Obs: {obstacle_dist:5.2f}m")
        
        in_intersection = False
        dyn_obs_ahead = False
        
        if "T3_Wind" in self.current_map:
            int_1 = (110.0 <= pos.x_val <= 130.0) and (-65.0 <= pos.y_val <= -45.0)
            int_2 = (110.0 <= pos.x_val <= 130.0) and (45.0 <= pos.y_val <= 65.0)
            int_3 = (60.0 <= pos.x_val <= 80.0) and (-65.0 <= pos.y_val <= -45.0)
            int_4 = (60.0 <= pos.x_val <= 80.0) and (45.0 <= pos.y_val <= 65.0)
            if int_1 or int_2 or int_3 or int_4:
                in_intersection = True
                
            if obstacle_dist < 10.0:
                obs1 = (16.0 <= pos.x_val <= 60.0) and (-6.0 <= pos.y_val <= 6.0)
                obs2 = (115.0 <= pos.x_val <= 125.0) and (-20.0 <= pos.y_val <= 20.0)
                if obs1 or obs2:
                    dyn_obs_ahead = True

        elif "T2_MediumCityBlocks" in self.current_map:
            if (-5.0 <= pos.x_val <= 25.0) and (-15.0 <= pos.y_val <= 15.0):
                in_intersection = True
            if obstacle_dist < 10.0 and (10.0 <= pos.x_val <= 65.0) and (-6.0 <= pos.y_val <= 6.0):
                dyn_obs_ahead = True

        just_entered = in_intersection and not self.prev_in_intersection
        self.prev_in_intersection = in_intersection
        done = False

        if current_speed < 0.4 and distance > 3.5:
            self.hover_counter += 1
        else:
            self.hover_counter = 0

        if self.hover_counter > 100: 
            done = True
            print("Hover timeout triggered! Agent cleared to prevent step starvation.")

        if distance < self.goal_threshold:
            done = True
            print("Target reached!")
        elif collision:
            done = True
            print("Collision!")
        elif out_of_bounds:
            done = True
            print("Out of bounds!")
        elif self.current_step >= self.max_steps:
            done = True
            print("Max steps reached!")
        
        if done:
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
            if distance < self.goal_threshold:
                self.continuous_start_x = float(self.target[0])
                self.continuous_start_y = float(self.target[1])
            else:
                self.continuous_start_x = 70.0
                self.continuous_start_y = 0.0

        info = {
            "success": distance < self.goal_threshold,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "action_magnitude": float(np.linalg.norm([vx, vy, vz, yaw_rate])),
            "action_change": action_change,
            "prev_distance": self.previous_distance,
            "curr_distance": distance,
            "stability": float(np.linalg.norm(ang_vel_np)),
            "obstacle_distance": obstacle_dist,
            "heading_obs": extracted_heading.tolist(),
            "Distance": sensor_dict.get("Distance", 20.0),
            "DistanceLeft": sensor_dict.get("DistanceLeft", 20.0),
            "DistanceRight": sensor_dict.get("DistanceRight", 20.0),
            "DistanceBack": sensor_dict.get("DistanceBack", 20.0),
            "DistanceDown": sensor_dict.get("DistanceDown", 20.0),
            "velocity": [vel.x_val, vel.y_val, vel.z_val],
            "goal_delta": (self.target - np.array([pos.x_val, pos.y_val, pos.z_val])).tolist(),
            "pos_x": pos.x_val,
            "pos_y": pos.y_val,
            "pos_z": pos.z_val,
            "pitch": pitch,
            "roll": roll,
            "yaw": yaw,
            "safety_bubble": getattr(self, "safety_distance", 5.0),
            "wind_vector": self.wind.tolist() if self.enable_wind else [0.0, 0.0, 0.0],
            "active_phase": self.ACTIVE_PHASE,
            "inside_intersection": in_intersection,
            "just_entered_intersection": just_entered,
            "dynamic_obs_ahead": dyn_obs_ahead,
            "current_episode": self.current_episode,
        }

        self.previous_distance = distance
        return next_state, 0.0, done, info

    def switch_map(self, map_name, force=False):
        if not force and self.current_map == map_name:
            print(f"✔️ Already on {map_name}. No restart needed.")
            return True

        print(f"\n--- 🛑 RELAY: Closing current EXE and starting {map_name} ---")
        os.system("taskkill /F /IM Blocks.exe /T >nul 2>&1")
        time.sleep(8) 

        exe_path = self.EXE_PATHS.get(map_name)
        subprocess.Popen([exe_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
        time.sleep(5)

        for attempt in range(1, 31):
            try:
                self.client = airsim.MultirotorClient(self.airsim_ip, self.rpc_port)
                self.client.confirmConnection()
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                
                self.current_map = map_name
                self._set_boundary_limits()
                print(f"--- ✅ Ready on {map_name} (Episode continues...) ---")
                return True
            except Exception:
                time.sleep(5)
        return False
