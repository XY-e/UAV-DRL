import numpy as np

# Configuration Constants
PROGRESS_CLIP_M = 2.0      

# Distinct Max Step Rewards for different phases
MAX_STEP_REWARD_PHASE_1_8 = 5.0      
MAX_STEP_REWARD_PHASE_9_12 = 15.0     

def compute_reward(
    prev_distance,
    curr_distance,
    reached_goal,
    collision,
    out_of_bounds,
    obstacle_distance=None,
    goal_delta=None,
    velocity=None,
    info=None,
    phase=9  
):
    reward = 0.0
    info = info if info is not None else {}

    # ==========================================================================
    # COMMON TERMINAL EVENTS (Applied to all phases)
    # ==========================================================================
    if reached_goal:
        vel_norm = np.linalg.norm(velocity) if velocity is not None else 0.0
        if vel_norm < 1.5:
            return 300.0  
        return 200.0      
        
    if collision:
        return -500.0     
        
    if out_of_bounds:
        return -500.0

    # ==========================================================================
    #                     PHASE 1 - 8 REWARD SYSTEM
    # ==========================================================================
    if phase <= 8:
        front = float(info.get("Distance", 20.0))     
        left  = float(info.get("DistanceLeft", 20.0))
        right = float(info.get("DistanceRight", 20.0))
        back  = float(info.get("DistanceBack", 20.0))  
        down  = float(info.get("DistanceDown", 20.0))

        if front < 1.0:
            reward -= 12.0
        elif front < 1.5:
            reward -= 4.0 * (2.5 - front)

        if down < 1.2:
            reward -= 8.0 * (1.2 - down)

        WALL_WARN_8 = 1.5
        WALL_DANGER_8 = 1.0
        WALL_CRITICAL_8 = 0.7
        
        if left < WALL_CRITICAL_8:
            reward -= 10.0 * (WALL_CRITICAL_8 - left)
        elif left < WALL_DANGER_8:
            reward -= 4.0 * (WALL_DANGER_8 - left)
        elif left < WALL_WARN_8:
            reward -= 1.5 * (WALL_WARN_8 - left)

        if right < WALL_CRITICAL_8:
            reward -= 10.0 * (WALL_CRITICAL_8 - right)
        elif right < WALL_DANGER_8:
            reward -= 4.0 * (WALL_DANGER_8 - right)
        elif right < WALL_WARN_8:
            reward -= 1.5 * (WALL_WARN_8 - right)

        def exp_penalty(d, scale=6.0, k=0.8):
            if 2.5 < d < scale:
                return -20.0 * np.exp(-k * d)
            return 0.0

        reward += exp_penalty(front)
        reward += exp_penalty(back)
        reward += exp_penalty(left)
        reward += exp_penalty(right)

        if down < 1.5:
            reward -= 5.0 / (down + 0.05)

        if prev_distance is not None and curr_distance is not None:
            delta = float(prev_distance - curr_distance)
            delta = np.clip(delta, -PROGRESS_CLIP_M, PROGRESS_CLIP_M)
            progress_norm = np.clip((delta + PROGRESS_CLIP_M) / (2 * PROGRESS_CLIP_M), 0.0, 1.0)
            scale = 8.0 - 6.0 * progress_norm
            reward += scale * delta
            
        alignment = 0.0
        vel_norm = 0.0
        goal_norm = 0.0
        if velocity is not None and goal_delta is not None:
            vel_norm = np.linalg.norm(velocity)
            goal_norm = np.linalg.norm(goal_delta)
            if vel_norm > 1e-6 and goal_norm > 1e-6:
                v = velocity / vel_norm
                g = goal_delta / goal_norm
                alignment = np.dot(v, g)
                reward += 1.2 * alignment

        corridor_width = left + right
        if corridor_width < 7.0:
            side_balance = abs(left - right)
            normalized_offset = side_balance / (corridor_width + 1e-6)
            center_reward = 4.0 * (1.0 - normalized_offset) ** 2
            reward += center_reward
            if normalized_offset > 0.35:
                reward -= 1.5 * (normalized_offset - 0.35)
            if corridor_width < 6.0:
                reward -= 0.5 * (6.0 - corridor_width)
        
        pitch = float(info.get("pitch", 0.0))
        roll = float(info.get("roll", 0.0))
        abs_pitch = abs(pitch)
        abs_roll = abs(roll)

        if abs_pitch <= 5.0: pass 
        elif abs_pitch <= 8.0: reward -= 0.1 * (abs_pitch - 5.0)
        else: reward -= 1.5 * (abs_pitch - 8.0)

        if abs_roll < 5.0: reward += 0.2 
        elif abs_roll <= 8.0: reward -= 0.1 * (abs_roll - 5.0)
        else: reward -= 1.5 * (abs_roll - 8.0)

        action_mag = float(info.get("action_magnitude", 0.0))
        action_change = float(info.get("action_change", 0.0)) 
        reward -= 0.05 * action_mag
        reward -= 0.1 * action_change

        if velocity is not None and goal_delta is not None and vel_norm > 1e-6 and goal_norm > 1e-6:
            lateral_speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
            lateral_penalty = -0.2 * lateral_speed * (1.0 - max(alignment, 0.0))
            reward += lateral_penalty

            vertical_speed = abs(velocity[2])
            g = goal_delta / goal_norm
            vertical_penalty = -0.1 * vertical_speed * (1.0 - abs(g[2]))
            reward += vertical_penalty

        wind_vector = info.get("wind_vector", [0.0, 0.0, 0.0])
        wind_strength = float(np.linalg.norm(wind_vector))
        if wind_strength > 0.5 and velocity is not None:
            drift = np.dot(velocity, wind_vector)
            if drift > 0: reward -= 1.5 * drift  

        reward -= 0.03  
        reward = min(reward, MAX_STEP_REWARD_PHASE_1_8)
        return float(reward)

    # ==========================================================================
    #                     PHASE 9 - 12 REWARD SYSTEM
    # ==========================================================================
    # ==========================================================================
    #                     PHASE 9 - 12 REWARD SYSTEM
    # ==========================================================================
    else:
        vel_norm = np.linalg.norm(velocity) if velocity is not None else 0.0
        forward_velocity = float(velocity[0]) if velocity is not None else 0.0
        vertical_speed = float(velocity[2]) if velocity is not None else 0.0
        pos_z = float(info.get("pos_z", -5.0)) 
        
        front = float(info.get("Distance", 20.0))    
        left  = float(info.get("DistanceLeft", 20.0))
        right = float(info.get("DistanceRight", 20.0))
        back  = float(info.get("DistanceBack", 20.0))  
        down  = float(info.get("DistanceDown", 20.0))

        yaw = float(info.get("yaw", 0.0))  
        inside_intersection = info.get("inside_intersection", False)
        dynamic_obs_ahead = info.get("dynamic_obs_ahead", False) 

        # 1. CEILING SAFETY (Enforces street-level containment)
        if pos_z < -7.2: 
            reward -= 10.0 * abs(-7.2 - pos_z)

        # 2. VELOCITY-TIED BRAKING (Anti-Kamikaze control)
        if front < 4.0 and forward_velocity > 1.4:
            reward -= 6.0 * forward_velocity * (4.0 - front)

        # 3. UP AND OVER (Evasion)
        if dynamic_obs_ahead and front < 7.0:
            climb_rate = -vertical_speed 
            if climb_rate > 0.15:
                reward += min((climb_rate * 5.0) + (forward_velocity * 1.5), 10.0)
            elif climb_rate < -0.1:
                reward -= 5.0 
            else:
                reward -= 3.0 
        else:
            if front < 1.5: reward -= 15.0  
            elif front < 2.5: reward -= 5.0 * (2.5 - front)

        # 4. FLOOR SAFETY
        if down < 1.2: reward -= 8.0 * (1.2 - down)

        # 5. SIDE WALLS (2.5m Cushion limits)
        WALL_WARN_9 = 2.5
        WALL_DANGER_9 = 2.0
        WALL_CRITICAL_9 = 1.0

        if left < WALL_CRITICAL_9: reward -= 15.0 * (WALL_CRITICAL_9 - left)
        elif left < WALL_DANGER_9: reward -= 8.0 * (WALL_DANGER_9 - left)
        elif left < WALL_WARN_9: reward -= 2.0 * (WALL_WARN_9 - left)

        if right < WALL_CRITICAL_9: reward -= 15.0 * (WALL_CRITICAL_9 - right)
        elif right < WALL_DANGER_9: reward -= 8.0 * (WALL_DANGER_9 - right)
        elif right < WALL_WARN_9: reward -= 2.0 * (WALL_WARN_9 - right)

        # 6. INTERSECTION TRANSIT
        if inside_intersection:
            if vel_norm > 1.6: reward -= vel_norm * 2.5
            elif 0.3 <= vel_norm <= 1.2: reward += 2.0  
            if info.get("just_entered_intersection", False): reward += 15.0

        # 7. GOAL PROGRESS
        if prev_distance is not None and curr_distance is not None:
            delta = float(prev_distance - curr_distance)
            delta = np.clip(delta, -PROGRESS_CLIP_M, PROGRESS_CLIP_M)
            progress_norm = np.clip((delta + PROGRESS_CLIP_M) / (2 * PROGRESS_CLIP_M), 0.0, 1.0)
            scale = 8.0 - 6.0 * progress_norm
            reward += scale * delta

        # 8. ANTI-HOVERING
        if vel_norm < 0.55 and curr_distance > 3.0:
            if inside_intersection: reward -= 0.4  
            else: reward -= 5.0  

        # 9. HOLONOMIC MOVEMENT ALIGNMENT & PHASE 12 SPEED INCENTIVE
        alignment = 0.0
        if velocity is not None and goal_delta is not None:
            goal_norm = np.linalg.norm(goal_delta)
            if vel_norm > 1e-6 and goal_norm > 1e-6:
                v = velocity / vel_norm
                g = goal_delta / goal_norm
                alignment = np.dot(v, g)
                
                # Boost velocity weighting explicitly during Phase 12 if tracking correctly
                if phase == 12 and alignment > 0.85 and front > 6.0:
                    # Reward higher nominal velocities when pointing straight down clear roads
                    reward += 3.5 * alignment * vel_norm
                else:
                    speed_multiplier = min(vel_norm, 2.0)
                    reward += 1.5 * alignment * speed_multiplier

        # 10. CORRIDOR CENTERING 
        corridor_width = left + right
        if corridor_width < 7.0:
            side_balance = abs(left - right)
            normalized_offset = side_balance / (corridor_width + 1e-6)
            center_reward = 4.0 * (1.0 - normalized_offset) ** 2
            reward += center_reward
            if normalized_offset > 0.35: reward -= 1.5 * (normalized_offset - 0.35)
            if corridor_width < 6.0: reward -= 0.5 * (6.0 - corridor_width)
        
        # 11. ORIENTATION STABILITY
        pitch = float(info.get("pitch", 0.0))
        roll = float(info.get("roll", 0.0))
        abs_pitch, abs_roll, abs_yaw = abs(pitch), abs(roll), abs(yaw)

        if abs_yaw > 2.0: reward -= abs_yaw * 0.5 
        if abs_pitch <= 5.0: pass 
        elif abs_pitch <= 8.0: reward -= 0.1 * (abs_pitch - 5.0)
        else: reward -= 1.5 * (abs_pitch - 8.0)

        if abs_roll < 5.0: reward += 0.2 
        elif abs_roll <= 8.0: reward -= 0.1 * (abs_roll - 5.0)
        else: reward -= 1.5 * (abs_roll - 8.0)

        # 12. SMOOTH CONTROL & ANTI-OSCILLATION
        action_mag = float(info.get("action_magnitude", 0.0))
        action_change = float(info.get("action_change", 0.0)) 
        reward -= 0.05 * action_mag
        reward -= 0.15 * action_change

        if velocity is not None and goal_delta is not None and vel_norm > 1e-6 and goal_norm > 1e-6:
            lateral_speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
            lateral_penalty = -0.2 * lateral_speed * (1.0 - max(alignment, 0.0))
            reward += lateral_penalty

            if not dynamic_obs_ahead:
                g = goal_delta / goal_norm
                vertical_penalty = -0.1 * abs(vertical_speed) * (1.0 - abs(g[2]))
                reward += vertical_penalty

        # 13. WIND / DRIFT ROBUSTNESS 
        wind_vector = info.get("wind_vector", [0.0, 0.0, 0.0])
        wind_strength = float(np.linalg.norm(wind_vector))
        if wind_strength > 0.5 and velocity is not None:
            drift = np.dot(velocity, wind_vector)
            if drift > 0: reward -= 1.5 * drift  

        # 14. TIME PRESSURE (Proportional efficiency drive)
        # 🚀 PHASE 12 UPGRADE: Elevate baseline pressure, but give proportional velocity cushion
        if phase == 12:
            reward -= 0.14  # Stronger base time cost pressure
            if vel_norm > 1.5 and alignment > 0.90:
                reward += 0.04  # Rebate penalty to reward high-speed direct cruise lines
        else:
            reward -= 0.08  # Default Phase 9-11 pressure

        reward = min(reward, MAX_STEP_REWARD_PHASE_9_12) 
        return float(reward)
    

