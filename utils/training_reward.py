import numpy as np

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
PROGRESS_CLIP_M = 2.0
MAX_STEP_REWARD_PHASE_1_8   = 5.0
MAX_STEP_REWARD_PHASE_9_12  = 12.0   # Cap to reduce variance in later phases

# Anti-hover penalty schedule: ramps up over episodes to give time to learn locomotion
_HOVER_PENALTY_MIN = 0.5    
_HOVER_PENALTY_MAX = 3.0    
_HOVER_RAMP_EP     = 300    

def _hover_penalty(info: dict) -> float:
    """Calculates the current episode-scaled anti-hover penalty magnitude."""
    ep = int(info.get("current_episode", _HOVER_RAMP_EP))
    t  = min(1.0, ep / _HOVER_RAMP_EP)
    return _HOVER_PENALTY_MIN + (_HOVER_PENALTY_MAX - _HOVER_PENALTY_MIN) * t


def compute_reward(
    prev_distance, curr_distance, reached_goal, collision, out_of_bounds,
    obstacle_distance=None, goal_delta=None, velocity=None, info=None, phase=9
):
    reward = 0.0
    info   = info if info is not None else {}

    # 1. Terminal Events (Highest Priority)
    if reached_goal:
        vel_norm = np.linalg.norm(velocity) if velocity is not None else 0.0
        return 300.0 if vel_norm < 1.5 else 200.0
    if collision:
        return -500.0
    if out_of_bounds:
        return -500.0

    # ==========================================================================
    # PHASE 1–8 REWARD SYSTEM
    # ==========================================================================
    if phase <= 8:
        front = float(info.get("Distance", 20.0))
        left  = float(info.get("DistanceLeft", 20.0))
        right = float(info.get("DistanceRight", 20.0))
        back  = float(info.get("DistanceBack", 20.0))
        down  = float(info.get("DistanceDown", 20.0))

        # Front obstacle collision warning
        if front < 1.0:
            reward -= 12.0
        elif front < 1.5:
            reward -= 4.0 * (2.5 - front)

        # Floor safety margin
        if down < 1.2:
            reward -= 8.0 * (1.2 - down)

        # Side wall proximity limits
        WALL_WARN_8, WALL_DANGER_8, WALL_CRITICAL_8 = 1.5, 1.0, 0.7
        for side in (left, right):
            if   side < WALL_CRITICAL_8: reward -= 10.0 * (WALL_CRITICAL_8 - side)
            elif side < WALL_DANGER_8:   reward -=  4.0 * (WALL_DANGER_8   - side)
            elif side < WALL_WARN_8:     reward -=  1.5 * (WALL_WARN_8     - side)

        # Exponential proximity penalty for all sides
        def exp_penalty(d, scale=6.0, k=0.8):
            return -20.0 * np.exp(-k * d) if 2.5 < d < scale else 0.0
        for d in (front, back, left, right):
            reward += exp_penalty(d)

        # Additional low-altitude penalty
        if down < 1.5:
            reward -= 5.0 / (down + 0.05)

        # Goal progression tracking
        if prev_distance is not None and curr_distance is not None:
            delta         = np.clip(float(prev_distance - curr_distance), -PROGRESS_CLIP_M, PROGRESS_CLIP_M)
            progress_norm = np.clip((delta + PROGRESS_CLIP_M) / (2 * PROGRESS_CLIP_M), 0.0, 1.0)
            reward       += (8.0 - 6.0 * progress_norm) * delta

        # Vector alignment to target
        alignment = vel_norm = goal_norm = 0.0
        if velocity is not None and goal_delta is not None:
            vel_norm  = np.linalg.norm(velocity)
            goal_norm = np.linalg.norm(goal_delta)
            if vel_norm > 1e-6 and goal_norm > 1e-6:
                alignment = np.dot(velocity / vel_norm, goal_delta / goal_norm)
                reward   += 1.2 * alignment

        # Corridor centering mechanics
        corridor_width = left + right
        if corridor_width < 7.0:
            normalized_offset = abs(left - right) / (corridor_width + 1e-6)
            reward += 4.0 * (1.0 - normalized_offset) ** 2
            if normalized_offset > 0.35: reward -= 1.5 * (normalized_offset - 0.35)
            if corridor_width < 6.0: reward -= 0.5 * (6.0 - corridor_width)

        # Orientation and tilt limits
        pitch, roll = abs(float(info.get("pitch", 0.0))), abs(float(info.get("roll", 0.0)))
        if pitch > 8.0:    reward -= 1.5 * (pitch - 8.0)
        elif pitch > 5.0:  reward -= 0.1 * (pitch - 5.0)
        
        if roll < 5.0:     reward += 0.2
        elif roll <= 8.0:  reward -= 0.1 * (roll  - 5.0)
        else:              reward -= 1.5 * (roll  - 8.0)

        # Control smoothness and jitter penalization
        reward -= 0.05 * float(info.get("action_magnitude", 0.0))
        reward -= 0.10 * float(info.get("action_change", 0.0))

        # Cross-track (lateral) and vertical deviation penalties
        if velocity is not None and goal_delta is not None and vel_norm > 1e-6 and goal_norm > 1e-6:
            reward -= 0.2 * np.sqrt(velocity[0]**2 + velocity[1]**2) * (1.0 - max(alignment, 0.0))
            reward -= 0.1 * abs(velocity[2]) * (1.0 - abs((goal_delta / goal_norm)[2]))

        # Wind drift penalization
        wind_vector = info.get("wind_vector", [0.0, 0.0, 0.0])
        if float(np.linalg.norm(wind_vector)) > 0.5 and velocity is not None:
            drift = np.dot(velocity, wind_vector)
            if drift > 0: reward -= 1.5 * drift

        # Constant step time pressure
        reward -= 0.03
        return float(min(reward, MAX_STEP_REWARD_PHASE_1_8))

    # ==========================================================================
    # PHASE 9–12 REWARD SYSTEM
    # ==========================================================================
    current_episode = int(info.get("current_episode", 999))

    # Phase 9 Bridge: Smoothly blend from Phase 8 rules to Phase 9 rules over 200 episodes
    BRIDGE_EP_END, BRIDGE_START_OLD = 200, 0.70
    if phase == 9 and current_episode <= BRIDGE_EP_END:
        weight_old = BRIDGE_START_OLD * (1.0 - (current_episode / BRIDGE_EP_END))
        
        r_old = compute_reward(prev_distance, curr_distance, False, False, False, obstacle_distance, goal_delta, velocity, info, phase=8)
        
        info_no_bridge = dict(info)
        info_no_bridge["current_episode"] = BRIDGE_EP_END + 1
        r_new = compute_reward(prev_distance, curr_distance, False, False, False, obstacle_distance, goal_delta, velocity, info_no_bridge, phase=9)

        return float(min((weight_old * r_old) + ((1.0 - weight_old) * r_new), MAX_STEP_REWARD_PHASE_9_12))

    # State Extraction
    vel_norm         = np.linalg.norm(velocity) if velocity is not None else 0.0
    forward_velocity = float(velocity[0]) if velocity is not None else 0.0
    vertical_speed   = float(velocity[2]) if velocity is not None else 0.0
    pos_z            = float(info.get("pos_z", -5.0))
    
    front, left, right, back, down = (float(info.get(k, 20.0)) for k in ("Distance", "DistanceLeft", "DistanceRight", "DistanceBack", "DistanceDown"))
    yaw                 = float(info.get("yaw", 0.0))
    inside_intersection = info.get("inside_intersection", False)
    dynamic_obs_ahead   = info.get("dynamic_obs_ahead", False)

    # 1. Ceiling safety (Dynamically shifted for Phase 9 high altitude)
    ceiling_limit = -12 if phase == 9 else -7.2
    if pos_z < ceiling_limit:
        reward -= 8.0 * abs(ceiling_limit - pos_z)

    # 2. Velocity-tied braking (Penalize speeding into walls)
    if front < 4.0 and forward_velocity > 1.4:
        reward -= 5.0 * forward_velocity * (4.0 - front)

    # 3. Dynamic obstacle evasion (Climb over obstacles)
    if dynamic_obs_ahead and front < 7.0:
        climb_rate = -vertical_speed
        if climb_rate > 0.15:    reward += min((climb_rate * 4.0) + (forward_velocity * 1.0), 8.0)
        elif climb_rate < -0.1:  reward -= 4.0
        else:                    reward -= 2.0
    else:
        if   front < 1.5: reward -= 12.0
        elif front < 2.5: reward -=  5.0 * (2.5 - front)

    # 4. Floor safety
    if down < 1.2:
        reward -= 8.0 * (1.2 - down)

    # 5. Side wall proximity (Stricter tolerances for narrow maps)
    WALL_WARN_9, WALL_DANGER_9, WALL_CRITICAL_9 = 2.5, 2.0, 1.0
    for side in (left, right):
        if   side < WALL_CRITICAL_9: reward -= 15.0 * (WALL_CRITICAL_9 - side)
        elif side < WALL_DANGER_9:   reward -=  8.0 * (WALL_DANGER_9   - side)
        elif side < WALL_WARN_9:     reward -=  2.0 * (WALL_WARN_9     - side)

    # 6. Intersection transit (Reward slow, safe navigation through cross-streets)
    if inside_intersection:
        if vel_norm > 1.6:            reward -= vel_norm * 2.0
        elif 0.3 <= vel_norm <= 1.2:  reward += 3.0
        if info.get("just_entered_intersection", False): reward += 4.0

    # 7. Goal progression tracking
    if prev_distance is not None and curr_distance is not None:
        delta         = np.clip(float(prev_distance - curr_distance), -PROGRESS_CLIP_M, PROGRESS_CLIP_M)
        progress_norm = np.clip((delta + PROGRESS_CLIP_M) / (2 * PROGRESS_CLIP_M), 0.0, 1.0)
        reward       += (8.0 - 6.0 * progress_norm) * delta

    # 8. Anti-hover penalty (Forces exploration, disabled slightly during intersection turns)
    if vel_norm < 0.55 and curr_distance > 3.0:
        reward -= 0.4 if inside_intersection else _hover_penalty(info)

    # 9. Holonomic alignment (Faces the target)
    alignment = goal_norm = 0.0
    if velocity is not None and goal_delta is not None:
        goal_norm = np.linalg.norm(goal_delta)
        if vel_norm > 1e-6 and goal_norm > 1e-6:
            alignment = np.dot(velocity / vel_norm, goal_delta / goal_norm)
            
            # Speed incentive gated on clear road logic for Phase 12
            if phase == 12 and alignment > 0.85 and front > 8.0 and curr_distance > 15.0:
                reward += 3.0 * alignment * min(vel_norm, 2.5)
            else:
                reward += 1.5 * alignment * min(vel_norm, 2.0)

    # 10. Corridor centering (Maintains middle of the street)
    corridor_width = left + right
    if corridor_width < 7.0:
        normalized_offset = abs(left - right) / (corridor_width + 1e-6)
        reward += 4.0 * (1.0 - normalized_offset) ** 2
        if normalized_offset > 0.35: reward -= 1.5 * (normalized_offset - 0.35)
        if corridor_width < 6.0:     reward -= 0.5 * (6.0 - corridor_width)

    # 11. Orientation stability (Strict Holonomic Lock)
    pitch, roll, abs_yaw = abs(float(info.get("pitch", 0.0))), abs(float(info.get("roll", 0.0))), abs(float(info.get("yaw", 0.0)))
    if abs_yaw > 0.1:      reward -= abs_yaw * 5.0
    
    if pitch > 8.0:        reward -= 1.5 * (pitch - 8.0)
    elif pitch > 5.0:      reward -= 0.1 * (pitch - 5.0)
    if roll < 5.0:         reward += 0.2
    elif roll <= 8.0:      reward -= 0.1 * (roll - 5.0)
    else:                  reward -= 1.5 * (roll - 8.0)

    # 12. Control smoothness
    reward -= 0.05 * float(info.get("action_magnitude", 0.0))
    reward -= 0.15 * float(info.get("action_change", 0.0))

    # 13. Lateral / vertical movement penalties
    if velocity is not None and goal_delta is not None and vel_norm > 1e-6 and goal_norm > 1e-6:
        reward -= 0.2 * np.sqrt(velocity[0]**2 + velocity[1]**2) * (1.0 - max(alignment, 0.0))
        if not dynamic_obs_ahead:
            reward -= 0.1 * abs(vertical_speed) * (1.0 - abs((goal_delta / goal_norm)[2]))

    # 14. Wind / Drift robustness
    wind_vector = info.get("wind_vector", [0.0, 0.0, 0.0])
    if float(np.linalg.norm(wind_vector)) > 0.5 and velocity is not None:
        drift = np.dot(velocity, wind_vector)
        if drift > 0: reward -= 1.5 * drift

    # 15. Time pressure
    if phase == 12:
        reward -= 0.12
        if vel_norm > 1.5 and alignment > 0.90 and front > 8.0 and curr_distance > 15.0:
            reward += 0.04 # Rebate for moving fast on clear roads
    else:
        reward -= 0.07

    # 16. Holonomic Diagonal Penalty (Anti-Corner Cutting)
    if phase >= 9 and velocity is not None:
        diagonal_factor = abs(float(velocity[0])) * abs(float(velocity[1]))
        if diagonal_factor > 0.15:
            reward -= diagonal_factor * 3.0

    return float(min(reward, MAX_STEP_REWARD_PHASE_9_12))

```
