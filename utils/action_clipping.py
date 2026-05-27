# import numpy as np

# MAX_VELOCITY = 3.0


# def clip_action(action):
#     return np.clip(action[:4], -MAX_VELOCITY, MAX_VELOCITY).astype(np.float32)

import numpy as np

# Velocity limits for position tracking (m/s)
MAX_VELOCITY = 3.0  

# Rotational agility speed limit (degrees/second)
MAX_YAW_RATE = 45.0 

def clip_action(action):
    # Construct a custom element-by-element boundary ceiling array
    low_bounds  = np.array([-MAX_VELOCITY, -MAX_VELOCITY, -MAX_VELOCITY, -MAX_YAW_RATE], dtype=np.float32)
    high_bounds = np.array([ MAX_VELOCITY,  MAX_VELOCITY,  MAX_VELOCITY,  MAX_YAW_RATE], dtype=np.float32)
    
    # Securely clamp the 4D action space array 
    return np.clip(action[:4], low_bounds, high_bounds).astype(np.float32)