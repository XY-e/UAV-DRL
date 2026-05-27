import airsim
import time
import os
import sys
import numpy as np

# Logic to find DroneEnv
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from skynav_integration.adapters.drone_env import DroneEnv

# ==========================================
# CONFIGURATION
MAP_NAME = "T3_Wind" 
MANUAL_EXE_PATHS = {
    "T1_OpenUrbanGrid": r"C:\Users\sieya\Downloads\UAV_DRL_Project_clean - Copy\UAV_DRL_Project\map\T1_OpenUrbanGrid\WindowsNoEditor\Blocks.exe",
    "T2_MediumCityBlocks": r"C:\Users\sieya\Downloads\UAV_DRL_Project_clean - Copy\UAV_DRL_Project\map\T2_MediumCityBlocks\WindowsNoEditor\Blocks.exe",
    "T3_Wind": r"C:\Users\sieya\Downloads\UAV_DRL_Project_clean - Copy\UAV_DRL_Project\map\T3_Wind\WindowsNoEditor\Blocks.exe"
}
# ==========================================

def run_verification():
    map_points = {
        "T1_OpenUrbanGrid": [
            [10.0, 2.0], [-5.5, 3.0], [-10.0, -4.0], [8.5, -5.5], 
            [8.0, 6.0], [19.0, -5.5],[39.0, 0.0], [36.0, -6.5], 
            [20.0, 5.6], [27.5, -5.1]
        ],
        "T2_MediumCityBlocks": [
            [25.0, 7.5], [57.0, 0.3], [47.5, 15.0], [42.0, -18.0], 
            [-10.5, 5.5], [-18.0, -7.5], [12.0, -12.0], [3.5, -26.0], 
            [15.3, -49.5], [10.0, 17.0], [21.5, 30.5], [5.0, 48.5]
        ],
        "T3_Wind": [
            [63.0, -30.5], [120.0, 7.3], [75.0, -55.0], [38.0, -45.0], 
            [80.5, 52.5], [0.0, -47.0], [69.0, 44.3], [0.7, 48.7], 
        ]
        # "T3_Wind": [
        #     [-12.0, 110.0], [69.0, 85.0], [168.0, 105.0], [145.0, 52.0], 
        #     [38.0, -50.0], [8.0, -105.0], [165.0, 0.0], [148.0, -52.0], 
        #     [128.5, -102.5], [63.0, 19.5]
        # ]
    }

    env = DroneEnv(visualize_target=True) 
    env.EXE_PATHS = MANUAL_EXE_PATHS 

    print(f"🔄 Launching: {MAP_NAME}...")
    env.switch_map(MAP_NAME)
    
    time.sleep(10) # Give more time for the EXE to stabilize
    client = airsim.MultirotorClient()
    client.confirmConnection()

    points = np.array(map_points.get(MAP_NAME, []))
    
    # 1. Draw Points FIRST
    print("📍 Drawing target markers...")
    client.simFlushPersistentMarkers()
    for pt in points:
        pos = airsim.Vector3r(pt[0], pt[1], -5.0) 
        # Reduced size for a cleaner look
        client.simPlotPoints(
            [pos], 
            color_rgba=[1.0, 0.0, 0.0, 1.0], 
            size=20.0,  # Adjusted from 35.0 to 20.0
            is_persistent=True
        )
        
        # Making the "pin" line thinner as well
        client.simPlotLineStrip(
            [airsim.Vector3r(pt[0], pt[1], -3.0), airsim.Vector3r(pt[0], pt[1], -7.0)],
            color_rgba=[1.0, 0.0, 0.0, 0.8], # Slightly transparent red
            thickness=3.0, # Adjusted from 6.0 to 3.0
            is_persistent=True
        )

    # 2. Dynamic View Selection
    print(f"🛰️ Setting Dynamic Camera View for {MAP_NAME}...")
    center_x, center_y = np.mean(points[:, 0]), np.mean(points[:, 1])
    
    # Define custom heights for each map
    if MAP_NAME == "T1_OpenUrbanGrid":
        # Map 1 is small (approx 40m). 60m is perfect for a close top-down view.
        view_height = -60.0 
    elif MAP_NAME == "T2_MediumCityBlocks":
        # Map 2 is medium. 120m height covers the blocks.
        view_height = -120.0
    elif MAP_NAME == "T3_Wind":
        # Map 3 is huge (180m+). Needs 250m to see all targets at once.
        view_height = -250.0
    else:
        view_height = -100.0 # Fallback default

    # Move the 'External Camera' 
    # Use Pose(Position, Orientation) where Pitch -1.57 is straight down
    camera_pose = airsim.Pose(
        airsim.Vector3r(center_x, center_y, view_height), 
        airsim.to_quaternion(-1.57, 0, 0)
    )
    
    # Apply to camera "0" (the main view)
    client.simSetCameraPose("0", camera_pose) 

    print(f"✅ View set! Camera is at {abs(view_height)}m altitude.")
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run_verification()