import argparse
import os
import random
import sys
import time
import numpy as np
import torch
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from algorithms.ppo_algorithm import PPOAgent, normalize_state
from skynav_integration.adapters.drone_env import DroneEnv
from training_logger import TrainingLogger
from utils.action_clipping import clip_action
from utils.training_reward import compute_reward

STATE_STD = {
    "T1_OpenUrbanGrid": np.array([
        30.0, 6.0, 8.0, 
        3.0, 3.0, 2.0, 
        3.14, 3.14, 3.14, 
        2.0, 2.0, 2.0, 
        40.0, 40.0, 40.0, 40.0, 40.0, 
        15.0, 5.0, 8.0, 
        1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0 
    ], dtype=np.float32),

    "T2_MediumCityBlocks": np.array([
        60.0, 55.0, 8.0, 
        3.0, 3.0, 2.0, 
        3.14, 3.14, 3.14, 
        2.0, 2.0, 2.0, 
        40.0, 40.0, 40.0, 40.0, 40.0, 
        30.0, 50.0, 8.0, 
        1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0 
    ], dtype=np.float32),

    "T3_Wind": np.array([
        130.0, 65.0, 8.0, 
        3.0, 3.0, 2.0, 
        3.14, 3.14, 3.14, 
        2.0, 2.0, 2.0, 
        40.0, 40.0, 40.0, 40.0, 40.0, 
        130.0, 65.0, 8.0, 
        1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0 
    ], dtype=np.float32)
}

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.ep_rewards = []
        self.ep_lengths = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.ep_rewards.clear()
        self.ep_lengths.clear()

def evaluate_policy(env, agent, mean, std, eval_episodes=10, base_seed=2026, curriculum_progress=1.0, log_phase="eval", log_episode=0, temperature=0.0):
    py_state = random.getstate()
    np_state = np.random.get_state()
    successes, collisions, oob = 0, 0, 0
    rewards, lengths = [], []
    eval_target_cache = None

    try:
        for idx in range(eval_episodes):
            random.seed(base_seed + idx)
            np.random.seed(base_seed + idx)
            env.set_curriculum_progress(curriculum_progress)
            
            env.set_episode_context(log_episode, phase=log_phase)
            state = env.reset() 
            
            if eval_target_cache is None:
                eval_target_cache = env.target.copy()
                
            done = False        
            episode_reward = 0.0 
            episode_steps = 0    
            last_info = {}

            while not done:
                state_norm = normalize_state(state, mean, std)
                state_tensor = torch.FloatTensor(state_norm).to(agent.device)
                with torch.no_grad():
                    if temperature == 0.0:
                        action, _ = agent.policy_old.act(state_tensor, deterministic=True)
                    else:
                        action_raw, _ = agent.policy_old.act(state_tensor, deterministic=False)
                        if temperature < 1.0:
                            mu, _ = agent.policy_old.act(state_tensor, deterministic=True)
                            action = (1.0 - temperature) * mu + temperature * action_raw
                        else:
                            action = action_raw

                action_np = action.detach().cpu().numpy().flatten()
                clipped_action = clip_action(action_np)
                next_state, _, done_env, info = env.step(clipped_action)
                reward = compute_reward(
                    prev_distance=info["prev_distance"], curr_distance=info["curr_distance"],
                    reached_goal=info["success"], collision=info["collision"], out_of_bounds=info["out_of_bounds"],
                    obstacle_distance=info.get("obstacle_distance"), velocity=info.get("velocity"),
                    goal_delta=info.get("goal_delta"), info=info,
                    phase=env.ACTIVE_PHASE
                )
                state = next_state
                episode_reward += reward
                episode_steps += 1
                last_info = info
                done = done_env or (episode_steps >= env.max_steps)

            rewards.append(float(episode_reward))
            lengths.append(episode_steps)
            if last_info and last_info.get("success"): successes += 1
            if last_info and last_info.get("collision"): collisions += 1
            if last_info and last_info.get("out_of_bounds"): oob += 1
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    return {
        "success_rate": successes / max(1, eval_episodes),
        "collision_rate": collisions / max(1, eval_episodes),
        "oob_rate": oob / max(1, eval_episodes),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_episode_length": int(np.mean(lengths)) if lengths else 0,
        "eval_target": eval_target_cache
    }

def _prepare_train_episode(env, episodes_completed, max_episodes):
    progress = min(1.0, (episodes_completed + 1) / max_episodes)
    env.set_curriculum_progress(progress)
    env.set_episode_context(episodes_completed + 1, phase="train")
    return progress

def _maybe_agent_update(agent, buffer, collected_steps, rollout_target_steps, episodes_done, total_target_episodes, current_map):
    if collected_steps >= rollout_target_steps or episodes_done >= total_target_episodes:
        if buffer.states:
            batch_size = 256 if "T3_Wind" in current_map else 64
            agent.update(buffer, mini_batch_size=batch_size)
            buffer.clear()
            return True
        if episodes_done >= total_target_episodes:
            buffer.clear()
            return True
    return False

def _run_eval_checkpoint(env, agent, mean, std, log_episode, max_episodes, all_rewards, all_lengths, logger, eval_episodes, total_collisions, total_successes, fallback_train_reward, successes_list):
    progress = log_episode / max_episodes
    window_rewards = all_rewards[-50:] if all_rewards else [0]
    window_lengths = all_lengths[-50:] if all_lengths else [0]

    logger.save_logs(log_episode, window_rewards, window_lengths, total_successes, successes_list, total_collisions, agent.policy)
    eval_metrics = evaluate_policy(env, agent, mean, std, eval_episodes=eval_episodes, curriculum_progress=progress, log_phase="eval", log_episode=log_episode, temperature=0.0)

    HARD_EVAL_MIN_EPISODE = 200
    eval_hard = None
    if log_episode >= HARD_EVAL_MIN_EPISODE:
        eval_hard = evaluate_policy(env, agent, mean, std, eval_episodes=max(3, eval_episodes // 2), base_seed=4040, curriculum_progress=1.0, log_phase="eval-hard", log_episode=log_episode, temperature=0.0)

    logger.log_evaluation(log_episode, eval_metrics)
    if eval_hard is not None: logger.log_hard_evaluation(log_episode, eval_hard)

    if env.ACTIVE_PHASE == 9:
        target = eval_metrics.get("eval_target")
        if target is None: target = [0.0, 0.0, 0.0]
        
        manifest_file = os.path.join(project_root, "phase9_checkpoint_manifest.csv")
        chkpt_dir = os.path.join(project_root, "checkpoints", "phase9")
        os.makedirs(chkpt_dir, exist_ok=True)
        
        if not os.path.exists(manifest_file):
            with open(manifest_file, "w") as f:
                f.write("Episode,Target_X,Target_Y,Target_Z,Success_Rate,Avg_Reward\n")
        
        with open(manifest_file, "a") as f:
            f.write(f"{log_episode},{target[0]:.2f},{target[1]:.2f},{target[2]:.2f},{eval_metrics['success_rate']:.2f},{eval_metrics['avg_reward']:.2f}\n")
        
        torch.save(agent.policy.state_dict(), os.path.join(chkpt_dir, f"ppo_phase9_ep{log_episode}.pt"))

    # print("Eval @ ep {} | success: {:.2%} | collision: {:.2%} | oob: {:.2%} | avg_len: {}".format(
        log_episode, eval_metrics["success_rate"], eval_metrics["collision_rate"], eval_metrics["oob_rate"], eval_metrics["avg_episode_length"]))
    if eval_hard is not None:
        print("Eval-Hard @ ep {} | success: {:.2%} | collision: {:.2%} | oob: {:.2%} | avg_len: {}".format(
            log_episode, eval_hard["success_rate"], eval_hard["collision_rate"], eval_hard["oob_rate"], eval_hard["avg_episode_length"]))

def train(env, agent, max_episodes=2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.policy.to(device)
    agent.policy_old.to(device)
    state_dim = 26
    mean = np.zeros(state_dim, dtype=np.float32)
    target_map = env.current_map if hasattr(env, 'current_map') else "T1_OpenUrbanGrid"
    std = STATE_STD[target_map].copy()
    buffer = RolloutBuffer()
    logger = TrainingLogger(agent, max_episodes)

    rollout_target_steps = 2048
    eval_interval = 20 if env.ACTIVE_PHASE == 9 else 40
    eval_episodes = 5
    collected_steps = 0
    successes_list = []
    total_collisions = 0
    total_successes = 0

    episodes_completed = 0
    next_eval_milestone = eval_interval
    all_train_ep_rewards, all_train_ep_lengths = [], []

    while episodes_completed < max_episodes:
        ep = episodes_completed + 1
        active_phase = env.ACTIVE_PHASE

        if active_phase == 1:
            target_map = "T1_OpenUrbanGrid"
        elif active_phase in [2, 3]:
            target_map = random.choices(["T1_OpenUrbanGrid", "T2_MediumCityBlocks"], weights=[0.3, 0.7])[0]
        elif active_phase == 4:
            target_map = random.choices(["T1_OpenUrbanGrid", "T2_MediumCityBlocks"], weights=[0.2, 0.8])[0]
        elif active_phase == 5:
            target_map = random.choices(["T1_OpenUrbanGrid", "T2_MediumCityBlocks"], weights=[0.2, 0.8])[0]
        elif active_phase == 6:
            target_map = random.choices(["T1_OpenUrbanGrid", "T2_MediumCityBlocks", "T3_Wind"], weights=[0.2, 0.3, 0.5])[0]
        elif active_phase == 7:
            if episodes_completed < 840:
                target_map = "T3_Wind"
            else:
                target_map = random.choices(["T1_OpenUrbanGrid", "T2_MediumCityBlocks", "T3_Wind"], weights=[0.2, 0.3, 0.5])[0]
        elif active_phase == 8:
            if episodes_completed < 1240:
                target_map = "T3_Wind"
            else:
                target_map = random.choices(["T1_OpenUrbanGrid", "T2_MediumCityBlocks", "T3_Wind"], weights=[0.2, 0.3, 0.5])[0]
        elif active_phase == 9:
            if episodes_completed < 1220:
                target_map = "T3_Wind"
            else:
                if episodes_completed % 40 == 0 or 'current_phase_map' not in locals():
                    current_phase_map = random.choices(["T1_OpenUrbanGrid", "T2_MediumCityBlocks", "T3_Wind"], weights=[0.20, 0.30, 0.50])[0]
                target_map = current_phase_map

        if env.current_map != target_map or episodes_completed == 0:
            if env.current_map != target_map:
                env.switch_map(target_map)
                print(f"Switched to {target_map}")
            std = STATE_STD[target_map].copy() 
            print(f"Scales refreshed for {target_map}")

        display_ep = episodes_completed + 1
        env.set_episode_context(display_ep, phase=f"Phase {active_phase}")       
        env.set_curriculum_progress(episodes_completed / max_episodes)

        print(f"Starting Phase {active_phase} - Episode {display_ep} on {env.current_map}")

        _prepare_train_episode(env, episodes_completed, max_episodes)
        state = env.reset()
        episode_reward, ep_steps, done, ep_success = 0.0, 0, False, 0

        while not done:
            agent.set_training_progress(min(1.0, (episodes_completed + 1) / max_episodes))
            state_norm = normalize_state(state, mean, std)
            state_tensor = torch.FloatTensor(state_norm).to(device)
            action, log_prob = agent.policy_old.act(state_tensor)
            action_np = action.detach().cpu().numpy().flatten()
            clipped_action = clip_action(action_np)

            next_state, _, done_env, info = env.step(clipped_action)
            reward = compute_reward(
                prev_distance=info["prev_distance"], curr_distance=info["curr_distance"], reached_goal=info["success"],
                collision=info["collision"], out_of_bounds=info["out_of_bounds"], obstacle_distance=info.get("obstacle_distance"),
                goal_delta=info.get("goal_delta"), velocity=info.get("velocity"), info=info,
                phase=active_phase
            )

            ep_steps += 1
            episode_reward += reward
            done = done_env or (ep_steps >= env.max_steps)

            buffer.states.append(state_norm)
            buffer.actions.append(action_np)
            buffer.logprobs.append(log_prob.detach().cpu())
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            collected_steps += 1

            if info.get("collision"): total_collisions += 1
            if info.get("success"):
                total_successes += 1
                ep_success = 1

            if not done: state = next_state

        successes_list.append(ep_success)
        buffer.ep_rewards.append(episode_reward)
        buffer.ep_lengths.append(ep_steps)
        all_train_ep_rewards.append(episode_reward)
        all_train_ep_lengths.append(ep_steps)
        episodes_completed += 1

        if _maybe_agent_update(agent, buffer, collected_steps, rollout_target_steps, episodes_completed, max_episodes, env.current_map):
            collected_steps = 0

        while next_eval_milestone <= episodes_completed:
            _run_eval_checkpoint(env, agent, mean, std, next_eval_milestone, max_episodes, all_train_ep_rewards, all_train_ep_lengths, logger, eval_episodes, total_collisions, total_successes, episode_reward, successes_list)
            next_eval_milestone += eval_interval

    logger.finalize_run(max_episodes, all_train_ep_rewards)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    success_rate = total_successes / max(1, max_episodes)
    best_model_dir = os.path.join(project_root, "best_models")
    os.makedirs(best_model_dir, exist_ok=True)
    
    filename = f"ppo_ep{max_episodes}_best_stage{env.ACTIVE_PHASE}_{timestamp}_sr{success_rate:.2f}.pt"
    final_save_path = os.path.join(best_model_dir, filename)
    torch.save(agent.policy.state_dict(), final_save_path)
    print(f"Phase {env.ACTIVE_PHASE} Complete. Model saved securely as: {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training with Auto-Continue.")
    parser.add_argument("--airsim-ip", type=str, default="127.0.0.1")
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--vehicle-name", type=str, default="Drone1")
    parser.add_argument("--no-visualize-target", action="store_true")
    parser.add_argument("--pretrain", type=str, default="")
    cli = parser.parse_args()

    env = DroneEnv(visualize_target=not cli.no_visualize_target, ip=cli.airsim_ip, port=cli.airsim_port, vehicle_name=cli.vehicle_name)
    try:
        print("\n--- Starting First Simulator Instance ---")
        success = env.switch_map("T1_OpenUrbanGrid", force=True)
        if not success: raise ConnectionError("Failed to launch AirSim .exe")
        print("Environment is active. Training will begin shortly...")
        time.sleep(2)
    except Exception as exc:
        raise ConnectionError(f"AirSim startup failed: {exc}") from exc

    agent = PPOAgent(state_dim=26, action_dim=4)

    if cli.pretrain:
        pretrain_path = os.path.join(project_root, cli.pretrain) if not os.path.isabs(cli.pretrain) else cli.pretrain
        print(f"Loading pretrained actor from: {pretrain_path}")
        loaded_state = torch.load(pretrain_path, map_location=agent.device)
        agent.policy.load_state_dict(loaded_state)
        agent.policy_old.load_state_dict(loaded_state)
        print(f"Successfully loaded pretrained weights!")

    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(agent.policy.state_dict(), os.path.join(model_dir, "ppo_initial.pt"))
    
    PHASE_TOTALS = {1: 520, 2: 560, 3: 320, 4: 560, 5: 320, 6: 400, 7: 1040, 8: 1640, 9: 1620}
    START_PHASE = env.ACTIVE_PHASE
    MAX_PHASE = 9

    print(f"\n{'='*60}\nINITIATING AUTOMATED PIPELINE (PHASE {START_PHASE} to {MAX_PHASE})\n{'='*60}\n")

    for phase in range(START_PHASE, MAX_PHASE + 1):
        env.ACTIVE_PHASE = phase
        target_episodes = PHASE_TOTALS.get(phase, 500)
        print(f"\n--- STARTING PHASE {phase} ({target_episodes} Episodes) ---")
        train(env, agent, max_episodes=target_episodes)
        print(f"--- PHASE {phase} FINISHED. CONTINUING TO NEXT PHASE... ---\n")
