import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from collections import deque, Counter

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type = int, default = 1, help = "Number of environments to spawn.")
parser.add_argument("--model_path", type = str, default = "mlp_model_door.pth", help = "Path to trained model")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("[DEBUG] Isaac Sim App Launched. Importing libraries...")

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene

# Import your config
from task_envs_door import DoorSceneCfg, PHYSICS_DT, RENDERING_DT

# === 8 Input Policy (Matches your trained model) ===
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # Input: 8 dims [Robot_X, Y, Z, Gripper, Door_X, Y, Z, Robot_Yaw]
        # Output: 11 dims
        self.net = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        return self.net(x)

class InferenceExperiment:
    def __init__(self, model_path):
        
        print(f"[DEBUG] Initializing Experiment. Checking model path: {model_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DEBUG] Using device: {self.device}")

        self.model = Policy().to(self.device)
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[DEBUG] SUCCESS: Loaded 8-INPUT model from {model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load model weights: {e}")
                exit()
        else:
            print(f"[ERROR] Model file NOT found at: {os.path.abspath(model_path)}")
            print("Please make sure 'mlp_model_door.pth' is in the same folder.")
            exit()
        
        self.model.eval()

        # 2. Initialize Sim
        print("[DEBUG] Building Simulation Context...")
        sim_cfg = sim_utils.SimulationCfg(device = args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([1.2, 0.0, 0.8], [0.5, 0.0, 0.2])
        self.sim.set_simulation_dt(physics_dt = PHYSICS_DT, rendering_dt = RENDERING_DT)
        self.sim_dt = self.sim.get_physics_dt()

        print("[DEBUG] Building Scene (DoorSceneCfg)...")
        scene_cfg = DoorSceneCfg(args_cli.num_envs, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)
        
        print("[DEBUG] Resetting Simulation (Waiting for assets to load)...")
        self.sim.reset()
        print("[DEBUG] Simulation Reset Complete.")

        # 3. Setup IK
        print("[DEBUG] Setting up IK Controller...")
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)
        self.ik_body = "gripper_center"
        self.robot_entity_cfg = SceneEntityCfg(
            "ur5e", 
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], 
            body_names=[self.ik_body]
        )
        self.robot_entity_cfg.resolve(self.scene)
        
        if self.scene["ur5e"].is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]
        
        print("[DEBUG] Setup Finished. Ready to run.")

    def get_eef_pos(self):
        return self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[:3]

    def get_door_pos(self):
        return torch.squeeze(self.scene["door"].data.root_link_pos_w).detach().cpu().numpy()

    def run(self):
        
        max_steps = 1000
        temp_dist_target = 0.0018
        rot_step = 0.05
        max_joint_change = 0.10
        episode_count = 0

        # [新增] 平滑缓冲区 (Smoothing Buffer)
        history_len = 5
        action_history = deque(maxlen=history_len)

        print(f"[DEBUG] Starting Main Loop. Max steps per episode: {max_steps}")

        while simulation_app.is_running():
            episode_count += 1
            print(f"\n[DEBUG] === Starting Episode {episode_count} ===")
            
            # === RESET LOGIC ===
            start_pos = np.array([0.4, 0.0, 0.35]) 
            base_quat = np.array([0, -np.sqrt(2)/2, np.sqrt(2)/2, 0]) 
            current_yaw = 0.0
            
            start_pose = np.concatenate([start_pos, base_quat])
            
            self.diff_ik_controller.reset()
            self.diff_ik_controller.set_command(torch.tensor(start_pose, device=self.sim.device))
            action_history.clear() # 清空历史动作

            print("[DEBUG] Warm-up: Moving robot to start position...")
            # Warm up loop
            for i in range(400):
                if not simulation_app.is_running(): return

                # 1. 计算当前位置误差
                cur_pos = self.get_eef_pos()
                dist_error = np.linalg.norm(cur_pos - start_pos)
                
                # 2. 如果误差小于 1cm，说明到位了，跳出循环
                if dist_error < 0.01: 
                    print(f"[DEBUG] Robot reached start position at step {i}.")
                    break
                
                jacobian = self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
                ee_pose_w = self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
                joint_pos = self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]
                joint_pos_des = self.diff_ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos)
                
                all_joint_pos_des = torch.zeros((1, 8))
                all_joint_pos_des[:, :6] = joint_pos_des
                all_joint_pos_des[:, 6:] = torch.tensor([0.05, 0.05]).to(self.sim.device)
                
                self.scene["ur5e"].set_joint_position_target(all_joint_pos_des)
                self.scene.write_data_to_sim()
                self.sim.step()

                self.scene.update(self.sim_dt)

            print("[DEBUG] Warm-up done. Starting Inference Control...")

            gripper_state = -1 
            is_rot_aligned = False
            
            # Evaluation Loop
            for step in range(max_steps):
                if not simulation_app.is_running(): break

                # 1. Get State
                current_robot_pos = self.get_eef_pos()
                current_door_pos = self.get_door_pos()
                
                # 构造 8 维输入
                obs_vector = np.array([
                    current_robot_pos[0],
                    current_robot_pos[1],
                    current_robot_pos[2],
                    gripper_state,
                    current_door_pos[0],
                    current_door_pos[1],
                    current_door_pos[2],
                    current_yaw
                ])
                
                # 2. Inference
                obs_tensor = torch.FloatTensor(obs_vector).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(obs_tensor)

                    # 1. 旋转锁定
                    if abs(current_yaw) > 1.45: is_rot_aligned = True
                    if is_rot_aligned:
                        logits[0, 8] = -1e9 # Block Stationary
                        logits[0, 9] = -1e9 # Block Rot
                        logits[0, 10] = -1e9

                 # === 2. 几何状态计算 ===
                    diff_y = abs(current_robot_pos[1] - current_door_pos[1])
                    diff_x = abs(current_robot_pos[0] - current_door_pos[0]) 
                    robot_z = current_robot_pos[2]
                    # 判断是否到位
                    is_x_aligned = diff_x > 0.20 # 到达边缘
                    is_y_aligned = diff_y < 0.02 # 对准中心

                    # === 3. 高空对准逻辑 (High Altitude Logic) ===
                    
                    if not (is_x_aligned and is_y_aligned):
                        # --- 阶段 A：还没对准 ---
                        # 严禁下降 (Block -Z)
                        logits[0, 5] = -1e9 
                        # 严禁抓取
                        logits[0, 7] = -1e9

                        # 细分：先 X 后 Y
                        if not is_x_aligned:
                            # 还没走到边缘：封锁 Y 轴，专心走直线 X
                            logits[0, 2] = -1e9
                            logits[0, 3] = -1e9
                        else:
                            # X 到了，Y 没对准：封锁 X 轴 (防止走过头)，专心调 Y
                            # (可选：如果不希望它停下，可以不封锁 X，但封锁更稳)
                            pass
                    
                    else:
                        # --- 阶段 B：XY 都对准了 (Vertical Descent) ---
                        # 允许下降 (Action 5)
                        
                        # 高度保护：到底之前不能抓
                        if current_robot_pos[2] < 0.28:
                            logits[0, 7] = 1e9  # 鼓励抓取
                            logits[0, 0] = -1e20
                            logits[0, 1] = -1e20
                            logits[0, 2] = -1e20
                            logits[0, 3] = -1e20
                            logits[0, 8] = -1e20 # Mask Stationary
                            logits[0, 9] = -1e20 # Mask Rot
                            logits[0, 10] = -1e20
                        else:
                            logits[0, 7] = -1e9 # 禁止抓取

                    raw_action_idx = torch.argmax(logits, dim=1).item()
                # 平滑
                action_history.append(raw_action_idx)
                if len(action_history) == history_len:
                    final_action_idx = Counter(action_history).most_common(1)[0][0]
                else:
                    final_action_idx = raw_action_idx
                
                if step % 50 == 0:
                    status_str = "ROT_OK" if is_rot_aligned else "ROTATING"
                    print(f"[DEBUG] Step {step} | {status_str} | Act: {final_action_idx} | X:{diff_x:.3f} Y:{diff_y:.3f} Z:{robot_z:.3f}")
                    
                target_pos = current_robot_pos.copy()
                is_stationary = False

                if final_action_idx == 0: target_pos[0] += temp_dist_target
                elif final_action_idx == 1: target_pos[0] -= temp_dist_target
                elif final_action_idx == 2: target_pos[1] += temp_dist_target
                elif final_action_idx == 3: target_pos[1] -= temp_dist_target
                elif final_action_idx == 4: target_pos[2] += temp_dist_target
                elif final_action_idx == 5: target_pos[2] -= temp_dist_target
                elif final_action_idx == 6: gripper_state = -1 
                elif final_action_idx == 7: gripper_state = 1  
                elif final_action_idx == 8: is_stationary = True
                elif final_action_idx == 9: current_yaw += rot_step 
                elif final_action_idx == 10: current_yaw -= rot_step 

                # 4. Compute Orientation
                base_rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]) 
                z_rot = R.from_euler('z', current_yaw)
                final_rot = base_rot * z_rot
                final_quat_scipy = final_rot.as_quat() 
                target_quat = np.array([final_quat_scipy[3], final_quat_scipy[0], final_quat_scipy[1], final_quat_scipy[2]])

                target_pose = np.concatenate([target_pos, target_quat])
                self.diff_ik_controller.set_command(torch.tensor(target_pose, device=self.sim.device))

                # 5. Physics Step
                jacobian = self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
                ee_pose_w = self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
                joint_pos = self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]
                joint_pos_des = self.diff_ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos)

                joint_changes = (joint_pos_des - joint_pos).detach().cpu().numpy()[0]
                if np.sum(np.abs(joint_changes) > max_joint_change) > 0:
                    scaled_joint_changes = joint_changes / (np.max(np.abs(joint_changes)) / max_joint_change)
                    scaled_joint_changes = torch.tensor(scaled_joint_changes).unsqueeze(0).to(joint_pos_des.device)
                    joint_pos_des = joint_pos + scaled_joint_changes

                all_joint_pos_des = torch.zeros((1, 8))
                all_joint_pos_des[:, :6] = joint_pos_des
                if gripper_state == -1:
                    all_joint_pos_des[:, 6:] = torch.tensor([0.05, 0.05]).to(self.sim.device)
                else:
                    all_joint_pos_des[:, 6:] = torch.tensor([0.0, 0.0]).to(self.sim.device)

                self.scene["ur5e"].set_joint_position_target(all_joint_pos_des)
                self.scene.write_data_to_sim()
                self.sim.step()
                self.scene.update(self.sim_dt)

            print("[DEBUG] Episode finished. Resetting Environment...")
            
            # Reset
            ur5e_state = self.scene["ur5e"].data.default_root_state.clone()
            self.scene["ur5e"].write_root_pose_to_sim(ur5e_state[:, :7])
            self.scene["ur5e"].write_root_velocity_to_sim(ur5e_state[:, 7:])
            joint_pos, joint_vel = (self.scene["ur5e"].data.default_joint_pos.clone(), self.scene["ur5e"].data.default_joint_vel.clone())
            self.scene["ur5e"].write_joint_state_to_sim(joint_pos, joint_vel)
            
            door_joint_pos = torch.zeros((self.scene["door"].num_instances, 1), device=self.sim.device)
            door_joint_vel = torch.zeros((self.scene["door"].num_instances, 1), device=self.sim.device)
            self.scene["door"].write_joint_state_to_sim(door_joint_pos, door_joint_vel)
            
            self.scene.reset()


if __name__ == "__main__":
    try:
        exp = InferenceExperiment(model_path=args_cli.model_path)
        exp.run()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Script crashed: {e}")