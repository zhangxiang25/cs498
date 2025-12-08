import argparse
import os
import sys
from isaaclab.app import AppLauncher
from collections import deque, Counter

# add argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type = int, default = 1, help = "Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
# === ADD THIS BLOCK ===
# Force CUDA initialization before Warp loads
try:
    print("[Fix] Forcing PyTorch CUDA context initialization...")
    z = torch.zeros(1).cuda()
    print("[Fix] CUDA context initialized successfully.")
except Exception as e:
    print(f"[Fix] Warning: Could not initialize CUDA context: {e}")
# ======================
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from scipy.spatial.transform import Rotation as R
# from scipy.spatial.transform import Rotation
from scipy.ndimage import ( # Needed for advanced image processing
    binary_opening, binary_closing, binary_fill_holes,
    generate_binary_structure, label,binary_dilation,
)

from task_envs import MP2SceneCfg, PHYSICS_DT, RENDERING_DT

# === 添加 Policy 类定义 (必须与训练脚本一致) ===
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # Input: 8 dims [Rob_X, Rob_Y, Rob_Z, Grip, Door_X, Door_Y, Door_Z, Yaw]
        # Output: 11 dims [Acts...]
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
# =================================================
# wrap everything into a class so it is easier to access things
class Experiment:

    def __init__(self):
        
        # initialize sim
        sim_cfg = sim_utils.SimulationCfg(device = args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([1.5, 0.0, 1.2], [0.0, 0.0, 0.15])

        # set time step size
        self.sim.set_simulation_dt(physics_dt = PHYSICS_DT, rendering_dt = RENDERING_DT)
        print("\nSim dt: {}\n".format(self.sim.get_physics_dt()))
        self.sim_dt = self.sim.get_physics_dt()
        
        # initialize scene
        scene_cfg = MP2SceneCfg(args_cli.num_envs, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)

        # reset simulation
        self.sim.reset()
        print("Setup complete...")

        # setup IK solver
        diff_ik_cfg = DifferentialIKControllerCfg(command_type = "pose", use_relative_mode = False, ik_method = "dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs = self.scene.num_envs, device = self.sim.device)
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

        # set up visualization
        self.fig, ax = plt.subplots()
        self.im = ax.imshow((np.ones((256, 256))*255).astype(np.uint8))

        # record robot pose
        self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]

        self.gripper_open_val = 0.04
        self.gripper_close_val = 0.0 # This is just a guess

        # === 新增：加载训练好的模型 ===
        self.model_path = "mlp_model_door.pth" # 确保路径正确
        self.policy = Policy()
        
        if os.path.exists(self.model_path):
            self.policy.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.policy.eval() # 切换到推理模式
            print(f"Loaded policy model from {self.model_path}")
        else:
            print(f"WARNING: Model not found at {self.model_path}")

        self.current_yaw = 0.0 # 记录当前的旋转角度
        
 

    def move_robot_joint (self, target_joint_pos, target_gripper_pos, count = 10, time_for_residual_movement = 5):
        '''
        Moves the robot to the given target joint angles. This function is blocking.
        '''

        initial = self.scene['ur5e'].data.joint_pos.clone()
        init_joint_pos = self.scene['ur5e'].data.joint_pos[:, :6].squeeze()
        init_gripper_pos = self.scene['ur5e'].data.joint_pos[:, 6:].squeeze()
        target = self.scene['ur5e'].data.joint_pos.clone()

        if target_gripper_pos is None:
            target[:, :6] = torch.tensor(target_joint_pos)
            target[:, 6:] = init_gripper_pos
        elif target_joint_pos is None:
            target[:, :6] = init_joint_pos
            target[:, 6:] = torch.tensor([target_gripper_pos, target_gripper_pos])
        else:
            target[:, :6] = torch.tensor(target_joint_pos)
            target[:, 6:] = torch.tensor([target_gripper_pos, target_gripper_pos])

        # print("Moving the robot through joint control...")
        for i in range (count):

            self.scene["ur5e"].set_joint_position_target((target - initial)/count*i + initial)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        # some buffer time for the robot to complete the movement
        # print("Waiting for any residual movement...")
        for i in range (time_for_residual_movement):
            self.scene["ur5e"].set_joint_position_target(target)

            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        # update robot pose
        self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]
    

    def move_robot_ik (self, target_pose, max_joint_change = 0.04, ik_tol = 1e-3, timeout_count = 100):
        '''
        Calls Isaac Lab's IK controller and moves the robot to a desired pose in 3D space.
        '''

        self.diff_ik_controller.set_command(torch.tensor(target_pose, device = self.sim.device))

        # print("Moving the robot through IK...")
        count = 0
        while simulation_app.is_running():

            # obtain quantities from simulation
            jacobian = self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
            ee_pose_w = self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = self.scene["ur5e"].data.root_state_w[:, 0:7]
            joint_pos = self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]

            # compute the joint commands
            joint_pos_des = self.diff_ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos)

            # apply actions
            joint_changes = (joint_pos_des - joint_pos).detach().cpu().numpy()[0]
            if np.sum(np.abs(joint_changes) > max_joint_change) > 0:
                scaled_joint_changes = joint_changes / (np.max(np.abs(joint_changes)) / max_joint_change)
                scaled_joint_changes = torch.tensor(scaled_joint_changes).unsqueeze(0).to(joint_pos_des.device)
                self.scene["ur5e"].set_joint_position_target(joint_pos + scaled_joint_changes, joint_ids = self.robot_entity_cfg.joint_ids)
            else:
                self.scene["ur5e"].set_joint_position_target(joint_pos_des, joint_ids = self.robot_entity_cfg.joint_ids)

            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

            count += 1

            # terminating condition
            cur_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
            if np.average(np.abs(target_pose - cur_pose)[:3]) < ik_tol and np.average(np.abs(target_pose - cur_pose)[3:]) < ik_tol:
                # update robot pose
                self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
                self.robot_pos = self.robot_pose[:3]
                self.robot_quat = self.robot_pose[3:]
                return
            
            if count >= timeout_count:
                # print("Movement terminated due to timeout.")
                self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
                self.robot_pos = self.robot_pose[:3]
                self.robot_quat = self.robot_pose[3:]
                return
            

    def sim_wait (self, count):
        # print("Waiting...")
        for _ in range (count):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)
        # update robot pose
        self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]


    def open_gripper(self):
        self.move_robot_joint(
            target_joint_pos=None, 
            target_gripper_pos=self.gripper_open_val,
            count=25, 
            time_for_residual_movement=5
        )
        
    def close_gripper(self):
        self.move_robot_joint(
            target_joint_pos=None, 
            target_gripper_pos=self.gripper_close_val,
            count=25, 
            time_for_residual_movement=5
        )

    # === 新增：辅助函数用于获取 Observation ===
    def get_door_pos(self):
        # 获取门基座的世界坐标
        return torch.squeeze(self.scene["door"].data.root_link_pos_w).detach().cpu().numpy()

    def get_eef_pos(self):
        # 获取末端执行器坐标
        return self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[:3]

    def run (self):
        '''
        Modified Logic: Uses a WHILE loop to continuously clear blocking cubes until none remain.
        '''
        # 1. Reset and Init
        self.scene.reset()
        fixed_quat = self.robot_quat.copy()
        
        self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]
        
        intrinsics = np.squeeze(self.scene["birdview_camera"].data.intrinsic_matrices.detach().cpu().numpy())
        extrinsics = np.array([
            [ 0,  1,  0, 0.5],
            [ 1,  0,  0,   0],
            [ 0,  0, -1, 1.2],
            [ 0,  0,  0,   1]
        ])

        # --- 2. Initial Robot Lift (Done once) ---
        print("Initializing Robot Position...")
        # Get the robot's starting X and Y position
        lift_pos = self.robot_pos.copy() 
        lift_pos[2] = 0.5
        # Execute the vertical lift
        self.move_robot_ik(np.concatenate([lift_pos, self.robot_quat]))
        self.sim_wait(20)

        away_pos = np.array([-0.2, 0.0, 0.5])
        
        # --- 3. The CLEARING Loop ---
        loop_iteration = 0
        while True:
            loop_iteration += 1
            print(f"\n======== Starting Clearing Pass {loop_iteration} ========")
            
            # 3.1 Move Robot to "Look" Position
            print("Moving to observation position...")
            self.move_robot_ik(np.concatenate([away_pos, self.robot_quat]))
            self.sim_wait(20)

            # 3.2 Capture and Process Image
            print("Capturing and processing scene...")
            color_raw = self.scene["birdview_camera"].data.output["rgb"].detach().cpu().numpy()[0]
            
            # Define Color Ranges (Moved inside loop or reused)
            h_red_low_1 = 0.00; h_red_high_1 = 0.04
            h_red_low_2 = 0.96; h_red_high_2 = 1.00
            h_green_low = 0.23; h_green_high = 0.44
            h_blue_low = 0.58; h_blue_high = 0.75
            h_yellow_low = 0.14; h_yellow_high = 0.18
            h_magenta_low = 0.82; h_magenta_high = 0.88
            s_min = 0.40; v_min = 0.20
            morph_iters = 2
            
            color = color_raw[:, :, :3] if color_raw.shape[2] == 4 else color_raw
            height, width, _ = color.shape
            depth_image = np.squeeze(self.scene["birdview_camera"].data.output["depth"].detach().cpu().numpy()[0])
            
            rgb01 = np.clip(color.astype(np.float32)/255.0, 0.0, 1.0)
            hsv = mcolors.rgb_to_hsv(rgb01)
            h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            
            # Generate Masks
            red1_raw = (h >= h_red_low_1) & (h <= h_red_high_1) & (s >= s_min) & (v >= v_min)
            red2_raw = (h >= h_red_low_2) & (h <= h_red_high_2) & (s >= s_min) & (v >= v_min)
            red_mask_raw = red1_raw | red2_raw
            green_mask_raw = (h >= h_green_low) & (h <= h_green_high) & (s >= s_min) & (v >= v_min)
            blue_mask_raw = (h >= h_blue_low) & (h <= h_blue_high) & (s >= s_min) & (v >= v_min)
            yellow_mask_raw = (h >= h_yellow_low) & (h <= h_yellow_high) & (s >= s_min) & (v >= v_min)
            magenta_mask_raw = (h >= h_magenta_low) & (h <= h_magenta_high) & (s >= s_min) & (v >= v_min)

            struct = generate_binary_structure(2, 2)
            red_mask = binary_closing(binary_fill_holes(binary_opening(red_mask_raw, structure=struct, iterations=morph_iters)), structure=struct, iterations=max(1, morph_iters // 2))
            green_mask = binary_closing(binary_fill_holes(binary_opening(green_mask_raw, structure=struct, iterations=morph_iters)), structure=struct, iterations=max(1, morph_iters // 2))
            blue_mask = binary_closing(binary_fill_holes(binary_opening(blue_mask_raw, structure=struct, iterations=morph_iters)), structure=struct, iterations=max(1, morph_iters // 2))
            yellow_mask = binary_closing(binary_fill_holes(binary_opening(yellow_mask_raw, structure=struct, iterations=morph_iters)), structure=struct, iterations=max(1, morph_iters // 2))
            magenta_mask = binary_closing(binary_fill_holes(binary_opening(magenta_mask_raw, structure=struct, iterations=morph_iters)), structure=struct, iterations=max(1, morph_iters // 2))

            # Generate Point Cloud
            fx = intrinsics[0,0]; fy = intrinsics[1,1]
            cx = intrinsics[0,2]; cy = intrinsics[1,2]
            y_pixel, x_pixel = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            z_c = depth_image
            x_c = z_c * (x_pixel - cx) / fx
            y_c = z_c * (y_pixel - cy) / fy
            points_camera = np.stack((x_c, y_c, z_c), axis=-1).reshape(-1, 3) 
            ones = np.ones((points_camera.shape[0], 1), dtype=points_camera.dtype)
            points_world_h = extrinsics @ np.concatenate([points_camera, ones], axis=1).T
            points_world = (points_world_h[:3,:] / np.clip(points_world_h[3,:], 1e-8, None)).T
            pc_world = points_world.reshape(height, width, 3)

            # 3.3 Analyze Cubes
            # --- RED CUBES ---
            lab_red, n_red = label(red_mask.astype(np.uint8), structure=struct)
            red_infos = []
            for cid in range(1, n_red + 1):
                idx_red = (lab_red == cid)
                pts_red = pc_world.reshape(-1, 3)[idx_red.flatten()]
                pts_red = pts_red[np.isfinite(pts_red).all(axis=1)]
                if pts_red.shape[0] == 0: continue
                
                top_z_red = float(np.nanpercentile(pts_red[:, 2], 95))
                low_z_red = float(np.nanpercentile(pts_red[:, 2], 5))
                # Top slice for accurate center
                top_slice_mask = pts_red[:, 2] > (top_z_red - 0.015) 
                pts_red_top = pts_red[top_slice_mask]
                if pts_red_top.shape[0] > 10:
                    centroid_xy_red = np.nanmedian(pts_red_top[:, :2], axis=0)
                else:
                    centroid_xy_red = np.nanmedian(pts_red[:, :2], axis=0)

                stats = {"centroid_xy": centroid_xy_red, "top_z": top_z_red, "low_z": low_z_red}
                # area used for sorting if needed, though simpler now
                area = (lab_red == cid).sum()
                red_infos.append({"cid": cid, "area": area, "stats": stats})

            # --- GREEN CUBES ---
            lab_green, n_green = label(green_mask.astype(np.uint8), structure=struct)
            green_infos = []
            for cid in range(1, n_green + 1):
                idx_green = (lab_green == cid)
                pts_green = pc_world.reshape(-1, 3)[idx_green.flatten()]
                pts_green = pts_green[np.isfinite(pts_green).all(axis=1)]
                if pts_green.shape[0] == 0: continue
                
                centroid_xy_green = np.nanmedian(pts_green[:, :2], axis=0)
                top_z_green = float(np.nanpercentile(pts_green[:, 2], 95))
                low_z_green = float(np.nanpercentile(pts_green[:, 2], 5))
                stats = {"centroid_xy": centroid_xy_green, "top_z": top_z_green, "low_z": low_z_green}
                
                d_center = float(np.linalg.norm(stats["centroid_xy"] - np.array([0.5, 0.0])))
                area = (lab_green == cid).sum()
                green_infos.append({"cid": cid, "area": area, "stats": stats, "d_center": d_center})

            # 3.4 Determine Blocking Status
            door_mask_with_holes = blue_mask | yellow_mask | magenta_mask
            door_mask = binary_dilation(door_mask_with_holes, structure=struct, iterations=10)
            
            red_blocking_infos = []
            green_blocking_infos = []
            blocking_ids_red = []
            blocking_ids_green = []

            # Check Red Blocking
            for info in red_infos:
                stats = info['stats']
                xyz_world = np.array([stats['centroid_xy'][0], stats['centroid_xy'][1], stats['top_z']])
                distances_sq = np.sum((pc_world - xyz_world)**2, axis=2)
                v, u = np.unravel_index(np.argmin(distances_sq), distances_sq.shape)
                if door_mask[v, u]:
                    red_blocking_infos.append(info)
                    blocking_ids_red.append(info['cid'])

            # Check Green Blocking
            for info in green_infos:
                stats = info['stats']
                xyz_world = np.array([stats['centroid_xy'][0], stats['centroid_xy'][1], stats['top_z']])
                distances_sq = np.sum((pc_world - xyz_world)**2, axis=2)
                v, u = np.unravel_index(np.argmin(distances_sq), distances_sq.shape)
                if door_mask[v, u]:
                    green_blocking_infos.append(info)
                    blocking_ids_green.append(info['cid'])

            print(f"Pass {loop_iteration} Analysis: Found {len(red_blocking_infos)} red and {len(green_blocking_infos)} green blocking cubes.")

            # --- EXIT CONDITION ---
            if len(red_blocking_infos) == 0 and len(green_blocking_infos) == 0:
                print("\nSUCCESS: No blocking cubes remaining. Door area is clear!")
                break
            
            # --- Identify Safe Spots ---
            table_min_x, table_max_x = 0.15, 0.85
            table_min_y, table_max_y = -0.35, 0.35
            edge_margin = 0.18

            # Safe Red
            safe_red_infos = []
            for info in red_infos:
                if info['cid'] in blocking_ids_red: continue
                cx, cy = info['stats']['centroid_xy']
                is_on_edge = (cx < table_min_x + edge_margin) or (cx > table_max_x - edge_margin) or \
                             (cy < table_min_y + edge_margin) or (cy > table_max_y - edge_margin)
                if not is_on_edge:
                    safe_red_infos.append(info)
            
            # Safe Green
            safe_green_infos = []
            for info in green_infos:
                if info['cid'] in blocking_ids_green: continue
                cx, cy = info['stats']['centroid_xy']
                is_on_edge = (cx < table_min_x + edge_margin) or (cx > table_max_x - edge_margin) or \
                             (cy < table_min_y + edge_margin) or (cy > table_max_y - edge_margin)
                if not is_on_edge:
                    safe_green_infos.append(info)

            # --- 3.5 Execute Moves (Red) ---
            num_to_move = min(len(red_blocking_infos), len(safe_red_infos))
            z_travel_height = 0.45
            z_hover_height = 0.45
            z_offset_grasp = 0.02
            z_offset_stack = 0.025

            for i in range(num_to_move):
                block_cube = red_blocking_infos[i]
                target_base = safe_red_infos[i]
                
                print(f"Moving RED Blocking ID {block_cube['cid']} to Safe ID {target_base['cid']}")
                
                pick_xy = block_cube['stats']['centroid_xy']
                pick_z = block_cube['stats']['top_z'] + z_offset_grasp
                
                place_xy = target_base['stats']['centroid_xy']
                place_z = target_base['stats']['top_z'] + 0.04 + z_offset_stack
                
                # Move Logic
                hover_pose = np.concatenate([pick_xy, [z_travel_height], fixed_quat])
                self.move_robot_ik(hover_pose)
                self.open_gripper()
                
                pick_pose = np.concatenate([pick_xy, [pick_z], fixed_quat])
                self.move_robot_ik(pick_pose)
                self.sim_wait(10)
                self.close_gripper()
                self.sim_wait(15)
                
                self.move_robot_ik(hover_pose) # Lift

                place_hover_pose = np.concatenate([place_xy, [z_travel_height], fixed_quat])
                self.move_robot_ik(place_hover_pose)
                
                place_pose = np.concatenate([place_xy, [place_z], fixed_quat])
                self.move_robot_ik(place_pose)
                self.sim_wait(15)
                self.open_gripper()
                self.sim_wait(20)
                
                self.move_robot_ik(place_hover_pose)

            # --- 3.6 Execute Moves (Green) ---
            num_green_to_move = min(len(green_blocking_infos), len(safe_green_infos))
            green_cube_height = 0.05 

            for i in range(num_green_to_move):
                block_cube = green_blocking_infos[i]
                target_base = safe_green_infos[i]
                
                print(f"Moving GREEN Blocking ID {block_cube['cid']} to Safe ID {target_base['cid']}")
                
                pick_xy = block_cube['stats']['centroid_xy']
                pick_z = block_cube['stats']['top_z'] + z_offset_grasp 
                
                place_xy = target_base['stats']['centroid_xy']
                place_z = target_base['stats']['top_z'] + green_cube_height + z_offset_stack
                
                hover_pose = np.concatenate([pick_xy, [z_travel_height], fixed_quat])
                self.move_robot_ik(hover_pose)
                self.open_gripper()
                
                pick_pose = np.concatenate([pick_xy, [pick_z], fixed_quat])
                self.move_robot_ik(pick_pose)
                self.sim_wait(10)
                self.close_gripper()
                self.sim_wait(15)

                self.move_robot_ik(hover_pose)

                place_hover_pose = np.concatenate([place_xy, [z_travel_height], fixed_quat])
                self.move_robot_ik(place_hover_pose)
                
                place_pose = np.concatenate([place_xy, [place_z], fixed_quat])
                self.move_robot_ik(place_pose)
                self.sim_wait(15)
                self.open_gripper()
                self.sim_wait(20)

                self.move_robot_ik(place_hover_pose)

            # Check if we are stuck (Blocking cubes exist but no safe spots)
            if (len(red_blocking_infos) > 0 and len(safe_red_infos) == 0) and \
               (len(green_blocking_infos) > 0 and len(safe_green_infos) == 0):
                print("WARNING: Stuck! Blocking cubes remain but no safe locations found.")
                break
                
            # Loop restarts here, re-scanning the environment
        print("Moving to Door Start Position...")
        start_pos = np.array([0.4, 0.0, 0.3]) 
        # Reset Yaw to 0 (pointing down)
        base_quat = np.array([0, -np.sqrt(2)/2, np.sqrt(2)/2, 0]) 
        self.current_yaw = 0.0
        
        start_pose = np.concatenate([start_pos, base_quat])
        self.move_robot_ik(start_pose, timeout_count=200, ik_tol=1e-2)
        print("Reached Start Position. Opening Gripper...")
        self.open_gripper() 
        self.sim_wait(30)

        
        print("Starting Enhanced Inference Loop...")

        while simulation_app.is_running():

            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        simulation_app.close()