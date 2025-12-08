import argparse
import os
import sys
from isaaclab.app import AppLauncher
from collections import deque, Counter

# add argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type = int, default = 1, help = "Number of environments to spawn.")
parser.add_argument("--model_path", type = str, default = "mlp_model_door.pth", help = "Path to trained door model") # Added model path arg
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
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

        # === ADDED: Load Policy Model ===
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_model = Policy().to(self.device)
        
        # Use provided arg or default to local file
        model_path = args_cli.model_path
        if os.path.exists(model_path):
            try:
                self.policy_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.policy_model.eval()
                print(f"[SUCCESS] Loaded Door Policy model from {model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load model weights: {e}")
        else:
            print(f"[WARNING] Model file NOT found at: {model_path}. Door opening phase will likely fail.")
        # ================================
    
    # === ADDED: Helper for Door Position ===
    def get_door_pos(self):
        # Assumes the door prim is named "door" in MP2SceneCfg
        return torch.squeeze(self.scene["door"].data.root_link_pos_w).detach().cpu().numpy()

    def get_current_eef_pos(self):
        # Helper to get fresh EEF pos for the policy loop
        return self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[:3]
    # =======================================
 

    def move_robot_joint (self, target_joint_pos, target_gripper_pos, count = 10, time_for_residual_movement = 5):
        '''
        Moves the robot to the given target joint angles. This function is blocking.

        Parameters:
            - target_joint_pos:     An numpy array of length 6 specifying the target values (in radians) of the ur5e robot body joints.
                                    If set to None, the robot body joints do not move.
            - target_gripper_pos:   A float value specifying the target value (in meters) of the ur5e robot gripper joint.
                                    If set to None, the robot gripper does not move.
            - count:                An integer specifying the number of desired simulation timesteps this movement will take.
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

        print("Moving the robot through joint control...")
        for i in range (count):

            self.scene["ur5e"].set_joint_position_target((target - initial)/count*i + initial)
            # scene["ur5e"].set_joint_position_target(target)

            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        # some buffer time for the robot to complete the movement
        print("Waiting for any residual movement...")
        for i in range (time_for_residual_movement):
            self.scene["ur5e"].set_joint_position_target(target)

            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        print("Movement completed. Deviation: {}".format((target - self.scene['ur5e'].data.joint_pos).squeeze().detach().cpu().numpy()))

        # update robot pose
        self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]
    

    def move_robot_ik (self, target_pose, max_joint_change = 0.04, ik_tol = 1e-3, timeout_count = 100):
        '''
        Calls Isaac Lab's IK controller and moves the robot to a desired pose in 3D space. This function is blocking.

        Parameters:
            - target_pose:          A length 7 numpy array of format [x, y, z, qw, qx, qy, qz], the desired pose in 3D to move the robot to.
            - max_joint_change:     A float value denoting the maximum change (in radians) for robot body joints. Setting this limit 
                                    helps prevent overshoot.
            - ik_tol:               A float value (in meters) denoting the distance between the current robot eef position and the target 
                                    robot eef position for the movement to be considered "at target".
            - timeout_count:        The number of simulation timesteps before the robot aborts the movement.
        '''

        self.diff_ik_controller.set_command(torch.tensor(target_pose, device = self.sim.device))

        print("Moving the robot through IK...")
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
                print("Movement completed. Deviation:", np.abs(target_pose - cur_pose)[:3])
                
                # update robot pose
                self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
                self.robot_pos = self.robot_pose[:3]
                self.robot_quat = self.robot_pose[3:]

                return
            
            if count >= timeout_count:
                print("Movement terminated due to timeout. Deviation:", np.abs(target_pose - cur_pose)[:3])
                
                # update robot pose
                self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
                self.robot_pos = self.robot_pose[:3]
                self.robot_quat = self.robot_pose[3:]
                
                return
            

    def sim_wait (self, count):
        '''
        Wait for a given number of timesteps in simulation.
        '''

        print("Waiting...")
        for _ in range (count):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        # update robot pose
        self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]


    def render_camera (self, camera_name):
        '''
        Can potentially be used to render and update camera images in a loop. You don't have to use it when completing this MP.
        '''

        cam_img = self.scene[camera_name].data.output["rgb"].detach().cpu().numpy()[0]

        self.im.set_data(cam_img)
        plt.pause(1e-6)
        self.fig.canvas.draw()
    
    def open_gripper(self):
        """Commands the gripper to open using joint control."""
        # Calls your existing function to move ONLY the gripper
        self.move_robot_joint(
            target_joint_pos=None,  # Arm doesn't move
            target_gripper_pos=self.gripper_open_val,
            count=15, # Use a few steps to make it a smooth open
            time_for_residual_movement=5
        )
        
    def close_gripper(self):
        """Commands the gripper to close using joint control."""
        # Calls your existing function to move ONLY the gripper
        self.move_robot_joint(
            target_joint_pos=None,  # Arm doesn't move
            target_gripper_pos=self.gripper_close_val,
            count=15, # Use a few steps to make it a smooth close
            time_for_residual_movement=5
        )
        
    # === ADDED: Image Processing and Cube Identification Helper ===
    def _identify_blocking_cubes(self, intrinsics, extrinsics, x_pixel, y_pixel, cx, cy, fx, fy, height, width):
        """
        Performs camera capture, image processing, 3D reconstruction, and identifies blocking and safe cubes.
        Returns all necessary lists and masks.
        """
        # 1. Capture Data
        color_raw = self.scene["birdview_camera"].data.output["rgb"].detach().cpu().numpy()[0]
        depth_image = np.squeeze(self.scene["birdview_camera"].data.output["depth"].detach().cpu().numpy()[0])
        color = color_raw[:, :, :3] if color_raw.shape[2] == 4 else color_raw
        
        # 2. Color and Morphology Parameters
        h_red_low_1 = 0.00; h_red_high_1 = 0.04
        h_red_low_2 = 0.96; h_red_high_2 = 1.00
        h_green_low = 0.23; h_green_high = 0.44
        h_blue_low = 0.58; h_blue_high = 0.75
        h_yellow_low = 0.14; h_yellow_high = 0.18
        h_magenta_low = 0.82; h_magenta_high = 0.88
        s_min = 0.40; v_min = 0.20
        morph_iters = 2
        struct = generate_binary_structure(2, 2)
        
        # 3. HSV Conversion and Raw Masks
        rgb01 = np.clip(color.astype(np.float32)/255.0, 0.0, 1.0)
        hsv = mcolors.rgb_to_hsv(rgb01)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        red1_raw = (h >= h_red_low_1) & (h <= h_red_high_1) & (s >= s_min) & (v >= v_min)
        red2_raw = (h >= h_red_low_2) & (h <= h_red_high_2) & (s >= s_min) & (v >= v_min)
        red_mask_raw = red1_raw | red2_raw
        green_mask_raw = (h >= h_green_low) & (h <= h_green_high) & (s >= s_min) & (v >= v_min)
        blue_mask_raw = (h >= h_blue_low) & (h <= h_blue_high) & (s >= s_min) & (v >= v_min)
        yellow_mask_raw = (h >= h_yellow_low) & (h <= h_yellow_high) & (s >= s_min) & (v >= v_min)
        magenta_mask_raw = (h >= h_magenta_low) & (h <= h_magenta_high) & (s >= s_min) & (v >= v_min)

        # 4. Morphological Processing
        red_mask = binary_opening(red_mask_raw, structure=struct, iterations=morph_iters)
        red_mask = binary_fill_holes(red_mask)
        red_mask = binary_closing(red_mask, structure=struct, iterations=max(1, morph_iters // 2))
        green_mask = binary_opening(green_mask_raw, structure=struct, iterations=morph_iters)
        green_mask = binary_fill_holes(green_mask)
        green_mask = binary_closing(green_mask, structure=struct, iterations=max(1, morph_iters // 2))
        blue_mask = binary_opening(blue_mask_raw, structure=struct, iterations=morph_iters)
        blue_mask = binary_fill_holes(blue_mask)
        blue_mask = binary_closing(blue_mask, structure=struct, iterations=max(1, morph_iters // 2))
        yellow_mask = binary_opening(yellow_mask_raw, structure=struct, iterations=morph_iters)
        yellow_mask = binary_fill_holes(yellow_mask)
        yellow_mask = binary_closing(yellow_mask, structure=struct, iterations=max(1, morph_iters // 2))
        magenta_mask = binary_opening(magenta_mask_raw, structure=struct, iterations=morph_iters)
        magenta_mask = binary_fill_holes(magenta_mask)
        magenta_mask = binary_closing(magenta_mask, structure=struct, iterations=max(1, morph_iters // 2))
        
        # Combine all door-related masks and dilate
        door_mask_with_holes = blue_mask | yellow_mask | magenta_mask
        door_mask = binary_dilation(door_mask_with_holes, structure=struct, iterations=10)

        # 5. 3D Reconstruction
        z_c = depth_image
        x_c = z_c * (x_pixel - cx) / fx
        y_c = z_c * (y_pixel - cy) / fy
        points_camera = np.stack((x_c, y_c, z_c), axis=-1).reshape(-1, 3) 
        ones = np.ones((points_camera.shape[0], 1), dtype=points_camera.dtype)
        points_world_h = extrinsics @ np.concatenate([points_camera, ones], axis=1).T
        points_world = (points_world_h[:3,:] / np.clip(points_world_h[3,:], 1e-8, None)).T
        pc_world = points_world.reshape(height, width, 3)
        
        # 6. Cube Information Extraction (Red and Green)
        red_infos = []
        lab_red, n_red = label(red_mask.astype(np.uint8), structure=struct);
        red_clusters_info = sorted([(cid, (lab_red == cid).sum()) for cid in range(1, n_red + 1)], key=lambda item: item[1], reverse=True)
        for cid, area in red_clusters_info:
            idx_red = (lab_red == cid)
            pts_red = pc_world.reshape(-1, 3)[idx_red.flatten()]
            pts_red = pts_red[np.isfinite(pts_red).all(axis=1)]
            if pts_red.shape[0] == 0: continue
            top_z_red = float(np.nanpercentile(pts_red[:, 2], 95))
            low_z_red = float(np.nanpercentile(pts_red[:, 2], 5))
            top_slice_mask = pts_red[:, 2] > (top_z_red - 0.015) 
            pts_red_top = pts_red[top_slice_mask]
            
            if pts_red_top.shape[0] > 10: 
                centroid_xy_red = np.nanmedian(pts_red_top[:, :2], axis=0)
            else:
                centroid_xy_red = np.nanmedian(pts_red[:, :2], axis=0)

            stats = {
                "centroid_xy": centroid_xy_red, "top_z": top_z_red, "low_z": low_z_red, 
                "size_xy": np.nanpercentile(pts_red[:, :2], 95, axis=0) - np.nanpercentile(pts_red[:, :2], 5, axis=0)
            }
            red_infos.append({"cid": cid, "area": area, "stats": stats})

        green_infos = []
        lab_green, n_green = label(green_mask.astype(np.uint8), structure=struct); 
        green_clusters_info = sorted([(cid, (lab_green == cid).sum()) for cid in range(1, n_green + 1)], key=lambda item: item[1], reverse=True)
        for cid, area in green_clusters_info:
            idx_green = (lab_green == cid); 
            pts_green = pc_world.reshape(-1, 3)[idx_green.flatten()]; 
            pts_green = pts_green[np.isfinite(pts_green).all(axis=1)]
            if pts_green.shape[0] == 0: continue
            centroid_xy_green = np.nanmedian(pts_green[:, :2], axis=0); 
            top_z_green = float(np.nanpercentile(pts_green[:, 2], 95)); 
            low_z_green = float(np.nanpercentile(pts_green[:, 2], 5))
            stats = {"centroid_xy": centroid_xy_green, "top_z": top_z_green, "low_z": low_z_green, "size_xy": np.nanpercentile(pts_green[:, :2], 95, axis=0) - np.nanpercentile(pts_green[:, :2], 5, axis=0)}
            d_center = float(np.linalg.norm(stats["centroid_xy"] - np.array([0.5, 0.0])))
            green_infos.append({"cid": cid, "area": area, "stats": stats, "d_center": d_center})
        green_infos.sort(key=lambda it: (it["d_center"], -it["area"])); 
        
        # 7. Blocking Cube Identification (Projection)
        red_blocking_infos = []
        green_blocking_infos = []
        table_min_x, table_max_x = 0.15, 0.85
        table_min_y, table_max_y = -0.35, 0.35
        edge_margin = 0.12 # For safe cubes
        proximity_margin = 0.03 # For blocking cubes

        door_points_3d = pc_world[door_mask]
        door_points_3d = door_points_3d[np.isfinite(door_points_3d).all(axis=1)]
        is_door_points_available = door_points_3d.shape[0] > 0

        for info in red_infos:
            stats = info['stats']
            # 使用方块顶部的 Z 坐标作为质心的 Z 坐标进行投影和距离计算
            xyz_world = np.array([stats['centroid_xy'][0], stats['centroid_xy'][1], stats['top_z']])
            
            # 1. 投影检查 (Projection Check): 检查质心最近的点是否在 door_mask 上
            distances_sq = np.sum((pc_world - xyz_world)**2, axis=2)
            v, u = np.unravel_index(np.argmin(distances_sq), distances_sq.shape)
            is_blocking_projection = door_mask[v, u]
            
            # 2. 三维距离检查 (3D Proximity Check): 检查方块质心与门结构的最小 3D 距离
            is_blocking_proximity = False
            if is_door_points_available:
                # 计算方块质心到所有门点云的最小距离
                min_dist_to_door = np.min(np.linalg.norm(door_points_3d - xyz_world, axis=1))
                is_blocking_proximity = min_dist_to_door < proximity_margin

            # 如果满足任一条件，则视为阻塞方块
            if is_blocking_projection or is_blocking_proximity:
                red_blocking_infos.append(info)

        for info in green_infos:
            stats = info['stats']
            xyz_world = np.array([stats['centroid_xy'][0], stats['centroid_xy'][1], stats['top_z']])
            
            # 1. 投影检查 (Projection Check)
            distances_sq = np.sum((pc_world - xyz_world)**2, axis=2)
            v, u = np.unravel_index(np.argmin(distances_sq), distances_sq.shape)
            is_blocking_projection = door_mask[v, u]
            
            # 2. 三维距离检查 (3D Proximity Check)
            is_blocking_proximity = False
            if is_door_points_available:
                min_dist_to_door = np.min(np.linalg.norm(door_points_3d - xyz_world, axis=1))
                is_blocking_proximity = min_dist_to_door < proximity_margin

            # 如果满足任一条件，则视为阻塞方块
            if is_blocking_projection or is_blocking_proximity:
                green_blocking_infos.append(info)
        
                
        # 8. Safe Cube Filtering
        blocking_red_ids = [info['cid'] for info in red_blocking_infos]
        safe_red_infos = []
        for info in red_infos:
            if info['cid'] in blocking_red_ids: continue
            cx, cy = info['stats']['centroid_xy']
            is_on_edge = (cx < table_min_x + edge_margin) or \
                         (cx > table_max_x - edge_margin) or \
                         (cy < table_min_y + edge_margin) or \
                         (cy > table_max_y - edge_margin)
            if not is_on_edge: safe_red_infos.append(info)


        blocking_green_ids = [info['cid'] for info in green_blocking_infos]
        safe_green_infos = []
        for info in green_infos:
            if info['cid'] in blocking_green_ids: continue
            cx, cy = info['stats']['centroid_xy']
            is_on_edge = (cx < table_min_x + edge_margin) or \
                         (cx > table_max_x - edge_margin) or \
                         (cy < table_min_y + edge_margin) or \
                         (cy > table_max_y - edge_margin)
            if not is_on_edge: safe_green_infos.append(info)

        return red_blocking_infos, green_blocking_infos, safe_red_infos, safe_green_infos, door_mask, pc_world

    # === ADDED: Pick and Place Helper ===
    def _execute_pick_and_place(self, blocking_infos, safe_infos, cube_height=0.04):
        """
        Executes the pick-and-place sequence for a list of blocking cubes.
        """
       
        fixed_quat = np.array([0.0, 1.0, 0.0, 0.0])
        z_travel_height = 0.4
        z_hover_height = 0.4
        z_offset_grasp = 0.015
        z_offset_stack = 0.028
        
        num_to_move = min(len(blocking_infos), len(safe_infos))
        
        for i in range(num_to_move):
            block_cube = blocking_infos[i]
            target_base = safe_infos[i]
            
            print(f"  -> Moving Blocking Cube ID {block_cube['cid']} to Safe Cube ID {target_base['cid']}")
            
            # --- Coordinate Calculation ---
            pick_xy = block_cube['stats']['centroid_xy']
            pick_z = block_cube['stats']['top_z'] + z_offset_grasp
            
            place_xy = target_base['stats']['centroid_xy']
            # New Z is Safe Cube's top + Cube Height + Stack Offset
            place_z = target_base['stats']['top_z'] + cube_height + z_offset_stack
            
            # --- Action Sequence (Manhattan Path) ---
            
            # 1. Move to Pick Hover
            hover_pose = np.concatenate([pick_xy, [z_travel_height], fixed_quat])
            self.move_robot_ik(hover_pose)
            self.open_gripper()
            
            # 2. Vertical Descent to Pick
            pick_pose = np.concatenate([pick_xy, [pick_z], fixed_quat])
            self.move_robot_ik(pick_pose)
            

            # 3. Grasp
            self.close_gripper()
            self.sim_wait(5)

            # 4. Vertical Lift
            self.move_robot_ik(hover_pose)

            # 5. Move to Place Hover
            place_hover_pose = np.concatenate([place_xy, [z_travel_height], fixed_quat])
            self.move_robot_ik(place_hover_pose)
            
            # 6. Vertical Descent to Place
            place_pose = np.concatenate([place_xy, [place_z], fixed_quat])
            self.move_robot_ik(place_pose)

            # 7. Release
            self.open_gripper()
            self.sim_wait(5)

            # 8. Retreat
            self.move_robot_ik(place_hover_pose)

    # === ADDED: Main Logic from eval_mlp_door.py ===
    def run_door_policy(self):
        print("\n\n[DOOR POLICY] Starting Door Opening Phase...")
        
        # 1. Warm-up: Move robot to start position defined in eval_mlp_door
        start_pos = np.array([0.4, 0.0, 0.35]) 
        base_quat = np.array([0, -np.sqrt(2)/2, np.sqrt(2)/2, 0]) 
        
        start_pose = np.concatenate([start_pos, base_quat])
        
        print("[DOOR POLICY] Moving to policy start position...")
        self.move_robot_ik(start_pose, timeout_count=200)
        self.open_gripper() # Ensure gripper is open

        # 2. Variables for Policy Loop
        max_steps = 600
        temp_dist_target = 0.0018
        rot_step = 0.05
        max_joint_change = 0.10
        
        # State tracking
        current_yaw = 0.0 # Policy assumes we start at 0 relative yaw from base_quat
        gripper_state = -1 # -1 is Open
        is_rot_aligned = False
        # === ADDED: Gripper 目标值和渐进速度 ===
        self.gripper_step = 0.005 # 每次只移动 0.005m，实现慢速关闭
        current_gripper_target = self.gripper_open_val
        # ======================================
        # Smoothing Buffer
        history_len = 5
        action_history = deque(maxlen=history_len)

        print("[DOOR POLICY] Handing over control to Neural Network...")
        
        # 3. Policy Loop
        for step in range(max_steps):
            if not simulation_app.is_running(): break
            
            # --- A. Get Observations ---
            current_robot_pos = self.get_current_eef_pos()
            current_door_pos = self.get_door_pos()
            
            # Construct 8-dim input
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

            # --- B. Inference ---
            obs_tensor = torch.FloatTensor(obs_vector).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.policy_model(obs_tensor)

                # --- C. Apply Heuristic Masks (Copied exactly from eval script) ---
                # 1. Rotation Locking
                if abs(current_yaw) > 1.45: is_rot_aligned = True
                if is_rot_aligned:
                    logits[0, 8] = -1e9 # Block Stationary
                    logits[0, 9] = -1e9 # Block Rot +
                    logits[0, 10] = -1e9 # Block Rot -
                if current_robot_pos[2] > 0.275:
                    logits[0,7] = -1e20 # Block Close Gripper at High Altitude
                # # 2. Geometric Alignment Logic
                # diff_y = abs(current_robot_pos[1] - current_door_pos[1])
                # diff_x = abs(current_robot_pos[0] - current_door_pos[0]) 
                
                # is_x_aligned = diff_x > 0.20 # Reached edge
                # is_y_aligned = diff_y < 0.02 # Centered

                # # 3. High Altitude Logic
                # if not (is_x_aligned and is_y_aligned):
                #     # Phase A: Not aligned yet
                #     logits[0, 5] = -1e9 # Block Down (-Z)
                #     logits[0, 7] = -1e9 # Block Close Gripper

                #     if not is_x_aligned:
                #         # Block Y motion, focus on X
                #         logits[0, 2] = -1e9
                #         logits[0, 3] = -1e9
                # else:
                #     # Phase B: Aligned (Descent)
                #     if current_robot_pos[2] < 0.28:
                #         # Close to handle -> Encourage Grasp
                #         logits[0, 7] = 1e20  
                #     else:
                #         logits[0, 7] = -1e9 # Too high to grasp

                raw_action_idx = torch.argmax(logits, dim=1).item()

            # --- D. Smoothing ---
            action_history.append(raw_action_idx)
            if len(action_history) == history_len:
                final_action_idx = Counter(action_history).most_common(1)[0][0]
            else:
                final_action_idx = raw_action_idx

            if step % 10 == 0:
                 print(f"[DOOR STEP {step}] Act: {final_action_idx} | Yaw: {current_yaw:.2f} | Z: {current_robot_pos[2]:.3f}")

            # --- E. Execute Action (Calculate Target) ---
            target_pos = current_robot_pos.copy()
            
            # Map index to action
            if final_action_idx == 0: target_pos[0] += temp_dist_target
            elif final_action_idx == 1: target_pos[0] -= temp_dist_target
            elif final_action_idx == 2: target_pos[1] += temp_dist_target
            elif final_action_idx == 3: target_pos[1] -= temp_dist_target
            elif final_action_idx == 4: target_pos[2] += temp_dist_target
            elif final_action_idx == 5: target_pos[2] -= temp_dist_target
            elif final_action_idx == 6: 
                gripper_state = -1 # Open
                current_gripper_target = self.gripper_open_val # 立即完全打开
            elif final_action_idx == 7: 
                gripper_state = 1  # Close
                # 渐进式关闭: 每次只向目标值靠近 self.gripper_step
                if current_gripper_target > self.gripper_close_val:
                    current_gripper_target -= self.gripper_step
                    current_gripper_target = max(current_gripper_target, self.gripper_close_val)
            elif final_action_idx == 8: pass # Stationary
            elif final_action_idx == 9: current_yaw += rot_step 
            elif final_action_idx == 10: current_yaw -= rot_step 

            # Calculate Orientation
            base_rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]) 
            z_rot = R.from_euler('z', current_yaw)
            final_rot = base_rot * z_rot
            final_quat_scipy = final_rot.as_quat() 
            target_quat = np.array([final_quat_scipy[3], final_quat_scipy[0], final_quat_scipy[1], final_quat_scipy[2]])

            target_pose = np.concatenate([target_pos, target_quat])
            
            # --- F. Low-Level Control (Non-Blocking IK) ---
            self.diff_ik_controller.set_command(torch.tensor(target_pose, device=self.sim.device))

            jacobian = self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
            ee_pose_w = self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
            joint_pos = self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]
            
            # Compute IK
            joint_pos_des = self.diff_ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos)

            # Limit joint changes (Safety)
            joint_changes = (joint_pos_des - joint_pos).detach().cpu().numpy()[0]
            if np.sum(np.abs(joint_changes) > max_joint_change) > 0:
                scaled_joint_changes = joint_changes / (np.max(np.abs(joint_changes)) / max_joint_change)
                scaled_joint_changes = torch.tensor(scaled_joint_changes).unsqueeze(0).to(joint_pos_des.device)
                joint_pos_des = joint_pos + scaled_joint_changes

            # Construct full command (Arm + Gripper)
            all_joint_pos_des = torch.zeros((1, 8))
            all_joint_pos_des[:, :6] = joint_pos_des
            
            # Handle Gripper
            # if gripper_state == -1:
            #     all_joint_pos_des[:, 6:] = torch.tensor([self.gripper_open_val, self.gripper_open_val]).to(self.sim.device)
            # else:
            #     all_joint_pos_des[:, 6:] = torch.tensor([self.gripper_close_val, self.gripper_close_val]).to(self.sim.device)
            gripper_val = current_gripper_target
            all_joint_pos_des[:, 6:] = torch.tensor([gripper_val, gripper_val]).to(self.sim.device)
            # Apply
            self.scene["ur5e"].set_joint_position_target(all_joint_pos_des)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        print("[DOOR POLICY] Max steps reached or app closed.")


    def run (self):
        '''
        You code goes here.
        '''
    
        # Reset the environment (Relies on default states defined in task_envs.py)
        self.scene.reset()
        # fixed_quat = self.robot_quat.copy()
        # Update internal robot pose trackers after reset
        self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]
        fixed_quat = np.array([0.0, 1.0, 0.0, 0.0])
        # birdview_camera intrinsic and extrinsic matrix
        intrinsics = np.squeeze(self.scene["birdview_camera"].data.intrinsic_matrices.detach().cpu().numpy())
        extrinsics = np.array([
            [ 0,  1,  0, 0.5],
            [ 1,  0,  0,   0],
            [ 0,  0, -1, 1.2],
            [ 0,  0,  0,   1]
        ])

        # move the robot out of the way for getting information from birdview camera
        self.sim_wait(20)
        
        # Get the robot's starting X and Y position
        lift_pos = self.robot_pos.copy() 
        lift_pos[2] = 0.5
        
        # Execute the vertical lift
        self.move_robot_ik(np.concatenate([lift_pos, fixed_quat]))
        self.sim_wait(20) # Wait for the lift to complete

        # Define the final "away" position
        away_pos = np.array([-0.2, 0.0, 0.5])
        
        # Execute the horizontal move
        self.move_robot_ik(np.concatenate([away_pos, fixed_quat]))
        self.sim_wait(20) # Wait for the move to complete

        # Setup 2D to 3D mapping constants
        height, width = 256, 256 # Assuming fixed resolution
        y_pixel, x_pixel = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        fx = intrinsics[0,0]; fy = intrinsics[1,1]
        cx = intrinsics[0,2]; cy = intrinsics[1,2]

        # === TWO-PASS CUBE CLEARING LOGIC ===
        MAX_PASSES = 2
        
        for pass_num in range(1, MAX_PASSES + 1):
            print(f"\n====================================================")
            print(f"========= STARTING CUBE CLEARING PASS {pass_num} =========")
            print(f"====================================================")

            # 1. RE-IDENTIFY CUBES
            (red_blocking_infos, green_blocking_infos, 
             safe_red_infos, safe_green_infos, door_mask, pc_world) = \
                self._identify_blocking_cubes(intrinsics, extrinsics, x_pixel, y_pixel, cx, cy, fx, fy, height, width)

            print(f"Pass {pass_num} Scan Results:")
            print(f"  Red Blocking: {len(red_blocking_infos)} | Safe Red: {len(safe_red_infos)}")
            print(f"  Green Blocking: {len(green_blocking_infos)} | Safe Green: {len(safe_green_infos)}")

            # If no cubes are blocking, break the loop early (only if pass_num == 1)
            if len(red_blocking_infos) == 0 and len(green_blocking_infos) == 0:
                print("No cubes blocking the door. Moving to next phase.")
                break
                
            # 2. RED CUBE PICK-AND-PLACE
            print(f"\n--- Red Cube Movement (Pass {pass_num}) ---")
            if len(red_blocking_infos) > 0 and len(safe_red_infos) > 0:
                self._execute_pick_and_place(red_blocking_infos, safe_red_infos, cube_height=0.04)
            else:
                print("Skipping Red Cube movement (not enough blocking or safe cubes).")

            # 3. GREEN CUBE PICK-AND-PLACE
            print(f"\n--- Green Cube Movement (Pass {pass_num}) ---")
            if len(green_blocking_infos) > 0 and len(safe_green_infos) > 0:
                # Green cubes are slightly larger (approx 0.05m high)
                self._execute_pick_and_place(green_blocking_infos, safe_green_infos, cube_height=0.05)
            else:
                print("Skipping Green Cube movement (not enough blocking or safe cubes).")

            print(f"\n========= PASS {pass_num} COMPLETED =========")
            self.move_robot_ik(np.concatenate([away_pos, self.robot_quat])) # Move away for next scan
            self.sim_wait(20)

        # 4. FINAL CHECK (Only performed if we did not break early)
        if len(red_blocking_infos) > 0 or len(green_blocking_infos) > 0:
             print("\n====================================================")
             print("=== FINAL CUBE CHECK AFTER TWO PASSES ===")
             print("====================================================")
             
             # Re-identify one final time to determine success/failure of clearing
             (red_blocking_infos_final, green_blocking_infos_final, 
              _, _, door_mask, pc_world) = \
                 self._identify_blocking_cubes(intrinsics, extrinsics, x_pixel, y_pixel, cx, cy, fx, fy, height, width)

             final_blocking_count = len(red_blocking_infos_final) + len(green_blocking_infos_final)

             if final_blocking_count == 0:
                 print("\nSUCCESS: All blocking cubes removed! Door area is clear.")
             else:
                 print(f"\nFAILURE: {final_blocking_count} cubes are still blocking the door.")

        # 5. EXECUTE DOOR POLICY
        self.run_door_policy()    
        
        # steps simulation but does not command the robot
        while simulation_app.is_running():

            # step environment
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        # this helps shut down the script correctly
        simulation_app.close()
        

if __name__ == "__main__":

    exp = Experiment()
    exp.run()