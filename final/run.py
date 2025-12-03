import argparse

from isaaclab.app import AppLauncher

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
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene

# from scipy.spatial.transform import Rotation
from scipy.ndimage import ( # Needed for advanced image processing
    binary_opening, binary_closing, binary_fill_holes,
    generate_binary_structure, label,binary_dilation,
)

from task_envs import MP2SceneCfg, PHYSICS_DT, RENDERING_DT


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
            count=25, # Use a few steps to make it a smooth open
            time_for_residual_movement=5
        )
        
    def close_gripper(self):
        """Commands the gripper to close using joint control."""
        # Calls your existing function to move ONLY the gripper
        self.move_robot_joint(
            target_joint_pos=None,  # Arm doesn't move
            target_gripper_pos=self.gripper_close_val,
            count=25, # Use a few steps to make it a smooth close
            time_for_residual_movement=5
        )

    def run (self):
        '''
        You code goes here.
        '''
    
        # Reset the environment (Relies on default states defined in task_envs.py)
        self.scene.reset()
        fixed_quat = self.robot_quat.copy()
        # Update internal robot pose trackers after reset
        self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]
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
        # Set the target Z-height to 0.5 (this lifts the arm)
        lift_pos[2] = 0.5
        
        # Execute the vertical lift
        self.move_robot_ik(np.concatenate([lift_pos, self.robot_quat]))
        self.sim_wait(20) # Wait for the lift to complete

        # Define the final "away" position (already at Z=0.5)
        away_pos = np.array([-0.2, 0.0, 0.5])
        
        # Execute the horizontal move
        self.move_robot_ik(np.concatenate([away_pos, self.robot_quat]))
        self.sim_wait(20) # Wait for the move to complete

        # render birdview camera image
        color_raw = self.scene["birdview_camera"].data.output["rgb"].detach().cpu().numpy()[0]
        #plt.imshow(color)
        
        # h_ : Hue ranges for detecting red and green in the HSV color space
        
        h_red_low_1 = 0.00
        h_red_high_1 = 0.04
        h_red_low_2 = 0.96
        h_red_high_2 = 1.00
        h_green_low = 0.23
        h_green_high = 0.44

        # Blue range (approx 210-270 degrees)
        h_blue_low = 0.58
        h_blue_high = 0.75
        # Yellow range (approx 50-70 degrees)
        h_yellow_low = 0.14  # Was 0.13
        h_yellow_high = 0.18 # Was 0.20
        
        h_magenta_low = 0.82
        h_magenta_high = 0.88

        # minimum thresholds for Saturation (S) and Value (V)
        s_min = 0.40
        v_min = 0.20
        # The number of iterations for morphological image processing operations, which are used to clean up noise
        morph_iters = 2
        hover_offset = 0.12
        lift_offset = 0.16
        grasp_depth = 0.02
        # Ensures the image is in a standard RGB format
        color = color_raw[:, :, :3] if color_raw.shape[2] == 4 else color_raw
        
        height, width, _ = color.shape
        # distance of each pixel from the camera, remove any extra,single-value dimensions
        depth_image = np.squeeze(self.scene["birdview_camera"].data.output["depth"].detach().cpu().numpy()[0])
        # Convert RGB to HSV color space
        rgb01 = np.clip(color.astype(np.float32)/255.0, 0.0, 1.0) # Normalize it
        hsv = mcolors.rgb_to_hsv(rgb01)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # initial binary masks
        red1_raw = (h >= h_red_low_1) & (h <= h_red_high_1) & (s >= s_min) & (v >= v_min)
        red2_raw = (h >= h_red_low_2) & (h <= h_red_high_2) & (s >= s_min) & (v >= v_min)
        red_mask_raw = red1_raw | red2_raw
        green_mask_raw = (h >= h_green_low) & (h <= h_green_high) & (s >= s_min) & (v >= v_min)
        blue_mask_raw = (h >= h_blue_low) & (h <= h_blue_high) & (s >= s_min) & (v >= v_min)
        yellow_mask_raw = (h >= h_yellow_low) & (h <= h_yellow_high) & (s >= s_min) & (v >= v_min)
        magenta_mask_raw = (h >= h_magenta_low) & (h <= h_magenta_high) & (s >= s_min) & (v >= v_min)

        struct = generate_binary_structure(2, 2)
        # removes small, isolated bright spots
        red_mask = binary_opening(red_mask_raw, structure=struct, iterations=morph_iters)
        # fills any black holes inside a larger white object
        red_mask = binary_fill_holes(red_mask)
        # fills small gaps and holes within an object
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

        # Visualizing the 2D Masks
        fig, axs = plt.subplots(1, 6, figsize=(30,5))
        axs[0].imshow(color)
        axs[0].set_title('Original Image')
        axs[1].imshow(red_mask, cmap='gray')
        axs[1].set_title('Red Cube Mask')
        axs[2].imshow(green_mask, cmap='gray')
        axs[2].set_title('Green Cube Mask')
        axs[3].imshow(blue_mask, cmap='gray')
        axs[3].set_title('Blue Frame Mask')
        
        axs[4].imshow(yellow_mask, cmap='gray')
        axs[4].set_title('Yellow Panel Mask')

        axs[5].imshow(magenta_mask,cmap='gray')
        axs[5].set_title('Magenta Handle Mask')

        plt.savefig("color_masks.png") # Save the figure instead
        plt.close()

        # Creating a 3D Point Cloud
        height, width = depth_image.shape
        
        # focal lengths of the camera lens
        fx = intrinsics[0,0]
        fy = intrinsics[1,1]
        # principal point
        cx = intrinsics[0,2]
        cy = intrinsics[1,2]
        # row index (y-coordinate) and column index (x-coordinate) at each position
        y_pixel, x_pixel = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        # Turns 2D pixels into 3D points in the camera's coordinate system.
        z_c = depth_image
        x_c = z_c * (x_pixel - cx) / fx
        y_c = z_c * (y_pixel - cy) / fy

        points_camera = np.stack((x_c, y_c, z_c), axis=-1).reshape(-1, 3) 
        # transforms the points from the camera's coordinate system to the global world coordinate system.
        ones = np.ones((points_camera.shape[0], 1), dtype=points_camera.dtype)
        points_world_h = extrinsics @ np.concatenate([points_camera, ones], axis=1).T
            
        points_world = (points_world_h[:3,:] / np.clip(points_world_h[3,:], 1e-8, None)).T
        pc_world = points_world.reshape(height, width, 3)
        
        # flatten the 2D masks into 1D arrays
        red_pixels = red_mask.flatten().astype(bool)
        green_pixels = green_mask.flatten().astype(bool)
        blue_pixels = blue_mask.flatten().astype(bool)    # Added blue
        yellow_pixels = yellow_mask.flatten().astype(bool)  # Added yellow
        magenta_pixels = magenta_mask.flatten().astype(bool) # Added magenta
    
        point_colors = np.full((points_world.shape[0], 3), [0.6, 0.6, 0.6])
        point_colors[red_pixels] = [1, 0, 0]
        point_colors[green_pixels] = [0, 1, 0]
        point_colors[blue_pixels] = [0, 0, 1]      # Blue
        point_colors[yellow_pixels] = [1, 1, 0]    # Yellow
        point_colors[magenta_pixels] = [1, 0, 1]

        fig = plt.figure()
        scene = fig.add_subplot(projection="3d")
        scene.scatter(
            points_world[:,0],
            points_world[:,1],
            points_world[:,2],
            c=point_colors,
            s=1.0
        )
        scene.set_xlabel("X World (m)")
        scene.set_ylabel("Y World (m)")
        scene.set_zlabel("Z World (m)")
        scene.set_title("Scene Point Cloud")
        scene.set_aspect("equal")
        scene.view_init(elev=60, azim=-45)

        plt.savefig("scene_point_cloud.png")
        plt.close(fig)

        # Scans the mask and finds all separate, contiguous white regions, assigning a unique ID to each one.
        lab_red, n_red = label(red_mask.astype(np.uint8), structure=struct);
        
        red_infos = []
        red_clusters_info = sorted([(cid, (lab_red == cid).sum()) for cid in range(1, n_red + 1)], key=lambda item: item[1], reverse=True)

        for cid, area in red_clusters_info:
            idx_red = (lab_red == cid)
            pts_red = pc_world.reshape(-1, 3)[idx_red.flatten()]
            pts_red = pts_red[np.isfinite(pts_red).all(axis=1)]
            if pts_red.shape[0] == 0: continue
            
            # 1. 先算出最高点 (Top Z)
            top_z_red = float(np.nanpercentile(pts_red[:, 2], 95))
            low_z_red = float(np.nanpercentile(pts_red[:, 2], 5))
            
            # 2. 【关键修正】只取顶部 1.5cm 范围内的点来计算 XY 中心
            # 这样可以排除掉侧面点对中心的干扰
            top_slice_mask = pts_red[:, 2] > (top_z_red - 0.015) 
            pts_red_top = pts_red[top_slice_mask]
            
            if pts_red_top.shape[0] > 10: # 确保有足够的点
                centroid_xy_red = np.nanmedian(pts_red_top[:, :2], axis=0)
            else:
                # 如果顶部点太少（异常情况），退回使用所有点
                centroid_xy_red = np.nanmedian(pts_red[:, :2], axis=0)

            stats = {
                "centroid_xy": centroid_xy_red, 
                "top_z": top_z_red, 
                "low_z": low_z_red, 
                "size_xy": np.nanpercentile(pts_red[:, :2], 95, axis=0) - np.nanpercentile(pts_red[:, :2], 5, axis=0)
            }
            
            red_infos.append({"cid": cid, "area": area, "stats": stats})

        # red_edge = np.clip(float(np.mean(np.abs(red_stats["size_xy"]))), 0.03, 0.06)
        red_edge = 0.04

        lab_green, n_green = label(green_mask.astype(np.uint8), structure=struct); 
        green_infos = []
        # Finds all green objects and sorts them by their area in descending order
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
            # calculates the Euclidean distance from the current green object's centroid to a fixed reference point in the world ([0.5, 0.0])
            d_center = float(np.linalg.norm(stats["centroid_xy"] - np.array([0.5, 0.0])))
            green_infos.append({"cid": cid, "area": area, "stats": stats, "d_center": d_center})

        # Sort by d_center in ascending order, 2. Sorting by area in descending order.
        green_infos.sort(key=lambda it: (it["d_center"], -it["area"])); 
        # first item in the list (green_infos[0]) is designated as the base block
        base_info, top_info = green_infos[0], green_infos[1]
        base_stats, top_stats = base_info["stats"], top_info["stats"]
        # green_edge = np.mean([np.clip(float(np.mean(np.abs(s["size_xy"]))), 0.03, 0.06) for s in [base_stats, top_stats]])
        green_edge = 0.05
        # Estimating the Table Height
        z_table_est = float(np.nanpercentile(pc_world[..., 2].reshape(-1), 1.0))

        # Combine all door-related masks to get the full door's 2D footprint
        door_mask_with_holes = blue_mask | yellow_mask | magenta_mask
        
        # We "grow" the mask to fill in gaps. 10 iterations is a good starting point.
        struct = generate_binary_structure(2, 2)
        door_mask = binary_dilation(door_mask_with_holes, structure=struct, iterations=10)
        
        print("Checking for blocking cubes using centroid projection...")
        
        # Create separate lists and masks
        red_blocking_infos = []
        green_blocking_infos = []
        red_blocking_mask = np.zeros_like(red_mask, dtype=bool) 
        green_blocking_mask = np.zeros_like(green_mask, dtype=bool) 
        
        # Get image dimensions for boundary checks
        height, width = door_mask.shape

        # Check all red cubes
        for info in red_infos:
            stats = info['stats']
            # Get the cube's 3D world centroid (X, Y, Z)
            xyz_world = np.array([stats['centroid_xy'][0], stats['centroid_xy'][1], stats['top_z']])
            
            # 1. Calculate the squared distance from the centroid to EVERY 3D point in the point cloud
            distances_sq = np.sum((pc_world - xyz_world)**2, axis=2)
            
            # 2. Find the (v, u) pixel coordinates of the closest point
            v, u = np.unravel_index(np.argmin(distances_sq), distances_sq.shape)

            # Check if that pixel (v, u) is on the door
            if door_mask[v, u]:
                print(f"  Found blocking RED cube (CID: {info['cid']})")
                red_blocking_infos.append(info)
                red_blocking_mask |= (lab_red == info['cid'])

        # Check all green cubes
        for info in green_infos:
            stats = info['stats']
            # Get the cube's 3D world centroid (X, Y, Z)
            xyz_world = np.array([stats['centroid_xy'][0], stats['centroid_xy'][1], stats['top_z']])

            # Find the closest pixel 
            distances_sq = np.sum((pc_world - xyz_world)**2, axis=2)
            v, u = np.unravel_index(np.argmin(distances_sq), distances_sq.shape)
            
            # Check if that pixel (v, u) is on the door
            if door_mask[v, u]:
                print(f"  Found blocking GREEN cube (CID: {info['cid']})")
                green_blocking_infos.append(info)
                green_blocking_mask |= (lab_green == info['cid'])

        # Visualize the new Blocking Masks
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(color)
        axs[0].set_title('Original Image')
        
        axs[1].imshow(red_blocking_mask, cmap='gray')
        axs[1].set_title('Red Blocking Cubes')
        
        axs[2].imshow(green_blocking_mask, cmap='gray')
        axs[2].set_title('Green Blocking Cubes')
        
        plt.savefig("blocking_cubes_masks.png")
        plt.close(fig)
        
        print(f"Found {len(red_blocking_infos)} red and {len(green_blocking_infos)} green blocking cubes.")

        # 1. 找到 Door Mask 的 3D 中心 (Find 3D center of the door)
        # 获取 door_mask 中所有像素的坐标
        door_indices = np.nonzero(door_mask)
        # 从 pc_world 中提取这些像素对应的 3D 点
        door_points_3d = pc_world[door_indices]
        # 过滤掉无效点 (inf/nan)
        door_points_3d = door_points_3d[np.isfinite(door_points_3d).all(axis=1)]
        door_center_3d = np.mean(door_points_3d, axis=0)
        print(f"Door Center 3D: {door_center_3d}")
        plt.figure()
        plt.imshow(door_mask, cmap='gray')
        plt.title('Door Mask (Dilated)')
        plt.savefig("door_mask.png")
        plt.close()
        print("Saved door_mask.png")

        blocking_ids = [info['cid'] for info in red_blocking_infos]
        
        # === 新增：定义桌子边界和安全边距 ===
        # 根据你提供的数据
        table_min_x, table_max_x = 0.15, 0.85
        table_min_y, table_max_y = -0.35, 0.35
        # 设置安全边距，例如 5cm (0.05m)，保证方块中心离边缘至少有这么远
        edge_margin = 0.12

        # === 修改：筛选 Safe Cubes (非阻挡 + 不在边缘) ===
        safe_red_infos = []
        for info in red_infos:
            # 1. 排除阻挡方块
            if info['cid'] in blocking_ids:
                continue
            
            # 2. 检查是否在桌子边缘
            cx, cy = info['stats']['centroid_xy']
            is_on_edge = (cx < table_min_x + edge_margin) or \
                         (cx > table_max_x - edge_margin) or \
                         (cy < table_min_y + edge_margin) or \
                         (cy > table_max_y - edge_margin)
            
            if is_on_edge:
                print(f"  Skipping unsafe cube ID {info['cid']} (Too close to edge: {cx:.2f}, {cy:.2f})")
                continue
            
            # 如果既不是阻挡，又不在边缘，则为安全
            safe_red_infos.append(info)
        # 定义距离计算函数
        
        print(f"Safe Red Cubes available: {len(safe_red_infos)}")

        # 3. 搬运循环 (Pick and Place Loop)
        # 我们只能搬运 min(blocking数量, safe数量) 个物体
        num_to_move = min(len(red_blocking_infos), len(safe_red_infos))
        
         
        
        # 定义高度参数
        z_travel_height = 0.35  # 搬运过程中的高空飞行高度 (非常安全)
        z_hover_height = 0.4   # 准备抓取/放置时的悬停高度
        
        # 抓取高度微调 (根据你之前的需求，这里设为负数表示“悬空抓取”，正数表示“压下去抓”)
        # 如果你想“垂直下降再抓住”，建议设为 0 或者 -0.005 (稍微接触表面)
        z_offset_grasp = 0.012 
        z_offset_stack = 0.025

        for i in range(num_to_move):
            block_cube = red_blocking_infos[i]
            target_base = safe_red_infos[i]
            
            print(f"Moving Blocking Cube ID {block_cube['cid']} to Safe Cube ID {target_base['cid']}")
            
            # --- 准备抓取数据 ---
            # 抓取位置
            pick_xy = block_cube['stats']['centroid_xy']
            pick_z = block_cube['stats']['top_z'] +z_offset_grasp
            
            # 放置位置 (目标是 Safe Cube 的顶部)
            place_xy = target_base['stats']['centroid_xy']
            place_z = target_base['stats']['top_z'] + 0.04 + z_offset_stack
            
            # --- 动作序列 ---
         

            # 2. 提升到安全高度 (Lift to Travel Height)
            # 先在当前位置垂直抬升，避免横向撞击
            current_xy = self.robot_pos[:2]
            # self.move_robot_ik(np.concatenate([current_xy, [z_travel_height], fixed_quat]))

            # 3. 平移到目标正上方 (Move XY to Hover)
            # 在高空平移，此时 Z 轴不变，只变 XY
            hover_pose = np.concatenate([pick_xy, [z_travel_height], fixed_quat])
            self.move_robot_ik(hover_pose)
            self.open_gripper()
            
            # 如果需要，可以再降到一个较低的悬停点 (Optional Pre-grasp Hover)
            pre_grasp_pose = np.concatenate([pick_xy, [z_hover_height], fixed_quat])
            # self.move_robot_ik(pre_grasp_pose)

            # 4. 垂直下降 (Vertical Descent)
            # 这里的关键是：XY 坐标不变，只改变 Z，且使用 fixed_quat 锁死姿态
            print("Descending vertically...")
            pick_pose = np.concatenate([pick_xy, [pick_z], fixed_quat])
            self.move_robot_ik(pick_pose)
            self.sim_wait(10) # 等待稳定

            # 5. 抓取 (Grasp)
            self.close_gripper()
            self.sim_wait(15)

            # 6. 垂直抬起 (Vertical Lift)
            # self.move_robot_ik(pre_grasp_pose)      # 回到低悬停点
            self.move_robot_ik(hover_pose)          # 回到高空飞行点

            # 7. 平移到放置点正上方 (Move to Place)
            place_hover_pose = np.concatenate([place_xy, [z_travel_height], fixed_quat])
            self.move_robot_ik(place_hover_pose)
            
            place_pre_pose = np.concatenate([place_xy, [z_hover_height], fixed_quat])
            # self.move_robot_ik(place_pre_pose)

            # 8. 垂直下降放置 (Vertical Descent to Place)
            place_pose = np.concatenate([place_xy, [place_z], fixed_quat])
            self.move_robot_ik(place_pose)
            self.sim_wait(15)

            # 9. 松开 (Release)
            self.open_gripper()
            self.sim_wait(20)

            # 10. 垂直撤离 (Retreat)
            self.move_robot_ik(place_hover_pose)
            # self.move_robot_ik(place_hover_pose)
        print("Red Cube clearing completed.")

        blocking_green_ids = [info['cid'] for info in green_blocking_infos]
        
        # 1. 筛选 Safe Green Cubes (非阻挡 + 不在边缘)
        safe_green_infos = []
        for info in green_infos:
            if info['cid'] in blocking_green_ids:
                continue
            
            cx, cy = info['stats']['centroid_xy']
            # 使用之前定义的边界参数
            is_on_edge = (cx < table_min_x + edge_margin) or \
                         (cx > table_max_x - edge_margin) or \
                         (cy < table_min_y + edge_margin) or \
                         (cy > table_max_y - edge_margin)
            
            if is_on_edge:
                print(f"  Skipping unsafe green cube ID {info['cid']} (Too close to edge)")
                continue
            
            safe_green_infos.append(info)

        
        print(f"Safe Green Cubes available: {len(safe_green_infos)}")

        # 3. 搬运循环
        num_green_to_move = min(len(green_blocking_infos), len(safe_green_infos))
        
        # 绿色方块通常比红色略大 (Red=0.04, Green=0.05)
        # 这个高度用于计算堆叠时的 Z 轴偏移
        green_cube_height = 0.05 

        for i in range(num_green_to_move):
            block_cube = green_blocking_infos[i]
            target_base = safe_green_infos[i]
            
            print(f"Moving Blocking Green Cube ID {block_cube['cid']} to Safe Green ID {target_base['cid']}")
            
            # --- 坐标计算 ---
            pick_xy = block_cube['stats']['centroid_xy']
            # 抓取高度保持一致
            pick_z = block_cube['stats']['top_z'] + z_offset_grasp 
            
            place_xy = target_base['stats']['centroid_xy']
            # 【注意】这里把 0.04 改为了 green_cube_height (0.05)，适应绿色方块的高度
            place_z = target_base['stats']['top_z'] + green_cube_height + z_offset_stack
            
            # --- 动作序列 (Manhattan Path) ---
            
            # 1. 移动到抓取点上方 (Hover)
            hover_pose = np.concatenate([pick_xy, [z_travel_height], fixed_quat])
            self.move_robot_ik(hover_pose)
            self.open_gripper()
            
            # 2. 垂直下降抓取 (Descend to Pick)
            print("Descending vertically (Green)...")
            pick_pose = np.concatenate([pick_xy, [pick_z], fixed_quat])
            self.move_robot_ik(pick_pose)
            self.sim_wait(10)

            # 3. 闭合夹爪 (Grasp)
            self.close_gripper()
            self.sim_wait(15)

            # 4. 垂直抬起 (Lift)
            self.move_robot_ik(hover_pose)

            # 5. 平移到放置点上方 (Move to Place Hover)
            place_hover_pose = np.concatenate([place_xy, [z_travel_height], fixed_quat])
            self.move_robot_ik(place_hover_pose)
            
            # 6. 垂直下降放置 (Descend to Place)
            place_pose = np.concatenate([place_xy, [place_z], fixed_quat])
            self.move_robot_ik(place_pose)
            self.sim_wait(15)

            # 7. 松开 (Release)
            self.open_gripper()
            self.sim_wait(20)

            # 8. 撤离 (Retreat)
            self.move_robot_ik(place_hover_pose)

        print("Green Cube clearing completed.")
        print("\n=== Performing Final Check ===")
        
        # 1. 把机器人移开 (Move robot away to clear view)
        self.move_robot_ik(np.concatenate([away_pos, self.robot_quat]))
        self.sim_wait(20)

        # 2. 重新获取图像数据 (Capture new data)
        color_raw = self.scene["birdview_camera"].data.output["rgb"].detach().cpu().numpy()[0]
        color = color_raw[:, :, :3] if color_raw.shape[2] == 4 else color_raw
        depth_image = np.squeeze(self.scene["birdview_camera"].data.output["depth"].detach().cpu().numpy()[0])
        
        # 3. 重新计算 Mask (Re-calculate Masks using existing thresholds)
        rgb01 = np.clip(color.astype(np.float32)/255.0, 0.0, 1.0)
        hsv = mcolors.rgb_to_hsv(rgb01)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Red Mask Check
        red1_check = (h >= h_red_low_1) & (h <= h_red_high_1) & (s >= s_min) & (v >= v_min)
        red2_check = (h >= h_red_low_2) & (h <= h_red_high_2) & (s >= s_min) & (v >= v_min)
        red_mask_check = red1_check | red2_check
        red_mask_check = binary_opening(red_mask_check, structure=struct, iterations=morph_iters)
        red_mask_check = binary_fill_holes(red_mask_check)
        red_mask_check = binary_closing(red_mask_check, structure=struct, iterations=max(1, morph_iters // 2))

        # Green Mask Check
        green_mask_check = (h >= h_green_low) & (h <= h_green_high) & (s >= s_min) & (v >= v_min)
        green_mask_check = binary_opening(green_mask_check, structure=struct, iterations=morph_iters)
        green_mask_check = binary_fill_holes(green_mask_check)
        green_mask_check = binary_closing(green_mask_check, structure=struct, iterations=max(1, morph_iters // 2))

        # 4. 重新生成点云 (Re-generate Point Cloud)
        z_c = depth_image
        x_c = z_c * (x_pixel - cx) / fx
        y_c = z_c * (y_pixel - cy) / fy
        points_camera = np.stack((x_c, y_c, z_c), axis=-1).reshape(-1, 3) 
        ones = np.ones((points_camera.shape[0], 1), dtype=points_camera.dtype)
        points_world_h = extrinsics @ np.concatenate([points_camera, ones], axis=1).T
        points_world = (points_world_h[:3,:] / np.clip(points_world_h[3,:], 1e-8, None)).T
        pc_world_check = points_world.reshape(height, width, 3)

        # 5. 检查剩余的 Blocking Cubes (Check remaining)
        final_blocking_count = 0
        
        # Check Red
        lab_red_check, n_red_check = label(red_mask_check.astype(np.uint8), structure=struct)
        for cid in range(1, n_red_check + 1):
            idx = (lab_red_check == cid)
            pts = pc_world_check.reshape(-1, 3)[idx.flatten()]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if pts.shape[0] == 0: continue
            
            # 计算新的质心
            centroid_xy = np.nanmedian(pts[:, :2], axis=0)
            centroid_z = float(np.nanpercentile(pts[:, 2], 95))
            
            # 投影回像素坐标
            xyz_world = np.array([centroid_xy[0], centroid_xy[1], centroid_z])
            distances_sq = np.sum((pc_world_check - xyz_world)**2, axis=2)
            v, u = np.unravel_index(np.argmin(distances_sq), distances_sq.shape)
            
            # 检查是否还在门上 (Reuse door_mask)
            if door_mask[v, u]:
                print(f"  [WARNING] Found remaining RED blocking cube!")
                final_blocking_count += 1

        # Check Green
        lab_green_check, n_green_check = label(green_mask_check.astype(np.uint8), structure=struct)
        for cid in range(1, n_green_check + 1):
            idx = (lab_green_check == cid)
            pts = pc_world_check.reshape(-1, 3)[idx.flatten()]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if pts.shape[0] == 0: continue
            
            centroid_xy = np.nanmedian(pts[:, :2], axis=0)
            centroid_z = float(np.nanpercentile(pts[:, 2], 95))
            
            xyz_world = np.array([centroid_xy[0], centroid_xy[1], centroid_z])
            distances_sq = np.sum((pc_world_check - xyz_world)**2, axis=2)
            v, u = np.unravel_index(np.argmin(distances_sq), distances_sq.shape)
            
            if door_mask[v, u]:
                print(f"  [WARNING] Found remaining GREEN blocking cube!")
                final_blocking_count += 1

        # 6. 输出最终结果
        if final_blocking_count == 0:
            print(f"\nSUCCESS: All blocking cubes removed! Door area is clear.")
        else:
            print(f"\nFAILURE: {final_blocking_count} cubes are still blocking the door.")
            
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
