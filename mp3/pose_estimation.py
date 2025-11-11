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
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene

from task_envs import MP3PoseEstimSceneCfg, PHYSICS_DT, RENDERING_DT

import open3d as o3d


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
        scene_cfg = MP3PoseEstimSceneCfg(args_cli.num_envs, env_spacing=2.0)
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
        self.robot_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]

        self.world_point_clouds = []


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

        initial = self.scene["ur5e"].data.joint_pos.clone()
        init_joint_pos = self.scene["ur5e"].data.joint_pos[:, :6].squeeze()
        init_gripper_pos = self.scene["ur5e"].data.joint_pos[:, 6:].squeeze()
        target = self.scene["ur5e"].data.joint_pos.clone()

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

        print("Movement completed. Deviation: {}".format((target - self.scene["ur5e"].data.joint_pos).squeeze().detach().cpu().numpy()))

        # update robot pose
        self.robot_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]
    

    def move_robot_ik (self, target_pose, max_joint_change = 0.04, ik_tol = 1e-3, timeout_count = 200):
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
            # compare rotation matrices for orientation
            cur_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
            cur_quat = cur_pose[3:]
            cur_rot_matrix = R.from_quat([cur_quat[1], cur_quat[2], cur_quat[3], cur_quat[0]]).as_matrix()
            target_quat = target_pose[3:]
            target_rot_matrix = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_matrix()

            if np.average(np.abs(target_pose - cur_pose)[:3]) < ik_tol and np.average(np.abs(target_rot_matrix - cur_rot_matrix)) < ik_tol:
                print("Movement completed. Deviation:", np.abs(target_pose - cur_pose))
                
                # update robot pose
                self.robot_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
                self.robot_pos = self.robot_pose[:3]
                self.robot_quat = self.robot_pose[3:]

                return
            
            if count >= timeout_count:
                print("Movement terminated due to timeout. Deviation:", np.abs(target_pose - cur_pose))
                
                # update robot pose
                self.robot_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
                self.robot_pos = self.robot_pose[:3]
                self.robot_quat = self.robot_pose[3:]
                
                return
            
# added gripper_target
    def sim_wait (self, count,gripper_target=None):
        '''
        Wait for a given number of timesteps in simulation.
        '''

        print("Waiting...")
        robot_articulation = self.scene.articulations["ur5e"]
        full_joint_target = robot_articulation.data.joint_pos.squeeze().clone()
        
        if gripper_target is not None and torch.is_tensor(gripper_target):
            # The UR5e arm has 6 joints. Gripper joints are usually the 7th and 8th (index 6 and 7).
            # Get the total number of joints (e.g., 8: 6 arm + 2 gripper)
            num_total_joints = full_joint_target.shape[0]
            
            # The index of the first gripper joint is num_total_joints - 2
            # Assuming a 6-DoF arm + a 2-DoF gripper (like Robotiq 2F-85)
            gripper_joint_ids_start = num_total_joints - 2
            
            # Update the gripper part of the full_joint_target tensor
            # gripper_target is expected to be a [2] tensor: [open_val, open_val]
            full_joint_target[gripper_joint_ids_start:num_total_joints] = gripper_target
        for _ in range (count):
            if gripper_target is not None and torch.is_tensor(gripper_target):
                # Continuously set the joint target to hold the open position
                robot_articulation.set_joint_position_target(full_joint_target)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        # update robot pose
        self.robot_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]

    def capture_and_process_frame(self, cam_intrinsics, T_eef_cam, 
                                  current_eef_pos, current_eef_quat, # <-- MODIFIED: Added arguments
                                  save_visualization=False, view_name=""):
        """Capture RGB-D frame, process to point cloud, and transform to world coordinates"""
        # Get camera data
        wrist_camera = self.scene.sensors["wrist_camera"]
        rgb = wrist_camera.data.output["rgb"][0].cpu().numpy()
        depth = wrist_camera.data.output["depth"][0].cpu().numpy()

        # Convert to Open3D format
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth)

        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=1.0, convert_rgb_to_intensity=False
        )
        
        # Create point cloud from RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam_intrinsics)
        
        # # Filter points (your logic from previous step)
        points_cam_frame = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        color_mask = (colors[:, 0] > 0.5) & (colors[:, 1] < 0.4) & (colors[:, 2] < 0.4)
        MIN_DEPTH = 0.1 
        MAX_DEPTH = 1.0 # Match depth_trunc limit (1.0 m)
        depth_mask = (points_cam_frame[:, 2] > MIN_DEPTH) & (points_cam_frame[:, 2] < MAX_DEPTH)
        combined_mask = color_mask & depth_mask
        pcd = pcd.select_by_index(np.where(combined_mask)[0])

        if len(pcd.points) == 0:
            print(f"[WARNING] Point cloud is empty after depth filtering for view: {view_name}. Check the MIN/MAX_DEPTH values.")
            return o3d.geometry.PointCloud()
       
        # Get end-effector pose in world frame from arguments
        eef_pos = current_eef_pos.cpu().numpy()      # Use argument, convert to numpy
        eef_quat_wxyz = current_eef_quat.cpu().numpy() # Use argument, convert to numpy
        
        # Create transformation matrix from end-effector to world
        T_world_eef = np.eye(4)
        
        q_w, q_x, q_y, q_z = eef_quat_wxyz
        
        eef_quat_xyzw = [q_x, q_y, q_z, q_w] # Just re-order, NO negation
        T_world_eef[:3, :3] = R.from_quat(eef_quat_xyzw).as_matrix()
        
        T_world_eef[:3, 3] = eef_pos


        # Transform point cloud to world coordinates: T_world_cam = T_world_eef * T_eef_cam
        T_world_cam = T_world_eef @ T_eef_cam
        pcd.transform(T_world_cam)

        self.world_point_clouds.append(pcd)
        # Save visualization if requested
        if save_visualization and view_name:
            # ... (saving logic remains the same) ...
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(rgb)
            plt.title(f'RGB Image - {view_name}')
            plt.subplot(122)
            plt.imshow(depth, cmap='gray')
            plt.title(f'Depth Image - {view_name}')
            plt.savefig(f'vase_{view_name}_rgbd.png')
            plt.close()
            # Note: o3d.visualization.draw_geometries blocks execution, 
            # so you might want to comment this out inside the loop
            # o3d.visualization.draw_geometries([pcd], window_name=f'Vase Point Cloud - {view_name}')
            o3d.io.write_point_cloud(f'vase_{view_name}_pcd.ply', pcd)

        return pcd

    
    def run (self):
        '''
        Your code goes here.
        '''
        
        # Define camera intrinsics and load the hand-eye calibration matrix

        # Get camera parameters from task_envs.py
        cam_cfg = self.scene.sensors["wrist_camera"].cfg
        height, width = cam_cfg.height, cam_cfg.width
        focal_length, h_aperture = cam_cfg.spawn.focal_length, cam_cfg.spawn.horizontal_aperture
        
        # Calculate focal length in pixels and the principal point
        fx = (focal_length / h_aperture) * width
        fy = fx 
        cx, cy = width / 2, height / 2
        cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # Load hand-eye calibration matrix T_eef_cam from your .npz file
        try:
            with np.load('hand_eye_calibration_result.npz') as data:
                T_eef_cam = data['T_eef_cam']
            print(f"T_eef_cam (translation): {T_eef_cam}")
        except (FileNotFoundError, KeyError) as e:
            print(f"ERROR: Failed to load hand-eye calibration file. {e}")
            simulation_app.close()
            return

        try:
            # The robot asset is named 'ur5e' in your scene config
            robot_articulation = self.scene.articulations["ur5e"]
            # The camera is attached to 'gripper_center' in your scene config
            eef_body_name = "gripper_center"
            eef_index = robot_articulation.body_names.index(eef_body_name)
        except (KeyError, ValueError, AttributeError) as e:
            print(f"ERROR: Could not find robot 'ur5e' or end-effector link '{eef_body_name}'. {e}")
            simulation_app.close()
            return
        # TODO: Pose estimation and grasping
        # Define the gripper open target (e.g., 0.08m)
        GRIPPER_OPEN_POS = 0.05 
        # Convert to tensor for use with joint control functions
        gripper_open_target = torch.tensor([GRIPPER_OPEN_POS, GRIPPER_OPEN_POS], device=self.sim.device)
        
        # Open the gripper using the existing joint control function
        # Move arm joints to current position (i.e., don't move), move gripper to open position
        print("\nOpening gripper to a fixed position...")
        current_arm_joints = self.scene["ur5e"].data.joint_pos[:, :6].squeeze().detach().cpu().numpy()
        self.move_robot_joint(
            target_joint_pos=current_arm_joints, 
            target_gripper_pos=GRIPPER_OPEN_POS, 
            count=100, # More steps for a smoother movement
            time_for_residual_movement=50
        )
        # Define target poses for capturing multiple views
        down_orientation = [0.0, 0.7071, 0.7071, 0.0] 
        
        vase_init_pos_tuple = self.scene.cfg.vase.init_state.pos
            # Convert the tuple to a numpy array
        vase_position = np.array(vase_init_pos_tuple)
        # Base position (center)
        base_x = vase_position[0]
        base_y = vase_position[1]
        base_z = 0.4

        NEW_OFFSET = 0.05 
        NEW_HEIGHT = 0.35 
        LOW_OFFSET = 0.3  
        LOW_HEIGHT = 0.2

        angled_orientation_right = [-0.5, 0.5, 0.5, 0.5] # [w, x, y, z]
        
        angled_orientation_left = [0.5,0.5,0.5,-0.5] # [w, x, y, z]
        # Keep the top-down orientation for simplicity, but adjust the base_z for the offset views.

        target_poses = [
            # Pose 1: Top-down (Center) - Keep original
            [base_x, base_y, base_z, *down_orientation], 
            
            # Pose 2: Front (+X offset, Lower, Further)
            [base_x + NEW_OFFSET, base_y, NEW_HEIGHT, *down_orientation],
            
            # Pose 3: Back (-X offset, Lower, Further)
            [base_x - NEW_OFFSET, base_y, NEW_HEIGHT, *down_orientation],
            
            # Pose 4: Right (-Y offset, Lower, Further)
            [base_x, base_y - NEW_OFFSET, NEW_HEIGHT, *down_orientation],
            
            # Pose 5: Left (+Y offset, Lower, Further)
            [base_x, base_y + NEW_OFFSET, NEW_HEIGHT, *down_orientation],

            # Pose 6: Low, Left
            [base_x+0.12, base_y + LOW_OFFSET, LOW_HEIGHT, *angled_orientation_left],

            # Pose 7: Top-down (Center) - Keep original
            [base_x, base_y, base_z, *down_orientation],

            # Pose 8: Low, Right
            [base_x+0.12, base_y - LOW_OFFSET, LOW_HEIGHT, *angled_orientation_right],
        ]
        
        # Updated view names to match
        view_names = ["top_down_center", "top_down_front", "top_down_back", "top_down_right", "top_down_left", "low_angled_left","top_down","low_angled_right"]
   

        # Capture point clouds from each view
        for i, (target_pose, view_name) in enumerate(zip(target_poses, view_names)):
            print(f"\nCapturing view {i+1}/{len(target_poses)}: {view_name}")
            
            # Move robot to target pose
            self.move_robot_ik(target_pose)
            self.sim_wait(20,gripper_target=gripper_open_target)  # Settle before capturing

            # We must get the pose *after* sim_wait, as this is when the robot has arrived
            # The .data attribute is updated by scene.update() (which sim_wait calls)
            robot_data = robot_articulation.data
            current_eef_pos = robot_data.body_pos_w[0, eef_index]   # Get pos from tensor
            current_eef_quat = robot_data.body_quat_w[0, eef_index] # Get quat from tensor [w, x, y, z]
            # Capture and process frame
            pcd = self.capture_and_process_frame(
                cam_intrinsics, T_eef_cam, 
                current_eef_pos, current_eef_quat,  # <-- MODIFIED: Pass new args
                save_visualization=True, view_name=view_name
            )
        # Combine all point clouds
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in self.world_point_clouds:
            combined_pcd += pcd

        # Downsample to reduce noise and redundancy
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.002)
        
        # Remove outliers
        cl, ind = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        combined_pcd = combined_pcd.select_by_index(ind)

        # Save and visualize combined point cloud
        o3d.io.write_point_cloud('combined_vase_pcd.ply', combined_pcd)
        print("Displaying combined point cloud...")
        o3d.visualization.draw_geometries([combined_pcd], window_name='Combined Vase Point Cloud')

        try:
            source_data = np.load('objects/vase.npz') 
            source_key = source_data.files[0]
            source_points = source_data[source_key] 
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_points)
            print("Successfully loaded source point cloud 'vase.npz'.")
        except (FileNotFoundError, KeyError) as e:
            print(f"ERROR: Failed to load 'vase.npz'. Ensure file exists and key is 'points'. {e}")
            simulation_app.close()
            return
        
        initial_transform = np.eye(4)
        initial_transform[:3, 3] = vase_position 
        
        source_pcd_aligned_guess = o3d.geometry.PointCloud(source_pcd)
        source_pcd_aligned_guess.transform(initial_transform)

        source_pcd_aligned_guess.paint_uniform_color([1, 0, 0]) 
        combined_pcd.paint_uniform_color([0, 0, 1])               

        
        print("Displaying point clouds *before* ICP registration (Red=Source, Blue=Target)...")
        o3d.visualization.draw_geometries([source_pcd_aligned_guess, combined_pcd], window_name='Before ICP (Red=Source, Blue=Target)')

        threshold = 0.02  
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)

        print("Running ICP...")
        
       
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd,        
            combined_pcd,        
            threshold,         
            initial_transform, 
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria
        )
        

        final_transform = icp_result.transformation

        print(f"ICP Fitness: {icp_result.fitness:.4f}, RMSE: {icp_result.inlier_rmse:.4f}")

        source_pcd_final = o3d.geometry.PointCloud(source_pcd)
        source_pcd_final.transform(final_transform)
        source_pcd_final.paint_uniform_color([1, 0, 0]) 

        
        o3d.visualization.draw_geometries([source_pcd_final, combined_pcd], window_name='After ICP (Red=Source, Blue=Target)')

        np.savez('icp_result.npz', T_world_vase=final_transform, icp_fitness=icp_result.fitness)
        print("ICP result (T_world_vase) saved to 'icp_result.npz' for use in Q5.")

        try:

            with np.load('icp_result.npz') as data:
                T_world_vase = data['T_world_vase']

            with np.load('T_vase_grasp.npz') as data:
                grasp_key = data.files[0] 
                T_vase_grasp = data[grasp_key]
        
        except (FileNotFoundError, KeyError) as e:
            print(f"ERROR: Failed to load .npz files for Q5. {e}")
            simulation_app.close()
            return

        T_world_grasp = T_world_vase @ T_vase_grasp

        print(T_world_grasp)

        grasp_pos = T_world_grasp[:3, 3]
        grasp_rot_matrix = T_world_grasp[:3, :3]
        r_grasp = R.from_matrix(grasp_rot_matrix)
        grasp_quat_xyzw = r_grasp.as_quat()
        grasp_quat_wxyz = np.array([grasp_quat_xyzw[3], grasp_quat_xyzw[0], grasp_quat_xyzw[1], grasp_quat_xyzw[2]])
        grasp_pose_7d = np.concatenate([grasp_pos, grasp_quat_wxyz])

        PRE_GRASP_OFFSET = 0.10 # 10 cm
        pre_grasp_pos = grasp_pos + np.array([0.0, 0.0, PRE_GRASP_OFFSET])
        pre_grasp_pose_7d = np.concatenate([pre_grasp_pos, grasp_quat_wxyz]) 

        lift_pose_7d = pre_grasp_pose_7d

        GRIPPER_CLOSE_POS = 0.0065
        gripper_close_target = torch.tensor([GRIPPER_CLOSE_POS, GRIPPER_CLOSE_POS], device=self.sim.device)

        print("\nMoving to Pre-Grasp Pose...")
        self.move_robot_ik(pre_grasp_pose_7d)
        self.sim_wait(50, gripper_target=gripper_open_target)

        print("Moving to Final Grasp Pose...")
        self.move_robot_ik(grasp_pose_7d)
        self.sim_wait(50, gripper_target=gripper_open_target)


        print("Closing Gripper...")

        self.move_robot_joint(
            target_joint_pos=None, 
            target_gripper_pos=GRIPPER_CLOSE_POS, 
            count=100,  
            time_for_residual_movement=50
        )
        print("Grasp complete.")

        print("Lifting object...")
        self.move_robot_ik(lift_pose_7d)
        self.sim_wait(80, gripper_target=gripper_close_target) 

        print("Q5 Grasp sequence complete.")

        # steps simulation but does not command the robot
        while simulation_app.is_running():
            
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        # this helps shut down the script correctly
        simulation_app.close()


if __name__ == "__main__":

    exp = Experiment()
    exp.run()