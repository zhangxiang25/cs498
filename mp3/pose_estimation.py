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
        # ==============================================================================
        # Q3 - 物体点云重建 (Object Point Cloud Reconstruction)
        # ==============================================================================
        # 中文注释：
        # 这个部分的目标是使用机器人手腕上的相机，从多个视角拍摄花瓶的RGB-D图像，
        # 然后将这些图像转换成点云，并最终融合成一个完整的花瓶三维模型。

        # 步骤 1: 定义相机内参和加载手眼标定矩阵
        # ------------------------------------------------
        # Step 1: Define camera intrinsics and load the hand-eye calibration matrix
        print("开始点云重建...")
        print("Starting point cloud reconstruction...")

        # 从 task_envs.py 中获取相机参数
        # Get camera parameters from task_envs.py
        cam_cfg = self.scene.sensors["wrist_camera"].cfg
        height, width = cam_cfg.height, cam_cfg.width
        focal_length, h_aperture = cam_cfg.spawn.focal_length, cam_cfg.spawn.horizontal_aperture
        
        # 计算以像素为单位的焦距和主点
        # Calculate focal length in pixels and the principal point
        fx = (focal_length / h_aperture) * width
        fy = fx  # 假设像素是正方形 Assuming square pixels
        cx, cy = width / 2, height / 2
        cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        print(f"相机内参 (fx, fy, cx, cy): ({fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f})")

        # 从你的 .npz 文件加载手眼标定矩阵 T_eef_cam
        # Load hand-eye calibration matrix T_eef_cam from your .npz file
        try:
            with np.load('hand_eye_calibration_result.npz') as data:
                # 假设矩阵在文件中的键是 'T_eef_cam'
                # Assuming the key for the matrix in the file is 'T_eef_cam'
                T_eef_cam = data['T_eef_cam']
            print("成功加载手眼标定矩阵 'hand_eye_calibration_result.npz'.")
            print(f"T_eef_cam (translation): {T_eef_cam}")
            print("Successfully loaded hand-eye calibration matrix from 'hand_eye_calibration_result.npz'.")
        except (FileNotFoundError, KeyError) as e:
            print(f"错误: 加载手眼标定文件失败。 {e}")
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
            print(f"错误: 无法找到机器人 'ur5e' 或末端执行器连杆 '{eef_body_name}'. {e}")
            print(f"ERROR: Could not find robot 'ur5e' or end-effector link '{eef_body_name}'. {e}")
            simulation_app.close()
            return
        # TODO: Pose estimation and grasping
        # 1. Define the gripper open target (e.g., 0.08m)
        GRIPPER_OPEN_POS = 0.05 
        # Convert to tensor for use with joint control functions
        gripper_open_target = torch.tensor([GRIPPER_OPEN_POS, GRIPPER_OPEN_POS], device=self.sim.device)
        
        # 2. Open the gripper using the existing joint control function
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
        # These poses are chosen to环绕花瓶, covering different angles and heights
        # This quaternion points the gripper/camera down
        down_orientation = [0.0, 0.7071, 0.7071, 0.0] 
        
        vase_init_pos_tuple = self.scene.cfg.vase.init_state.pos
            
            # 将元组 (tuple) 转换为 numpy 数组
            # Convert the tuple to a numpy array
        vase_position = np.array(vase_init_pos_tuple)
        # Base position (center)
        base_x = vase_position[0]
        base_y = vase_position[1]
        base_z = 0.4 # 保持之前设定的 0.5米 高度
        
        # Offset for side views
        offset = 0.1 # 10 cm offset

        NEW_OFFSET = 0.05 # New, larger offset (25cm)
        NEW_HEIGHT = 0.35 # New, lower height (35cm)
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
            [base_x, base_y + NEW_OFFSET, NEW_HEIGHT, *down_orientation]
        ]
        
        # Updated view names to match
        view_names = ["top_down_center", "top_down_front", "top_down_back", "top_down_right", "top_down_left"]
   
        
        LOW_OFFSET = 0.3  # 更远 (30cm) 以获得更好的角度
        LOW_HEIGHT = 0.2 # 更低 (z=0.30m), (Raised from 0.25 to 0.30 to avoid collision)
        
        angled_orientation_right = [-0.5, 0.5, 0.5, 0.5] # [w, x, y, z]
        
        # angled_orientation_left = [0.130, -0.130, -0.698, 0.698] # [w, x, y, z]
        angled_orientation_left = [0.5,0.5,0.5,-0.5] # [w, x, y, z]

        # Add new poses to the list
        target_poses.extend([
            
            # Pose 9: Low, Left
            [base_x+0.12, base_y + LOW_OFFSET, LOW_HEIGHT, *angled_orientation_left],
            # Pose 1: Top-down (Center) - Keep original
            [base_x, base_y, base_z, *down_orientation],
            # Pose 8: Low, Right
            [base_x+0.12, base_y - LOW_OFFSET, LOW_HEIGHT, *angled_orientation_right],
        ])
        
        view_names.extend([
            "low_angled_left",
            "top_down",
            "low_angled_right"
        ])

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
        print("\n显示合并后的点云...")
        print("Displaying combined point cloud...")
        o3d.visualization.draw_geometries([combined_pcd], window_name='Combined Vase Point Cloud')

        # Q4 - 迭代最近点 (Iterative Closest Point)
        # ==============================================================================
        print("\n开始 Q4 ICP 配准...")
        print("Starting Q4 ICP Registration...")

        # ------------------------------------------------
        # 步骤 1: 加载源点云 (模型)
        # ------------------------------------------------
        try:
            # 假设 vase.npz 文件在同一个目录下
            source_data = np.load('objects/vase.npz') 
            # 假设 .npz 文件中的键是 'points'
            source_key = source_data.files[0]
            source_points = source_data[source_key] 
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_points)
            print("成功加载源点云 'vase.npz'.")
            print("Successfully loaded source point cloud 'vase.npz'.")
        except (FileNotFoundError, KeyError) as e:
            print(f"错误: 加载 'vase.npz' 失败. 请确保文件存在并且键是 'points'. {e}")
            print(f"ERROR: Failed to load 'vase.npz'. Ensure file exists and key is 'points'. {e}")
            simulation_app.close()
            return
        
        # ------------------------------------------------
        # 步骤 2: 准备 ICP - 初始猜测 (Pre-alignment)
        # ------------------------------------------------
        # ICP 需要一个好的初始猜测。我们使用仿真中花瓶的*初始*位置作为猜测。
        # 'vase_position' 已经在 Q3 的开头部分从 self.scene.cfg.vase.init_state.pos 中获取了
        
        # 创建一个初始变换矩阵 (仅平移，无旋转)
        # 这是 T_world_model_initial
        initial_transform = np.eye(4)
        initial_transform[:3, 3] = vase_position 
        
        # 创建一个源点云的副本用于可视化 "Before"
        source_pcd_aligned_guess = o3d.geometry.PointCloud(source_pcd)
        source_pcd_aligned_guess.transform(initial_transform)

        # ------------------------------------------------
        # 步骤 3: 可视化 "Before"
        # ------------------------------------------------
        # 给点云上色以便区分
        source_pcd_aligned_guess.paint_uniform_color([1, 0, 0]) # 源点云 = 红色
        combined_pcd.paint_uniform_color([0, 0, 1])               # 目标点云 = 蓝色

        print("显示 ICP 配准 *之前* 的点云 (红色=源, 蓝色=目标)...")
        print("Displaying point clouds *before* ICP registration (Red=Source, Blue=Target)...")
        o3d.visualization.draw_geometries(
            [source_pcd_aligned_guess, combined_pcd], 
            window_name='[Q4] Before ICP (Red=Source, Blue=Target)'
        )

        # ------------------------------------------------
        # 步骤 4: 执行 ICP
        # ------------------------------------------------
        # (为你报告准备的) ICP 工作原理:
        # 1. 寻找对应点：对于源点云中的每个点，在目标点云中找到*最近*的邻居点。
        # 2. 最小化误差：计算一个刚体变换（旋转+平移），该变换能使这些对应点对之间的
        #    距离（通常是平方和）最小化。
        # 3. 应用变换：将计算出的变换应用于*源*点云。
        # 4. 迭代：重复步骤 1-3，直到变换收敛（变化很小）或达到最大迭代次数。

        threshold = 0.02  # 2cm - 对应点对之间的最大距离
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)

        print("正在运行 ICP...")
        print("Running ICP...")
        
        # 我们提供了*原始*的 source_pcd (在原点) 和 initial_transform (我们的猜测)
        # ICP 算法将使用 initial_transform 作为起点
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd,        # 原始源点云 (在原点)
            combined_pcd,        # 目标点云 (在世界坐标系)
            threshold,         # 最大对应距离
            initial_transform, # 我们的初始猜测
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria
        )
        
        # 最终的变换矩阵 (T_world_vase)
        final_transform = icp_result.transformation
        print("ICP 完成.")
        print("ICP finished.")
        print("ICP 最终变换矩阵 (T_world_vase):")
        print(final_transform)
        print(f"ICP Fitness: {icp_result.fitness:.4f}, RMSE: {icp_result.inlier_rmse:.4f}")

        # ------------------------------------------------
        # 步骤 5: 可视化 "After"
        # ------------------------------------------------
        # 创建一个新的源点云副本并应用*最终*的变换
        source_pcd_final = o3d.geometry.PointCloud(source_pcd)
        source_pcd_final.transform(final_transform)
        source_pcd_final.paint_uniform_color([1, 0, 0]) # 源点云 = 红色
        # target_pcd 仍然是蓝色

        print("显示 ICP 配准 *之后* 的点云 (红色=源, 蓝色=目标)...")
        print("Displaying point clouds *after* ICP registration (Red=Source, Blue=Target)...")
        o3d.visualization.draw_geometries(
            [source_pcd_final, combined_pcd], 
            window_name='[Q4] After ICP (Red=Source, Blue=Target)'
        )

        # ------------------------------------------------
        # 步骤 6: 保存结果供 Q5 使用
        # ------------------------------------------------
        # 这个 final_transform 就是 Q5 需要的 T_world_vase (或 T_world_model)
        np.savez('icp_result.npz', T_world_vase=final_transform, icp_fitness=icp_result.fitness)
        print("ICP 结果 (T_world_vase) 已保存到 'icp_result.npz' 供 Q5 使用.")
        print("ICP result (T_world_vase) saved to 'icp_result.npz' for use in Q5.")

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