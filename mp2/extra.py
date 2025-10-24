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

# -- NEW IMPORTS FOR EXTRA CREDIT --
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation
# ------------------------------------

from scipy.ndimage import ( # Needed for advanced image processing
    binary_opening, binary_closing, binary_fill_holes,
    generate_binary_structure, label
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
        self.update_robot_pose()
    
    def update_robot_pose(self):
        """Updates the stored robot end-effector pose."""
        self.robot_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]

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

        print("Moving the robot through joint control...")
        for i in range (count):

            self.scene["ur5e"].set_joint_position_target((target - initial)/count*i + initial)

            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        print("Waiting for any residual movement...")
        for i in range (time_for_residual_movement):
            self.scene["ur5e"].set_joint_position_target(target)

            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        print("Movement completed. Deviation: {}".format((target - self.scene['ur5e'].data.joint_pos).squeeze().detach().cpu().numpy()))
        self.update_robot_pose()
    

    def move_robot_ik (self, target_pose, max_joint_change = 0.04, ik_tol = 1e-3, timeout_count = 100):
        '''
        Calls Isaac Lab's IK controller and moves the robot to a desired pose in 3D space. This function is blocking.
        '''

        self.diff_ik_controller.set_command(torch.tensor(target_pose, device = self.sim.device))

        print("Moving the robot through IK...")
        count = 0
        while simulation_app.is_running():
            jacobian = self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
            ee_pose_w = self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
            joint_pos = self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]
            joint_pos_des = self.diff_ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos)

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

            cur_pose = self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
            pos_diff = np.linalg.norm((target_pose - cur_pose)[:3])
            
            if pos_diff < ik_tol:
                print("Movement completed. Position deviation:", pos_diff)
                self.update_robot_pose()
                return
            
            if count >= timeout_count:
                print("Movement terminated due to timeout. Position deviation:", pos_diff)
                self.update_robot_pose()
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
        self.update_robot_pose()
    
    # -- NEW HELPER FUNCTION FOR EXTRA CREDIT --
    def get_6d_pose_from_points(self, points):
        """
        Calculates the 6D pose (position and orientation) of a point cloud using PCA.
        
        Args:
            points (np.ndarray): An (N, 3) array of 3D points.

        Returns:
            tuple: A tuple containing:
                - pos (np.ndarray): The 3D position (centroid) [x, y, z].
                - quat (np.ndarray): The orientation quaternion [qw, qx, qy, qz].
        """
        if points.shape[0] < 3:
            return None, None
        # 1. Calculate centroid for position
        # Using median is slightly more robust to outliers than mean
        centroid = np.median(points, axis=0)
        
        # 2. Use PCA to find the principal axis (longest dimension) of the object
        pca = PCA(n_components=3)
        pca.fit(points)
        
        # The first principal component is a vector along the longest dimension of the cube
        primary_axis_3d = pca.components_[0]
        
        # 3. Project this axis onto the XY plane to find the yaw
        # We only care about the direction in the plane of the table
        primary_axis_2d = primary_axis_3d[:2]
        
        # 4. Calculate the yaw angle from this 2D vector
        # np.arctan2 is used to get the angle in radians from the vector's x and y components
        yaw_angle = np.arctan2(primary_axis_2d[1], primary_axis_2d[0])
        
        # 4. --- THIS IS THE KEY FIX ---
        # We need to combine two rotations:
        #   a) A fixed rotation to make the gripper point straight down.
        #      This is typically -90 degrees (-pi/2 radians) around the Y-axis.
        #   b) The dynamic yaw rotation to align with the cube.
        
        # Create the "point down" rotation object
        # This is our base orientation for a top-down grasp
        point_down_rotation = Rotation.from_euler('x', np.pi)
        
        # Create the yaw rotation object
        yaw_rotation = Rotation.from_euler('z', yaw_angle)
        
        # Combine them: first apply the point-down, then the yaw
        # The order of multiplication is important!
        final_rotation = yaw_rotation * point_down_rotation
        
        # 5. Convert the final combined rotation to the required quaternion format
        quat_xyzw = final_rotation.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return centroid, quat_wxyz

    def run (self):
        '''
        This modified run method solves the extra credit task:
        1. It identifies all cubes and calculates their full 6D pose (position + orientation).
        2. It defines a target stacking location and a standard orientation for all cubes.
        3. It picks the base green cube and places it at the target location with the standard orientation.
        4. It then stacks the red and the second green cube on top, aligning each one.
        '''
        # -- Perception Parameters --
        h_red_low_1, h_red_high_1 = 0.00, 0.04
        h_red_low_2, h_red_high_2 = 0.96, 1.00
        h_green_low, h_green_high = 0.23, 0.44
        s_min, v_min = 0.40, 0.20
        morph_iters = 2
        struct = generate_binary_structure(2, 2)

        # -- Task Planning Parameters --
        GRIPPER_OPEN_POS = 0.04
        GRIPPER_CLOSED_POS = 0.0
        hover_offset = 0.12
        lift_offset = 0.15
        red_edge, green_edge = 0.04, 0.05
        STACK_XY_POS = np.array([0.5, 0.0]) # Designated spot for the final stack
        TARGET_QUAT = np.array([1.0, 0.0, 0.0, 0.0]) # World-aligned [qw, qx, qy, qz]

        # 1. --- SENSE: Get camera data and build point cloud ---
        self.sim_wait(20)
        target_robot_pos = self.robot_pos - np.array([0.15, 0.2, 0.0])
        self.move_robot_ik(np.concatenate([target_robot_pos, self.robot_quat]))
        self.sim_wait(20)

        intrinsics = np.squeeze(self.scene["birdview_camera"].data.intrinsic_matrices.detach().cpu().numpy())
        extrinsics = np.array([[0,1,0,0.5], [1,0,0,0], [0,0,-1,1.2], [0,0,0,1]])
        color_raw = self.scene["birdview_camera"].data.output["rgb"].detach().cpu().numpy()[0][..., :3]
        depth_image = np.squeeze(self.scene["birdview_camera"].data.output["depth"].detach().cpu().numpy()[0])
        
        hsv = mcolors.rgb_to_hsv(np.clip(color_raw.astype(np.float32)/255.0, 0.0, 1.0))
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        red_mask = binary_closing(binary_fill_holes(binary_opening(((h >= h_red_low_1) & (h <= h_red_high_1) | (h >= h_red_low_2) & (h <= h_red_high_2)) & (s >= s_min) & (v >= v_min), structure=struct, iterations=morph_iters)))
        green_mask = binary_closing(binary_fill_holes(binary_opening(((h >= h_green_low) & (h <= h_green_high)) & (s >= s_min) & (v >= v_min), structure=struct, iterations=morph_iters)))

        height, width = depth_image.shape
        fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
        y_pixel, x_pixel = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        z_c = depth_image
        x_c = z_c * (x_pixel - cx) / fx
        y_c = z_c * (y_pixel - cy) / fy
        points_camera = np.stack((x_c, y_c, z_c), axis=-1).reshape(-1, 3) 
        points_world_h = extrinsics @ np.concatenate([points_camera, np.ones((points_camera.shape[0], 1))], axis=1).T
        pc_world = (points_world_h[:3,:] / np.clip(points_world_h[3,:], 1e-8, None)).T.reshape(height, width, 3)
        z_table_est = float(np.nanpercentile(pc_world[..., 2].reshape(-1), 1.0))

        # 2. --- PERCEIVE & PLAN: Find 6D poses of all cubes ---
        objects = []
        lab_red, n_red = label(red_mask, structure=struct)
        for cid in range(1, n_red + 1):
            idx = (lab_red == cid).flatten()
            pts = pc_world.reshape(-1, 3)[idx]
            pos, quat = self.get_6d_pose_from_points(pts)
            if pos is not None:
                objects.append({"color": "red", "pos": pos, "quat": quat})
        
        lab_green, n_green = label(green_mask, structure=struct)
        for cid in range(1, n_green + 1):
            idx = (lab_green == cid).flatten()
            pts = pc_world.reshape(-1, 3)[idx]
            pos, quat = self.get_6d_pose_from_points(pts)
            if pos is not None:
                objects.append({"color": "green", "pos": pos, "quat": quat})

        # Sort green blocks by distance to center to select base vs top
        green_cubes = sorted([o for o in objects if o['color'] == 'green'], key=lambda o: np.linalg.norm(o['pos'][:2] - STACK_XY_POS))
        red_cube = next(o for o in objects if o['color'] == 'red')
        base_green_cube, top_green_cube = green_cubes[0], green_cubes[1]

        print(f"Found {len(objects)} objects. Base green at {base_green_cube['pos']}. Starting stack...")

        # 3. --- ACT: Execute the aligned stacking sequence ---
        
        # Function to perform a full pick and place cycle
        def pick_and_place(pick_obj, place_xy, place_z, target_quat):
            # PICK SEQUENCE
            self.move_robot_joint(None, GRIPPER_OPEN_POS, count=30)
            pick_pos, pick_quat = pick_obj['pos'], pick_obj['quat']
            self.move_robot_ik(np.concatenate([pick_pos[:2], [pick_pos[2] + hover_offset], pick_quat])) # Hover over pick
            self.move_robot_ik(np.concatenate([pick_pos, pick_quat])) # Move to grasp
            self.move_robot_joint(None, GRIPPER_CLOSED_POS, count=50)
            self.sim_wait(20)
            self.move_robot_ik(np.concatenate([pick_pos[:2], [pick_pos[2] + lift_offset], pick_quat])) # Lift
            
            # PLACE SEQUENCE
            place_pos = np.array([place_xy[0], place_xy[1], place_z])
            self.move_robot_ik(np.concatenate([place_pos[:2], [place_pos[2] + hover_offset], target_quat])) # Hover over place
            self.move_robot_ik(np.concatenate([place_pos, target_quat])) # Move to place
            self.move_robot_joint(None, GRIPPER_OPEN_POS, count=50)
            self.sim_wait(20)
            self.move_robot_ik(np.concatenate([place_pos[:2], [place_pos[2] + lift_offset], target_quat])) # Retract
        
        # --- STACKING ACTIONS ---
        # Action 1: Move base green cube to the designated stacking spot
        print("\n--- Moving Base Green Cube ---")
        base_place_z = z_table_est + green_edge / 2
        pick_and_place(base_green_cube, STACK_XY_POS, base_place_z, TARGET_QUAT)

        # Action 2: Stack red cube on top of the base green cube
        print("\n--- Stacking Red Cube ---")
        red_place_z = base_place_z + green_edge / 2 + red_edge / 2
        pick_and_place(red_cube, STACK_XY_POS, red_place_z, TARGET_QUAT)

        # Action 3: Stack the top green cube on the red cube
        print("\n--- Stacking Top Green Cube ---")
        top_green_place_z = red_place_z + red_edge / 2 + green_edge / 2
        pick_and_place(top_green_cube, STACK_XY_POS, top_green_place_z, TARGET_QUAT)
        
        print("\nStacking complete!")

        # Keep simulation running
        while simulation_app.is_running():
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        simulation_app.close()


if __name__ == "__main__":
    exp = Experiment()
    exp.run()