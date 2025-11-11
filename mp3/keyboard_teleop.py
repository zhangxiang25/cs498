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
import matplotlib
matplotlib.use('Qt5Agg')  # <-- Use 'Qt5Agg' instead of 'TkAgg'
import matplotlib.pyplot as plt
from pynput.keyboard import Listener
from scipy.spatial.transform import Rotation as R

import cv2
from pupil_apriltags import Detector
import os

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene

from task_envs import MP3CalibSceneCfg, PHYSICS_DT, RENDERING_DT


class RobotKeyboardController:

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
        scene_cfg = MP3CalibSceneCfg(args_cli.num_envs, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)

        # reset simulation
        self.sim.reset()
        print("Setup complete...")

        # setup IK solver
        self.ik_body = "gripper_center"
        diff_ik_cfg = DifferentialIKControllerCfg(command_type = "pose", use_relative_mode = False, ik_method = "dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs = self.scene.num_envs, device = self.sim.device)
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

        # for showing image continuously
        self.fig, ax = plt.subplots()
        self.im = ax.imshow((np.ones((256, 256))*255).astype(np.uint8))

        # teleop state indicators
        self.d_pressed = False
        self.z_pressed = False
        self.x_pressed = False
        self.c_pressed = False
        self.g_pressed = False
        self.v_pressed = False
        self.comma_pressed = False
        self.period_pressed = False
        self.l_pressed = False
        self.colon_pressed = False
        self.quote_pressed = False
        self.slash_pressed = False

        self.apriltag_detector = Detector(families="tag36h11", nthreads=1)
        self.tag_size = 0.16
        self.eef_poses = []
        self.tag_poses = []

        # ADDED: Define output directory for images ---
        self.output_dir = "calibration_images"
        os.makedirs(self.output_dir, exist_ok=True) # Add this line
        self.b_pressed = False # For saving data
        #

    def on_press (self, key):
        try:
            if key.char == "d":
                self.d_pressed = True
            
            elif key.char == "x":
                self.x_pressed = True

            elif key.char == "z":
                self.z_pressed = True

            elif key.char == "c":
                self.c_pressed = True

            elif key.char == "g":
                self.g_pressed = True
            
            elif key.char == "v":
                self.v_pressed = True

            elif key.char == ",":
                self.comma_pressed = True

            elif key.char == ".":
                self.period_pressed = True

            elif key.char == "l":
                self.l_pressed = True
            
            elif key.char == ";":
                self.colon_pressed = True
            
            elif key.char == "'":
                self.quote_pressed = True

            elif key.char == "/":
                self.slash_pressed = True
            elif key.char == "b":
                self.b_pressed = True
        except:
            pass
    

    def on_release (self, key):
        try:
            if key.char == "d":
                self.d_pressed = False
            
            elif key.char == "x":
                self.x_pressed = False

            elif key.char == "z":
                self.z_pressed = False

            elif key.char == "c":
                self.c_pressed = False

            elif key.char == "g":
                self.g_pressed = False
            
            elif key.char == "v":
                self.v_pressed = False

            elif key.char == ",":
                self.comma_pressed = False

            elif key.char == ".":
                self.period_pressed = False

            elif key.char == "l":
                self.l_pressed = False
            
            elif key.char == ";":
                self.colon_pressed = False
            
            elif key.char == "'":
                self.quote_pressed = False

            elif key.char == "/":
                self.slash_pressed = False
            
            elif key.char == "b": 
                self.b_pressed = False
        
        except:
            pass

        
    def teleop (self):

        # initialize keyboard listener
        listener = Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        listener.start()
        print("\nTeleoperation started\n")

        print("Position control:")
        print(" - Press d and x to move the robot along the X axis")
        print(" - Press z and c to move the robot along the Y axis")
        print(" - Press v and g to move the robot along the Z axis\n")
        
        print("Orientation control:")
        print(" - Press l and ; to rotate the robot around the X axis")
        print(" - Press ' and / to rotate the robot around the Y axis")
        print(" - Press , and . to rotate the robot around the Z axis\n")

        # teleop params
        max_joint_change = 0.1
        temp_dist_target = 0.01
        temp_rot_target = 2.

        is_stationary = True
        recorded_stationary_pose = False
        stationary_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        
        #
        save_key_was_down = False
        # teleop
        while simulation_app.is_running():

            # record last stationary pose to prevent error accumulation
            if not (self.d_pressed or self.x_pressed or self.z_pressed or self.c_pressed or self.g_pressed or self.v_pressed) and \
                not (self.l_pressed or self.colon_pressed or self.quote_pressed or self.comma_pressed or self.period_pressed or self.slash_pressed) :
                is_stationary = True
            else:
                is_stationary = False
                
            if is_stationary:
                if not recorded_stationary_pose:
                    stationary_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
                    recorded_stationary_pose = True
            else:
                recorded_stationary_pose = False

            # set new robot pose according to keyboard inputs
            # position
            delta_dists = np.zeros((3,))
            if self.d_pressed:
                delta_dists[0] -= temp_dist_target
            if self.x_pressed:
                delta_dists[0] += temp_dist_target
            if self.z_pressed:
                delta_dists[1] -= temp_dist_target
            if self.c_pressed:
                delta_dists[1] += temp_dist_target
            if self.g_pressed:
                delta_dists[2] += temp_dist_target
            if self.v_pressed:
                delta_dists[2] -= temp_dist_target

            # orientation
            delta_euler_angles = np.zeros((3,))
            if self.comma_pressed:
                delta_euler_angles[2] -= temp_rot_target
            if self.period_pressed:
                delta_euler_angles[2] += temp_rot_target
            if self.l_pressed:
                delta_euler_angles[0] += temp_rot_target
            if self.colon_pressed:
                delta_euler_angles[0] -= temp_rot_target
            if self.quote_pressed:
                delta_euler_angles[1] -= temp_rot_target
            if self.slash_pressed:
                delta_euler_angles[1] += temp_rot_target

            # get current robot pose
            cur_quat = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[3:]
            cur_pos = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[:3]
            T_world_eef = np.eye(4)
            T_world_eef[:3, :3] = R.from_quat([cur_quat[1], cur_quat[2], cur_quat[3], cur_quat[0]]).as_matrix()
            T_world_eef[:3, 3] = cur_pos

            # define temp_frame located at the eef, but with axis aligned with world frame
            T_world_tempframeold = np.eye(4)
            T_world_tempframeold[:3, 3] = cur_pos.copy()
            
            # orientation change
            T_tempframe_eef = np.linalg.inv(T_world_tempframeold) @ T_world_eef
            T_tempframeold_tempframenew = np.eye(4)
            T_tempframeold_tempframenew[:3, :3] = R.from_euler("xyz", delta_euler_angles, degrees=True).as_matrix()
            T_world_eefnew = T_world_tempframeold @ T_tempframeold_tempframenew @ T_tempframe_eef

            # position change
            T_world_eefnew[:3, 3] += delta_dists

            # format conversion for isaaclab's ik controller
            target_pos = T_world_eefnew[:3, 3]
            target_quat_raw = R.from_matrix(T_world_eefnew[:3, :3]).as_quat()
            target_quat = [target_quat_raw[3], target_quat_raw[0], target_quat_raw[1], target_quat_raw[2]]
            target_pose = np.array(target_pos.tolist() + target_quat)

            # IK controller
            if is_stationary:
                self.diff_ik_controller.set_command(torch.tensor(stationary_pose, device = self.sim.device))
            else:
                self.diff_ik_controller.set_command(torch.tensor(target_pose, device = self.sim.device))
            jacobian = self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
            ee_pose_w = self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
            joint_pos = self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]

            # compute joint commands
            joint_pos_des = self.diff_ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos)

            # limit joint changes to prevent overshoot
            joint_changes = (joint_pos_des - joint_pos).detach().cpu().numpy()[0]
            if np.sum(np.abs(joint_changes) > max_joint_change) > 0:
                scaled_joint_changes = joint_changes / (np.max(np.abs(joint_changes)) / max_joint_change)
                scaled_joint_changes = torch.tensor(scaled_joint_changes).unsqueeze(0).to(joint_pos_des.device)
                joint_pos_des = joint_pos + scaled_joint_changes

            # all joints, including gripper joint
            all_joint_pos_des = torch.zeros((1, 8))
            all_joint_pos_des[:, :6] = joint_pos_des
            # keep gripper open to minimize occlusion
            all_joint_pos_des[:, 6:] = torch.tensor([0.05, 0.05]).to(self.sim.device)

            # send command to robot
            self.scene["ur5e"].set_joint_position_target(all_joint_pos_des)


            # TODO: apriltag detection and calib dataset collection
            # Gets the camera's intrinsic matrix
            intrinsics = np.squeeze(self.scene["wrist_camera"].data.intrinsic_matrices.detach().cpu().numpy())
            # Gets the RGBA image from the wrist camera
            wrist_cam_img = self.scene["wrist_camera"].data.output["rgb"].detach().cpu().numpy()[0]

            # Convert RGBA image from Isaac Lab to grayscale for AprilTag detection
            gray_img = cv2.cvtColor(wrist_cam_img, cv2.COLOR_RGBA2GRAY)

            # Define camera parameters (fx, fy, cx, cy) from the intrinsic matrix
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            camera_params = (fx, fy, cx, cy)

            # Detect AprilTags
            detections = self.apriltag_detector.detect(
                gray_img,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=self.tag_size,
            )
            
            # Create a color image from the RGBA buffer for visualization using OpenCV
            vis_img = cv2.cvtColor(wrist_cam_img, cv2.COLOR_RGBA2BGR)

            # Create a boolean falg, which is true only on when the b key is first pressed
            save_attempted = self.b_pressed and not save_key_was_down

            # Loop over detected tags and draw visualizations
            for tag in detections:
                # Draw the bounding box (in green)
                corners = tag.corners.astype(int) # Gets the four corner points of the tag
                cv2.polylines(vis_img, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

                # Get the estimated pose (rotation matrix(R) and translation vector(t))
                pose_R, pose_t = tag.pose_R, tag.pose_t

                # Convert rotation matrix to rotation vector for OpenCV's projectPoints function
                rvec, _ = cv2.Rodrigues(pose_R)
                tvec = pose_t

                # Define 3D points for the pose axes in the tag's local coordinate frame
                axis_len = 0.5 * self.tag_size  # Visualize axes at half the tag's size
                axis_points = np.float32([
                    [0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]
                ]).reshape(-1, 3)

                # Project the 3D axes points onto the 2D image plane
                img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, intrinsics, distCoeffs=None)
                img_points = np.round(img_points).astype(int)

                # Draw the projected axes on the image
                origin = tuple(img_points[0].ravel())
                # X-axis in Red, Y-axis in Green, Z-axis in Blue
                cv2.line(vis_img, origin, tuple(img_points[1].ravel()), (0, 0, 255), 3)
                cv2.line(vis_img, origin, tuple(img_points[2].ravel()), (0, 255, 0), 3)
                cv2.line(vis_img, origin, tuple(img_points[3].ravel()), (255, 0, 0), 3)

            if save_attempted:
                if detections: 
                    # Use the first detected tag for saving the pose data
                    tag = detections[0]
                    T_cam_tag = np.eye(4)
                    T_cam_tag[:3, :3] = tag.pose_R
                    T_cam_tag[:3, 3] = tag.pose_t.flatten()

                    self.eef_poses.append(T_world_eef) # end-effector's pose
                    self.tag_poses.append(T_cam_tag)

                    # Save the annotated image
                    image_filename = f"calibration_image_{len(self.eef_poses)}.png"
                    image_filepath = os.path.join(self.output_dir, image_filename)
                    cv2.imwrite(image_filepath, vis_img)
                    self.im.set_data(vis_img)
                    plt.pause(0.001)
                    self.fig.canvas.draw()

                    print(f"SAVED POSE #{len(self.eef_poses)}/15 and image '{image_filename}'")
                    if len(self.eef_poses) == 15:
                        self.save_calibration()
                        self.calibration_done = True # Set flag to prevent re-running
                else:
                    # Provide feedback if 'b' is pressed but no tag is visible
                    self.im.set_data(vis_img)
                    plt.pause(0.001)
                    self.fig.canvas.draw()
                    print("Save key pressed, but NO AprilTag detected!")

            # Update the key state at the end of the frame to prevent multiple saves
            save_key_was_down = self.b_pressed
            # Convert from BGR (OpenCV's default) to RGB for Matplotlib display
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

            # step simulation
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

    def save_calibration(self):

        # Save the collected data to .npz file
        filepath = "calibration_data.npz"
        np.savez(
            filepath,
            eef_poses=np.array(self.eef_poses),
            tag_poses=np.array(self.tag_poses),
        )
        print(f"Data saved to '{filepath}'")

if __name__ == "__main__":

    keyboard_controller = RobotKeyboardController()
    keyboard_controller.teleop()
