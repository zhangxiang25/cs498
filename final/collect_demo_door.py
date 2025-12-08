import argparse
import os
import shutil
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pynput.keyboard import Key, Listener
from scipy.spatial.transform import Rotation as R

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

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene

# !!! IMPORT YOUR NEW DOOR CONFIG HERE !!!
from task_envs_door import DoorSceneCfg, PHYSICS_DT, RENDERING_DT


class Experiment:

    def __init__(self, dataset_dir = None):

        if (dataset_dir is not None) and (not os.path.exists(dataset_dir)):
            os.makedirs(dataset_dir)
        self.dataset_dir = dataset_dir
        
        # initialize sim
        sim_cfg = sim_utils.SimulationCfg(device = args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        # Adjust camera view to see the door better
        self.sim.set_camera_view([1.2, 0.0, 0.8], [0.5, 0.0, 0.2])

        # set time step size
        self.sim.set_simulation_dt(physics_dt = PHYSICS_DT, rendering_dt = RENDERING_DT)
        print("\nSim dt: {}\n".format(self.sim.get_physics_dt()))
        self.sim_dt = self.sim.get_physics_dt()

        # initialize scene (Use DoorSceneCfg)
        scene_cfg = DoorSceneCfg(args_cli.num_envs, env_spacing=2.0)
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

        # teleop state indicators
        self.d_pressed = False
        self.z_pressed = False
        self.x_pressed = False
        self.c_pressed = False
        self.g_pressed = False
        self.v_pressed = False
        self.a_pressed = False
        
        # New rotation keys
        self.e_pressed = False # Rotate Left (CCW)
        self.r_pressed = False # Rotate Right (CW)
        
        self.changed_gripper_state = False

        # saving and resetting scene
        self.close_cur_episode = False
        self.save_cur_episode = False


    def on_press (self, key):
        try:
            if key.char == "d": self.d_pressed = True
            elif key.char == "x": self.x_pressed = True
            elif key.char == "z": self.z_pressed = True
            elif key.char == "c": self.c_pressed = True
            elif key.char == "g": self.g_pressed = True
            elif key.char == "v": self.v_pressed = True
            elif key.char == "a": self.a_pressed = True
            
            # Rotation keys
            elif key.char == "e": self.e_pressed = True
            elif key.char == "r": self.r_pressed = True
            
            elif key.char == "s":
                self.close_cur_episode = True
                self.save_cur_episode = True
            elif key.char == "q":
                self.close_cur_episode = True
                self.save_cur_episode = False
        except:
            pass
    

    def on_release (self, key):
        try:
            if key.char == "d": self.d_pressed = False
            elif key.char == "x": self.x_pressed = False
            elif key.char == "z": self.z_pressed = False
            elif key.char == "c": self.c_pressed = False
            elif key.char == "g": self.g_pressed = False
            elif key.char == "v": self.v_pressed = False
            elif key.char == "e": self.e_pressed = False
            elif key.char == "r": self.r_pressed = False
            elif key.char == "a":
                self.a_pressed = False
                self.changed_gripper_state = False
        except:
            pass

    # Helper function to get Robot End-Effector Position
    def get_eef_pos (self):
        return self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[:3]

    # Helper function to get Door Position (Base of the door)
    def get_door_pos (self):
        return torch.squeeze(self.scene["door"].data.root_link_pos_w).detach().cpu().numpy()

    def render_camera (self, camera_name):
        cam_img = self.scene[camera_name].data.output["rgb"].detach().cpu().numpy()[0]
        self.im.set_data(cam_img)
        plt.pause(1e-6)
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.draw()

    def teleop (self):

        # to prevent overwriting existing data
        episode_counter = len(os.listdir(self.dataset_dir)) if self.dataset_dir is not None and os.path.exists(self.dataset_dir) else 0

        while True:
            
            # ================= MODIFICATION START =================
            # 1. 强制定义每轮初始位置 (Forced Initial Position)
            start_pos = np.array([0.4, 0.0, 0.3]) 
            
            # 2. Reset Yaw to 0 (pointing down)
            # This quaternion corresponds to [0, -1, 1, 0] unnormalized (w, x, y, z order dependent on controller, usually wxyz in Isaac)
            # Based on your script's logic:
            base_quat = np.array([0, -np.sqrt(2)/2, np.sqrt(2)/2, 0]) 
            current_yaw = 0.0
            
            # 3. Construct Start Pose
            # Concatenate Position + Quaternion
            start_pose = np.concatenate([start_pos, base_quat])
            
            # 4. Reset Controller and Force Move
            self.diff_ik_controller.reset()
            self.diff_ik_controller.set_command(torch.tensor(start_pose, device = self.sim.device))
            
            print(f"\nEpisode {episode_counter}: Resetting robot to start position {start_pos}...")

            # 5. Warm-up Loop: 自动运行 60 步，让机器人物理移动到起点
            # 如果不加这一段，机器人不会真的瞬移过去，而是在你按下按键时才开始修正
            for _ in range(60):
                # 获取当前状态
                jacobian = self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
                ee_pose_w = self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
                joint_pos = self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]
                
                # 计算 IK
                joint_pos_des = self.diff_ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos)
                
                # 设置关节目标 (默认 gripper 打开)
                all_joint_pos_des = torch.zeros((1, 8))
                all_joint_pos_des[:, :6] = joint_pos_des
                all_joint_pos_des[:, 6:] = torch.tensor([0.05, 0.05]).to(self.sim.device)

                self.scene["ur5e"].set_joint_position_target(all_joint_pos_des)
                self.scene.write_data_to_sim()
                self.sim.step()
            
            print("Robot ready. Starting Teleop...")
            # ================= MODIFICATION END =================
            
            
            # initialize keyboard listener
            listener = Listener(on_press=self.on_press, on_release=self.on_release)
            listener.start()
            print("\nTeleoperation started\n")

            print("Control robot:")
            print(" - d/x: +/- X axis")
            print(" - z/c: +/- Y axis")
            print(" - v/g: +/- Z axis")
            print(" - e/r: Rotate Left/Right (Yaw)")
            print(" - a: Open/Close Gripper")
            print(" - s: Save Episode")
            print(" - q: Discard Episode\n")

            # teleop params
            max_joint_change = 0.10
            temp_dist_target = 0.0018
            rot_step = 0.05 # Radians per step for rotation

            is_stationary = True
            recorded_stationary_pose = False
            stationary_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()

            gripper_state = -1 # -1 open, 1 close

            self.close_cur_episode = False
            self.save_cur_episode = False

            # clear demo trajectory
            state_observations = []
            actions = []

            last_saved_robot_pos = self.get_eef_pos()
            last_saved_gripper_state = gripper_state
            last_saved_door_pos = self.get_door_pos()
            last_saved_yaw = current_yaw # Track yaw changes
            
            pos_change_threshold = 0.004 
            rot_change_threshold = 0.04 # Save point if rotated enough

            # teleop loop
            while simulation_app.is_running():
                
                if self.close_cur_episode:
                    
                    if self.save_cur_episode and self.dataset_dir is not None and len(state_observations) != 0:
                        # Append end padding
                        end_observation = state_observations[-1].copy()
                        end_observation[3] = -1 
                        end_action = np.zeros(11) # CHANGED to 11
                        end_action[8] = 1 # Stationary action index is now 8

                        cur_image = self.scene["birdview_camera"].data.output["rgb"].detach().cpu().numpy()[0]
                        cur_image = cv2.cvtColor(cur_image, cv2.COLOR_RGB2BGR)

                        for i in range (10):
                            if not os.path.exists("{}/demo_{}/images".format(self.dataset_dir, episode_counter)):
                                os.makedirs("{}/demo_{}/images".format(self.dataset_dir, episode_counter))
                            
                            cv2.imwrite("{}/demo_{}/images/{}.png".format(self.dataset_dir, episode_counter, len(state_observations)), cur_image)
                            state_observations.append(end_observation)
                            actions.append(end_action)

                        np.savez("{}/demo_{}/states.npz".format(self.dataset_dir, episode_counter), state_observations=state_observations, actions=actions)
                        print("\nSaved episode {}.\n".format(episode_counter))
                        episode_counter += 1

                    elif len(state_observations) == 0:
                        if (self.dataset_dir is not None) and (os.path.exists("{}/demo_{}".format(self.dataset_dir, episode_counter))):
                            shutil.rmtree("{}/demo_{}".format(self.dataset_dir, episode_counter))
                        print("\nEmpty trajectory.\n")
                    else:
                        if (self.dataset_dir is not None) and (os.path.exists("{}/demo_{}".format(self.dataset_dir, episode_counter))):
                            shutil.rmtree("{}/demo_{}".format(self.dataset_dir, episode_counter))
                        print("\nDiscarded.\n")

                    # ===== RESET =====
                    ur5e_state = self.scene["ur5e"].data.default_root_state.clone()
                    self.scene["ur5e"].write_root_pose_to_sim(ur5e_state[:, :7])
                    self.scene["ur5e"].write_root_velocity_to_sim(ur5e_state[:, 7:])
                    joint_pos, joint_vel = (self.scene["ur5e"].data.default_joint_pos.clone(), self.scene["ur5e"].data.default_joint_vel.clone())
                    self.scene["ur5e"].write_joint_state_to_sim(joint_pos, joint_vel)
                    door_joint_pos = torch.zeros((self.scene["door"].num_instances, 1), device=self.sim.device)
                    door_joint_vel = torch.zeros((self.scene["door"].num_instances, 1), device=self.sim.device)
                    self.scene["door"].write_joint_state_to_sim(door_joint_pos, door_joint_vel)
                    
                    self.scene.reset()
                    listener.stop()
                    break
                
                # Check stationary
                if not (self.d_pressed or self.x_pressed or self.z_pressed or self.c_pressed or self.g_pressed or self.v_pressed or self.e_pressed or self.r_pressed):
                    is_stationary = True
                else:
                    is_stationary = False
                    
                if is_stationary:
                    if not recorded_stationary_pose:
                        stationary_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
                        recorded_stationary_pose = True
                else:
                    recorded_stationary_pose = False

                # 1. Update Position Targets
                temp_target = stationary_pose[:3].copy()
                cur_pos = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[:3]
                
                if self.d_pressed: temp_target[0] = cur_pos[0] - temp_dist_target
                elif self.x_pressed: temp_target[0] = cur_pos[0] + temp_dist_target
                elif self.z_pressed: temp_target[1] = cur_pos[1] - temp_dist_target
                elif self.c_pressed: temp_target[1] = cur_pos[1] + temp_dist_target
                elif self.g_pressed: temp_target[2] = cur_pos[2] + temp_dist_target
                elif self.v_pressed: temp_target[2] = cur_pos[2] - temp_dist_target
                
                # 2. Update Rotation Targets (Yaw)
                if self.e_pressed: current_yaw += rot_step
                elif self.r_pressed: current_yaw -= rot_step
                
                # 3. Update Gripper
                if self.a_pressed:
                    if not self.changed_gripper_state:
                        gripper_state *= -1
                        self.changed_gripper_state = True

                # Calculate Target Orientation (Base * Z-Rotation)
                # Convert base quat to matrix, rotate, convert back
                base_rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]) # Scalar-last for scipy
                z_rot = R.from_euler('z', current_yaw)
                final_rot = base_rot * z_rot
                final_quat_scipy = final_rot.as_quat() # xyzw
                # Isaac expects wxyz (or sometimes xyzw depending on controller, but IK usually takes wxyz tensor, here we use numpy for prep)
                # The code below `target_pose` concat seems to expect [x,y,z, w, x, y, z] based on previous context "qw, qx, qy, qz"
                target_quat = np.array([final_quat_scipy[3], final_quat_scipy[0], final_quat_scipy[1], final_quat_scipy[2]])

                # Construct Pose
                target_pos = temp_target
                target_pose = np.array(target_pos.tolist() + target_quat.tolist())

                # IK Command
                if is_stationary:
                    self.diff_ik_controller.set_command(torch.tensor(stationary_pose, device = self.sim.device))
                else:
                    self.diff_ik_controller.set_command(torch.tensor(target_pose, device = self.sim.device))
                
                jacobian = self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
                ee_pose_w = self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
                joint_pos = self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]
                joint_pos_des = self.diff_ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos)

                # Limit overshoot
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

                # Update stationary pose
                cur_pos = self.get_eef_pos()
                # If moved position, update pos; if rotated, update orientation in stationary pose
                if not is_stationary:
                    # We reconstruct stationary pose from current physics state to avoid drift
                    # But we must ensure "stationary" means "user not pressing key"
                    # For rotation, we just rely on `current_yaw` variable which holds state
                    if self.d_pressed or self.x_pressed or self.z_pressed or self.c_pressed or self.g_pressed or self.v_pressed:
                         stationary_pose[:3] = cur_pos
                    
                    # Update stationary pose orientation from current_yaw
                    stationary_pose[3:] = target_quat

                # === DATA SAVING LOGIC ===
                cur_image = self.scene["birdview_camera"].data.output["rgb"].detach().cpu().numpy()[0]
                cur_image = cv2.cvtColor(cur_image, cv2.COLOR_RGB2BGR)
                
                dist_moved = np.sqrt(np.sum((cur_pos - last_saved_robot_pos)**2))
                rot_moved = abs(current_yaw - last_saved_yaw)

                if (dist_moved > pos_change_threshold) or (rot_moved > rot_change_threshold) or (gripper_state != last_saved_gripper_state):

                    if self.z_pressed or self.x_pressed or self.c_pressed or self.d_pressed or self.v_pressed or self.g_pressed or \
                        self.e_pressed or self.r_pressed or (gripper_state != last_saved_gripper_state):

                        # === OBSERVATION (8 DIMENSIONS) ===
                        # Added current_yaw at the end
                        cur_observation = np.array([
                            last_saved_robot_pos[0],
                            last_saved_robot_pos[1],
                            last_saved_robot_pos[2],
                            last_saved_gripper_state,
                            last_saved_door_pos[0],
                            last_saved_door_pos[1],
                            last_saved_door_pos[2],
                            last_saved_yaw # <--- NEW: Rotation State
                        ])

                        # === ACTION (11 DIMENSIONS) ===
                        # [x+, x-, y+, y-, z+, z-, open, close, stationary, rot+, rot-]
                        cur_action = np.zeros(11)
                        if self.x_pressed: cur_action[0] = 1
                        elif self.d_pressed: cur_action[1] = 1
                        elif self.c_pressed: cur_action[2] = 1
                        elif self.z_pressed: cur_action[3] = 1
                        elif self.g_pressed: cur_action[4] = 1
                        elif self.v_pressed: cur_action[5] = 1
                        elif gripper_state != last_saved_gripper_state:
                            if gripper_state == -1: cur_action[6] = 1
                            elif gripper_state == 1: cur_action[7] = 1
                        elif self.e_pressed: cur_action[9] = 1 # Rotate Left
                        elif self.r_pressed: cur_action[10] = 1 # Rotate Right

                        # Save
                        if (self.dataset_dir is not None) and (not os.path.exists("{}/demo_{}/images".format(self.dataset_dir, episode_counter))):
                            os.makedirs("{}/demo_{}/images".format(self.dataset_dir, episode_counter))

                        if self.dataset_dir is not None:
                            cv2.imwrite("{}/demo_{}/images/{}.png".format(self.dataset_dir, episode_counter, len(state_observations)), cur_image)
                            state_observations.append(cur_observation)
                            actions.append(cur_action)

                            print(f"Pt {len(state_observations)} | Act: {np.argmax(cur_action)} | Yaw: {current_yaw:.2f}")
                            
                            last_saved_robot_pos = self.get_eef_pos()
                            last_saved_gripper_state = gripper_state
                            last_saved_door_pos = self.get_door_pos()
                            last_saved_yaw = current_yaw


if __name__ == "__main__":

    # TODO: change to your dataset path
    dataset_dir = r'C:\IsaacLab\cs498\final\image'

    exp = Experiment(dataset_dir = dataset_dir)
    exp.teleop()