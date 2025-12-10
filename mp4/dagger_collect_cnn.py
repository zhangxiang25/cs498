import argparse
import os
import shutil
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pynput.keyboard import Key, Listener
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from task_envs import MP4SceneCfg, PHYSICS_DT, RENDERING_DT

from train_cnn import Policy

class DaggerExperiment:

    def __init__(self, dataset_dir, model_path):
        self.dataset_dir = dataset_dir
        self.model_path = model_path

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        if not os.path.exists(self.model_path):
            print(f"\nerror: Model not found at {self.model_path}")
            print("Please run 'python train_mlp.py' to train the model first.\n")
            exit()


        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([1.1, 0.0, 0.7], [0.0, 0.0, 0.0])
        self.sim.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
        self.sim_dt = self.sim.get_physics_dt()

        scene_cfg = MP4SceneCfg(args_cli.num_envs, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        print("Setup complete...")

        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)
        self.ik_body = "gripper_center"
        self.robot_entity_cfg = SceneEntityCfg(
            "ur5e", 
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], 
            body_names=[self.ik_body]
        )
        self.robot_entity_cfg.resolve(self.scene)
        self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] if not self.scene["ur5e"].is_fixed_base else self.robot_entity_cfg.body_ids[0] - 1

        self.fig, ax = plt.subplots()
        self.im = ax.imshow((np.ones((256, 256))*255).astype(np.uint8))

        self.d_pressed = False
        self.x_pressed = False
        self.z_pressed = False
        self.c_pressed = False
        self.g_pressed = False
        self.v_pressed = False
        self.a_pressed = False
        self.changed_gripper_state = False


        self.policy_mode = True      
        self.collecting_data = False 
        self.save_episode = False
        self.reset_episode = False

    def on_press(self, key):
        try:
            char = key.char
            if char == "k":
                if self.policy_mode:
                    print("\n[Intervention] Switching to manual control and starting data recording...")
                    self.policy_mode = False
                    self.collecting_data = True

            elif char == "d": self.d_pressed = True
            elif char == "x": self.x_pressed = True
            elif char == "z": self.z_pressed = True
            elif char == "c": self.c_pressed = True
            elif char == "g": self.g_pressed = True
            elif char == "v": self.v_pressed = True
            elif char == "a": self.a_pressed = True
            
            elif char == "s": 
                if not self.policy_mode: # Only allow saving after intervention
                    self.save_episode = True
                    self.reset_episode = True
            elif char == "q": 
                self.save_episode = False
                self.reset_episode = True

        except AttributeError:
            pass

    def on_release(self, key):
        try:
            char = key.char
            if char == "d": self.d_pressed = False
            elif char == "x": self.x_pressed = False
            elif char == "z": self.z_pressed = False
            elif char == "c": self.c_pressed = False
            elif char == "g": self.g_pressed = False
            elif char == "v": self.v_pressed = False
            elif char == "a": 
                self.a_pressed = False
                self.changed_gripper_state = False
        except AttributeError:
            pass

    def get_eef_pos(self):
        return self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[:3]

    def get_red_cube_pos(self):
        return torch.squeeze(self.scene["cubeA"].data.root_link_pos_w).detach().cpu().numpy()

    def get_green_cube_pos(self):
        return torch.squeeze(self.scene["cubeB"].data.root_link_pos_w).detach().cpu().numpy()

    def run_dagger(self):
        print(f"Loading model from {self.model_path}...")
        model = Policy()
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.eval()
        model.to('cpu')

        existing_demos = [d for d in os.listdir(self.dataset_dir) if d.startswith('demo_')]
        episode_counter = len(existing_demos)
        print(f"Next demo will be saved as demo_{episode_counter}")

        while simulation_app.is_running():
        
            # Reset episode logic
            ur5e_state = self.scene["ur5e"].data.default_root_state.clone()
            self.scene["ur5e"].write_root_pose_to_sim(ur5e_state[:, :7])
            self.scene["ur5e"].write_root_velocity_to_sim(ur5e_state[:, 7:])
            
            joint_pos, joint_vel = self.scene["ur5e"].data.default_joint_pos.clone(), self.scene["ur5e"].data.default_joint_vel.clone()
            self.scene["ur5e"].write_joint_state_to_sim(joint_pos, joint_vel)

            # Randomize cube positions
            center_x, center_y, noise = 0.5, 0.0, 0.025
            rx = center_x - 0.03 + (np.random.random() - 0.5)*2 * noise
            ry = center_y - 0.1 + (np.random.random() - 0.5)*2 * noise
            gx = center_x + 0.03 + (np.random.random() - 0.5)*2 * noise
            gy = center_y + 0.1 + (np.random.random() - 0.5)*2 * noise
            
            self.scene["cubeA"].write_root_pose_to_sim(torch.tensor([rx, ry, 0.225, 1, 0, 0, 0])[None, :])
            self.scene["cubeA"].write_root_velocity_to_sim(torch.tensor([0, 0, 0, 0, 0, 0])[None, :])
            self.scene["cubeB"].write_root_pose_to_sim(torch.tensor([gx, gy, 0.225, 1, 0, 0, 0])[None, :])
            self.scene["cubeB"].write_root_velocity_to_sim(torch.tensor([0, 0, 0, 0, 0, 0])[None, :])
            
            self.scene.reset()
            
            # Move to starting position
            target_quat = np.array([0, -np.sqrt(2), np.sqrt(2), 0])
            target_robot_pos = np.array([0.4, 0.0, 0.355])
            
            self.diff_ik_controller.set_command(torch.tensor(np.concatenate([target_robot_pos, target_quat]), device=self.sim.device))
            for _ in range(50): 
                jacobian = self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
                ee_pose_w = self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
                joint_pos = self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]
                joint_pos_des = self.diff_ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos)
                self.scene["ur5e"].set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
                self.scene.write_data_to_sim()
                self.sim.step()
                self.scene.update(self.sim_dt)

            # Start DAgger episode
            listener = Listener(on_press=self.on_press, on_release=self.on_release)
            listener.start()
            
            print(f"\n--- Episode {episode_counter} start ---")
            print("Mode: Policy (robot moves automatically)")
            print("Press 'k': Intervene (take control and start recording)")
            print("Press 's': Save (only valid after intervention)")
            print("Press 'q': Skip/Abort episode\n")

            self.policy_mode = True
            self.collecting_data = False
            self.reset_episode = False
            self.save_episode = False
            
            gripper_state = -1 # Gripper open state
            stationary_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
            
            # Data buffers
            state_observations = []
            actions = []
            last_saved_pos = self.get_eef_pos()
            last_gripper = gripper_state
            pos_threshold = 0.005

        
            while simulation_app.is_running():
                if self.reset_episode:
                    break

                # Get the raw image from the simulator
                # Shape comes in as (1, Height, Width, 3)
                raw_img_tensor = self.scene["robotview_camera"].data.output["rgb"].clone()
                
                # 1. Get state (Still needed for recording ground truth in the dataset)
                current_eef = self.get_eef_pos()
                red_pos = self.get_red_cube_pos()
                green_pos = self.get_green_cube_pos()
                
                obs_np = np.array([
                    current_eef[0], current_eef[1], current_eef[2], gripper_state,
                    red_pos[0], red_pos[1], red_pos[2],
                    green_pos[0], green_pos[1]
                ])

                # 2. Decide action (policy vs teleoperation)
                action_idx = 8 # Default to stationary
                
                if self.policy_mode:
                    # Rearrange from (Batch, H, W, Channel) to (Batch, Channel, H, W)
                    # And normalize to [0, 1]
                    img_input = raw_img_tensor.permute(0, 3, 1, 2).float() / 255.0
                    
                    # Ensure input is on the same device as the model (CPU in your script)
                    img_input = img_input.to('cpu')

                    with torch.no_grad():
                        # The CNN model expects the image tensor, not obs_np
                        pred = model(img_input).squeeze(0).cpu().numpy()
                    action_idx = np.argmax(pred)
                else:
                    # Manual teleoperation
                    if self.x_pressed: action_idx = 0
                    elif self.d_pressed: action_idx = 1
                    elif self.c_pressed: action_idx = 2
                    elif self.z_pressed: action_idx = 3
                    elif self.g_pressed: action_idx = 4
                    elif self.v_pressed: action_idx = 5
                    elif self.a_pressed and not self.changed_gripper_state:
                        gripper_state *= -1
                        self.changed_gripper_state = True
                        action_idx = 7 if gripper_state == 1 else 6 
                    
                # 3. Execute action (apply to simulation)
                temp_dist = 0.003
                temp_target = stationary_pose[:3].copy()
                
                # If moving, update stationary anchor point
                cur_real_pos = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[:3]
                
                if action_idx in [0, 1]: stationary_pose[0] = cur_real_pos[0]
                if action_idx in [2, 3]: stationary_pose[1] = cur_real_pos[1]
                if action_idx in [4, 5]: stationary_pose[2] = cur_real_pos[2]
                
                # Apply increments
                if action_idx == 0: temp_target[0] += temp_dist
                elif action_idx == 1: temp_target[0] -= temp_dist
                elif action_idx == 2: temp_target[1] += temp_dist
                elif action_idx == 3: temp_target[1] -= temp_dist
                elif action_idx == 4: temp_target[2] += temp_dist
                elif action_idx == 5: temp_target[2] -= temp_dist
                elif action_idx == 6: gripper_state = -1 # Explicitly open
                elif action_idx == 7: gripper_state = 1  # Explicitly close

                # IK solver command
                T_new = np.eye(4)
                T_new[:3, :3] = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_matrix()
                T_new[:3, 3] = temp_target
                target_pose_ik = np.array(T_new[:3, 3].tolist() + target_quat.tolist())

                if action_idx == 8: # Stationary
                    self.diff_ik_controller.set_command(torch.tensor(stationary_pose, device=self.sim.device))
                else:
                    self.diff_ik_controller.set_command(torch.tensor(target_pose_ik, device=self.sim.device))

                # Compute and apply joint targets
                jacobian = self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
                ee_pose_w = self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
                joint_pos = self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]
                joint_pos_des = self.diff_ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos)
                
                joint_changes = (joint_pos_des - joint_pos).detach().cpu().numpy()[0]
                max_change = 0.1
                if np.sum(np.abs(joint_changes) > max_change) > 0:
                    scale = joint_changes / (np.max(np.abs(joint_changes)) / max_change)
                    joint_pos_des = joint_pos + torch.tensor(scale).unsqueeze(0).to(joint_pos_des.device)

                full_joint_target = torch.zeros((1, 8)).to(self.sim.device)
                full_joint_target[:, :6] = joint_pos_des
                full_joint_target[:, 6:] = torch.tensor([0.05, 0.05] if gripper_state == -1 else [0.0, 0.0]).to(self.sim.device)
                
                self.scene["ur5e"].set_joint_position_target(full_joint_target)
                self.scene.write_data_to_sim()
                self.sim.step()
                self.scene.update(self.sim_dt)

                # 4. Record data (only during intervention collection)
                if self.collecting_data:
                    dist = np.linalg.norm(current_eef - last_saved_pos)
                    
                    if dist > pos_threshold or gripper_state != last_gripper or action_idx in [6, 7]:
                        # Encode one-hot action
                        one_hot = np.zeros(9)
                        one_hot[action_idx] = 1
                        
                        state_observations.append(obs_np)
                        actions.append(one_hot)
                        
                        # Save image (to match dataset structure)
                        img_dir = os.path.join(self.dataset_dir, f"demo_{episode_counter}", "images")
                        if not os.path.exists(img_dir): os.makedirs(img_dir)
                        
                        # Convert tensor back to numpy for saving with OpenCV
                        cur_image = raw_img_tensor[0].detach().cpu().numpy()
                        cur_image = cv2.cvtColor(cur_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(img_dir, f"{len(state_observations)}.png"), cur_image)
                        print(f"Recording... Frames: {len(state_observations)} | Action: {action_idx}")
                        
                        last_saved_pos = current_eef
                        last_gripper = gripper_state

            listener.stop()

            # Save / Discard
            if self.save_episode and len(state_observations) > 0:
                save_path = os.path.join(self.dataset_dir, f"demo_{episode_counter}", "states.npz")
                np.savez(save_path, state_observations=state_observations, actions=actions)
                print(f"\nSaved demo_{episode_counter} with {len(state_observations)} frames of correction data.\n")
                episode_counter += 1
            elif self.reset_episode:
                print("\n>>> Episode skipped/abandoned.\n")
                # Clean up created folders
                demo_p = os.path.join(self.dataset_dir, f"demo_{episode_counter}")
                if os.path.exists(demo_p): shutil.rmtree(demo_p)

if __name__ == "__main__":
    # Point to your existing dataset here
    DATASET_DIR = r'C:\IsaacLab\cs498\mp4\image'
    MODEL_PATH = "policy_model.pth" 

    dagger = DaggerExperiment(DATASET_DIR, MODEL_PATH)
    dagger.run_dagger()
