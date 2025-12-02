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
from pynput.keyboard import Key, Listener
from scipy.spatial.transform import Rotation as R
import cv2

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene

from task_envs import MP4SceneCfg, PHYSICS_DT, RENDERING_DT


class Experiment:

    def __init__(self, policy, model_path):

        self.policy = policy
        self.model_path = model_path

        if policy not in ["mlp", "cnn"]:
            print("Unrecognized policy! Please select between [\"mlp\", \"cnn\"].")
            exit()
        
        # initialize sim
        sim_cfg = sim_utils.SimulationCfg(device = args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([1.1, 0.0, 0.7], [0.0, 0.0, 0.0])

        # set time step size
        self.sim.set_simulation_dt(physics_dt = PHYSICS_DT, rendering_dt = RENDERING_DT)
        print("\nSim dt: {}\n".format(self.sim.get_physics_dt()))
        self.sim_dt = self.sim.get_physics_dt()

        # initialize scene
        scene_cfg = MP4SceneCfg(args_cli.num_envs, env_spacing=2.0)
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

        # resetting scene
        self.start_new_episode = False


    def on_press (self, key):
        try:
            if key.char == "q":
                self.start_new_episode = True
        except:
            pass
    

    def on_release (self, key):
        pass


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
            

    def sim_wait (self, count):
        '''
        Wait for a given number of timesteps in simulation.
        '''

        print("Waiting...")
        for _ in range (count):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)


    def render_camera (self, camera_name):
        '''
        Render and visualize camera images in a loop
        '''

        cam_img = self.scene[camera_name].data.output["rgb"].detach().cpu().numpy()[0]

        self.im.set_data(cam_img)
        plt.pause(1e-6)
        # Disconnect all key press event handlers so we can use more keys for teleop
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.draw()


    def get_eef_pos (self):
        return self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[:3]


    def get_red_cube_pos (self):
        return torch.squeeze(self.scene["cubeA"].data.root_link_pos_w).detach().cpu().numpy()
    

    def get_green_cube_pos (self):
        return torch.squeeze(self.scene["cubeB"].data.root_link_pos_w).detach().cpu().numpy()


    def inference (self):

        while True:

            # TODO: model definition
            # ==========================================================================================

            if self.policy == "mlp":
                from train_mlp import Policy
                model = Policy()  # TODO: change to your policy definition

            elif self.policy == "cnn":
                from train_cnn import Policy
                model = Policy()  # TODO: change to your policy definition
            
            model.load_state_dict(torch.load(self.model_path))
            model.eval()
            model.to('cpu')

            # ==========================================================================================
            
            # we do not need to change the robot eef orientation
            target_quat = np.array([0, -np.sqrt(2), np.sqrt(2), 0])

            # move robot to starting position
            target_robot_pos = np.array([0.4, 0.0, 0.355])
            self.move_robot_ik(np.concatenate([target_robot_pos, target_quat]))

            # initialize keyboard listener
            listener = Listener(
                on_press=self.on_press,
                on_release=self.on_release
            )
            listener.start()

            # inference params
            max_joint_change = 0.1  # larger = faster movement
            temp_dist_target = 0.003  # larger = faster movement

            stationary_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()

            # gripper state indicator, -1 for open and 1 for close
            gripper_state = -1

            self.start_new_episode = False
            print("\nInference started. Press q to start a new episode.\n")

            # inference
            while simulation_app.is_running():
                
                if self.start_new_episode:

                    # DO NOT MODIFY THE RESET CODE
                    
                    # ===== reset robot =====

                    # set root state
                    ur5e_state = self.scene["ur5e"].data.default_root_state.clone()
                    # ur5e_state[:, :3] += self.scene.env_origins
                    self.scene["ur5e"].write_root_pose_to_sim(ur5e_state[:, :7])
                    self.scene["ur5e"].write_root_velocity_to_sim(ur5e_state[:, 7:])

                    # set joint states
                    joint_pos, joint_vel = (
                        self.scene["ur5e"].data.default_joint_pos.clone(),
                        self.scene["ur5e"].data.default_joint_vel.clone(),
                    )
                    self.scene["ur5e"].write_joint_state_to_sim(joint_pos, joint_vel)


                    # ===== reset cubes =====

                    # randomize cube poses
                    center_x = 0.5
                    center_y = 0.0
                    cube_pos_noise = 0.025

                    # red cube
                    red_cube_x = center_x - 0.03 + (np.random.random() - 0.5)*2 * cube_pos_noise
                    red_cube_y = center_y - 0.1 + (np.random.random() - 0.5)*2 * cube_pos_noise
                    self.scene["cubeA"].write_root_pose_to_sim(torch.tensor([red_cube_x, red_cube_y, 0.225, 1, 0, 0, 0])[None, :])
                    self.scene["cubeA"].write_root_velocity_to_sim(torch.tensor([0, 0, 0, 0, 0, 0])[None, :])

                    # green cube
                    green_cube_x = center_x + 0.03 + (np.random.random() - 0.5)*2 * cube_pos_noise
                    green_cube_y = center_y + 0.1 + (np.random.random() - 0.5)*2 * cube_pos_noise
                    self.scene["cubeB"].write_root_pose_to_sim(torch.tensor([green_cube_x, green_cube_y, 0.225, 1, 0, 0, 0])[None, :])
                    self.scene["cubeB"].write_root_velocity_to_sim(torch.tensor([0, 0, 0, 0, 0, 0])[None, :])


                    self.scene.reset()
                    listener.stop()

                    break
                

                # TODO: predict action. you might want to modify the input based on your model definition
                # ==========================================================================================

                # compile observations
                # input("Press enter to predict next action\n")
                if self.policy == "mlp":
                    # get current observations
                    cur_observation = np.array([
                        self.get_eef_pos()[0],
                        self.get_eef_pos()[1],
                        self.get_eef_pos()[2],
                        gripper_state,
                        self.get_red_cube_pos()[0],
                        self.get_red_cube_pos()[1],
                        self.get_red_cube_pos()[2],
                        self.get_green_cube_pos()[0],
                        self.get_green_cube_pos()[1],
                    ])
                    model_input = torch.tensor(cur_observation).to(torch.float32)
                
                elif self.policy == "cnn":
                    cur_image = self.scene["robotview_camera"].data.output["rgb"].detach().cpu().numpy()[0]
                    cur_image = cv2.cvtColor(cur_image, cv2.COLOR_RGB2BGR)

                    model_input = torch.tensor(cur_image).to(torch.float32).permute(2, 0, 1) / 255.0  # normalize to [0, 1]
                    model_input = torch.unsqueeze(model_input, 0)  # add batch dimension
                # ==========================================================================================    
                # predict
                with torch.no_grad():
                    action_pred = model(model_input).numpy()
                
                # ==========================================================================================

                
                # print(np.round(action_pred, 3), "\n")
                action_index = np.argmax(action_pred)  # [+x, -x, +y, -y, +z, -z, gripper_open, gripper_close, stationary]

                # set new robot pose according to model prediction
                # position
                temp_target = stationary_pose[:3].copy()
                cur_pos = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[:3]
                if action_index == 0:
                    temp_target[0] = cur_pos[0] + temp_dist_target

                elif action_index == 1:
                    temp_target[0] = cur_pos[0] - temp_dist_target

                elif action_index == 2:
                    temp_target[1] = cur_pos[1] + temp_dist_target

                elif action_index == 3:
                    temp_target[1] = cur_pos[1] - temp_dist_target

                elif action_index == 4:
                    temp_target[2] = cur_pos[2] + temp_dist_target

                elif action_index == 5:
                    temp_target[2] = cur_pos[2] - temp_dist_target

                elif action_index == 6:
                    gripper_state = -1
                    self.move_robot_joint(None, 0.05)
                
                elif action_index == 7:
                    gripper_state = 1
                    self.move_robot_joint(None, 0)

                # position change
                T_world_eefnew = np.eye(4)
                T_world_eefnew[:3, :3] = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_matrix()
                T_world_eefnew[:3, 3] = temp_target

                # format conversion for isaaclab's ik controller
                target_pos = T_world_eefnew[:3, 3]
                target_pose = np.array(target_pos.tolist() + target_quat.tolist())

                # IK controller
                if action_index == 8:
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
                
                # set gripper joint target according to gripper_state
                if gripper_state == -1:
                    all_joint_pos_des[:, 6:] = torch.tensor([0.05, 0.05]).to(self.sim.device)
                else:
                    all_joint_pos_des[:, 6:] = torch.tensor([0.0, 0.0]).to(self.sim.device)

                # send command to robot
                self.scene["ur5e"].set_joint_position_target(all_joint_pos_des)

                # visualize robotview camera
                # self.render_camera("robotview_camera")

                # step simulation
                self.scene.write_data_to_sim()
                self.sim.step()
                self.scene.update(self.sim_dt)

                # update stationary pose
                cur_pos = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[:3]
                if action_index == 0 or action_index == 1:  # if moved in x direction
                    stationary_pose[0] = cur_pos[0]

                elif action_index == 2 or action_index == 3:  # if moved in y direction
                    stationary_pose[1] = cur_pos[1]

                elif action_index == 4 or action_index == 5:  # if moved in z direction
                    stationary_pose[2] = cur_pos[2]


if __name__ == "__main__":

    # TODO: change to your model path
    # model_path = "mlp_model.pth"
    model_path = "policy_model.pth"

    exp = Experiment(policy = "cnn", model_path = model_path)
    exp.inference()