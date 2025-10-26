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
        self.robot_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]


    def run (self):
        '''
        Your code goes here.
        '''


        # TODO: Pose estimation and grasping


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