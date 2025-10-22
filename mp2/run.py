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

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene

import cv2
from scipy.spatial.transform import Rotation
from scipy import ndimage

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
            

    def run (self):
        '''
        You code goes here.
        '''
        # birdview_camera intrinsic and extrinsic matrix
        intrinsics = np.squeeze(self.scene["birdview_camera"].data.intrinsic_matrices.detach().cpu().numpy())
        #extrinsics = np.array([
        #    [ 0,  1,  0, 0.5],
        #    [ 1,  0,  0,   0],
        #    [ 0,  0, -1, 1.2],
        #    [ 0,  0,  0,   1]
        #])

        extrinsics = np.array([
            [ 0,  1,  0, 0],
            [ -1,  0,  0,  0.5],
            [ 0,  0, -1, 1.2],
            [ 0,  0,  0,   1]
        ])

        # move the robot out of the way for getting information from birdview camera
        self.sim_wait(20)
        target_robot_pos = self.robot_pos - np.array([0.15, 0.2, 0.0])
        self.move_robot_ik(np.concatenate([target_robot_pos, self.robot_quat]))
        self.sim_wait(20)

        # render birdview camera image
        color = self.scene["birdview_camera"].data.output["rgb"].detach().cpu().numpy()[0]
        plt.imshow(color)
        
        # TODO: move the robot to make the stack
        hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)

        lower_red1 = np.array([0, 120, 40])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 40])
        upper_red2 = np.array([180, 255, 255])
        lower_green = np.array([40, 100, 50])
        upper_green = np.array([80, 255, 255])

        mask_red1 =cv2.inRange(hsv, lower_red1, upper_red1) # check every element in hsv, whether it is staying between lower bound and opper bound
        # if satisfied with three condition(h,s,v), set it to 255
        mask_red2 =cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2) # Combined 2 masks
        green_mask = cv2.inRange(hsv, lower_green, upper_green)


        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(color)
        axs[0].set_title('Original Image')
        axs[1].imshow(red_mask, cmap='gray')
        axs[1].set_title('Red Cube Mask')
        axs[2].imshow(green_mask, cmap='gray')
        axs[2].set_title('Green Cube Mask')
        plt.savefig("color_masks.png") # Save the figure instead
        plt.close()

        depth_image = np.squeeze(self.scene["birdview_camera"].data.output["depth"].detach().cpu().numpy()[0])
        height, width = depth_image.shape

        fx = intrinsics[0,0]
        fy = intrinsics[1,1]
        cx = intrinsics[0,2]
        cy = intrinsics[1,2]

        y_pixel, x_pixel = np.indices((height, width))
        z_c = depth_image
        x_c = z_c * (x_pixel - cx) / fx
        y_c = z_c * (y_pixel - cy) / fy

        points_camera = np.stack((x_c, y_c, z_c), axis=-1).reshape(-1, 3)
        points_camera_h = np.hstack((points_camera, np.ones((points_camera.shape[0], 1))))

        points_world_h = (np.linalg.inv(extrinsics) @ points_camera_h.T).T
        points_world = points_world_h[:, :3]

        red_pixels = red_mask.flatten().astype(bool)
        green_pixels = green_mask.flatten().astype(bool)

        red_points = points_world[red_pixels]
        green_points = points_world[green_pixels]
       
        point_colors = np.full((points_world.shape[0], 3), [0.5, 0.5, 0.5])
        point_colors[red_pixels] = [1, 0, 0]
        point_colors[green_pixels] = [0, 1, 0]

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
        scene.set_title("Scene Point Cloud (Full)")
        scene.set_aspect("equal")
        scene.view_init(elev=60, azim=-45)

        plt.savefig("scene_point_cloud_full.png")
        plt.close(fig)

        # Calculate centroid from the raw red mask data
        CUBE_HEIGHT = 0.06
        red_centroid = None
        
        # red_centroid = np.mean(red_points, axis=0)
        if red_points.shape[0] > 0:
            r_top_z = np.max(red_points[:, 2])

                # 2. The X and Y of the center is the average of the visible points
            r_center_x = np.mean(red_points[:, 0])
            
            r_center_y = np.mean(red_points[:, 1])

            # 3. The true Z center is half a cube height below the top surface
            
            r_center_z = r_top_z - (CUBE_HEIGHT / 2.0)

                # 4. Assemble the corrected, more accurate centroid
            red_centroid = np.array([r_center_x, r_center_y, r_center_z])
            print(f"CALCULATED RED CUBE CENTROID : {red_centroid}")   

        # Find centroids for all green obstacles
        green_centroids = []
        labeled_mask, num_features = ndimage.label(green_mask)
        if num_features > 0:
   
            for i in range(1, num_features + 1):
                blob_mask = (labeled_mask.flatten() == i)
                green_points = points_world[blob_mask]
                if green_points.shape[0] > 0:
                    top_z = np.max(green_points[:, 2])
                    center_x = np.mean(green_points[:, 0])
                    center_y = np.mean(green_points[:, 1])
                    center_z = top_z - (CUBE_HEIGHT / 2.0)
                    corrected_centroid = np.array([center_x, center_y, center_z])
                    green_centroids.append(corrected_centroid)
        


        if red_centroid is not None:
            # 定义关键高度
            pre_grasp_height = 0.12  # 预抓取高度：立方体上方10厘米
            grasp_height = 0.03      # 抓取高度：立方体上方2厘米
            GRIPPER_OPEN_POS = 0.035    # 夹爪完全张开的位置 (m)
            GRIPPER_CLOSED_POS = 0.0   # 夹爪完全闭合的位置 (m)
            PRE_MOVE_HEIGHT = 0.12 # Safe height for moves

             # The best base is the one most isolated from other cubes, offering more room to maneuver.
            best_base = None
            max_isolation_dist = -1
            all_cubes = [red_centroid] + green_centroids
            for candidate_base in green_centroids:
                dist_sum = 0
                for other_cube in all_cubes:
                    if not np.array_equal(candidate_base, other_cube):
                        dist_sum += np.linalg.norm(candidate_base - other_cube)
                if dist_sum > max_isolation_dist:
                    max_isolation_dist = dist_sum
                    best_base = candidate_base
            base_green_cube = best_base

            remaining_green = [gc for gc in green_centroids if not np.array_equal(gc, base_green_cube)]
            top_green_cube = remaining_green[0]
            print(f"Base green cube selected at: {base_green_cube}")
            print(f"Top green cube selected at: {top_green_cube}")

            avg_obstacle_vector = np.array([0.0, 0.0, 0.0])
            if green_centroids:
                for gc in green_centroids:
                    avg_obstacle_vector += (gc - red_centroid)
                avg_obstacle_vector /= len(green_centroids)

            if np.linalg.norm(avg_obstacle_vector) > 1e-6:
                approach_vector = -avg_obstacle_vector
            else:
                approach_vector = np.array([1.0, 0.0, 0.0])
            
            # 将接近向量投影到XY平面并归一化
            approach_vector_xy = approach_vector[:2] / np.linalg.norm(approach_vector[:2])
            # 定义夹爪自身的坐标系
            z_axis = np.array([0.0, 0.0, -1.0]) # Z轴永远指向正下方
            x_axis = np.array([approach_vector_xy[0], approach_vector_xy[1], 0.0]) # X轴（前方）对准接近方向
            y_axis = np.cross(z_axis, x_axis) # Y轴（侧方）通过叉乘计算得出

            # 构建旋转矩阵并转换为四元数
            rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
            quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
            target_quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

            fixed_orientation = Rotation.from_euler('xyz', [180, 0, 90], degrees=True).as_quat()[[3, 0, 1, 2]]




            self.move_robot_joint(target_joint_pos=None, target_gripper_pos=GRIPPER_OPEN_POS, count=50)
            self.sim_wait(30)

            pre_grasp_pos = red_centroid + np.array([0.0, 0.0, pre_grasp_height])
            pre_grasp_pose = np.concatenate([pre_grasp_pos, target_quat_wxyz])
            self.move_robot_ik(pre_grasp_pose)
            self.sim_wait(30)

            grasp_pos = red_centroid + np.array([0.0, 0.0, grasp_height])
            grasp_pose = np.concatenate([grasp_pos, target_quat_wxyz])
            self.move_robot_ik(grasp_pose)
            self.sim_wait(30)
  
            self.move_robot_joint(target_joint_pos=None, target_gripper_pos=GRIPPER_CLOSED_POS, count=50)
            self.sim_wait(50)

            print("Picking the red cube")
            self.move_robot_ik(pre_grasp_pose) # Move back up to the pre-grasp height
            self.sim_wait(50)

            print("Placeing the red cube")
            place_pos_red = base_green_cube + np.array([0, 0, CUBE_HEIGHT])
            pre_place_pos = place_pos_red + np.array([0.0, 0.0, PRE_MOVE_HEIGHT])
            # self.move_robot_ik(np.concatenate([pre_place_pos, fixed_orientation])); self.sim_wait(50)
            self.move_robot_ik(np.concatenate([pre_place_pos, fixed_orientation])); self.sim_wait(50)
            self.move_robot_ik(np.concatenate([place_pos_red, fixed_orientation])); self.sim_wait(50)
            self.move_robot_joint(None, GRIPPER_OPEN_POS, count=50); self.sim_wait(50)
            self.move_robot_ik(np.concatenate([pre_place_pos, fixed_orientation])); self.sim_wait(40)

            # === MOVE 2: Place TOP GREEN cube on RED cube ===
            print(f"--- Picking TOP GREEN cube at {top_green_cube} ---")
            pre_grasp_pos = top_green_cube + np.array([0.0, 0.0, PRE_MOVE_HEIGHT])
            grasp_pos = top_green_cube + np.array([0.0, 0.0, grasp_height])
            self.move_robot_ik(np.concatenate([pre_grasp_pos, fixed_orientation])); self.sim_wait(40)
            self.move_robot_ik(np.concatenate([grasp_pos, fixed_orientation])); self.sim_wait(40)
            self.move_robot_joint(None, GRIPPER_CLOSED_POS, count=50); self.sim_wait(50)
            self.move_robot_ik(np.concatenate([pre_grasp_pos, fixed_orientation])); self.sim_wait(40)
            
            place_pos_green_top = base_green_cube + np.array([0, 0, CUBE_HEIGHT * 2])
            print(f"--- Placing TOP GREEN cube at {place_pos_green_top} ---")
            pre_place_pos = place_pos_green_top + np.array([0.0, 0.0, PRE_MOVE_HEIGHT])
            self.move_robot_ik(np.concatenate([pre_place_pos, fixed_orientation])); self.sim_wait(50)
            self.move_robot_ik(np.concatenate([place_pos_green_top, fixed_orientation])); self.sim_wait(50)
            self.move_robot_joint(None, GRIPPER_OPEN_POS, count=50); self.sim_wait(50)
            self.move_robot_ik(np.concatenate([pre_place_pos, fixed_orientation])); self.sim_wait(40)

            print(">>> STACKING SEQUENCE COMPLETE! <<<")

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

    """
    # Clean the masks to remove noise before finding contours
        kernel = np.ones((5, 5), np.uint8)
        clean_red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        clean_red_mask = cv2.morphologyEx(clean_red_mask, cv2.MORPH_CLOSE, kernel)
        clean_green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        clean_green_mask = cv2.morphologyEx(clean_green_mask, cv2.MORPH_CLOSE, kernel)

        # Find the largest red contour to get an accurate centroid
        red_centroid = None
        contours_red, _ = cv2.findContours(clean_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_red:
            largest_contour = max(contours_red, key=cv2.contourArea)
            final_red_mask = np.zeros_like(clean_red_mask)
            cv2.drawContours(final_red_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            red_pixels = final_red_mask.flatten().astype(bool)
            red_points = points_world[red_pixels]
            if red_points.shape[0] > 0:
                #red_centroid = np.mean(red_points, axis=0)
                # 1. Find the Z coordinate of the cube's top surface
                top_z = np.max(red_points[:, 2])

                # 2. The X and Y of the center is the average of the visible points
                center_x = np.mean(red_points[:, 0])
                center_y = np.mean(red_points[:, 1])

                # 3. The true Z center is half a cube height below the top surface
                CUBE_HEIGHT = 0.05 
                center_z = top_z - (CUBE_HEIGHT / 2.0)

                # 4. Assemble the corrected, more accurate centroid
                red_centroid = np.array([center_x, center_y, center_z])
                print(f"CALCULATED RED CUBE CENTROID: {red_centroid}")
                """
    
    """
    contours_green, _ = cv2.findContours(clean_green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_green:
            for contour in contours_green:
                mask = np.zeros_like(clean_green_mask)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                green_pixels = mask.flatten().astype(bool)
                green_points = points_world[green_pixels]
                if green_points.shape[0] > 0:
                    top_z = np.max(green_points[:, 2])
                    center_x = np.mean(green_points[:, 0])
                    center_y = np.mean(green_points[:, 1])
                    CUBE_HEIGHT = 0.05
                    center_z = top_z - (CUBE_HEIGHT / 2.0)
                    corrected_centroid = np.array([center_x, center_y, center_z])
                    green_centroids.append(corrected_centroid)
                    #green_centroids.append(np.mean(green_points, axis=0))
                    """
    
    """
    avg_obstacle_vector = np.array([0.0, 0.0, 0.0])
            if green_centroids:
                for gc in green_centroids:
                    avg_obstacle_vector += (gc - red_centroid)
                avg_obstacle_vector /= len(green_centroids)

            if np.linalg.norm(avg_obstacle_vector) > 1e-6:
                approach_vector = -avg_obstacle_vector
            else:
                approach_vector = np.array([1.0, 0.0, 0.0])
                """