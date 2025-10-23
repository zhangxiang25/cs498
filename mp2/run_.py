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
        extrinsics = np.array([
            [ 0,  1,  0, 0.5],
            [ 1,  0,  0,   0],
            [ 0,  0, -1, 1.2],
            [ 0,  0,  0,   1]
        ])

        # move the robot out of the way for getting information from birdview camera
        self.sim_wait(20)
        target_robot_pos = self.robot_pos - np.array([0.15, 0.2, 0.0])
        self.move_robot_ik(np.concatenate([target_robot_pos, self.robot_quat]))
        self.sim_wait(20)

        # render birdview camera image
        color_raw = self.scene["birdview_camera"].data.output["rgb"].detach().cpu().numpy()[0]
        #plt.imshow(color)
        
        # TODO: move the robot to make the stack

        # --- Q1: HSV 颜色阈值 (0~1) 与形态学操作参数 ---
        h_red_low_1 = 0.00
        h_red_high_1 = 0.04
        h_red_low_2 = 0.96
        h_red_high_2 = 1.00
        h_green_low = 0.23
        h_green_high = 0.44
        s_min = 0.40
        v_min = 0.20
        morph_iters = 2

        # --- Q3: 抓取控制 ---
        hover_offset = 0.12
        lift_offset = 0.15
        grasp_depth = 0.02
        

        color = color_raw[:, :, :3] if color_raw.shape[2] == 4 else color_raw
        
       
        height, width, _ = color.shape
        depth_image = np.squeeze(self.scene["birdview_camera"].data.output["depth"].detach().cpu().numpy()[0])
        # Convert RGB to HSV color space
        rgb01 = np.clip(color.astype(np.float32)/255.0, 0.0, 1.0)
        hsv = mcolors.rgb_to_hsv(rgb01)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        s_min, v_min, morph_iters = 0.40, 0.20, 2
        red1_raw = (h >= h_red_low_1) & (h <= h_red_high_1) & (s >= s_min) & (v >= v_min)
        red2_raw = (h >= h_red_low_2) & (h <= h_red_high_2) & (s >= s_min) & (v >= v_min)
        red_mask_raw = red1_raw | red2_raw
        green_mask_raw = (h >= h_green_low) & (h <= h_green_high) & (s >= s_min) & (v >= v_min)

        struct = generate_binary_structure(2, 2)
        red_mask = binary_opening(red_mask_raw, structure=struct, iterations=morph_iters)
        red_mask = binary_fill_holes(red_mask)
        red_mask = binary_closing(red_mask, structure=struct, iterations=max(1, morph_iters // 2))
        green_mask = binary_opening(green_mask_raw, structure=struct, iterations=morph_iters)
        green_mask = binary_fill_holes(green_mask)
        green_mask = binary_closing(green_mask, structure=struct, iterations=max(1, morph_iters // 2))
        

# Now your red_mask and green_mask are much cleaner!

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(color)
        axs[0].set_title('Original Image')
        axs[1].imshow(red_mask, cmap='gray')
        axs[1].set_title('Red Cube Mask')
        axs[2].imshow(green_mask, cmap='gray')
        axs[2].set_title('Green Cube Mask')
        plt.savefig("color_masks.png") # Save the figure instead
        plt.close()

        
        height, width = depth_image.shape
        #

        fx = intrinsics[0,0]
        fy = intrinsics[1,1]
        cx = intrinsics[0,2]
        cy = intrinsics[1,2]
        # test
        y_pixel, x_pixel = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
     
        z_c = depth_image
        x_c = z_c * (x_pixel - cx) / fx
        y_c = z_c * (y_pixel - cy) / fy

        points_camera = np.stack((x_c, y_c, z_c), axis=-1).reshape(-1, 3) #flat pc
        
        ones = np.ones((points_camera.shape[0], 1), dtype=points_camera.dtype)
        points_world_h = extrinsics @ np.concatenate([points_camera, ones], axis=1).T
        pc_world = (points_world_h[:3, :] / np.clip(points_world_h[3, :], 1e-8, None)).T.reshape(height, width, 3)
        #

       
        points_world = (points_world_h[:3,:] / np.clip(points_world_h[3,:], 1e-8, None)).T
        
        pc_world = points_world.reshape(height, width, 3)
        #
        red_pixels = red_mask.flatten().astype(bool)
        green_pixels = green_mask.flatten().astype(bool)
       
        point_colors = np.full((points_world.shape[0], 3), [0.6, 0.6, 0.6])
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
        scene.set_title("Scene Point Cloud")
        scene.set_aspect("equal")
        scene.view_init(elev=60, azim=-45)

        plt.savefig("scene_point_cloud.png")
        plt.close(fig)

        
        # test
        lab_red, n_red = label(red_mask.astype(np.uint8), structure=struct); 
        red_cid_main, _ = max([(cid, (lab_red == cid).sum()) for cid in range(1, n_red + 1)], key=lambda item: item[1])
        idx_red = (lab_red == red_cid_main).reshape(-1); 
        pts_red = pc_world.reshape(-1, 3)[idx_red]; 
        pts_red = pts_red[np.isfinite(pts_red).all(axis=1)]
        centroid_xy_red = np.nanmedian(pts_red[:, :2], axis=0); 
        top_z_red = float(np.nanpercentile(pts_red[:, 2], 95)); 
        low_z_red = float(np.nanpercentile(pts_red[:, 2], 5))
        red_stats = {"centroid_xy": centroid_xy_red, "top_z": top_z_red, "low_z": low_z_red, "size_xy": np.nanpercentile(pts_red[:, :2], 95, axis=0) - np.nanpercentile(pts_red[:, :2], 5, axis=0)}
        red_edge = np.clip(float(np.mean(np.abs(red_stats["size_xy"]))), 0.03, 0.06)

        lab_green, n_green = label(green_mask.astype(np.uint8), structure=struct); 
        green_infos = []
        green_clusters_info = sorted([(cid, (lab_green == cid).sum()) for cid in range(1, n_green + 1)], key=lambda item: item[1], reverse=True)
        for cid, area in green_clusters_info[:2]:
            idx_green = (lab_green == cid); 
            pts_green = pc_world.reshape(-1, 3)[idx_green.flatten()]; 
            pts_green = pts_green[np.isfinite(pts_green).all(axis=1)]
            if pts_green.shape[0] == 0: continue
            centroid_xy_green = np.nanmedian(pts_green[:, :2], axis=0); 
            top_z_green = float(np.nanpercentile(pts_green[:, 2], 95)); 
            low_z_green = float(np.nanpercentile(pts_green[:, 2], 5))
            stats = {"centroid_xy": centroid_xy_green, "top_z": top_z_green, "low_z": low_z_green, "size_xy": np.nanpercentile(pts_green[:, :2], 95, axis=0) - np.nanpercentile(pts_green[:, :2], 5, axis=0)}
            d_center = float(np.linalg.norm(stats["centroid_xy"] - np.array([0.5, 0.0])))
            green_infos.append({"cid": cid, "area": area, "stats": stats, "d_center": d_center})
        green_infos.sort(key=lambda it: (it["d_center"], -it["area"])); 
        base_info, top_info = green_infos[0], green_infos[1]
        base_stats, top_stats = base_info["stats"], top_info["stats"]
        green_edge = np.mean([np.clip(float(np.mean(np.abs(s["size_xy"]))), 0.03, 0.06) for s in [base_stats, top_stats]])

        z_table_est = float(np.nanpercentile(pc_world[..., 2].reshape(-1), 1.0))
        #
        
        if centroid_xy_red is not None:
            
            GRIPPER_OPEN_POS = 0.04    # 夹爪完全张开的位置 (m)
            GRIPPER_CLOSED_POS = 0.0   # 夹爪完全闭合的位置 (m)
            place_clearance = 0.015

            # test
            print("--- 阶段 1: 抓取红色方块 ---")
            h_red = max(0.01, red_stats["top_z"] - red_stats["low_z"]); grasp_z_red = max(z_table_est + 0.25 * h_red, red_stats["top_z"] - grasp_depth)
            hover_pos_red = np.array([red_stats["centroid_xy"][0], red_stats["centroid_xy"][1], red_stats["top_z"] + hover_offset])
            grasp_pos_red = np.array([red_stats["centroid_xy"][0], red_stats["centroid_xy"][1], grasp_z_red])
            lift_pos_red = np.array([red_stats["centroid_xy"][0], red_stats["centroid_xy"][1], red_stats["top_z"] + lift_offset])
            # self.set_gripper(gripper_open); 
            self.move_robot_joint(target_joint_pos=None, target_gripper_pos=GRIPPER_OPEN_POS, count=50)
            self.move_robot_ik(np.concatenate([hover_pos_red, self.robot_quat])); 
            self.sim_wait(15)
            self.move_robot_ik(np.concatenate([grasp_pos_red, self.robot_quat])); 
            self.sim_wait(10)
            self.move_robot_joint(target_joint_pos=None, target_gripper_pos=GRIPPER_CLOSED_POS, count=50)
            self.sim_wait(20)
            self.move_robot_ik(np.concatenate([lift_pos_red, self.robot_quat])); 
            self.sim_wait(15)
            #

            # TEST
            print("--- 阶段 2: 将红色方块放置在基座绿色方块上 ---")
            target_center_z_red = float(base_stats["top_z"] + 0.6 * red_edge + place_clearance)
            post_lift_z_red = float(base_stats["top_z"] + red_edge + lift_offset)
            hover_place_pos_red = np.array([base_stats["centroid_xy"][0], base_stats["centroid_xy"][1], target_center_z_red + hover_offset])
            place_pos_red = np.array([base_stats["centroid_xy"][0], base_stats["centroid_xy"][1], target_center_z_red])
            post_place_pos_red = np.array([base_stats["centroid_xy"][0], base_stats["centroid_xy"][1], post_lift_z_red])
            self.move_robot_ik(np.concatenate([hover_place_pos_red, self.robot_quat])); 
            self.sim_wait(10)
            self.move_robot_ik(np.concatenate([place_pos_red, self.robot_quat])); 
            self.sim_wait(6)
            self.move_robot_joint(target_joint_pos=None, target_gripper_pos=GRIPPER_OPEN_POS, count=50)
            self.sim_wait(20)
            self.move_robot_ik(np.concatenate([post_place_pos_red, self.robot_quat])); 
            self.sim_wait(10)
            
            #test
            print("--- 阶段 3: 抓取顶部绿色方块 ---")
            h_green_top = max(0.01, top_stats["top_z"] - top_stats["low_z"]); grasp_z_green = max(z_table_est + 0.25 * h_green_top, top_stats["top_z"] - grasp_depth)
            hover_pos_green = np.array([top_stats["centroid_xy"][0], top_stats["centroid_xy"][1], top_stats["top_z"] + hover_offset])
            grasp_pos_green = np.array([top_stats["centroid_xy"][0], top_stats["centroid_xy"][1], grasp_z_green])
            lift_pos_green = np.array([top_stats["centroid_xy"][0], top_stats["centroid_xy"][1], top_stats["top_z"] + lift_offset])
            self.move_robot_ik(np.concatenate([hover_pos_green, self.robot_quat])); 
            self.sim_wait(10)
            self.move_robot_ik(np.concatenate([grasp_pos_green, self.robot_quat])); 
            self.sim_wait(8)
            self.move_robot_joint(target_joint_pos=None, target_gripper_pos=GRIPPER_CLOSED_POS, count=50) 
            self.sim_wait(20)
            self.move_robot_ik(np.concatenate([lift_pos_green, self.robot_quat])); self.sim_wait(12)

            # test
            print("--- 阶段 4: 将顶部绿色方块放置在红色方块上 ---")
            red_top_after_place = base_stats["top_z"] + red_edge
            target_center_z_green = float(red_top_after_place + 0.6 * green_edge + place_clearance)
            post_lift_z_green = float(red_top_after_place + green_edge + lift_offset)
            hover_place_pos_green = np.array([base_stats["centroid_xy"][0], base_stats["centroid_xy"][1], target_center_z_green + hover_offset])
            place_pos_green = np.array([base_stats["centroid_xy"][0], base_stats["centroid_xy"][1], target_center_z_green])
            post_place_pos_green = np.array([base_stats["centroid_xy"][0], base_stats["centroid_xy"][1], post_lift_z_green])
            self.move_robot_ik(np.concatenate([hover_place_pos_green, self.robot_quat])); 
            self.sim_wait(10)
            self.move_robot_ik(np.concatenate([place_pos_green, self.robot_quat])); 
            self.sim_wait(6)
            self.move_robot_joint(target_joint_pos=None, target_gripper_pos=GRIPPER_OPEN_POS, count=50)
            self.sim_wait(20)
            self.move_robot_ik(np.concatenate([post_place_pos_green, self.robot_quat])); self.sim_wait(10)
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
