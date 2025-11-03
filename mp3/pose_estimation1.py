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


class Experiment:
    def __init__(self):
        # 初始化仿真
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([1.5, 0.0, 1.2], [0.0, 0.0, 0.15])

        # 设置时间步
        self.sim.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
        print(f"\nSim dt: {self.sim.get_physics_dt()}\n")
        self.sim_dt = self.sim.get_physics_dt()

        # 初始化场景
        scene_cfg = MP3PoseEstimSceneCfg(args_cli.num_envs, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)

        # 重置仿真
        self.sim.reset()
        print("Setup complete...")

        # 设置IK求解器
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self.diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device
        )
        self.ik_body = "gripper_center"
        self.robot_entity_cfg = SceneEntityCfg(
            "ur5e",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            body_names=[self.ik_body]
        )
        self.robot_entity_cfg.resolve(self.scene)
        if self.scene["ur5e"].is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # 设置可视化
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow((np.ones((256, 256)) * 255).astype(np.uint8))

        # 记录机器人位姿
        self.update_robot_pose()

        # 存储所有视角的点云
        self.all_point_clouds = []


    def update_robot_pose(self):
        """更新机器人末端执行器位姿"""
        body_idx = self.scene["ur5e"].find_bodies(self.ik_body)[0][0]
        self.robot_pose = self.scene["ur5e"].data.body_state_w[0, body_idx, :7].detach().cpu().numpy()
        self.robot_pos = self.robot_pose[:3]
        self.robot_quat = self.robot_pose[3:]


    def move_robot_joint(self, target_joint_pos, target_gripper_pos, count=10, time_for_residual_movement=5):
        # 保持原有实现不变
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
        for i in range(count):
            self.scene["ur5e"].set_joint_position_target((target - initial) / count * i + initial)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        # 等待残余运动
        print("Waiting for any residual movement...")
        for i in range(time_for_residual_movement):
            self.scene["ur5e"].set_joint_position_target(target)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        print("Movement completed. Deviation:", (target - self.scene["ur5e"].data.joint_pos).squeeze().detach().cpu().numpy())
        self.update_robot_pose()


    def move_robot_ik(self, target_pose, max_joint_change=0.04, ik_tol=1e-3, timeout_count=200):
        # 保持原有实现不变
        self.diff_ik_controller.set_command(torch.tensor(target_pose, device=self.sim.device))

        print("Moving the robot through IK...")
        count = 0
        while simulation_app.is_running():
            # 获取仿真数据
            jacobian = self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
            ee_pose_w = self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
            joint_pos = self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]

            # 计算关节指令
            joint_pos_des = self.diff_ik_controller.compute(
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos
            )

            # 应用动作
            joint_changes = (joint_pos_des - joint_pos).detach().cpu().numpy()[0]
            if np.sum(np.abs(joint_changes) > max_joint_change) > 0:
                scaled_joint_changes = joint_changes / (np.max(np.abs(joint_changes)) / max_joint_change)
                scaled_joint_changes = torch.tensor(scaled_joint_changes).unsqueeze(0).to(joint_pos_des.device)
                self.scene["ur5e"].set_joint_position_target(
                    joint_pos + scaled_joint_changes, joint_ids=self.robot_entity_cfg.joint_ids
                )
            else:
                self.scene["ur5e"].set_joint_position_target(
                    joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
                )

            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

            count += 1

            # 检查是否到达目标
            cur_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
            cur_quat = cur_pose[3:]
            cur_rot_matrix = R.from_quat([cur_quat[1], cur_quat[2], cur_quat[3], cur_quat[0]]).as_matrix()
            target_quat = target_pose[3:]
            target_rot_matrix = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_matrix()

            if (np.average(np.abs(target_pose - cur_pose)[:3]) < ik_tol and 
                np.average(np.abs(target_rot_matrix - cur_rot_matrix)) < ik_tol):
                print("Movement completed. Deviation:", np.abs(target_pose - cur_pose))
                self.update_robot_pose()
                return
            
            if count >= timeout_count:
                print("Movement terminated due to timeout. Deviation:", np.abs(target_pose - cur_pose))
                self.update_robot_pose()
                return


    def sim_wait(self, count):
        # 保持原有实现不变
        print("Waiting...")
        for _ in range(count):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)
        self.update_robot_pose()


    def get_vase_position(self):
        """获取花瓶在世界坐标系中的位置"""
        vase_prim = self.scene["vase"]
        vase_pose, _ = vase_prim.get_world_poses()
        vase_pose = vase_pose.squeeze().cpu().numpy()
        print(f"Vase position detected: {vase_pose}")
        return vase_pose


    # def compute_camera_poses(self, vase_pos, radius=0.2, height=0.3, num_positions=5):
    #     """计算围绕花瓶的5个相机位姿"""
    #     camera_poses = []
    #     for i in range(num_positions):
    #         # 计算环绕角度（0°、72°、144°、216°、288°）
    #         angle = np.deg2rad(i * (360 / num_positions))
            
    #         # 计算相机位置（极坐标转笛卡尔坐标）
    #         cam_x = vase_pos[0] + radius * np.cos(angle)
    #         cam_y = vase_pos[1] + radius * np.sin(angle)
    #         cam_z = height  # 固定高度
    #         cam_pos = np.array([cam_x, cam_y, cam_z])
            
    #         # 计算相机朝向（指向花瓶中心）
    #         direction = vase_pos - cam_pos
    #         direction /= np.linalg.norm(direction)  # 归一化方向向量
            
    #         # 计算相机旋转（保持水平，上方向为Z轴）
    #         up = np.array([0, 0, 1])
    #         right = np.cross(up, direction)
    #         right /= np.linalg.norm(right)
    #         new_up = np.cross(direction, right)
            
    #         # 构建旋转矩阵并转换为四元数（w, x, y, z）
    #         rot_matrix = np.array([right, new_up, direction]).T
    #         cam_rot = R.from_matrix(rot_matrix).as_quat()  # (x,y,z,w)
    #         cam_rot = np.array([cam_rot[3], cam_rot[0], cam_rot[1], cam_rot[2]])  # 转换为(w, x, y, z)
            
    #         # 组合位姿 [x, y, z, qw, qx, qy, qz]
    #         camera_pose = np.concatenate([cam_pos, cam_rot])
    #         camera_poses.append(camera_pose)
    #         print(f"Computed camera pose {i+1}: {camera_pose[:3]}")
        
    #     return camera_poses
    
    def compute_camera_poses(self, vase_pos, radius=0.4, height_offset=0.1, num_positions=5, target_z_offset=0.2):

        camera_poses = []
        # This is the point the camera will look at (the vase's center)
        target_point = vase_pos + np.array([0, 0, target_z_offset])

        for i in range(num_positions):
            angle = np.deg2rad(i * (360 / num_positions))
            
            # Calculate camera position
            cam_x = vase_pos[0] + radius * np.cos(angle)
            cam_y = vase_pos[1] + radius * np.sin(angle)
            # NEW: Camera height is now relative to the vase's Z position
            cam_z = vase_pos[2] + height_offset
            cam_pos = np.array([cam_x, cam_y, cam_z])
            
            # Calculate camera orientation (pointing at the new target_point)
            direction = target_point - cam_pos
            direction /= np.linalg.norm(direction)
            
            up = np.array([0, 0, 1])
            if np.allclose(np.abs(direction), up):
                right = np.array([1, 0, 0])
            else:
                right = np.cross(up, direction)
                right /= np.linalg.norm(right)
            new_up = np.cross(direction, right)
            
            # Build rotation matrix and convert to quaternion
            rot_matrix = np.array([right, new_up, direction]).T
            cam_rot = R.from_matrix(rot_matrix).as_quat()  # (x,y,z,w)
            cam_rot = np.array([cam_rot[3], cam_rot[0], cam_rot[1], cam_rot[2]])  # (w, x, y, z)
            
            # Combine into the final pose array
            camera_pose = np.concatenate([cam_pos, cam_rot])
            camera_poses.append(camera_pose)
            print(f"Computed DESIRED CAMERA pose {i+1}: {camera_pose[:3]}")
        
        return camera_poses

    def capture_and_process_image(self, T_eef_cam):
        """捕获并处理RGBD图像，转换为点云并转换到世界坐标系"""
        # 获取相机数据
        cam_data = self.scene.sensors["wrist_camera"].data
        rgb_img = cam_data.output["rgb"][0].cpu().numpy()
        depth_img = cam_data.output["depth"][0].cpu().numpy()
        depth_img = depth_img.squeeze(-1)

        # 显示RGB图像
        self.ax.imshow(rgb_img)
        self.fig.canvas.draw()
        plt.pause(0.1)

        # 1. 深度图反投影为点云（相机坐标系）
        height, width = depth_img.shape
        cam_cfg = self.scene.sensors["wrist_camera"].cfg
        focal_length, h_aperture = cam_cfg.spawn.focal_length, cam_cfg.spawn.horizontal_aperture
        
        # 计算内参
        fx = (focal_length / h_aperture) * width
        fy = fx
        cx, cy = width / 2, height / 2

        # 生成像素坐标网格
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.flatten()
        v = v.flatten()
        z = depth_img.flatten()

        # 过滤无效深度值
        valid_mask = z > 0.01  # 排除过近的点
        u, v, z = u[valid_mask], v[valid_mask], z[valid_mask]

        # 转换为相机坐标系下的点云
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_cam = np.column_stack([x, y, z])

        # 2. 颜色阈值过滤（提取红色花瓶）
        rgb_flat = rgb_img.reshape(-1, 3)[valid_mask]
        # 红色阈值（根据task_envs中设置的花瓶颜色(1.0, 0.0, 0.0)调整）
        red_threshold = 0.8
        green_blue_threshold = 0.3
        color_mask = (rgb_flat[:, 0] > red_threshold) & \
                     (rgb_flat[:, 1] < green_blue_threshold) & \
                     (rgb_flat[:, 2] < green_blue_threshold)
        
        filtered_points_cam = points_cam[color_mask]
        filtered_colors = rgb_flat[color_mask] / 255.0  # 归一化到[0,1]

        # 3. 转换到世界坐标系
        # 获取末端执行器位姿
        eef_pos = self.robot_pos
        eef_quat = self.robot_quat  # (qw, qx, qy, qz)
        
        # 构建末端执行器到世界的变换矩阵
        R_eef = R.from_quat([eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]]).as_matrix()
        T_eef_world = np.eye(4)
        T_eef_world[:3, :3] = R_eef
        T_eef_world[:3, 3] = eef_pos

        # 点云从相机坐标系转换到世界坐标系
        points_hom = np.hstack([filtered_points_cam, np.ones((filtered_points_cam.shape[0], 1))])  # 齐次坐标
        points_eef = (T_eef_cam @ points_hom.T).T[:, :3]  # 相机->末端执行器
        points_hom_eef = np.hstack([points_eef, np.ones((points_eef.shape[0], 1))])
        points_world = (T_eef_world @ points_hom_eef.T).T[:, :3]  # 末端执行器->世界

        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        return pcd


    def run(self):
        """主运行函数"""
        print("开始点云重建...")
        print("Starting point cloud reconstruction...")

        # 步骤1: 定义相机内参和加载手眼标定矩阵
        try:
            with np.load('hand_eye_calibration_result.npz') as data:
                T_eef_cam = data['T_eef_cam']
            print("成功加载手眼标定矩阵 'hand_eye_calibration_result.npz'.")
        except (FileNotFoundError, KeyError) as e:
            print(f"错误: 加载手眼标定文件失败。 {e}")
            simulation_app.close()
            return


        # 步骤2: 获取花瓶位置
        vase_pos = self.get_vase_position()

        # 步骤3: 计算5个环绕拍摄位置
        camera_poses = self.compute_camera_poses(vase_pos)

        # 步骤4: 移动到每个位置并拍摄
        for i, target_pose in enumerate(camera_poses):
            print(f"\n===== 拍摄位置 {i+1}/5 =====")

            # 移动机器人到目标位置
            self.move_robot_ik(target_pose, max_joint_change=0.05, timeout_count=300)
            self.sim_wait(20)  # 等待稳定
            
            # 捕获并处理图像
            pcd = self.capture_and_process_image(T_eef_cam)
            self.all_point_clouds.append(pcd)
            
            # 保存当前视角的点云
            o3d.io.write_point_cloud(f"vase_view_{i+1}.pcd", pcd)
            print(f"已保存视角 {i+1} 的点云")

        # 步骤5: 融合所有点云
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in self.all_point_clouds:
            combined_pcd += pcd
        
        # 下采样去除冗余点
        voxel_size = 0.002
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"融合后的点云包含 {len(combined_pcd.points)} 个点")

        # 保存合并后的点云
        o3d.io.write_point_cloud("combined_vase_point_cloud.pcd", combined_pcd)
        print("已保存合并后的花瓶点云")

        # 可视化合并后的点云
        o3d.visualization.draw_geometries([combined_pcd], window_name="Combined Vase Point Cloud")

        # 保持仿真运行
        while simulation_app.is_running():
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

        # 关闭仿真
        simulation_app.close()


if __name__ == "__main__":

    exp = Experiment()
    exp.run()