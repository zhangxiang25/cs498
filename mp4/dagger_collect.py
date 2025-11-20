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

# --- 参数解析 ---
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="生成的环境数量")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac Lab 导入 ---
import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from task_envs import MP4SceneCfg, PHYSICS_DT, RENDERING_DT

# 尝试导入 Policy 类
try:
    from train_mlp import Policy
except ImportError:
    print("\n错误: 无法从 train_mlp.py 导入 Policy。")
    print("请确保 train_mlp.py 在同一目录下。\n")
    exit()


class DaggerExperiment:

    def __init__(self, dataset_dir, model_path):
        self.dataset_dir = dataset_dir
        self.model_path = model_path

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        if not os.path.exists(self.model_path):
            print(f"\n错误: 在 {self.model_path} 未找到模型")
            print("请先运行 'python train_mlp.py' 训练模型\n")
            exit()

        # 初始化仿真
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([1.1, 0.0, 0.7], [0.0, 0.0, 0.0])
        self.sim.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
        self.sim_dt = self.sim.get_physics_dt()

        # 初始化场景
        scene_cfg = MP4SceneCfg(args_cli.num_envs, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        print("设置完成...")

        # 设置 IK 控制器
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

        # 可视化 (可选)
        self.fig, ax = plt.subplots()
        self.im = ax.imshow((np.ones((256, 256))*255).astype(np.uint8))

        # --- 控制标志 ---
        self.d_pressed = False
        self.x_pressed = False
        self.z_pressed = False
        self.c_pressed = False
        self.g_pressed = False
        self.v_pressed = False
        self.a_pressed = False
        self.changed_gripper_state = False

        # DAgger 专用标志
        self.policy_mode = True      # 初始为策略控制模式
        self.collecting_data = False # 初始不收集数据 (直到介入)
        self.save_episode = False
        self.reset_episode = False

    def on_press(self, key):
        try:
            char = key.char
            if char == "k":
                # 介入触发器：切换到手动控制并开始录制
                if self.policy_mode:
                    print("\n[介入] 切换到手动控制并开始录制数据...")
                    self.policy_mode = False
                    self.collecting_data = True
            
            # 标准遥操作键位
            elif char == "d": self.d_pressed = True
            elif char == "x": self.x_pressed = True
            elif char == "z": self.z_pressed = True
            elif char == "c": self.c_pressed = True
            elif char == "g": self.g_pressed = True
            elif char == "v": self.v_pressed = True
            elif char == "a": self.a_pressed = True
            
            # 回合管理
            elif char == "s": 
                if not self.policy_mode: # 只有在介入后才允许保存
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
        # 加载策略
        print(f"正在从 {self.model_path} 加载模型...")
        model = Policy()
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.eval()
        model.to('cpu')

        # 查找下一个 demo 索引
        existing_demos = [d for d in os.listdir(self.dataset_dir) if d.startswith('demo_')]
        episode_counter = len(existing_demos)
        print(f"下一个演示将保存为 demo_{episode_counter}")

        while simulation_app.is_running():
            
            # --- 重置回合逻辑 ---
            ur5e_state = self.scene["ur5e"].data.default_root_state.clone()
            self.scene["ur5e"].write_root_pose_to_sim(ur5e_state[:, :7])
            self.scene["ur5e"].write_root_velocity_to_sim(ur5e_state[:, 7:])
            
            joint_pos, joint_vel = self.scene["ur5e"].data.default_joint_pos.clone(), self.scene["ur5e"].data.default_joint_vel.clone()
            self.scene["ur5e"].write_joint_state_to_sim(joint_pos, joint_vel)

            # 随机化方块位置
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
            
            # 移动到起始位置
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

            # --- 启动 DAgger 回合 ---
            listener = Listener(on_press=self.on_press, on_release=self.on_release)
            listener.start()
            
            print(f"\n--- 回合 {episode_counter} 开始 ---")
            print("模式: 策略 (机器人自动移动)")
            print("按 'k' 键: 介入 (接管控制并开始录制)")
            print("按 's' 键: 保存 (仅在介入后有效)")
            print("按 'q' 键: 跳过/放弃")

            self.policy_mode = True
            self.collecting_data = False
            self.reset_episode = False
            self.save_episode = False
            
            gripper_state = -1 # 开启状态
            stationary_pose = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
            
            # 数据缓存
            state_observations = []
            actions = []
            last_saved_pos = self.get_eef_pos()
            last_gripper = gripper_state
            pos_threshold = 0.005

            # 循环：推理 / 遥操作
            while simulation_app.is_running():
                if self.reset_episode:
                    break
                
                # 1. 获取状态
                current_eef = self.get_eef_pos()
                red_pos = self.get_red_cube_pos()
                green_pos = self.get_green_cube_pos()
                
                obs_np = np.array([
                    current_eef[0], current_eef[1], current_eef[2], gripper_state,
                    red_pos[0], red_pos[1], red_pos[2],
                    green_pos[0], green_pos[1]
                ])

                # 2. 决定动作 (策略 vs 遥操作)
                action_idx = 8 # 默认为静止
                
                if self.policy_mode:
                    # 模型推理
                    with torch.no_grad():
                        inp = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
                        pred = model(inp).squeeze(0).cpu().numpy()
                    action_idx = np.argmax(pred)
                else:
                    # 手动遥操作
                    if self.x_pressed: action_idx = 0
                    elif self.d_pressed: action_idx = 1
                    elif self.c_pressed: action_idx = 2
                    elif self.z_pressed: action_idx = 3
                    elif self.g_pressed: action_idx = 4
                    elif self.v_pressed: action_idx = 5
                    elif self.a_pressed and not self.changed_gripper_state:
                        gripper_state *= -1
                        self.changed_gripper_state = True
                        action_idx = 7 if gripper_state == 1 else 6 # 关闭 或 开启
                    
                # 3. 执行动作 (应用到仿真)
                temp_dist = 0.003
                temp_target = stationary_pose[:3].copy()
                
                # 如果在移动，更新静止锚点
                cur_real_pos = self.scene["ur5e"].data.body_state_w[0, self.scene["ur5e"].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()[:3]
                
                if action_idx in [0, 1]: stationary_pose[0] = cur_real_pos[0]
                if action_idx in [2, 3]: stationary_pose[1] = cur_real_pos[1]
                if action_idx in [4, 5]: stationary_pose[2] = cur_real_pos[2]
                
                # 应用增量
                if action_idx == 0: temp_target[0] += temp_dist
                elif action_idx == 1: temp_target[0] -= temp_dist
                elif action_idx == 2: temp_target[1] += temp_dist
                elif action_idx == 3: temp_target[1] -= temp_dist
                elif action_idx == 4: temp_target[2] += temp_dist
                elif action_idx == 5: temp_target[2] -= temp_dist
                elif action_idx == 6: gripper_state = -1 # 显式开启
                elif action_idx == 7: gripper_state = 1  # 显式关闭

                # IK 解算器命令
                T_new = np.eye(4)
                T_new[:3, :3] = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_matrix()
                T_new[:3, 3] = temp_target
                target_pose_ik = np.array(T_new[:3, 3].tolist() + target_quat.tolist())

                if action_idx == 8: # 静止
                    self.diff_ik_controller.set_command(torch.tensor(stationary_pose, device=self.sim.device))
                else:
                    self.diff_ik_controller.set_command(torch.tensor(target_pose_ik, device=self.sim.device))

                # 计算并应用关节目标
                jacobian = self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
                ee_pose_w = self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
                joint_pos = self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]
                joint_pos_des = self.diff_ik_controller.compute(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, joint_pos)
                
                # 关节限制/过冲保护
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

                # 4. 记录数据 (仅在介入收集时)
                if self.collecting_data:
                    dist = np.linalg.norm(current_eef - last_saved_pos)
                    
                    if dist > pos_threshold or gripper_state != last_gripper or action_idx in [6, 7]:
                        # 编码独热动作
                        one_hot = np.zeros(9)
                        one_hot[action_idx] = 1
                        
                        state_observations.append(obs_np)
                        actions.append(one_hot)
                        
                        # 保存图像 (以匹配数据集结构)
                        img_dir = os.path.join(self.dataset_dir, f"demo_{episode_counter}", "images")
                        if not os.path.exists(img_dir): os.makedirs(img_dir)
                        
                        cur_image = self.scene["robotview_camera"].data.output["rgb"].detach().cpu().numpy()[0]
                        cur_image = cv2.cvtColor(cur_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(img_dir, f"{len(state_observations)}.png"), cur_image)

                        print(f"正在录制... 点数: {len(state_observations)} | 动作: {action_idx}")
                        
                        last_saved_pos = current_eef
                        last_gripper = gripper_state

            listener.stop()

            # --- 保存 / 放弃 ---
            if self.save_episode and len(state_observations) > 0:
                save_path = os.path.join(self.dataset_dir, f"demo_{episode_counter}", "states.npz")
                np.savez(save_path, state_observations=state_observations, actions=actions)
                print(f"\n>>> 已保存 demo_{episode_counter}，包含 {len(state_observations)} 帧修正数据。\n")
                episode_counter += 1
            elif self.reset_episode:
                print("\n>>> 回合已跳过/放弃。\n")
                # 清理已创建的文件夹
                demo_p = os.path.join(self.dataset_dir, f"demo_{episode_counter}")
                if os.path.exists(demo_p): shutil.rmtree(demo_p)

if __name__ == "__main__":
    # 这里指向您现有的数据集
    DATASET_DIR = r'C:\IsaacLab\cs498\mp4\image'
    MODEL_PATH = "mlp_model.pth" # 或者 mlp_model_best.pth

    dagger = DaggerExperiment(DATASET_DIR, MODEL_PATH)
    dagger.run_dagger()