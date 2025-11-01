# TODO: hand-eye calibration
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def load_calibration_data(filepath: str = "calibration_data.npz") -> tuple:
    """
    加载手眼标定所需的数据集（机械臂末端位姿与相机下的AprilTag位姿）。
    
    Args:
        filepath: 标定数据文件路径，默认从原代码生成的"calibration_data.npz"读取
    
    Returns:
        eef_poses: 机械臂末端在世界坐标系下的位姿列表，形状为(N, 4, 4)
        tag_poses: AprilTag在相机坐标系下的位姿列表，形状为(N, 4, 4)
    """
    try:
        data = np.load(filepath)
        eef_poses = data["eef_poses"]
        tag_poses = data["tag_poses"]
        print(f"成功加载标定数据，共包含 {len(eef_poses)} 组位姿数据")
        
        # 验证数据格式有效性
        if eef_poses.shape[1:] != (4, 4) or tag_poses.shape[1:] != (4, 4):
            raise ValueError("位姿数据格式错误，需为4x4变换矩阵")
        if len(eef_poses) < 2:
            raise ValueError("数据量不足，至少需要2组位姿才能计算相对运动")
        
        return eef_poses, tag_poses
    
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到标定数据文件，请确保'{filepath}'存在于当前目录")
    except KeyError:
        raise KeyError("数据文件缺少关键字段，需包含'eef_poses'和'tag_poses'")


def compute_relative_motions(eef_poses: np.ndarray, tag_poses: np.ndarray) -> tuple:
    """
    计算机械臂末端的相对运动（A矩阵）和Tag相对于相机的相对运动（B矩阵），用于AX=XB求解。
    
    Args:
        eef_poses: 机械臂末端位姿列表 (N, 4, 4)
        tag_poses: Tag在相机下的位姿列表 (N, 4, 4)
    
    Returns:
        R_gripper2base: 机械臂末端相对运动的旋转矩阵列表 (N-1, 3, 3)
        t_gripper2base: 机械臂末端相对运动的平移向量列表 (N-1, 3, 1)
        R_target2cam: Tag相对相机运动的旋转矩阵列表 (N-1, 3, 3)
        t_target2cam: Tag相对相机运动的平移向量列表 (N-1, 3, 1)
    """
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    # 遍历所有相邻位姿对，计算相对变换
    for i in range(len(eef_poses) - 1):
        # A: 机械臂末端从第i帧到第i+1帧的相对变换（base -> eef1 -> eef2）
        T_world_eef1 = eef_poses[i]
        T_world_eef2 = eef_poses[i + 1]
        T_eef1_eef2 = np.linalg.inv(T_world_eef1) @ T_world_eef2  # 相对变换矩阵
        R_gripper2base.append(T_eef1_eef2[:3, :3])
        t_gripper2base.append(T_eef1_eef2[:3, 3].reshape(3, 1))  # 转为列向量

        # B: Tag从第i帧到第i+1帧相对于相机的相对变换（cam -> tag2 -> tag1）
        T_cam_tag1 = tag_poses[i]
        T_cam_tag2 = tag_poses[i + 1]
        T_tag1_tag2 = T_cam_tag1 @ np.linalg.inv(T_cam_tag2)  # 相对变换矩阵
        R_target2cam.append(T_tag1_tag2[:3, :3])
        t_target2cam.append(T_tag1_tag2[:3, 3].reshape(3, 1))  # 转为列向量

    print(f"成功计算 {len(R_gripper2base)} 组相对运动矩阵")
    return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam


def solve_hand_eye_calibration(R_gripper2base: list, t_gripper2base: list,
                               R_target2cam: list, t_target2cam: list,
                               method: int = cv2.CALIB_HAND_EYE_TSAI) -> np.ndarray:
    """
    执行手眼标定，求解相机相对于机械臂末端的变换矩阵 T_eef_cam（X矩阵）。
    
    Args:
        R_gripper2base: 机械臂末端相对旋转列表
        t_gripper2base: 机械臂末端相对平移列表
        R_target2cam: Tag相对相机旋转列表
        t_target2cam: Tag相对相机平移列表
        method: 标定方法，默认使用TSAI算法（鲁棒性强）
    
    Returns:
        T_eef_cam: 相机在机械臂末端坐标系下的位姿矩阵（4x4），表示 eef -> cam 的变换
    """
    # 调用OpenCV手眼标定API求解
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method
    )

    # 组装4x4变换矩阵（旋转+平移）
    T_eef_cam = np.eye(4)
    T_eef_cam[:3, :3] = R_cam2gripper  # 旋转矩阵：eef -> cam
    T_eef_cam[:3, 3] = t_cam2gripper.flatten()  # 平移向量：eef -> cam

    return T_eef_cam


def save_calibration_result(T_eef_cam: np.ndarray, save_path: str = "hand_eye_calibration_result.npz") -> None:
    """
    保存手眼标定结果（变换矩阵）到文件，并打印格式化结果。
    
    Args:
        T_eef_cam: 相机相对于末端的变换矩阵（4x4）
        save_path: 结果保存路径
    """
    # 保存矩阵到npz文件
    np.savez(save_path, T_eef_cam=T_eef_cam)
    print(f"\n标定结果已保存到: {save_path}")

    # 格式化打印结果（旋转矩阵+平移向量）
    print("\n=== 手眼标定结果 ===")
    print("相机相对于机械臂末端的变换矩阵 T_eef_cam（eef -> cam）：")
    np.set_printoptions(precision=6, suppress=True)  # 保留6位小数，取消科学计数法
    print(T_eef_cam)

    # 额外打印旋转的欧拉角（便于直观理解姿态）
    r = R.from_matrix(T_eef_cam[:3, :3])
    euler_angles = r.as_euler("xyz", degrees=True)
    print(f"\n旋转姿态（欧拉角，XYZ顺序，单位：度）：")
    print(f"Roll: {euler_angles[0]:.2f}°, Pitch: {euler_angles[1]:.2f}°, Yaw: {euler_angles[2]:.2f}°")
    print(f"平移向量（单位：米）：")
    print(f"X: {T_eef_cam[0, 3]:.6f}, Y: {T_eef_cam[1, 3]:.6f}, Z: {T_eef_cam[2, 3]:.6f}")


def main():
    """主函数：执行完整手眼标定流程"""
    print("=== 开始手眼标定 ===")
    try:
        # 1. 加载标定数据
        eef_poses, tag_poses = load_calibration_data()
        
        # 2. 计算相对运动矩阵
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam = compute_relative_motions(eef_poses, tag_poses)
        
        # 3. 求解手眼变换矩阵
        T_eef_cam = solve_hand_eye_calibration(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
        
        # 4. 保存并打印结果
        save_calibration_result(T_eef_cam)
        
        print("\n=== 手眼标定完成 ===")

    except Exception as e:
        print(f"\n标定过程出错：{str(e)}")
        print("请检查数据文件或重新采集足够的标定数据")


if __name__ == "__main__":
    main()
