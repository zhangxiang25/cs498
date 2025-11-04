# TODO: hand-eye calibration
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def load_calibration_data(filepath: str = "calibration_data.npz") -> tuple:

    data = np.load(filepath)
    eef_poses = data["eef_poses"] # end-effector's pose relative to robot's base
    tag_poses = data["tag_poses"] # AprilTag pose relative to camera
 
    return eef_poses, tag_poses
    
def compute_relative_motions(eef_poses: np.ndarray, tag_poses: np.ndarray) -> tuple:
  
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for i in range(len(eef_poses) - 1):
        T_world_eef1 = eef_poses[i]
        T_world_eef2 = eef_poses[i + 1]
        # Calculates the transformation of the end-effector from its first position to its second.
        T_eef1_eef2 = np.linalg.inv(T_world_eef1) @ T_world_eef2  
        R_gripper2base.append(T_eef1_eef2[:3, :3])
        t_gripper2base.append(T_eef1_eef2[:3, 3].reshape(3, 1))  

        T_cam_tag1 = tag_poses[i]
        T_cam_tag2 = tag_poses[i + 1]
        # calculates the change in the tag's pose as seen by the camera
        T_tag1_tag2 = T_cam_tag2 @ np.linalg.inv(T_cam_tag1) 
        R_target2cam.append(T_tag1_tag2[:3, :3])
        t_target2cam.append(T_tag1_tag2[:3, 3].reshape(3, 1))  

    return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam


def solve_hand_eye_calibration(R_gripper2base: list, t_gripper2base: list,
                               R_target2cam: list, t_target2cam: list,
                               method: int = cv2.CALIB_HAND_EYE_TSAI) -> np.ndarray:

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method
    )
    # Assembles these two parts into a single 4x4 transformation matrix
    T_eef_cam = np.eye(4)
    T_eef_cam[:3, :3] = R_cam2gripper 
    T_eef_cam[:3, 3] = t_cam2gripper.flatten() 

    return T_eef_cam


def save_calibration_result(T_eef_cam: np.ndarray, save_path: str = "hand_eye_calibration_result.npz") -> None:

    np.savez(save_path, T_eef_cam=T_eef_cam)
    np.set_printoptions(precision=6, suppress=True)  
    print(T_eef_cam)

def main():

    eef_poses, tag_poses = load_calibration_data()
    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam = compute_relative_motions(eef_poses, tag_poses)

    T_eef_cam = solve_hand_eye_calibration(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)

    save_calibration_result(T_eef_cam)


if __name__ == "__main__":
    main()
