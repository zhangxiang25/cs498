import argparse, os, json
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation

from isaaclab.app import AppLauncher

# ---------------- argparse ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--save_dir", type=str, default="./outputs_q4")

# Q1: HSV thresholds (0~1) & morphology
parser.add_argument("--h_red_low_1", type=float, default=0.00)
parser.add_argument("--h_red_high_1", type=float, default=0.04)
parser.add_argument("--h_red_low_2", type=float, default=0.96)
parser.add_argument("--h_red_high_2", type=float, default=1.00)
parser.add_argument("--h_green_low", type=float, default=0.23)
parser.add_argument("--h_green_high", type=float, default=0.44)
parser.add_argument("--s_min", type=float, default=0.40)
parser.add_argument("--v_min", type=float, default=0.20)
parser.add_argument("--morph_iters", type=int, default=2)

# Q2: point cloud viz controls
parser.add_argument("--pc_stride", type=int, default=2)
parser.add_argument("--max_points_plot", type=int, default=150000)

# Q3: grasp controls
parser.add_argument("--hover_offset", type=float, default=0.12)
parser.add_argument("--lift_offset", type=float, default=0.15)
parser.add_argument("--grasp_depth", type=float, default=0.02)
parser.add_argument("--gripper_open", type=float, default=0.04)
parser.add_argument("--gripper_close", type=float, default=0.0)

# Q4: placing controls
parser.add_argument("--place_clearance", type=float, default=0.005, help="extra half-height margin when placing (m)")
parser.add_argument("--stability_wait_s", type=float, default=3.0, help="how long to wait after last release (sec)")

# Isaac launcher flags passthrough
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---------------- launch Isaac ----------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene

from scipy.ndimage import (
    binary_opening, binary_closing, binary_fill_holes,
    generate_binary_structure, label
)

from task_envs import MP2SceneCfg, PHYSICS_DT, RENDERING_DT

# ---------------- Experiment ----------------
class Experiment:
    def __init__(self):
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([1.5,0.0,1.2],[0.0,0.0,0.15])
        self.sim.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
        print("\nSim dt:", self.sim.get_physics_dt())
        self.sim_dt=self.sim.get_physics_dt()
        self.scene=InteractiveScene(MP2SceneCfg(args_cli.num_envs, env_spacing=2.0))
        self.sim.reset(); print("Setup complete...")
        diff_ik_cfg=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller=DifferentialIKController(diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)
        self.ik_body="gripper_center"
        self.robot_entity_cfg=SceneEntityCfg("ur5e",
            joint_names=["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"],
            body_names=[self.ik_body])
        self.robot_entity_cfg.resolve(self.scene)
        self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]-1 if self.scene["ur5e"].is_fixed_base else self.robot_entity_cfg.body_ids[0]
        self._update_robot_pose()

    def _update_robot_pose(self):
        self.robot_pose=self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
        self.robot_pos=self.robot_pose[:3]; self.robot_quat=self.robot_pose[3:]

    def sim_wait(self, n):
        for _ in range(n):
            self.scene.write_data_to_sim(); self.sim.step(); self.scene.update(self.sim_dt)
        self._update_robot_pose()

    def move_robot_joint(self, target_joint_pos, target_gripper_pos, count=10, time_for_residual_movement=5):
        initial=self.scene['ur5e'].data.joint_pos.clone()
        init_joint_pos=self.scene['ur5e'].data.joint_pos[:, :6].squeeze()
        init_gripper_pos=self.scene['ur5e'].data.joint_pos[:, 6:].squeeze()
        target=self.scene['ur5e'].data.joint_pos.clone()
        if target_gripper_pos is None:
            target[:, :6]=torch.tensor(target_joint_pos); target[:, 6:]=init_gripper_pos
        elif target_joint_pos is None:
            target[:, :6]=init_joint_pos; target[:, 6:]=torch.tensor([target_gripper_pos, target_gripper_pos])
        else:
            target[:, :6]=torch.tensor(target_joint_pos); target[:, 6:]=torch.tensor([target_gripper_pos, target_gripper_pos])
        for i in range(count):
            self.scene["ur5e"].set_joint_position_target((target-initial)/count*i+initial)
            self.scene.write_data_to_sim(); self.sim.step(); self.scene.update(self.sim_dt)
        for _ in range(time_for_residual_movement):
            self.scene["ur5e"].set_joint_position_target(target)
            self.scene.write_data_to_sim(); self.sim.step(); self.scene.update(self.sim_dt)
        self._update_robot_pose()

    def move_robot_ik(self, target_pose, max_joint_change=0.04, ik_tol=1e-3, timeout_count=250):
        self.diff_ik_controller.set_command(torch.tensor(target_pose, device=self.sim.device))
        cnt=0
        while simulation_app.is_running():
            jac=self.scene["ur5e"].root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
            ee=self.scene["ur5e"].data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
            q =self.scene["ur5e"].data.joint_pos[:, self.robot_entity_cfg.joint_ids]
            qd=self.diff_ik_controller.compute(ee[:,0:3], ee[:,3:7], jac, q)
            d=(qd-q).detach().cpu().numpy()[0]
            if (np.abs(d)>max_joint_change).any():
                scl=d/(np.max(np.abs(d))/max_joint_change); scl=torch.tensor(scl).unsqueeze(0).to(qd.device)
                self.scene["ur5e"].set_joint_position_target(q + scl, joint_ids=self.robot_entity_cfg.joint_ids)
            else:
                self.scene["ur5e"].set_joint_position_target(qd, joint_ids=self.robot_entity_cfg.joint_ids)
            self.scene.write_data_to_sim(); self.sim.step(); self.scene.update(self.sim_dt)
            cnt+=1
            cur=self.scene['ur5e'].data.body_state_w[0, self.scene['ur5e'].find_bodies(self.ik_body)[0][0], :7].detach().cpu().numpy()
            if np.average(np.abs(target_pose-cur)[:3])<ik_tol and np.average(np.abs(target_pose-cur)[3:])<ik_tol:
                self._update_robot_pose(); return
            if cnt>=timeout_count: self._update_robot_pose(); return

    def set_gripper(self, width, settle=12):
        self.move_robot_joint(None, float(width), count=settle, time_for_residual_movement=settle)

    def run(self):
        os.makedirs(args_cli.save_dir, exist_ok=True)
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")

        # clear camera view
        self.sim_wait(20)
        target=self.robot_pos - np.array([0.15,0.2,0.0])
        self.move_robot_ik(np.concatenate([target, self.robot_quat]))
        self.sim_wait(20)

        # =================================================================================
        # ========== INLINED PERCEPTION LOGIC (EQUIVALENT TO `perceive()`) ==========
        # =================================================================================
        
        # --- Get Camera Data ---
        color_raw = self.scene["birdview_camera"].data.output["rgb"].detach().cpu().numpy()[0]
        K = np.squeeze(self.scene["birdview_camera"].data.intrinsic_matrices.detach().cpu().numpy())
        
        # Inlined logic from `_ensure_rgb`
        color = color_raw[:, :, :3] if color_raw.shape[2] == 4 else color_raw
        height, width, _ = color.shape
        
        # Inlined logic from `get_depth_from_camera` and `_squeeze_depth_to_hw`
        depth, depth_type = (None, None)
        cam_outs = self.scene["birdview_camera"].data.output
        for key, mode in [("depth", "z"), ("linear_depth", "z"), ("distance_to_camera", "ray")]:
            if key in cam_outs:
                d = cam_outs[key].detach().cpu().numpy()
                d_squeezed = np.squeeze(d)
                while d_squeezed.ndim > 2:
                    d_squeezed = np.squeeze(d_squeezed[0])
                depth = d_squeezed.astype(np.float32, copy=False)
                depth_type = mode
                break
        if depth is None:
            raise KeyError("No supported depth output found in camera data.")

        # --- Inlined Color Segmentation (from `get_red_green_masks`) ---
        rgb01 = np.clip(color.astype(np.float32)/255.0, 0.0, 1.0)
        hsv = mcolors.rgb_to_hsv(rgb01)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        red1_raw = (h >= args_cli.h_red_low_1) & (h <= args_cli.h_red_high_1) & (s >= args_cli.s_min) & (v >= args_cli.v_min)
        red2_raw = (h >= args_cli.h_red_low_2) & (h <= args_cli.h_red_high_2) & (s >= args_cli.s_min) & (v >= args_cli.v_min)
        red_mask_raw = red1_raw | red2_raw
        green_mask_raw = (h >= args_cli.h_green_low) & (h <= args_cli.h_green_high) & (s >= args_cli.s_min) & (v >= args_cli.v_min)
        
        # Inlined mask cleaning
        struct = generate_binary_structure(2,2)
        red_mask = binary_opening(red_mask_raw, structure=struct, iterations=args_cli.morph_iters)
        red_mask = binary_fill_holes(red_mask)
        red_mask = binary_closing(red_mask, structure=struct, iterations=max(1, args_cli.morph_iters//2))
        
        green_mask = binary_opening(green_mask_raw, structure=struct, iterations=args_cli.morph_iters)
        green_mask = binary_fill_holes(green_mask)
        green_mask = binary_closing(green_mask, structure=struct, iterations=max(1, args_cli.morph_iters//2))
        
        # --- Inlined Deprojection (from `deproject_depth_to_camera_points`) ---
        H, W = depth.shape
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        xn = (xs-cx)/fx; yn=(ys-cy)/fy
        if depth_type == "z":
            Z = depth; X = xn * Z; Y = yn * Z
        else: # "ray" mode
            denom = np.sqrt(xn**2 + yn**2 + 1.0)
            t = depth / denom
            X = xn * t; Y = yn * t; Z = t
        pc_cam = np.stack([X,Y,Z], axis=-1).astype(np.float32)
        invalid_depth = ~np.isfinite(depth) | (depth <= 0)
        pc_cam[invalid_depth] = np.nan
        
        # --- Inlined World Transformation (from `transform_camera_to_world`) ---
        T = np.array([[0,1,0,0.5],[1,0,0,0],[0,0,-1,1.2],[0,0,0,1]], dtype=np.float32)
        H, W, _ = pc_cam.shape
        flat_pc = pc_cam.reshape(-1,3)
        ones = np.ones((flat_pc.shape[0], 1), dtype=flat_pc.dtype)
        Ph = np.concatenate([flat_pc, ones], axis=1).T
        Pw = T @ Ph
        P = (Pw[:3,:] / np.clip(Pw[3,:], 1e-8, None)).T
        pc_world = P.reshape(H,W,3)

        # =================================================================================
        # ========== INLINED VISUALIZATION & ANALYSIS LOGIC ==========
        # =================================================================================

        # --- Inlined `save_and_viz_q1` ---
        os.makedirs(args_cli.save_dir, exist_ok=True)
        fig_panel, ax_panel = plt.subplots(1, 3, figsize=(14,5))
        ax_panel[0].imshow(color); ax_panel[0].set_title("Birdview RGB")
        ax_panel[1].imshow(red_mask, cmap="gray"); ax_panel[1].set_title("Red Mask")
        ax_panel[2].imshow(green_mask, cmap="gray"); ax_panel[2].set_title("Green Mask")
        for a in ax_panel: a.axis("off")
        fig_panel.tight_layout()
        fig_panel.savefig(os.path.join(args_cli.save_dir, f"q1_panel_{ts}.png"), dpi=150)
        plt.close(fig_panel)

        # --- Inlined `plot_world_point_cloud` ---
        H,W,_=pc_world.shape; ys_stride=np.arange(0,H,max(1,args_cli.pc_stride)); xs_stride=np.arange(0,W,max(1,args_cli.pc_stride))
        YS,XS=np.meshgrid(ys_stride,xs_stride,indexing='ij'); idx_stride=(YS,XS)
        pts=pc_world[idx_stride].reshape(-1,3); mr=red_mask[idx_stride].reshape(-1); mg=green_mask[idx_stride].reshape(-1)
        val=np.isfinite(pts).all(axis=1); pts=pts[val]; mr=mr[val]; mg=mg[val]
        if pts.shape[0]>args_cli.max_points_plot:
            sel=np.linspace(0,pts.shape[0]-1,args_cli.max_points_plot).astype(int); pts=pts[sel]; mr=mr[sel]; mg=mg[sel]
        plot_colors=np.tile(np.array([[0.6,0.6,0.6]]),(pts.shape[0],1)); plot_colors[mr]=np.array([1,0,0]); plot_colors[mg]=np.array([0,1,0])
        fig_pc=plt.figure(figsize=(8,7)); ax_pc=fig_pc.add_subplot(111,projection='3d')
        ax_pc.scatter(pts[:,0],pts[:,1],pts[:,2],s=1,c=plot_colors,depthshade=False)
        ax_pc.set_xlabel("X (world)"); ax_pc.set_ylabel("Y (world)"); ax_pc.set_zlabel("Z (world)")
        ax_pc.set_title("World-frame Point Cloud"); ax_pc.view_init(35,50)
        fig_pc.tight_layout(); fig_path = os.path.join(args_cli.save_dir, f"q2_world_pointcloud_{ts}.png"); fig_pc.savefig(fig_path, dpi=200); plt.close(fig_pc)
        print("[Q2] Saved world-frame point cloud:", fig_path)

        # --- Inlined Cube Stats Calculation ---
        # Red stats
        lab_red, n_red = label(red_mask.astype(np.uint8), structure=struct)
        if n_red == 0: raise RuntimeError("No red cluster found in scene.")
        red_clusters_info = [(cid, (lab_red == cid).sum()) for cid in range(1, n_red + 1)]
        red_cid_main, _ = max(red_clusters_info, key=lambda item: item[1])
        red_mask_main = (lab_red == red_cid_main)
        
        idx_red = red_mask_main.reshape(-1); pts_red = pc_world.reshape(-1,3)[idx_red]; pts_red = pts_red[np.isfinite(pts_red).all(axis=1)]
        if pts_red.shape[0] == 0: raise RuntimeError("Red cluster has no valid 3D points.")
        xy_red=pts_red[:,:2]; z_red=pts_red[:,2]
        centroid_xy_red=np.nanmedian(xy_red,axis=0); top_z_red=float(np.nanpercentile(z_red,95)); low_z_red=float(np.nanpercentile(z_red,5))
        bbox_min_red=np.nanpercentile(xy_red,5,axis=0); bbox_max_red=np.nanpercentile(xy_red,95,axis=0); size_xy_red=bbox_max_red-bbox_min_red
        red_stats = {"centroid_xy":centroid_xy_red,"top_z":top_z_red,"low_z":low_z_red,"size_xy":size_xy_red}
        red_edge = np.clip(float(np.mean(np.abs(red_stats["size_xy"]))), 0.03, 0.06)

        # Green stats and selection
        lab_green, n_green = label(green_mask.astype(np.uint8), structure=struct)
        if n_green < 2: raise RuntimeError(f"Expected two green clusters but found {n_green}")
        
        green_infos = []
        table_center = np.array([0.5, 0.0])
        green_clusters_info = [(cid, (lab_green == cid).sum()) for cid in range(1, n_green + 1)]
        green_clusters_info.sort(key=lambda item: item[1], reverse=True) # Sort by size

        for cid, area in green_clusters_info[:2]: # Process two largest green clusters
            idx_green = (lab_green == cid); pts_green=pc_world.reshape(-1,3)[idx_green.flatten()]; pts_green=pts_green[np.isfinite(pts_green).all(axis=1)]
            if pts_green.shape[0] == 0: continue
            xy_green=pts_green[:,:2]; z_green=pts_green[:,2]
            centroid_xy_green=np.nanmedian(xy_green,axis=0); top_z_green=float(np.nanpercentile(z_green,95)); low_z_green=float(np.nanpercentile(z_green,5))
            bbox_min_green=np.nanpercentile(xy_green,5,axis=0); bbox_max_green=np.nanpercentile(xy_green,95,axis=0); size_xy_green=bbox_max_green-bbox_min_green
            stats = {"centroid_xy":centroid_xy_green,"top_z":top_z_green,"low_z":low_z_green,"size_xy":size_xy_green}
            d_center = float(np.linalg.norm(stats["centroid_xy"] - table_center))
            green_infos.append({"cid": cid, "area": area, "stats": stats, "d_center": d_center})
        
        if len(green_infos) < 2: raise RuntimeError("Could not compute stats for both green cubes.")
        green_infos.sort(key=lambda it: (it["d_center"], -it["area"]))
        base_info, top_info = green_infos[0], green_infos[1]
        base_stats, top_stats = base_info["stats"], top_info["stats"]
        green_edge = np.mean([np.clip(float(np.mean(np.abs(s["size_xy"]))), 0.03, 0.06) for s in [base_stats, top_stats]])

        # =================================================================================
        # ========== INLINED MANIPULATION LOGIC ==========
        # =================================================================================

        # ----- Pick RED -----
        print("--- Stage 1: Picking RED cube ---")
        z_table_est = float(np.nanpercentile(pc_world[...,2].reshape(-1), 1.0))
        h_red = max(0.01, red_stats["top_z"] - red_stats["low_z"])
        grasp_z_red = max(z_table_est + 0.25*h_red, red_stats["top_z"] - args_cli.grasp_depth)
        
        hover_pos_red = np.array([red_stats["centroid_xy"][0], red_stats["centroid_xy"][1], red_stats["top_z"] + args_cli.hover_offset])
        grasp_pos_red = np.array([red_stats["centroid_xy"][0], red_stats["centroid_xy"][1], grasp_z_red])
        lift_pos_red  = np.array([red_stats["centroid_xy"][0], red_stats["centroid_xy"][1], red_stats["top_z"] + args_cli.lift_offset])

        self.set_gripper(args_cli.gripper_open, settle=10)
        self.move_robot_ik(np.concatenate([hover_pos_red, self.robot_quat])); self.sim_wait(15)
        self.move_robot_ik(np.concatenate([grasp_pos_red, self.robot_quat])); self.sim_wait(10)
        self.set_gripper(args_cli.gripper_close, settle=18); self.sim_wait(12)
        self.move_robot_ik(np.concatenate([lift_pos_red, self.robot_quat])); self.sim_wait(15)

        # ----- Place RED on base GREEN -----
        print("--- Stage 2: Placing RED on BASE GREEN cube ---")
        target_center_z_red = float(base_stats["top_z"] + 0.6*red_edge + args_cli.place_clearance)
        post_lift_z_red = float(base_stats["top_z"] + red_edge + args_cli.lift_offset)
        
        hover_place_pos_red = np.array([base_stats["centroid_xy"][0], base_stats["centroid_xy"][1], target_center_z_red + args_cli.hover_offset])
        place_pos_red = np.array([base_stats["centroid_xy"][0], base_stats["centroid_xy"][1], target_center_z_red])
        post_place_pos_red = np.array([base_stats["centroid_xy"][0], base_stats["centroid_xy"][1], post_lift_z_red])

        self.move_robot_ik(np.concatenate([hover_place_pos_red, self.robot_quat])); self.sim_wait(10)
        self.move_robot_ik(np.concatenate([place_pos_red, self.robot_quat])); self.sim_wait(6)
        self.set_gripper(args_cli.gripper_open, settle=14); self.sim_wait(18)
        self.move_robot_ik(np.concatenate([post_place_pos_red, self.robot_quat])); self.sim_wait(10)

        # ----- Pick TOP GREEN -----
        print("--- Stage 3: Picking TOP GREEN cube ---")
        h_green_top = max(0.01, top_stats["top_z"] - top_stats["low_z"])
        grasp_z_green = max(z_table_est + 0.25 * h_green_top, top_stats["top_z"] - args_cli.grasp_depth)

        hover_pos_green = np.array([top_stats["centroid_xy"][0], top_stats["centroid_xy"][1], top_stats["top_z"] + args_cli.hover_offset])
        grasp_pos_green = np.array([top_stats["centroid_xy"][0], top_stats["centroid_xy"][1], grasp_z_green])
        lift_pos_green  = np.array([top_stats["centroid_xy"][0], top_stats["centroid_xy"][1], top_stats["top_z"] + args_cli.lift_offset])

        self.move_robot_ik(np.concatenate([hover_pos_green, self.robot_quat])); self.sim_wait(10)
        self.move_robot_ik(np.concatenate([grasp_pos_green, self.robot_quat])); self.sim_wait(8)
        self.set_gripper(args_cli.gripper_close, settle=18); self.sim_wait(12)
        self.move_robot_ik(np.concatenate([lift_pos_green, self.robot_quat])); self.sim_wait(12)

        # ----- Place TOP GREEN on RED -----
        print("--- Stage 4: Placing TOP GREEN on RED cube ---")
        red_top_after_place = base_stats["top_z"] + red_edge
        target_center_z_green = float(red_top_after_place + 0.6*green_edge + args_cli.place_clearance)
        post_lift_z_green = float(red_top_after_place + green_edge + args_cli.lift_offset)
        
        hover_place_pos_green = np.array([base_stats["centroid_xy"][0], base_stats["centroid_xy"][1], target_center_z_green + args_cli.hover_offset])
        place_pos_green = np.array([base_stats["centroid_xy"][0], base_stats["centroid_xy"][1], target_center_z_green])
        post_place_pos_green = np.array([base_stats["centroid_xy"][0], base_stats["centroid_xy"][1], post_lift_z_green])

        self.move_robot_ik(np.concatenate([hover_place_pos_green, self.robot_quat])); self.sim_wait(10)
        self.move_robot_ik(np.concatenate([place_pos_green, self.robot_quat])); self.sim_wait(6)
        self.set_gripper(args_cli.gripper_open, settle=14); self.sim_wait(18)
        self.move_robot_ik(np.concatenate([post_place_pos_green, self.robot_quat])); self.sim_wait(10)

        # --- Final stability wait ---
        wait_steps = int(max(1.0, args_cli.stability_wait_s) / PHYSICS_DT)
        print(f"Waiting {wait_steps} steps for stack stability...")
        self.sim_wait(wait_steps)
        print("Stacking sequence finished.")

        # keep viewer open
        while simulation_app.is_running():
            self.scene.write_data_to_sim(); self.sim.step(); self.scene.update(self.sim_dt)
        simulation_app.close()

if __name__=="__main__":
    exp=Experiment()
    exp.run()