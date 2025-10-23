

import argparse, os, json
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
parser.add_argument("--pick_target", type=str, default="red", choices=["red","green"])
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



# ---------------- Utilities: generic ----------------
def _squeeze_depth_to_hw(arr):
    a = np.asarray(arr); a = np.squeeze(a)
    while a.ndim > 2:
        if a.shape[0] == 1: a = np.squeeze(a[0]); continue
        if a.shape[-1] == 1: a = np.squeeze(a[...,0]); continue
        if a.ndim == 3 and a.shape[0] <= 8: a = np.squeeze(a[0]); continue
        a = np.squeeze(a)
        if a.ndim > 2:
            a = np.squeeze(a[0] if a.shape[0] <= a.shape[-1] else a[...,0])
    if a.ndim != 2: raise ValueError(f"Depth must be 2D after squeeze, got {a.shape}")
    return a.astype(np.float32, copy=False)

def _ensure_rgb(img):
    if img.ndim != 3: raise ValueError(f"Expected HxWxC image, got shape {img.shape}")
    return img[:, :, :3] if img.shape[2] == 4 else img


# ---------------- Q1: Color Thresholding ----------------
def rgb_to_hsv01(rgb_uint8):
    rgb01 = np.clip(rgb_uint8.astype(np.float32)/255.0, 0.0, 1.0)
    return mcolors.rgb_to_hsv(rgb01)

def make_mask_from_hsv(hsv, h_low, h_high, s_min, v_min):
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (h>=h_low)&(h<=h_high)&(s>=s_min)&(v>=v_min)

def clean_mask(mask, morph_iters=2):
    struct = generate_binary_structure(2,2)
    out = mask.copy()
    if morph_iters>0: out = binary_opening(out, structure=struct, iterations=morph_iters)
    out = binary_fill_holes(out)
    if morph_iters>0: out = binary_closing(out, structure=struct, iterations=max(1,morph_iters//2))
    return out

def get_red_green_masks(rgb_uint8, args):
    hsv = rgb_to_hsv01(_ensure_rgb(rgb_uint8))
    red1 = make_mask_from_hsv(hsv, args.h_red_low_1, args.h_red_high_1, args.s_min, args.v_min)
    red2 = make_mask_from_hsv(hsv, args.h_red_low_2, args.h_red_high_2, args.s_min, args.v_min)
    red = clean_mask(red1|red2, args.morph_iters)
    green = clean_mask(make_mask_from_hsv(hsv, args.h_green_low, args.h_green_high, args.s_min, args.v_min), args.morph_iters)
    return red, green

def save_and_viz_q1(rgb_uint8, red_mask, green_mask, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rgb_path = os.path.join(save_dir, f"rgb_{ts}.png"); plt.imsave(rgb_path, rgb_uint8)
    red_path = os.path.join(save_dir, f"mask_red_{ts}.png"); plt.imsave(red_path, red_mask.astype(np.uint8), cmap="gray")
    green_path = os.path.join(save_dir, f"mask_green_{ts}.png"); plt.imsave(green_path, green_mask.astype(np.uint8), cmap="gray")
    overlay = rgb_uint8.copy().astype(np.float32)/255.0
    ov_red = overlay.copy(); ov_green = overlay.copy()
    ov_red[red_mask,0] = 1.0; ov_green[green_mask,1] = 1.0
    plt.imsave(os.path.join(save_dir, f"overlay_red_{ts}.png"), np.clip(ov_red,0,1))
    plt.imsave(os.path.join(save_dir, f"overlay_green_{ts}.png"), np.clip(ov_green,0,1))
    fig,ax = plt.subplots(1,3,figsize=(14,5))
    ax[0].imshow(rgb_uint8); ax[0].set_title("Birdview RGB")
    ax[1].imshow(red_mask, cmap="gray"); ax[1].set_title("Red Mask")
    ax[2].imshow(green_mask, cmap="gray"); ax[2].set_title("Green Mask")
    for a in ax: a.axis("off")
    fig.tight_layout(); panel=os.path.join(save_dir, f"q1_panel_{ts}.png"); fig.savefig(panel, dpi=150); plt.close(fig)
    return {"rgb": rgb_path, "mask_red": red_path, "mask_green": green_path, "panel": panel}


# ---------------- Q2: Depth Deprojection ----------------
def get_depth_from_camera(camera):
    outs = camera.data.output
    def _grab(key, mode):
        d = outs[key].detach().cpu().numpy(); d = _squeeze_depth_to_hw(d); return d, mode
    if "distance_to_image_plane" in outs: return _grab("distance_to_image_plane","z")
    if "depth" in outs: return _grab("depth","z")
    if "linear_depth" in outs: return _grab("linear_depth","z")
    if "distance_to_camera" in outs: return _grab("distance_to_camera","ray")
    raise KeyError("No supported depth output found in camera.data.output")

def deproject_depth_to_camera_points(depth, K, mode="z"):
    H,W = depth.shape; fx,fy = float(K[0,0]), float(K[1,1]); cx,cy = float(K[0,2]), float(K[1,2])
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xn = (xs-cx)/fx; yn=(ys-cy)/fy
    if mode=="z":
        Z = depth; X=xn*Z; Y=yn*Z
    else:
        denom=np.sqrt(xn**2+yn**2+1.0); t=depth/denom; X=xn*t; Y=yn*t; Z=t
    pc = np.stack([X,Y,Z],axis=-1).astype(np.float32)
    inv = ~np.isfinite(depth)|(depth<=0); pc[inv]=np.nan; return pc

def transform_camera_to_world(pc_cam, T):
    H,W,_=pc_cam.shape; flat=pc_cam.reshape(-1,3); ones=np.ones((flat.shape[0],1),dtype=flat.dtype)
    Ph = np.concatenate([flat,ones],axis=1).T; Pw = T @ Ph; P = (Pw[:3,:]/np.clip(Pw[3,:],1e-8,None)).T
    return P.reshape(H,W,3)

def plot_world_point_cloud(pc_world, red_mask, green_mask, save_path, stride=2, max_points=150000):
    H,W,_=pc_world.shape; ys=np.arange(0,H,max(1,stride)); xs=np.arange(0,W,max(1,stride))
    YS,XS=np.meshgrid(ys,xs,indexing='ij'); idx=(YS,XS)
    pts=pc_world[idx].reshape(-1,3); mr=red_mask[idx].reshape(-1); mg=green_mask[idx].reshape(-1)
    val=np.isfinite(pts).all(axis=1); pts=pts[val]; mr=mr[val]; mg=mg[val]
    if pts.shape[0]>max_points:
        sel=np.linspace(0,pts.shape[0]-1,max_points).astype(int); pts=pts[sel]; mr=mr[sel]; mg=mg[sel]
    colors=np.tile(np.array([[0.6,0.6,0.6]]),(pts.shape[0],1)); colors[mr]=np.array([1,0,0]); colors[mg]=np.array([0,1,0])
    fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111,projection='3d')
    ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=1,c=colors,depthshade=False)
    ax.set_xlabel("X (world)"); ax.set_ylabel("Y (world)"); ax.set_zlabel("Z (world)")
    ax.set_title("World-frame Point Cloud (gray scene, red/green cubes)"); ax.view_init(35,50)
    fig.tight_layout(); fig.savefig(save_path,dpi=200); plt.close(fig)


# ---------------- Q3 helpers ----------------
def estimate_table_height(pc_world):
    z=pc_world[...,2].reshape(-1); z=z[np.isfinite(z)]
    return float(np.nanpercentile(z,1.0)) if z.size else 0.0

def clusters_from_mask(mask):
    struct=generate_binary_structure(2,2)
    lab,n = label(mask.astype(np.uint8), structure=struct)
    clusters=[]
    for i in range(1,n+1):
        idx=(lab==i); clusters.append((i,int(idx.sum()),idx))
    clusters.sort(key=lambda t:t[1], reverse=True); return clusters, lab

def cluster_stats_from_pc(mask, pc_world):
    idx=mask.reshape(-1); pts=pc_world.reshape(-1,3)[idx]; pts=pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0]==0: return None
    xy=pts[:,:2]; z=pts[:,2]
    centroid_xy=np.nanmedian(xy,axis=0); top_z=float(np.nanpercentile(z,95)); low_z=float(np.nanpercentile(z,5))
    bbox_min=np.nanpercentile(xy,5,axis=0); bbox_max=np.nanpercentile(xy,95,axis=0); size_xy=bbox_max-bbox_min
    return {"centroid_xy":centroid_xy,"top_z":top_z,"low_z":low_z,"size_xy":size_xy,"num_points":int(pts.shape[0])}


# ---------------- Q4 helpers (Placing) ----------------
def size_from_stats(stats):
    """Approximate cube edge length from XY extent (robust)."""
    if stats is None: return None
    s = float(np.mean(np.abs(stats["size_xy"])))
    # clamp to reasonable cube sizes (m) to avoid outliers
    return float(np.clip(s, 0.03, 0.06))

def choose_base_and_top_green(green_mask, pc_world):
    """Return (base, top) stats dicts and masks for two green cubes.
    Selection rule:
      1) Prefer the green cube closer to table center (0.5, 0.0) as BASE (more margin to edges).
      2) Tie-breaker by more points (denser, less occluded).
    """
    (clusters, lab) = clusters_from_mask(green_mask)
    if len(clusters) < 2:
        raise RuntimeError("Expected two green clusters but found {}".format(len(clusters)))
    # collect stats
    infos = []
    table_center = np.array([0.5, 0.0])  # from environment setup
    for cid, area, idx in clusters[:2]:
        stats = cluster_stats_from_pc(idx, pc_world)
        if stats is None: continue
        d_center = float(np.linalg.norm(np.array(stats["centroid_xy"]) - table_center))
        side = size_from_stats(stats)
        infos.append({"cid": cid, "area": area, "mask": idx, "stats": stats, "d_center": d_center, "side": side})
    if len(infos) < 2:
        raise RuntimeError("Could not compute stats for both green cubes.")
    # rank: closer to center first, then more points
    infos.sort(key=lambda it: (it["d_center"], -it["area"]))
    base, top = infos[0], infos[1]
    return base, top, lab

def make_single_cluster_mask(lab, cid):
    return (lab == cid)

def plan_place_over_support(support_xy, support_top_z, cube_edge, hover_offset=0.12, clearance=0.005, lift_offset=0.15):
    """Plan a vertical place motion to put the cube centered over the support top."""
    cx, cy = float(support_xy[0]), float(support_xy[1])
    target_center_z = float(support_top_z + 0.5*cube_edge + clearance)
    post_lift_z = float(support_top_z + cube_edge + lift_offset)
    return {
        "hover": [cx, cy, target_center_z + hover_offset],
        "place": [cx, cy, target_center_z],
        "post":  [cx, cy, post_lift_z]
    }


# ---------------- Experiment ----------------
class Experiment:
    def __init__(self):
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([1.5,0.0,1.2],[0.0,0.0,0.15])
        self.sim.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
        print("\\nSim dt:", self.sim.get_physics_dt())
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

    # -------- Perception (Q1 + Q2) --------
    def perceive(self):
        color=self.scene["birdview_camera"].data.output["rgb"].detach().cpu().numpy()[0]; color=_ensure_rgb(color)
        depth,depth_type=get_depth_from_camera(self.scene["birdview_camera"])
        K=np.squeeze(self.scene["birdview_camera"].data.intrinsic_matrices.detach().cpu().numpy())

        # Q1 masks
        red,green=get_red_green_masks(color, args_cli)

        # Q2 deprojection
        pc_cam=deproject_depth_to_camera_points(depth, K, mode="z" if depth_type=="z" else "ray")

        # Provided extrinsics (camera->world); replace with starter's constant if different
        T=np.array([[0,1,0,0.5],[1,0,0,0],[0,0,-1,1.2],[0,0,0,1]],dtype=np.float32)
        pc_world=transform_camera_to_world(pc_cam, T)
        return color, red, green, pc_world, K, T, depth, depth_type

    # -------- Grasp planner (Q3) --------
    def plan_grasp(self, pc_world, mask, target_name):
        table_z=estimate_table_height(pc_world)
        (clusters, _) = clusters_from_mask(mask)
        if not clusters: raise RuntimeError(f"No {target_name} cluster found.")
        _, area, idx = clusters[0]
        stats=cluster_stats_from_pc(idx, pc_world)
        if stats is None: raise RuntimeError(f"{target_name} cluster has no valid 3D points.")
        cx,cy = stats["centroid_xy"]; top_z=stats["top_z"]; low_z=stats["low_z"]; size_xy=stats["size_xy"]; h=max(0.01, top_z-low_z)
        hover_z=top_z+args_cli.hover_offset
        grasp_z=max(table_z+0.25*h, top_z-args_cli.grasp_depth)
        return {
            "target": target_name,
            "centroid_xy":[float(cx),float(cy)],
            "table_z":float(table_z), "top_z":float(top_z), "height":float(h),
            "size_xy":[float(size_xy[0]), float(size_xy[1])],
            "hover":[float(cx),float(cy),float(hover_z)],
            "grasp":[float(cx),float(cy),float(grasp_z)],
            "lift":[float(cx),float(cy),float(top_z+args_cli.lift_offset)],
            "area_px":int(area),
            "num_points":int(stats["num_points"])
        }

    def execute_grasp(self, plan):
        hover=np.concatenate([np.array(plan["hover"]), self.robot_quat])
        grasp=np.concatenate([np.array(plan["grasp"]), self.robot_quat])
        lift =np.concatenate([np.array(plan["lift"]),  self.robot_quat])
        self.set_gripper(args_cli.gripper_open, settle=10)
        self.move_robot_ik(hover, ik_tol=1e-3); self.sim_wait(10)
        self.move_robot_ik(grasp, ik_tol=8e-4); self.sim_wait(8)
        self.set_gripper(args_cli.gripper_close, settle=18); self.sim_wait(12)
        self.move_robot_ik(lift, ik_tol=1e-3); self.sim_wait(12)

    # -------- Placing (Q4) --------
    def execute_place(self, place_plan):
        hover=np.concatenate([np.array(place_plan["hover"]), self.robot_quat])
        place=np.concatenate([np.array(place_plan["place"]), self.robot_quat])
        post =np.concatenate([np.array(place_plan["post"]),  self.robot_quat])
        self.move_robot_ik(hover, ik_tol=1e-3); self.sim_wait(10)
        self.move_robot_ik(place, ik_tol=8e-4); self.sim_wait(6)
        # open to release and let cube settle
        self.set_gripper(args_cli.gripper_open, settle=14); self.sim_wait(18)
        self.move_robot_ik(post, ik_tol=1e-3); self.sim_wait(10)

    # -------- Main --------
    def run(self):
        os.makedirs(args_cli.save_dir, exist_ok=True)
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")

        # clear camera view
        self.sim_wait(20)
        target=self.robot_pos - np.array([0.15,0.2,0.0])
        self.move_robot_ik(np.concatenate([target, self.robot_quat]))
        self.sim_wait(20)

        # Perceive (Q1+Q2)
        color, red, green, pc_world, K, T, depth, depth_type = self.perceive()

        # Save Q1 visualizations
        q1_paths = save_and_viz_q1(color, red, green, args_cli.save_dir)

        # Save Q2 arrays and world-frame point cloud figure
        npz_path=os.path.join(args_cli.save_dir, f"q2_pointcloud_{ts}.npz")
        np.savez_compressed(npz_path, intrinsics=K, extrinsics=T, depth=depth, depth_type=depth_type,
                            pc_world=pc_world, red_mask=red, green_mask=green)
        fig_path = os.path.join(args_cli.save_dir, f"q2_world_pointcloud_{ts}.png")
        plot_world_point_cloud(pc_world, red, green, fig_path, stride=max(1,args_cli.pc_stride), max_points=args_cli.max_points_plot)
        print("[Q2] Saved arrays:", npz_path)
        print("[Q2] Saved world-frame point cloud:", fig_path)

        # ----- Compute stats for all cubes (before any manipulation) -----
        # Red stats
        (red_clusters, _) = clusters_from_mask(red)
        if not red_clusters:
            raise RuntimeError("No red cluster found in scene.")
        _, _, red_mask_main = red_clusters[0]
        red_stats = cluster_stats_from_pc(red_mask_main, pc_world)
        red_edge = size_from_stats(red_stats) if red_stats else 0.04  # fallback

        # Greens: choose base & top
        base_info, top_info, lab = choose_base_and_top_green(green, pc_world)
        base_stats, top_stats = base_info["stats"], top_info["stats"]
        green_edge = np.mean([size_from_stats(base_stats), size_from_stats(top_stats)])

        # Persist plan summary
        plan_summary = {
            "red_edge_est": red_edge,
            "green_edge_est": green_edge,
            "base_green": {"cid": base_info["cid"], "centroid_xy": list(map(float, base_stats["centroid_xy"])), "top_z": float(base_stats["top_z"])},
            "top_green":  {"cid": top_info["cid"],  "centroid_xy": list(map(float, top_stats["centroid_xy"])),  "top_z": float(top_stats["top_z"])},
        }
        with open(os.path.join(args_cli.save_dir, f"q4_plan_summary_{ts}.json"), "w") as f:
            json.dump(plan_summary, f, indent=2)

        # ----- Q3: Pick RED -----
        red_pick_plan = self.plan_grasp(pc_world, red, "red")
        with open(os.path.join(args_cli.save_dir, f"q3_grasp_plan_red_{ts}.json"), "w") as f:
            json.dump(red_pick_plan, f, indent=2)
        self.execute_grasp(red_pick_plan)

        # ----- Q4: Place RED on base GREEN -----
        place_red = plan_place_over_support(
            support_xy=base_stats["centroid_xy"],
            support_top_z=base_stats["top_z"],
            cube_edge=red_edge,
            hover_offset=args_cli.hover_offset,
            clearance=args_cli.place_clearance,
            lift_offset=args_cli.lift_offset,
        )
        with open(os.path.join(args_cli.save_dir, f"q4_place_red_on_base_{ts}.json"), "w") as f:
            json.dump(place_red, f, indent=2)
        self.execute_place(place_red)

        # ----- Q3: Pick TOP GREEN (the non-base one) -----
        # build a mask that only contains the 'top' green cluster
        top_mask_only = make_single_cluster_mask(lab, top_info["cid"])
        green_top_pick_plan = self.plan_grasp(pc_world, top_mask_only, "green-top")
        with open(os.path.join(args_cli.save_dir, f"q3_grasp_plan_green_top_{ts}.json"), "w") as f:
            json.dump(green_top_pick_plan, f, indent=2)
        self.execute_grasp(green_top_pick_plan)

        # ----- Q4: Place TOP GREEN on RED -----
        red_top_after_place = base_stats["top_z"] + red_edge
        place_green_top = plan_place_over_support(
            support_xy=base_stats["centroid_xy"],   # stack centered
            support_top_z=red_top_after_place,
            cube_edge=green_edge,
            hover_offset=args_cli.hover_offset,
            clearance=args_cli.place_clearance,
            lift_offset=args_cli.lift_offset,
        )
        with open(os.path.join(args_cli.save_dir, f"q4_place_green_on_red_{ts}.json"), "w") as f:
            json.dump(place_green_top, f, indent=2)
        self.execute_place(place_green_top)

        # ----- Q4: Final stability wait (>= 3s) -----
        wait_steps = int(max(1.0, args_cli.stability_wait_s) / PHYSICS_DT)
        print(f"[Q4] Waiting {wait_steps} steps for stack stability...")
        self.sim_wait(wait_steps)

        print("[Q4] Stacking sequence finished (target: green-red-green).")

        # keep viewer open
        while simulation_app.is_running():
            self.scene.write_data_to_sim(); self.sim.step(); self.scene.update(self.sim_dt)
        simulation_app.close()


if __name__=="__main__":
    exp=Experiment()
    exp.run()
