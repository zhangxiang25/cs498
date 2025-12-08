import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg


PHYSICS_DT = 1./60.
RENDERING_DT = 1./60.


UR5E_CONFIG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path = "{}/ur5e_usd/ur5e.usd".format(os.path.dirname(os.path.abspath(__file__))),
        activate_contact_sensors = True,
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        joint_pos = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
        pos = (0.0, 0.0, 0.0)
    ),
    actuators = {
        "ur5e_joints": ImplicitActuatorCfg(
            joint_names_expr = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            velocity_limit = 1e8,
            effort_limit = 1e8,
            stiffness = 3e5,
            damping = 5e3,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr = ["finger_joint_1", "finger_joint_2"],
            effort_limit = 1e2,
            velocity_limit = 1000.0,
            stiffness = 1e3,
            damping = 0.0,
        ),
    },
)

# This points to the USD file you just created.
# We assume you created a 'door_usd' folder next to this env.py file.
DOOR_CONFIG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        # This path finds the 'door_usd' folder in the same directory as this script
        usd_path = "{}/door_usd/door.usd".format(os.path.dirname(os.path.abspath(__file__))),
        activate_contact_sensors = True,
        
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        joint_pos = {
            "hinge_joint": 0.0  # This MUST match the joint name from the editor
        },
    ),
    actuators = {
        "door_hinge": ImplicitActuatorCfg(
            joint_names_expr = ["hinge_joint"], # Must match the joint name
            effort_limit = 1e2,
            velocity_limit = 1000.0,
            stiffness = 100.0, # Lower stiffness so it can be pushed
            damping = 10.0,
        ),
    },

)

@configclass
class MP2SceneCfg(InteractiveSceneCfg):

    def __init__(self, *args, **kwargs):
        super(MP2SceneCfg, self).__init__(*args, **kwargs)

        # Ground-plane
        self.ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        # lights
        self.dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )

        # robot
        self.ur5e = UR5E_CONFIG.replace(prim_path="/World/ur5e")

        # birdview camera
        self.birdview_camera = CameraCfg(
            prim_path = "/World/birdview_camera",
            update_period = PHYSICS_DT,
            height = 256,
            width = 256,
            data_types = ["rgb", "depth"],
            spawn = sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-3, 1.0e5)
            ),
            offset = CameraCfg.OffsetCfg(
                pos = (0.5, 0.0, 1.2),
                rot = (0.0, 0.7071068, 0.7071068, 0.0),
                convention="ros",
            ),
        )

        # shared parameters
        table_color = (1.0, 1.0, 1.0)
        
        # table properties (for randomization)
        table_center_pos = (0.5, 0.0, 0.2)
        table_size = (0.7, 0.7, 0.02)
        table_top_z = table_center_pos[2] + table_size[2] / 2.0  # z = 0.21
        table_min_x = table_center_pos[0] - table_size[0] / 2.0  # 0.15
        table_max_x = table_center_pos[0] + table_size[0] / 2.0  # 0.85
        table_min_y = table_center_pos[1] - table_size[1] / 2.0  # -0.35
        table_max_y = table_center_pos[1] + table_size[1] / 2.0  # 0.35

        # --- Randomization for Door (Modified) ---
        # The door asset's built-in height is 0.2, so its center is 0.1 above its base.
        # We place its base on the table (z=0.21), so the center z is 0.21 + 0.1 = 0.31
        door_z = table_top_z + 0.01

        # modification 1: 减小旋转随机性
        # 原来是 180 度 (全朝向随机)，现在改为 0 度 (固定) 或很小的值 (如 5 度)
        door_rot_noise = 0.0  
        
        # modification 2: 减小位置随机性
        # 不再使用 margin 和 min/max 计算，而是直接基于桌子中心 (table_center_pos) 加一点点噪声
        # table_center_pos 在代码前面定义过，通常是 (0.5, 0.0, 0.2)
        pos_noise = 0.01  # 仅 1cm 的随机误差，几乎相当于固定位置
        
        # 在桌子中心 X 附近微调
        door_x = table_center_pos[0] + (np.random.random() - 0.5) * 2 * pos_noise
        # 在桌子中心 Y 附近微调 (稍微偏一点可能方便机器人抓，视情况调整)
        door_y = table_center_pos[1] + (np.random.random() - 0.5) * 2 * pos_noise
        
        # Random rotation (yaw only)
        # 基础角度 -90.0 度意味着门面朝向 Y 轴（侧对着机器人）或者 X 轴，具体取决于你的 USD 坐标系。
        # 如果你发现这角度不好抓，可以手动改这个 -90.0 为 0.0 或 180.0
        door_rot_euler = np.array([-90.0, 0.0, 0.0])
        door_rot_euler[2] += (np.random.random() - 0.5) * 2. * door_rot_noise
        door_rot_quat = R.from_euler("xyz", door_rot_euler, degrees=True).as_quat()

        # Cube properties
        red_cube_size = 0.04
        green_cube_size = 0.05
        # Correct Z-position: table_top_z (0.21) + half_height
        red_cube_z = table_top_z + (red_cube_size / 2.0)     # 0.23
        green_cube_z = table_top_z + (green_cube_size / 2.0)   # 0.235
        cube_rot_noise = 180.

        placed_positions = []
        MIN_SEPARATION = 0.07  # Min distance (center-to-center) between any two cubes
        MAX_RETRIES = 100      # Max attempts to find a free spot for one cube

        for i in range(10):
            
            valid_position_found = False
            for _ in range(MAX_RETRIES):
                # 1. Generate a candidate position
                red_x = np.random.uniform(table_min_x + red_cube_size / 2.0, table_max_x - red_cube_size / 2.0)
                red_y = np.random.uniform(table_min_y + red_cube_size / 2.0, table_max_y - red_cube_size / 2.0)
                candidate_pos = np.array([red_x, red_y])
                
                # 2. Check if it's too close to any already placed cubes
                is_too_close = False
                for placed_pos in placed_positions:
                    dist = np.linalg.norm(candidate_pos - placed_pos)
                    if dist < MIN_SEPARATION:
                        is_too_close = True
                        break # It's too close, try a new spot
                
                # 3. If it's not too close, accept it
                if not is_too_close:
                    valid_position_found = True
                    break # Found a valid spot
            
            if not valid_position_found:
                print(f"Warning: Could not find a clear spot for red_cube_{i+1}. It may be too close to others.")
            
            # 4. Add the new position to our list
            placed_positions.append(np.array([red_x, red_y]))
 
            # Random rotation
            cube_rot_euler = np.array([0.0, 0.0, 0.0])
            cube_rot_euler[2] += (np.random.random() - 0.5) * 2. * cube_rot_noise
            cube_rot_quat = R.from_euler("xyz", cube_rot_euler, degrees=True).as_quat()
            
            # Create the cube config (uses the validated red_x and red_y)
            cube_cfg = AssetBaseCfg(
                prim_path = f'/World/red_cube_{i+1}', # Dynamic prim path (e.g., /World/red_cube_1)
                spawn = sim_utils.MeshCuboidCfg(
                    size = (red_cube_size, red_cube_size, red_cube_size),
                    rigid_props = sim_utils.RigidBodyPropertiesCfg(),
                    mass_props = sim_utils.MassPropertiesCfg(mass = 0.5),
                    collision_props = sim_utils.CollisionPropertiesCfg(),
                    visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = (1.0, 0.0, 0.0)),
                ),
                init_state = AssetBaseCfg.InitialStateCfg(
                    pos = (red_x, red_y, red_cube_z),
                    rot = (cube_rot_quat[3], cube_rot_quat[0], cube_rot_quat[1], cube_rot_quat[2]),
                )
            )
            # Use setattr to dynamically create self.red_cube_1, self.red_cube_2, etc.
            setattr(self, f'red_cube_{i+1}', cube_cfg)

        for i in range(10):
   
            valid_position_found = False
            for _ in range(MAX_RETRIES):
                # 1. Generate a candidate position
                green_x = np.random.uniform(table_min_x + green_cube_size / 2.0, table_max_x - green_cube_size / 2.0)
                green_y = np.random.uniform(table_min_y + green_cube_size / 2.0, table_max_y - green_cube_size / 2.0)
                candidate_pos = np.array([green_x, green_y])
                
                # 2. Check if it's too close to any already placed cubes (both red and green)
                is_too_close = False
                for placed_pos in placed_positions:
                    dist = np.linalg.norm(candidate_pos - placed_pos)
                    if dist < MIN_SEPARATION:
                        is_too_close = True
                        break # It's too close, try a new spot
                
                # 3. If it's not too close, accept it
                if not is_too_close:
                    valid_position_found = True
                    break # Found a valid spot
            
            if not valid_position_found:
                print(f"Warning: Could not find a clear spot for green_cube_{i+1}. It may be too close to others.")

            # 4. Add the new position to our list
            placed_positions.append(np.array([green_x, green_y]))
            
            # Random rotation
            cube_rot_euler = np.array([0.0, 0.0, 0.0])
            cube_rot_euler[2] += (np.random.random() - 0.5) * 2. * cube_rot_noise
            cube_rot_quat = R.from_euler("xyz", cube_rot_euler, degrees=True).as_quat()
            
    
            # Create the cube config (uses the validated green_x and green_y)
            cube_cfg = AssetBaseCfg(
                prim_path = f'/World/green_cube_{i+1}', # Dynamic prim path (e.g., /World/green_cube_1)
                spawn = sim_utils.MeshCuboidCfg(
                    size = (green_cube_size, green_cube_size, green_cube_size),
                    rigid_props = sim_utils.RigidBodyPropertiesCfg(),
                    mass_props = sim_utils.MassPropertiesCfg(mass = 0.5),
                    collision_props = sim_utils.CollisionPropertiesCfg(),
                    visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = (0.0, 0.5, 0.0)),
                ),
                init_state = AssetBaseCfg.InitialStateCfg(
                    pos = (green_x, green_y, green_cube_z),
                    rot = (cube_rot_quat[3], cube_rot_quat[0], cube_rot_quat[1], cube_rot_quat[2]),
                )
            )
            # Use setattr to dynamically create self.green_cube_1, self.green_cube_2, etc.
            setattr(self, f'green_cube_{i+1}', cube_cfg)
       

        self.door = DOOR_CONFIG.replace(
            prim_path = '/World/door',
            init_state = ArticulationCfg.InitialStateCfg(
                pos = (door_x, door_y, door_z), # Use the random pos
                rot = (door_rot_quat[3], door_rot_quat[0], door_rot_quat[1], door_rot_quat[2]), # Use the random rot
                joint_pos = { "hinge_joint": 0.0 } # Start closed
            )
            
        )

        # table
        self.table = AssetBaseCfg(
            prim_path = '/World/table',
            spawn = sim_utils.MeshCuboidCfg(
                size = table_size, # Use variable
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 5.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = table_center_pos, # Use variable
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_1 = AssetBaseCfg(
            prim_path = '/World/table_leg_1',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.19),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.165, 0.335, 0.095),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_2 = AssetBaseCfg(
            prim_path = '/World/table_leg_2',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.19),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.835, 0.335, 0.095),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_3 = AssetBaseCfg(
            prim_path = '/World/table_leg_3',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.19),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.165, -0.335, 0.095),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_4 = AssetBaseCfg(
            prim_path = '/World/table_leg_4',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.19),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.835, -0.335, 0.095),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
