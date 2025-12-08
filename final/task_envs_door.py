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

# Assumes 'door_usd' folder is next to this file
DOOR_CONFIG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
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
            joint_names_expr = ["hinge_joint"], 
            effort_limit = 1e2,
            velocity_limit = 1000.0,
            stiffness = 100.0, 
            damping = 10.0,
        ),
    },

)

@configclass
class DoorSceneCfg(InteractiveSceneCfg):

    def __init__(self, *args, **kwargs):
        super(DoorSceneCfg, self).__init__(*args, **kwargs)

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
        
        # table properties
        table_center_pos = (0.5, 0.0, 0.2)
        table_size = (0.7, 0.7, 0.02)
        table_top_z = table_center_pos[2] + table_size[2] / 2.0  # z = 0.21
        table_min_x = table_center_pos[0] - table_size[0] / 2.0  
        table_max_x = table_center_pos[0] + table_size[0] / 2.0  
        table_min_y = table_center_pos[1] - table_size[1] / 2.0 
        table_max_y = table_center_pos[1] + table_size[1] / 2.0  

        # --- Randomization for Door (Modified for easier training) ---
        # The door asset's built-in height is 0.2, so its center is 0.1 above its base.
        # We place its base on the table (z=0.21), so the center z is 0.21 + 0.1 = 0.31
        door_z = table_top_z + 0.01

        # Modification 1: Zero rotation noise (Fixed orientation)
        door_rot_noise = 0.0  
        
        # Modification 2: Minimal position noise (Centered on table)
        pos_noise = 0.01  # 1cm noise
        
        # Center the door on the table with tiny jitter
        door_x = table_center_pos[0] + (np.random.random() - 0.5) * 2 * pos_noise
        door_y = table_center_pos[1] + (np.random.random() - 0.5) * 2 * pos_noise
        
        # Fixed rotation (yaw only)
        # -90 degrees typically faces the Y-axis. Adjust if needed based on your USD.
        door_rot_euler = np.array([-90.0, 0.0, 0.0])
        door_rot_euler[2] += (np.random.random() - 0.5) * 2. * door_rot_noise
        door_rot_quat = R.from_euler("xyz", door_rot_euler, degrees=True).as_quat()

        # Add Door to Scene
        self.door = DOOR_CONFIG.replace(
            prim_path = '/World/door',
            init_state = ArticulationCfg.InitialStateCfg(
                pos = (door_x, door_y, door_z), 
                rot = (door_rot_quat[3], door_rot_quat[0], door_rot_quat[1], door_rot_quat[2]), 
                joint_pos = { "hinge_joint": 0.0 } # Start closed
            )
        )

        # table
        self.table = AssetBaseCfg(
            prim_path = '/World/table',
            spawn = sim_utils.MeshCuboidCfg(
                size = table_size, 
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),
                mass_props = sim_utils.MassPropertiesCfg(mass = 5.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = table_center_pos,
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_1 = AssetBaseCfg(
            prim_path = '/World/table_leg_1',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.19),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),
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
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),
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
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),
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
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.835, -0.335, 0.095),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )