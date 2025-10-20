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

        # randomize cube poses
        center_x = 0.5
        center_y = 0.0
        cube_pos_noise = 0.05
        red_cube_x = center_x + (np.random.random() - 0.5)*2 * cube_pos_noise
        red_cube_y = center_y + (np.random.random() - 0.5)*2 * cube_pos_noise
        green_cube_1_angle = np.random.random() * 2*np.pi
        green_cube_dist = 0.055
        cube_rot_noise = 180.

        # red cube
        cube_rot_euler = np.array([0.0, 0.0, 0.0])
        cube_rot_euler[2] += (np.random.random() - 0.5) * 2. * cube_rot_noise
        cube_rot_quat = R.from_euler("xyz", cube_rot_euler, degrees=True).as_quat()
        self.cubeA = AssetBaseCfg(
            prim_path = '/World/cubeA',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.04, 0.04, 0.04),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(),
                mass_props = sim_utils.MassPropertiesCfg(mass = 0.5),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = (1.0, 0.0, 0.0)),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (red_cube_x, red_cube_y, 0.225),
                rot = (cube_rot_quat[3], cube_rot_quat[0], cube_rot_quat[1], cube_rot_quat[2]),
            )
        )

        # green cube 1
        green_cube_1_x = red_cube_x + np.cos(green_cube_1_angle) * green_cube_dist
        green_cube_1_y = red_cube_y + np.sin(green_cube_1_angle) * green_cube_dist
        cube_rot_euler = np.array([0.0, 0.0, 0.0])
        cube_rot_euler[2] += (np.random.random() - 0.5) * 2. * cube_rot_noise
        cube_rot_quat = R.from_euler("xyz", cube_rot_euler, degrees=True).as_quat()
        self.cubeB = AssetBaseCfg(
            prim_path = "/World/cubeB",
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.05, 0.05, 0.05),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(),
                mass_props = sim_utils.MassPropertiesCfg(mass = 0.5),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = (0.0, 0.5, 0.0)),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (green_cube_1_x, green_cube_1_y, 0.23),
                rot = (cube_rot_quat[3], cube_rot_quat[0], cube_rot_quat[1], cube_rot_quat[2]),
            )
        )

        # green cube 2
        min_separation = 90.*np.pi/180.
        max_separation = 2*np.pi - min_separation
        separation = np.random.random() * (max_separation - min_separation) + min_separation
        green_cube_2_angle = green_cube_1_angle + separation
        green_cube_2_x = red_cube_x + np.cos(green_cube_2_angle) * green_cube_dist
        green_cube_2_y = red_cube_y + np.sin(green_cube_2_angle) * green_cube_dist
        cube_rot_euler = np.array([0.0, 0.0, 0.0])
        cube_rot_euler[2] += (np.random.random() - 0.5) * 2. * cube_rot_noise
        cube_rot_quat = R.from_euler("xyz", cube_rot_euler, degrees=True).as_quat()
        self.cubeC = AssetBaseCfg(
            prim_path = "/World/cubeC",
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.05, 0.05, 0.05),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(),
                mass_props = sim_utils.MassPropertiesCfg(mass = 0.5),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = (0.0, 0.5, 0.0)),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (green_cube_2_x, green_cube_2_y, 0.23),
                rot = (cube_rot_quat[3], cube_rot_quat[0], cube_rot_quat[1], cube_rot_quat[2]),
            )
        )

        # print(red_cube_x, red_cube_y)
        # print(green_cube_1_x, green_cube_1_y)
        # print(green_cube_2_x, green_cube_2_y)
        # print(np.sqrt((green_cube_1_x - red_cube_x)**2 + (green_cube_1_y - red_cube_y)**2))
        # print(np.sqrt((green_cube_2_x - red_cube_x)**2 + (green_cube_2_y - red_cube_y)**2))

        # table
        self.table = AssetBaseCfg(
            prim_path = '/World/table',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.7, 0.7, 0.02),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 5.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.5, 0.0, 0.2),
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