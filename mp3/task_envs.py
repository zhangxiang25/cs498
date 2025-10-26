import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg, FrameTransformerCfg


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
class MP3CalibSceneCfg(InteractiveSceneCfg):

    def __init__(self, *args, **kwargs):
        super(MP3CalibSceneCfg, self).__init__(*args, **kwargs)

        # Ground-plane
        self.ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        # lights
        self.dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )

        # robot
        self.ur5e = UR5E_CONFIG.replace(prim_path="/World/ur5e")

        # wrist camera
        self.wrist_camera = CameraCfg(
            prim_path = "/World/ur5e/gripper_center/cam",
            update_period = PHYSICS_DT,
            height = 256,
            width = 256,
            data_types = ["rgb", "depth"],
            spawn = sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-3, 1.0e5)
            ),
            offset = CameraCfg.OffsetCfg(
                pos = (0.0, 0.06, -0.09),
                rot = (0.9659258, 0.258819, 0, 0),
                convention="ros",
            ),
        )

        # apriltag
        self.apriltag = AssetBaseCfg(
            prim_path = "/World/apriltag",
            spawn = sim_utils.UsdFileCfg(
                usd_path = "{}/objects/apriltag.usd".format(os.path.dirname(os.path.abspath(__file__))),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.5, 0.0, 0.11),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )

        # shared parameters
        table_color = (1.0, 1.0, 1.0)

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
                pos = (0.57, 0.0, 0.1),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_1 = AssetBaseCfg(
            prim_path = '/World/table_leg_1',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.09),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.235, 0.335, 0.045),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_2 = AssetBaseCfg(
            prim_path = '/World/table_leg_2',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.09),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.905, 0.335, 0.045),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_3 = AssetBaseCfg(
            prim_path = '/World/table_leg_3',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.09),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.235, -0.335, 0.045),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_4 = AssetBaseCfg(
            prim_path = '/World/table_leg_4',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.09),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.905, -0.335, 0.045),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )


@configclass
class MP3PoseEstimSceneCfg(InteractiveSceneCfg):

    def __init__(self, *args, **kwargs):
        super(MP3PoseEstimSceneCfg, self).__init__(*args, **kwargs)

        # Ground-plane
        self.ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        # lights
        self.dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )

        # robot
        self.ur5e = UR5E_CONFIG.replace(prim_path="/World/ur5e")

        # wrist camera
        self.wrist_camera = CameraCfg(
            prim_path = "/World/ur5e/gripper_center/cam",
            update_period = PHYSICS_DT,
            height = 256,
            width = 256,
            data_types = ["rgb", "depth"],
            spawn = sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-3, 1.0e5)
            ),
            offset = CameraCfg.OffsetCfg(
                pos = (0.0, 0.06, -0.09),
                rot = (0.9659258, 0.258819, 0, 0),
                convention="ros",
            ),
        )

        # vase pos randomization
        center_x = 0.45
        center_y = 0
        pos_noise = 0.05
        pos_x = center_x + (np.random.random() - 0.5)*2 * pos_noise
        pos_y = center_y + (np.random.random() - 0.5)*2 * pos_noise

        rot_z_noise = 30.
        rot_z = (np.random.random() - 0.5)*2 * rot_z_noise
        obj_quat = R.from_euler("xyz", [0, 0, rot_z], degrees=True).as_quat()

        # vase
        self.vase = AssetBaseCfg(
            prim_path = "/World/vase",
            spawn = sim_utils.UsdFileCfg(
                usd_path = "{}/objects/vase.usd".format(os.path.dirname(os.path.abspath(__file__))),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = (1.0, 0.0, 0.0)),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (pos_x, pos_y, 0.11),
                rot = (obj_quat[3], obj_quat[0], obj_quat[1], obj_quat[2]),
            )
        )

        # shared parameters
        table_color = (1.0, 1.0, 1.0)

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
                pos = (0.57, 0.0, 0.1),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_1 = AssetBaseCfg(
            prim_path = '/World/table_leg_1',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.09),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.235, 0.335, 0.045),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_2 = AssetBaseCfg(
            prim_path = '/World/table_leg_2',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.09),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.905, 0.335, 0.045),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_3 = AssetBaseCfg(
            prim_path = '/World/table_leg_3',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.09),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.235, -0.335, 0.045),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )
        self.table_leg_4 = AssetBaseCfg(
            prim_path = '/World/table_leg_4',
            spawn = sim_utils.MeshCuboidCfg(
                size = (0.03, 0.03, 0.09),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),  # does not move in the environment
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = table_color),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos = (0.905, -0.335, 0.045),
                rot = (1.0, 0.0, 0.0, 0.0),
            )
        )