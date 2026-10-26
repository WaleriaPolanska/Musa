
"""Просто посмотреть на Музу: последняя модель NeuroMechFly (flygym 2.1).

Мозга и ходьбы здесь нет — муха стоит, модель можно крутить мышкой.

Запуск:  ./run.sh musa/viewer.py
"""

import sys

import mujoco.viewer

from flygym.anatomy import (
    ActuatedDOFPreset,
    ContactBodiesPreset,
    JointPreset,
    Skeleton,
    AxisOrder,
)
from flygym.compose import (
    ActuatorType,
    FlatGroundWorld,
    KinematicPosePreset,
    NeuroMechFly,
)
from flygym.rendering import launch_interactive_viewer
from flygym.utils.math import Rotation3D


if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
    sys.exit(
        "На macOS интерактивный вьювер работает только под mjpython.\n"
        "Запускай так:  ./run.sh musa/viewer.py"
    )

# Полная модель: все биологические сочленения, ноги под управлением.
skeleton = Skeleton(
    joint_preset=JointPreset.ALL_BIOLOGICAL,
    axis_order=AxisOrder.YAW_PITCH_ROLL,
)

fly = NeuroMechFly()
fly.add_joints(skeleton, KinematicPosePreset.NEUTRAL)

fly.add_actuators(
    skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY),
    ActuatorType.POSITION,
    neutral_input=KinematicPosePreset.NEUTRAL,
    kp=50.0,
    ctrlrange=(-3.14, 3.14),
)

fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
fly.colorize()
fly.add_tracking_camera(name="trackingcam")

world = FlatGroundWorld()
world.add_fly(
    fly,
    (0, 0, 0.8),
    Rotation3D("quat", (1, 0, 0, 0)),
    bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
)

mj_model, mj_data = world.compile()

print("🪰 МУЗА — модель flygym 2.1. Закрой окно, чтобы выйти.")

launch_interactive_viewer(mj_model, mj_data)
