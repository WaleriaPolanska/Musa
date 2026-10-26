import numpy as np
import mujoco
import mujoco.viewer

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import (
    CPGController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
    make_tripod_cpg_network,
)
from flygym.utils.math import Rotation3D


# =========================
# MUSA
# =========================

fly = make_locomotion_fly(
    name="musa",
    add_adhesion=True,
    colorize=True,
)

world = FlatGroundWorld()

world.add_fly(
    fly,
    [0, 0, 0.5],
    Rotation3D("quat", [1, 0, 0, 0]),
)

# =========================
# НАШ СВЕТ
# =========================

light_pos = np.array([0.0, -2.0, 2.5], dtype=float)

world.mjcf_root.worldbody.add_light(
    name="musa_light",
    pos=light_pos.tolist(),
    dir=[0, 0, -1],
    diffuse=[1, 1, 1],
    specular=[1, 1, 1],
    ambient=[0.01, 0.01, 0.01],
    castshadow=True,
)

marker = world.mjcf_root.worldbody.add_body(
    name="light_marker",
    pos=light_pos.tolist(),
)

marker.add_geom(
    name="light_marker_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[0.08, 0, 0],
    rgba=[1, 1, 0.05, 1],
    contype=0,
    conaffinity=0,
)

sim = Simulation(world)

model = sim.mj_model
data = sim.mj_data


# =========================
# CPG
# =========================

preprogrammed_steps = PreprogrammedSteps()

dof_order = fly.get_actuated_jointdofs_order("position")

cpg_network = make_tripod_cpg_network(
    timestep=sim.timestep,
    intrinsic_amplitude=1.0,
    coupling_strength=10.0,
    convergence_coef=20.0,
    seed=0,
)

controller = CPGController(
    cpg_network=cpg_network,
    preprogrammed_steps=preprogrammed_steps,
    output_dof_order=dof_order,
)


# =========================
# LIGHT SENSORS
# =========================

def sensors():
    x = light_pos[0]

    # Три рецептора.
    left = np.exp(-((x + 1.0) ** 2) / 0.5)
    center = np.exp(-(x ** 2) / 0.5)
    right = np.exp(-((x - 1.0) ** 2) / 0.5)

    values = np.array([left, center, right])

    return values / max(values.max(), 1e-8)


def show_sensors():

    left, center, right = sensors()

    turn = right - left

    if turn < -0.15:
        direction = "TURN LEFT  ←"
    elif turn > 0.15:
        direction = "TURN RIGHT →"
    else:
        direction = "CENTER     ↑"

    print(
        "\033[2J\033[H"
        "🪰 MUSA — LIGHT BRAIN\n"
        "========================\n\n"
        f"LEFT    {'█' * int(left * 10):10s} {left:.2f}\n"
        f"CENTER  {'█' * int(center * 10):10s} {center:.2f}\n"
        f"RIGHT   {'█' * int(right * 10):10s} {right:.2f}\n\n"
        f"NEURAL TURN SIGNAL: {turn:+.2f}\n"
        f"BEHAVIOR: {direction}\n\n"
        "← →  move light\n"
        "SPACE  center light\n"
        "ESC    quit\n"
    )


# =========================
# KEYBOARD
# =========================

light_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_LIGHT,
    "musa_light",
)

marker_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "light_marker",
)


def move_light(dx):

    light_pos[0] = np.clip(
        light_pos[0] + dx,
        -2.5,
        2.5,
    )

    model.light_pos[light_id][:] = light_pos
    model.body_pos[marker_id][:] = light_pos

    show_sensors()


def key_callback(key):

    if key == 263:       # LEFT
        move_light(-0.25)

    elif key == 262:     # RIGHT
        move_light(+0.25)

    elif key == 32:      # SPACE
        light_pos[0] = 0.0

        model.light_pos[light_id][:] = light_pos
        model.body_pos[marker_id][:] = light_pos

        show_sensors()


# =========================
# START
# =========================

print(
    """
🪰 MUSA — LIGHT SENSOR

Муза стоит на месте.
Свет управляет только сенсорным сигналом.

←  свет влево
→  свет вправо
SPACE  центр

"""
)

show_sensors()


# =========================
# VIEWER
# =========================

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():

        # CPG продолжает работать,
        # но мы НЕ даём ему толкать Музу вперёд.
        action = controller.step()

        angles = np.array(
            action.joint_angles,
            copy=True,
        )

        # Очень маленький поворотный сигнал.
        left, center, right = sensors()
        turn = right - left

        # Только симметричная небольшая модуляция.
        # Это пока демонстрация сенсорного -> моторного пути.
        n = min(6, len(angles))

        angles[:n] += turn * 0.015

        action = LocomotionAction(
            joint_angles=angles,
            adhesion_onoff=action.adhesion_onoff,
        )

        apply_locomotion_action(
            sim,
            fly.name,
            action,
        )

        sim.step()

        viewer.sync()
