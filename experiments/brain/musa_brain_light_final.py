import time
import numpy as np
import mujoco
import mujoco.viewer

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import (
    HybridTurningController,
    HybridControllerObservation,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)
from flygym.utils.math import Rotation3D


# ============================================================
# MUSA
# ============================================================

fly = make_locomotion_fly(
    name="musa",
    add_adhesion=True,
    colorize=True,
)

world = FlatGroundWorld()

world.add_fly(
    fly,
    [0, 0, 0.8],
    Rotation3D("quat", [1, 0, 0, 0]),
)

sim = Simulation(world)


# ============================================================
# НАСТОЯЩИЙ TURNING CONTROLLER FLYGYM
# ============================================================

preprogrammed_steps = PreprogrammedSteps()

dof_order = fly.get_actuated_jointdofs_order("position")

controller = HybridTurningController(
    timestep=sim.timestep,
    preprogrammed_steps=preprogrammed_steps,
    output_dof_order=dof_order,
)


# ============================================================
# INITIAL POSE
# ============================================================

sim.reset()
controller.reset(seed=0)

# ============================================================
# LIGHT
# ============================================================

light_x = 2.0
light_y = 0.0
light_z = 1.5

LIGHT_STEP = 0.5
LIGHT_LIMIT = 8.0


# ============================================================
# STATE
# ============================================================

walking = False


# ============================================================
# THORAX
# ============================================================

thorax_id = model_body_id = model_body_id = None


# Получаем MuJoCo body ID после создания Simulation.
thorax_id = sim.mj_model.body(
    "musa/c_thorax"
).id


# ============================================================
# BRAIN
# ============================================================

def get_descending_signal():

    """
    Превращает положение света в два канала мозга:

        LEFT
        RIGHT

    Координаты света переводятся из мировых координат
    в систему координат грудной клетки Музы.
    """

    # Положение грудной клетки.
    thorax_pos = sim.mj_data.xpos[thorax_id].copy()

    # Матрица ориентации грудной клетки.
    thorax_rot = sim.mj_data.xmat[
        thorax_id
    ].reshape(3, 3)

    # Свет относительно Музы в WORLD.
    light_world = np.array([
        light_x,
        light_y,
        light_z,
    ])

    relative_world = (
        light_world -
        thorax_pos
    )

    # WORLD -> BODY.
    relative_body = (
        thorax_rot.T @ relative_world
    )

    forward = relative_body[0]
    lateral = relative_body[1]

    # Чем ближе свет и чем сильнее он сбоку,
    # тем сильнее соответствующий сенсор.
    distance = np.linalg.norm(
        relative_body[:2]
    )

    distance_gain = np.exp(
        -distance / 6.0
    )

    # Передние объекты важнее задних.
    forward_gain = np.clip(
        (forward + 1.0) / 3.0,
        0.15,
        1.0,
    )

    # LEFT / RIGHT sensory neurons.
    left_activation = np.exp(
        -((lateral + 1.2) ** 2) / 2.0
    )

    right_activation = np.exp(
        -((lateral - 1.2) ** 2) / 2.0
    )

    left_activation *= (
        distance_gain *
        forward_gain
    )

    right_activation *= (
        distance_gain *
        forward_gain
    )

    # Минимальная активность,
    # чтобы CPG продолжал ходить.
    base = 0.65

    left_signal = base + 1.5 * left_activation
    right_signal = base + 1.5 * right_activation

    return np.array([
        left_signal,
        right_signal,
    ])


# ============================================================
# KEYBOARD
# ============================================================

def key_callback(key):

    global light_x
    global light_y
    global walking

    # SPACE
    if key == 32:

        walking = not walking

        print(
            "\n🪰 WALK:",
            "ON" if walking else "OFF"
        )

    # LEFT
    elif key == 263:

        light_x -= LIGHT_STEP

    # RIGHT
    elif key == 262:

        light_x += LIGHT_STEP

    # UP
    elif key == 265:

        light_y += LIGHT_STEP

    # DOWN
    elif key == 264:

        light_y -= LIGHT_STEP

    # R — вернуть свет
    elif key == 82:

        light_x = 2.0
        light_y = 0.0

    light_x = np.clip(
        light_x,
        -LIGHT_LIMIT,
        LIGHT_LIMIT,
    )

    light_y = np.clip(
        light_y,
        -LIGHT_LIMIT,
        LIGHT_LIMIT,
    )


# ============================================================
# START
# ============================================================

print()
print("🪰 MUSA — BRAIN / LIGHT / TURNING")
print("=" * 60)
print()
print("SPACE     ходьба ON/OFF")
print("← → ↑ ↓   движение света")
print("R         вернуть свет")
print()
print("🧠 HybridTurningController подключён.")
print("🟡 Свет является входом мозга.")
print()


# ============================================================
# VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
    sim.mj_model,
    sim.mj_data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():

        # ----------------------------------------------------
        # BRAIN
        # ----------------------------------------------------

        descending_signal = (
            get_descending_signal()
        )


        # ----------------------------------------------------
        # MOTOR SYSTEM
        # ----------------------------------------------------

        if walking:

            obs = (
                HybridControllerObservation
                .from_sim(
                    sim,
                    fly.name,
                )
            )

            action = controller.step(
                descending_signal,
                obs,
            )

            apply_locomotion_action(
                sim,
                fly.name,
                action,
            )


        # ----------------------------------------------------
        # PHYSICS
        # ----------------------------------------------------

        sim.step()


        # ----------------------------------------------------
        # LIGHT
        # ----------------------------------------------------

        viewer.user_scn.ngeom = 0

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],

            type=mujoco.mjtGeom.mjGEOM_SPHERE,

            size=np.array([
                0.30,
                0,
                0,
            ]),

            pos=np.array([
                light_x,
                light_y,
                light_z,
            ]),

            mat=np.eye(3).flatten(),

            rgba=np.array([
                1.0,
                1.0,
                0.0,
                1.0,
            ]),
        )

        viewer.user_scn.ngeom = 1


        # ----------------------------------------------------
        # MONITOR
        # ----------------------------------------------------

        print(
            "\r"
            f"L={descending_signal[0]:.2f} "
            f"R={descending_signal[1]:.2f} "
            f"| LIGHT "
            f"X={light_x:+.1f} "
            f"Y={light_y:+.1f}",
            end="",
            flush=True,
        )


        viewer.sync()

        time.sleep(0.005)
