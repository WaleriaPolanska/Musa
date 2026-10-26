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

model = sim.mj_model
data = sim.mj_data


# ============================================================
# CONTROLLER
# ============================================================

steps = PreprogrammedSteps()

dof_order = fly.get_actuated_jointdofs_order("position")

controller = HybridTurningController(
    timestep=sim.timestep,
    preprogrammed_steps=steps,
    output_dof_order=dof_order,
    enable_adhesion=True,
)

controller.reset(seed=0)
sim.reset()


# ============================================================
# LIGHT
# ============================================================

light_x = 2.0
light_y = 0.0
light_z = 1.5

LIGHT_STEP = 1.0
LIGHT_LIMIT = 10.0


# ============================================================
# STATE
# ============================================================

walking = False


# ============================================================
# THORAX
# ============================================================

thorax_id = model.body("musa/c_thorax").id


# ============================================================
# BRAIN
# ============================================================

def brain():

    thorax_pos = data.xpos[thorax_id].copy()
    thorax_rot = data.xmat[thorax_id].reshape(3, 3)

    light_world = np.array([
        light_x,
        light_y,
        light_z,
    ])

    # Свет относительно корпуса.
    body = thorax_rot.T @ (
        light_world - thorax_pos
    )

    forward = body[0]
    side = body[1]

    # ========================================================
    # ВИЗУАЛЬНЫЙ НЕЙРОН
    # ========================================================

    # Сильная боковая чувствительность.
    steering = np.tanh(side / 1.2)

    # Свет впереди усиливает реакцию.
    forward_gain = np.clip(
        (forward + 1.0) / 2.5,
        0.5,
        1.0,
    )

    steering *= forward_gain

    # ========================================================
    # СИЛЬНЫЙ TURN ДЛЯ ВСЕГО TRIPOD
    # ========================================================

    TURN = 0.90

    left_signal = 1.0 - TURN * steering
    right_signal = 1.0 + TURN * steering

    # Сохраняем нормальное направление CPG.
    # Но даём сильную разницу между сторонами.

    left_signal = np.clip(
        left_signal,
        0.20,
        1.80,
    )

    right_signal = np.clip(
        right_signal,
        0.20,
        1.80,
    )

    return np.array(
        [left_signal, right_signal],
        dtype=float,
    )


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

    # ←
    # Свет влево.
    elif key == 263:

        light_y += LIGHT_STEP

    # →
    # Свет вправо.
    elif key == 262:

        light_y -= LIGHT_STEP

    # ↑
    # Свет вперёд.
    elif key == 265:

        light_x += LIGHT_STEP

    # ↓
    # Свет назад.
    elif key == 264:

        light_x -= LIGHT_STEP

    # R
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
print("🪰 MUSA — FINAL BRAIN")
print("=" * 60)
print()
print("SPACE     = ходьба")
print("← →       = свет влево / вправо")
print("↑ ↓       = свет вперёд / назад")
print("R         = центр")
print()
print("🧠 HybridTurningController")
print("🟡 visual light sensor")
print("🦿 six-leg CPG")
print()


# ============================================================
# VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():

        # ----------------------------------------------------
        # BRAIN
        # ----------------------------------------------------

        descending_signal = brain()


        # ----------------------------------------------------
        # WALKING
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
                0.0,
                0.0,
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
                0.1,
                1.0,
            ]),
        )

        viewer.user_scn.ngeom = 1


        # ----------------------------------------------------
        # MONITOR
        # ----------------------------------------------------

        print(
            "\r"
            f"WALK={'ON ' if walking else 'OFF'} "
            f"| L={descending_signal[0]:.2f} "
            f"R={descending_signal[1]:.2f} "
            f"| X={light_x:+.1f} "
            f"Y={light_y:+.1f}",
            end="",
            flush=True,
        )

        viewer.sync()

        time.sleep(0.005)
