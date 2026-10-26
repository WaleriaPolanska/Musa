import time
import numpy as np
import mujoco
import mujoco.viewer

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import make_locomotion_fly
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
    [0, 0, 0.5],
    Rotation3D("quat", [1, 0, 0, 0]),
)

sim = Simulation(world)

model = sim.mj_model
data = sim.mj_data


# ============================================================
# LIGHT
# ============================================================

light_x = 0.0
light_y = 0.0
light_z = 1.5


# ============================================================
# SENSORS
# ============================================================

def sensors():

    x = light_x

    left = np.exp(-((x + 2.0) ** 2) / 4.0)
    center = np.exp(-(x ** 2) / 4.0)
    right = np.exp(-((x - 2.0) ** 2) / 4.0)

    values = np.array([
        left,
        center,
        right,
    ])

    values /= max(values.max(), 1e-8)

    return values


# ============================================================
# BRAIN
# ============================================================

def brain():

    left, center, right = sensors()

    left_neuron = left
    right_neuron = right

    turn = right_neuron - left_neuron

    return float(np.clip(turn, -1.0, 1.0))


# ============================================================
# KEYBOARD
# ============================================================

def key_callback(key):

    global light_x, light_y

    if key == 263:
        light_x -= 0.5

    elif key == 262:
        light_x += 0.5

    elif key == 265:
        light_y += 0.5

    elif key == 264:
        light_y -= 0.5

    elif key == 32:
        light_x = 0.0
        light_y = 0.0

    light_x = np.clip(light_x, -10.0, 10.0)
    light_y = np.clip(light_y, -10.0, 10.0)


# ============================================================
# START
# ============================================================

print()
print("🪰 MUSA — PURE BRAIN REACTION TEST")
print("=" * 50)
print()
print("← → ↑ ↓  двигают шар")
print("SPACE    центр")
print()
print("CPG ОТКЛЮЧЕН.")
print("Муза НЕ ДОЛЖНА УХОДИТЬ.")
print()
print("TURN управляет только маленькой реакцией суставов.")
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

        turn = brain()


        # ----------------------------------------------------
        # MOTOR SIGNAL
        # ----------------------------------------------------

        # Маленькая команда.
        motor = 0.15 * turn


        # ----------------------------------------------------
        # DIRECT MOTOR TEST
        # ----------------------------------------------------

        # Берём первые доступные суставы модели.
        #
        # Важно:
        # это НЕ CPG.
        # Мы просто проверяем, что изменение
        # control/qpos действительно видно телу.

        if model.nu > 0:

            n = min(
                6,
                model.nu,
            )

            for i in range(n):

                sign = 1.0 if i % 2 == 0 else -1.0

                data.ctrl[i] = sign * motor


        # ----------------------------------------------------
        # PHYSICS
        # ----------------------------------------------------

        mujoco.mj_step(
            model,
            data,
        )


        # ----------------------------------------------------
        # DRAW LIGHT
        # ----------------------------------------------------

        viewer.user_scn.ngeom = 0

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([
                0.35,
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

        left, center, right = sensors()

        print(
            "\r"
            f"L={left:.2f} "
            f"C={center:.2f} "
            f"R={right:.2f} "
            f"| TURN={turn:+.2f} "
            f"| MOTOR={motor:+.3f}",
            end="",
            flush=True,
        )

        viewer.sync()

        time.sleep(0.01)
