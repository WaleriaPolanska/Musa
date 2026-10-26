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
    [0, 0, 0.7],
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

    values = np.array([left, center, right])

    values /= max(values.max(), 1e-8)

    return values


# ============================================================
# BRAIN
# ============================================================

def brain():

    left, center, right = sensors()

    turn = right - left

    return float(np.clip(turn, -1.0, 1.0))


# ============================================================
# KEYBOARD
# ============================================================

def key_callback(key):

    global light_x, light_y

    if key == 263:       # LEFT
        light_x -= 0.5

    elif key == 262:     # RIGHT
        light_x += 0.5

    elif key == 265:     # UP
        light_y += 0.5

    elif key == 264:     # DOWN
        light_y -= 0.5

    elif key == 32:      # SPACE
        light_x = 0.0
        light_y = 0.0

    light_x = np.clip(light_x, -10.0, 10.0)
    light_y = np.clip(light_y, -10.0, 10.0)


# ============================================================
# START
# ============================================================

print()
print("🪰 MUSA — BRAIN STAND TEST")
print("=" * 55)
print()
print("CPG: OFF")
print("Муса не должна самостоятельно идти.")
print()
print("← → ↑ ↓  — шар")
print("SPACE    — центр")
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

        turn = brain()

        # ----------------------------------------------------
        # Очень маленькая реакция
        # ----------------------------------------------------

        steering = 0.08 * turn

        # Только coxa-yaw:
        #
        # LF = 0
        # LM = 7
        # LH = 14
        # RF = 21
        # RM = 28
        # RH = 35

        left = [0, 7, 14]
        right = [21, 28, 35]


        for i in left:
            data.ctrl[i] = -steering


        for i in right:
            data.ctrl[i] = steering


        # Остальные моторы НЕ трогаем.
        # Ставим их в текущую позицию,
        # чтобы они не получали случайных команд.

        for i in range(42):
            if i not in left and i not in right:
                data.ctrl[i] = data.qpos[7 + i]


        mujoco.mj_step(
            model,
            data,
        )


        # ----------------------------------------------------
        # LIGHT
        # ----------------------------------------------------

        viewer.user_scn.ngeom = 0

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.35, 0, 0]),
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

        left_s, center_s, right_s = sensors()

        print(
            "\r"
            f"L={left_s:.2f} "
            f"C={center_s:.2f} "
            f"R={right_s:.2f} "
            f"| TURN={turn:+.2f} "
            f"| MOTOR={steering:+.3f}",
            end="",
            flush=True,
        )

        viewer.sync()

        time.sleep(0.01)
