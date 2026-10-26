import time
import numpy as np
import mujoco
import mujoco.viewer

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import make_locomotion_fly
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

sim = Simulation(world)

model = sim.mj_model
data = sim.mj_data


# =========================
# LIGHT POSITION
# =========================

light_x = 0.0
light_y = -1.0
light_z = 1.5


# =========================
# SENSORS
# =========================

def get_sensors(x):

    left = np.exp(-((x + 1.0) ** 2) / 0.5)
    center = np.exp(-(x ** 2) / 0.5)
    right = np.exp(-((x - 1.0) ** 2) / 0.5)

    values = np.array([left, center, right])

    values /= max(values.max(), 1e-8)

    return values


def show_sensors():

    left, center, right = get_sensors(light_x)

    if left > right + 0.15:
        behavior = "← LIGHT LEFT"
    elif right > left + 0.15:
        behavior = "LIGHT RIGHT →"
    else:
        behavior = "LIGHT CENTER"

    print(
        "\r"
        f"L={left:.2f}  "
        f"C={center:.2f}  "
        f"R={right:.2f}   "
        f"| {behavior}        ",
        end="",
        flush=True,
    )


# =========================
# KEYBOARD
# =========================

def key_callback(key):

    global light_x, light_y

    if key == 263:       # LEFT
        light_x -= 0.20

    elif key == 262:     # RIGHT
        light_x += 0.20

    elif key == 265:     # UP
        light_y += 0.20

    elif key == 264:     # DOWN
        light_y -= 0.20

    elif key == 32:      # SPACE
        light_x = 0.0
        light_y = -1.0

    light_x = np.clip(light_x, -3.0, 3.0)
    light_y = np.clip(light_y, -3.0, 2.0)

    show_sensors()


# =========================
# VIEWER
# =========================

print()
print("🪰 MUSA — LIGHT BRAIN")
print("======================")
print()
print("← →  свет влево / вправо")
print("↑ ↓  свет ближе / дальше")
print("SPACE  центр")
print()
print("Муза пока НЕ ДВИГАЕТСЯ.")
print("Жёлтая точка — наш свет.")
print()

show_sensors()


with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():

        # --------------------------------
        # Рисуем нашу движущуюся точку
        # --------------------------------

        viewer.user_scn.ngeom = 0

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.20, 0, 0]),
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

        viewer.sync()

        time.sleep(0.01)
