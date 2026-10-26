import time
import numpy as np
import mujoco
import mujoco.viewer

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import make_locomotion_fly
from flygym.utils.math import Rotation3D


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


# ------------------------------------------------------------
# Свет
# ------------------------------------------------------------

light_x = 0.0
light_y = 0.0
light_z = 1.5


# ------------------------------------------------------------
# Сенсоры
# ------------------------------------------------------------

def sensors():

    dx = light_x
    dy = light_y

    distance = np.sqrt(dx * dx + dy * dy)

    # Насколько свет близко
    intensity = np.exp(-(distance ** 2) / 4.0)

    # Лево / центр / право
    left = np.exp(-((dx + 2.0) ** 2 + dy ** 2) / 4.0)
    center = np.exp(-(dx ** 2 + dy ** 2) / 4.0)
    right = np.exp(-((dx - 2.0) ** 2 + dy ** 2) / 4.0)

    values = np.array([
        left,
        center,
        right,
    ])

    values /= max(values.max(), 1e-8)

    return values, intensity


# ------------------------------------------------------------
# Мозг
# ------------------------------------------------------------

def brain():

    (left, center, right), intensity = sensors()

    left_neuron = left
    right_neuron = right

    turn = right_neuron - left_neuron

    return left, center, right, intensity, turn


# ------------------------------------------------------------
# Клавиатура
# ------------------------------------------------------------

def key_callback(key):

    global light_x, light_y

    if key == 263:
        light_x -= 0.3

    elif key == 262:
        light_x += 0.3

    elif key == 265:
        light_y += 0.3

    elif key == 264:
        light_y -= 0.3

    elif key == 32:
        light_x = 0.0
        light_y = 0.0

    light_x = np.clip(light_x, -10.0, 10.0)
    light_y = np.clip(light_y, -10.0, 10.0)


print()
print("🪰 MUSA — SENSORS + BRAIN")
print("=" * 45)
print()
print("← → ↑ ↓ — двигают свет")
print("SPACE — центр")
print()
print("Муза пока НЕ ДВИГАЕТСЯ.")
print()


with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():

        left, center, right, intensity, turn = brain()

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

        print(
            "\r"
            f"L={left:.2f} "
            f"C={center:.2f} "
            f"R={right:.2f} "
            f"| LIGHT={intensity:.2f} "
            f"| TURN={turn:+.2f}    ",
            end="",
            flush=True,
        )

        viewer.sync()

        time.sleep(0.02)
