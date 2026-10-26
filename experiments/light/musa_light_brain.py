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
light_y = -1.0
light_z = 1.5


# ============================================================
# SENSORS
# ============================================================

def get_sensors(x):

    left = np.exp(-((x + 1.0) ** 2) / 0.5)
    center = np.exp(-(x ** 2) / 0.5)
    right = np.exp(-((x - 1.0) ** 2) / 0.5)

    values = np.array([
        left,
        center,
        right,
    ])

    values /= max(values.max(), 1e-8)

    return values


# ============================================================
# SIMPLE NEURAL LAYER
# ============================================================

def neural_turn_signal():

    left, center, right = get_sensors(light_x)

    left_neuron = left
    right_neuron = right

    turn = right_neuron - left_neuron

    return float(np.clip(turn, -1.0, 1.0))


# ============================================================
# TERMINAL DISPLAY
# ============================================================

def show_brain():

    left, center, right = get_sensors(light_x)

    turn = neural_turn_signal()

    if turn < -0.15:
        behavior = "TURN LEFT  ←"

    elif turn > 0.15:
        behavior = "TURN RIGHT →"

    else:
        behavior = "CENTER     ↑"

    print(
        "\r"
        f"L={left:.2f}  "
        f"C={center:.2f}  "
        f"R={right:.2f}  "
        f"| TURN={turn:+.2f}  "
        f"| {behavior}       ",
        end="",
        flush=True,
    )


# ============================================================
# MOVE LIGHT
# ============================================================

def move_light(dx, dy):

    global light_x, light_y

    light_x += dx
    light_y += dy

    light_x = np.clip(
        light_x,
        -3.0,
        3.0,
    )

    light_y = np.clip(
        light_y,
        -3.0,
        2.0,
    )

    show_brain()


def reset_light():

    global light_x, light_y

    light_x = 0.0
    light_y = -1.0

    show_brain()


# ============================================================
# KEYBOARD
# ============================================================

def key_callback(key):

    if key == 263:
        move_light(-0.20, 0)

    elif key == 262:
        move_light(+0.20, 0)

    elif key == 265:
        move_light(0, +0.20)

    elif key == 264:
        move_light(0, -0.20)

    elif key == 32:
        reset_light()


# ============================================================
# START
# ============================================================

print()
print("🪰 MUSA — LIGHT BRAIN")
print("=" * 45)
print()
print("←  свет влево")
print("→  свет вправо")
print("↑  свет дальше")
print("↓  свет ближе")
print("SPACE  центр")
print()
print("Муза пока неподвижна.")
print("Сейчас тестируем только сенсоры и нейрон.")
print()

show_brain()


# ============================================================
# VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():

        # --------------------------------------------
        # DRAW LIGHT
        # --------------------------------------------

        viewer.user_scn.ngeom = 0

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([
                0.20,
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

        # --------------------------------------------
        # UPDATE VIEW
        # --------------------------------------------

        viewer.sync()

        time.sleep(0.01)
