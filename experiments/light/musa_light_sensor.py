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


# =========================
# НАШ СВЕТ
# =========================

light_pos = np.array(
    [0.0, -2.0, 2.5],
    dtype=float,
)

world.mjcf_root.worldbody.add_light(
    name="musa_light",
    pos=light_pos.tolist(),
    dir=[0, 0, -1],
    diffuse=[1, 1, 1],
    specular=[1, 1, 1],
    ambient=[0.01, 0.01, 0.01],
)


# ЯРКАЯ ТОЧКА — ВИДИМЫЙ МАРКЕР СВЕТА
marker = world.mjcf_root.worldbody.add_body(
    name="light_marker",
    pos=light_pos.tolist(),
)

marker.add_geom(
    name="light_marker_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[0.20, 0, 0],
    rgba=[1, 1, 0, 1],
    contype=0,
    conaffinity=0,
)


# =========================
# SIMULATION
# =========================

sim = Simulation(world)

model = sim.mj_model
data = sim.mj_data

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


# =========================
# СЕНСОРЫ
# =========================

def get_sensors():

    x = light_pos[0]

    left = np.exp(
        -((x + 1.0) ** 2) / 0.5
    )

    center = np.exp(
        -(x ** 2) / 0.5
    )

    right = np.exp(
        -((x - 1.0) ** 2) / 0.5
    )

    values = np.array(
        [left, center, right]
    )

    return values / values.max()


def show_sensors():

    left, center, right = get_sensors()

    print(
        "\r"
        f"💡 L={left:.2f}   "
        f"C={center:.2f}   "
        f"R={right:.2f}   ",
        end="",
        flush=True,
    )


# =========================
# ДВИЖЕНИЕ СВЕТА
# =========================

def move_light(dx, dy):

    light_pos[0] = np.clip(
        light_pos[0] + dx,
        -3.0,
        3.0,
    )

    light_pos[1] = np.clip(
        light_pos[1] + dy,
        -3.0,
        1.0,
    )

    model.light_pos[light_id][:] = light_pos
    model.body_pos[marker_id][:] = light_pos

    show_sensors()


def reset_light():

    light_pos[:] = [0.0, -2.0, 2.5]

    model.light_pos[light_id][:] = light_pos
    model.body_pos[marker_id][:] = light_pos

    show_sensors()


# =========================
# KEYBOARD
# =========================

def key_callback(key):

    if key == 263:       # ←
        move_light(-0.25, 0)

    elif key == 262:     # →
        move_light(+0.25, 0)

    elif key == 265:     # ↑
        move_light(0, +0.25)

    elif key == 264:     # ↓
        move_light(0, -0.25)

    elif key == 32:      # SPACE
        reset_light()


# =========================
# START
# =========================

print()
print("🪰 MUSA — LIGHT SENSOR")
print("=" * 35)
print()
print("← →  свет влево / вправо")
print("↑ ↓  свет вперёд / назад")
print("SPACE  центр")
print()
print("Муза пока НЕ ДВИГАЕТСЯ.")
print("Двигаем только свет.")
print()

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

        # Никакого mj_step!
        # Поэтому Муза физически неподвижна.

        viewer.sync()
