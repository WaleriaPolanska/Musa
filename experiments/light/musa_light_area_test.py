import time
import numpy as np
import mujoco
import mujoco.viewer

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import make_locomotion_fly
from flygym.utils.math import Rotation3D


# ============================================================
# MUSA — только как объект сцены
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
# СВЕТ
# ============================================================

light_x = 0.0
light_y = 0.0
light_z = 2.0


def key_callback(key):

    global light_x, light_y

    step = 1.0

    if key == 263:       # ←
        light_x -= step

    elif key == 262:     # →
        light_x += step

    elif key == 265:     # ↑
        light_y += step

    elif key == 264:     # ↓
        light_y -= step

    elif key == 32:      # SPACE
        light_x = 0.0
        light_y = 0.0

    # Очень большая область.
    light_x = np.clip(light_x, -30.0, 30.0)
    light_y = np.clip(light_y, -30.0, 30.0)


print()
print("🟡 MUSA LIGHT AREA TEST")
print("=" * 50)
print()
print("← → ↑ ↓ — перемещение света")
print("SPACE — центр")
print()
print("Диапазон света: -30 ... +30")
print("Моторы Музы НЕ УПРАВЛЯЮТСЯ.")
print()


with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():

        # Никаких команд моторам.
        # Только физика сцены.

        mujoco.mj_step(
            model,
            data,
        )


        # ----------------------------------------------------
        # ЖЁЛТЫЙ ШАР
        # ----------------------------------------------------

        viewer.user_scn.ngeom = 0

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([
                0.5,
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


        print(
            f"\r🟡 X={light_x:+.1f} "
            f"Y={light_y:+.1f}",
            end="",
            flush=True,
        )

        viewer.sync()

        time.sleep(0.01)
