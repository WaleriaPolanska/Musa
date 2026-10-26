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


light_x = 0.0
light_y = 0.0
light_z = 1.5


def key_callback(key):

    global light_x, light_y

    if key == 263:       # ←
        light_x -= 0.3

    elif key == 262:     # →
        light_x += 0.3

    elif key == 265:     # ↑
        light_y += 0.3

    elif key == 264:     # ↓
        light_y -= 0.3

    elif key == 32:      # SPACE
        light_x = 0.0
        light_y = 0.0

    light_x = np.clip(light_x, -10.0, 10.0)
    light_y = np.clip(light_y, -10.0, 10.0)


print()
print("🟡 LIGHT KEYBOARD")
print()
print("← → ↑ ↓ — движение точки")
print("SPACE — центр")
print()


with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():

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

        viewer.sync()

        time.sleep(0.01)
