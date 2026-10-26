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


print()
print("🪰 MUSA — TEST MOVING LIGHT")
print("Жёлтая точка должна двигаться слева направо.")
print()


with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    t = 0.0

    while viewer.is_running():

        x = 2.0 * np.sin(t)

        # Очищаем пользовательские объекты.
        viewer.user_scn.ngeom = 0

        # Создаём движущуюся жёлтую точку.
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.20, 0, 0]),
            pos=np.array([x, -1.0, 1.5]),
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 1.0, 0.0, 1.0]),
        )

        viewer.user_scn.ngeom = 1

        print(
            f"\r💡 X = {x:+.2f}",
            end="",
            flush=True,
        )

        viewer.sync()

        t += 0.03
        time.sleep(0.03)
