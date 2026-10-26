import time
import numpy as np
import mujoco
import mujoco.viewer

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import (
    CPGController,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
    make_tripod_cpg_network,
)
from flygym.utils.math import Rotation3D


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
# CPG
# ============================================================

steps = PreprogrammedSteps()

dof_order = fly.get_actuated_jointdofs_order("position")

network = make_tripod_cpg_network(
    timestep=sim.timestep,
    intrinsic_amplitude=1.0,
    coupling_strength=10.0,
    convergence_coef=20.0,
    seed=0,
)

controller = CPGController(
    cpg_network=network,
    preprogrammed_steps=steps,
    output_dof_order=dof_order,
)


# ============================================================
# LIGHT
# ============================================================

light_x = 1.0
light_y = 0.0
light_z = 1.5


walking = False


def key_callback(key):

    global walking
    global light_x
    global light_y

    # SPACE — ходьба ON/OFF
    if key == 32:
        walking = not walking

        print(
            "\nCPG:",
            "ON" if walking else "OFF"
        )

    # СВЕТ
    elif key == 263:
        light_x -= 0.25

    elif key == 262:
        light_x += 0.25

    elif key == 265:
        light_y += 0.25

    elif key == 264:
        light_y -= 0.25

    elif key == 257:
        light_x = 1.0
        light_y = 0.0


print()
print("🪰 MUSA — CPG STOP TEST")
print("=" * 50)
print()
print("При запуске CPG ВЫКЛЮЧЕН.")
print()
print("SPACE — включить/выключить CPG")
print("← → ↑ ↓ — двигать шар")
print()
print("Сначала проверь Музу в состоянии STOP.")
print()


with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():

        if walking:

            action = controller.step()

            apply_locomotion_action(
                sim,
                fly.name,
                action,
            )

        sim.step()


        # ====================================================
        # LIGHT
        # ====================================================

        viewer.user_scn.ngeom = 0

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([
                0.30,
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


        viewer.sync()

        time.sleep(0.005)
