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
# KEYBOARD
# ============================================================

walking = True


def key_callback(key):

    global walking

    # SPACE = стоп / старт
    if key == 32:
        walking = not walking

        print(
            "\n🪰 ХОДЬБА:",
            "ON" if walking else "OFF"
        )


# ============================================================
# START
# ============================================================

print()
print("🪰 MUSA — ЧИСТЫЙ CPG WALK TEST")
print("=" * 50)
print()
print("CPG управляет ногами.")
print("Никакого data.ctrl вручную.")
print()
print("SPACE — остановить / запустить CPG")
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

        if walking:

            action = controller.step()

            apply_locomotion_action(
                sim,
                fly.name,
                action,
            )

        sim.step()

        viewer.sync()

        time.sleep(0.005)
