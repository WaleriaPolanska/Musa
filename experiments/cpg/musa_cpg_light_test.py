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
# СВЕТ
# ============================================================

light_x = 1.0
light_y = 0.0
light_z = 1.5

# Пока небольшая зона вокруг Музы.
LIGHT_LIMIT = 3.0


# ============================================================
# КЛАВИАТУРА
# ============================================================

def key_callback(key):

    global light_x, light_y

    if key == 263:       # ←
        light_x -= 0.25

    elif key == 262:     # →
        light_x += 0.25

    elif key == 265:     # ↑
        light_y += 0.25

    elif key == 264:     # ↓
        light_y -= 0.25

    elif key == 32:      # SPACE
        light_x = 1.0
        light_y = 0.0

    light_x = np.clip(
        light_x,
        -LIGHT_LIMIT,
        LIGHT_LIMIT,
    )

    light_y = np.clip(
        light_y,
        -LIGHT_LIMIT,
        LIGHT_LIMIT,
    )


print()
print("🪰 MUSA — CPG + LIGHT")
print("=" * 50)
print()
print("CPG: ON")
print("Муза ходит самостоятельно.")
print()
print("← → ↑ ↓ — перемещение света")
print("SPACE — вернуть свет")
print()
print("Свет пока НЕ управляет Музой.")
print("Мы только проверяем обе системы одновременно.")
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

        # ----------------------------------------------------
        # CPG — единственный источник управления ногами
        # ----------------------------------------------------

        action = controller.step()

        apply_locomotion_action(
            sim,
            fly.name,
            action,
        )

        sim.step()


        # ----------------------------------------------------
        # ШАР
        # ----------------------------------------------------

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


        print(
            "\r"
            f"LIGHT X={light_x:+.2f} "
            f"Y={light_y:+.2f}",
            end="",
            flush=True,
        )

        viewer.sync()

        time.sleep(0.005)
