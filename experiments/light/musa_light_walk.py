import time
import numpy as np
import mujoco
import mujoco.viewer

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import (
    CPGController,
    LocomotionAction,
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
    [0, 0, 0.5],
    Rotation3D("quat", [1, 0, 0, 0]),
)

sim = Simulation(world)

model = sim.mj_model
data = sim.mj_data


# ============================================================
# CPG
# ============================================================

preprogrammed_steps = PreprogrammedSteps()

dof_order = fly.get_actuated_jointdofs_order("position")

cpg_network = make_tripod_cpg_network(
    timestep=sim.timestep,
    intrinsic_amplitude=1.0,
    coupling_strength=10.0,
    convergence_coef=20.0,
    seed=0,
)

controller = CPGController(
    cpg_network=cpg_network,
    preprogrammed_steps=preprogrammed_steps,
    output_dof_order=dof_order,
)


# ============================================================
# LIGHT
# ============================================================

light_x = 0.0
light_y = -1.0
light_z = 1.5


# ============================================================
# SENSORS
# ============================================================

def get_sensors():

    x = light_x

    left = np.exp(-((x + 1.0) ** 2) / 0.5)
    center = np.exp(-(x ** 2) / 0.5)
    right = np.exp(-((x - 1.0) ** 2) / 0.5)

    values = np.array([left, center, right])

    return values / max(values.max(), 1e-8)


def neural_turn():

    left, center, right = get_sensors()

    return float(
        np.clip(
            right - left,
            -1.0,
            1.0,
        )
    )


# ============================================================
# LIGHT CONTROL
# ============================================================

def move_light(dx, dy):

    global light_x, light_y

    light_x += dx
    light_y += dy

    light_x = np.clip(light_x, -10.0, 10.0)
    light_y = np.clip(light_y, -10.0, 10.0)



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
        global light_x, light_y
        light_x = 0.0
        light_y = -1.0


# ============================================================
# START
# ============================================================

print()
print("🪰 MUSA — LIGHT → BRAIN → CPG")
print("=" * 45)
print()
print("← →  двигают свет")
print("↑ ↓  двигают свет по глубине")
print("SPACE  центр света")
print()
print("Теперь CPG управляет ногами.")
print("Свет даёт небольшой сигнал поворота.")
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
        # 1. Мозг получает свет
        # ----------------------------------------------------

        turn = neural_turn()


        # ----------------------------------------------------
        # 2. CPG генерирует ходьбу
        # ----------------------------------------------------

        action = controller.step()

        angles = np.array(
            action.joint_angles,
            copy=True,
        )


        # ----------------------------------------------------
        # 3. Очень слабая реакция на свет
        # ----------------------------------------------------

        # Симметричная модуляция.
        # Пока коэффициент специально маленький.

        steering = 0.01 * turn

        n = min(
            12,
            len(angles),
        )

        angles[:n] += steering


        # ----------------------------------------------------
        # 4. Создаём новый action
        # ----------------------------------------------------

        action = LocomotionAction(
            joint_angles=angles,
            adhesion_onoff=action.adhesion_onoff,
        )


        # ----------------------------------------------------
        # 5. Передаём действие Музе
        # ----------------------------------------------------

        apply_locomotion_action(
            sim,
            fly.name,
            action,
        )

        sim.step()


        # ----------------------------------------------------
        # 6. Рисуем свет
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # 7. Информация
        # ----------------------------------------------------

        left, center, right = get_sensors()

        print(
            f"\r💡 L={left:.2f} "
            f"C={center:.2f} "
            f"R={right:.2f} "
            f"| 🧠 TURN={turn:+.2f}",
            end="",
            flush=True,
        )

        viewer.sync()

        time.sleep(0.005)
