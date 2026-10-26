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
light_y = 0.0
light_z = 1.5


# ============================================================
# SENSORS
# ============================================================

def sensors():

    dx = light_x
    dy = light_y

    left = np.exp(
        -((dx + 2.0) ** 2 + dy ** 2) / 4.0
    )

    center = np.exp(
        -(dx ** 2 + dy ** 2) / 4.0
    )

    right = np.exp(
        -((dx - 2.0) ** 2 + dy ** 2) / 4.0
    )

    values = np.array([
        left,
        center,
        right,
    ])

    values /= max(values.max(), 1e-8)

    return values


# ============================================================
# BRAIN
# ============================================================

def brain():

    left, center, right = sensors()

    left_neuron = left
    right_neuron = right

    turn = right_neuron - left_neuron

    return float(np.clip(turn, -1.0, 1.0))


# ============================================================
# LIGHT KEYBOARD
# ============================================================

def key_callback(key):

    global light_x, light_y

    if key == 263:       # LEFT
        light_x -= 0.3

    elif key == 262:     # RIGHT
        light_x += 0.3

    elif key == 265:     # UP
        light_y += 0.3

    elif key == 264:     # DOWN
        light_y -= 0.3

    elif key == 32:      # SPACE
        light_x = 0.0
        light_y = 0.0

    light_x = np.clip(light_x, -10.0, 10.0)
    light_y = np.clip(light_y, -10.0, 10.0)


# ============================================================
# START
# ============================================================

print()
print("🪰 MUSA — BRAIN → CPG → MOTORS")
print("=" * 55)
print()
print("← → ↑ ↓  двигают свет")
print("SPACE    центр")
print()
print("Теперь CPG действительно управляет ногами.")
print("Свет добавляет небольшую команду поворота.")
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
        # 1. СВЕТ → СЕНСОРЫ → МОЗГ
        # ----------------------------------------------------

        turn = brain()


        # ----------------------------------------------------
        # 2. CPG → БАЗОВОЕ ДВИЖЕНИЕ
        # ----------------------------------------------------

        action = controller.step()

        # LocomotionAction frozen,
        # поэтому делаем независимую копию.
        angles = np.array(
            action.joint_angles,
            dtype=float,
            copy=True,
        )


        # ----------------------------------------------------
        # 3. МОЗГ → МОДУЛЯЦИЯ МОТОРОВ
        # ----------------------------------------------------

        # Очень маленькая величина для первого теста.
        steering = 0.04 * turn

        # Левая/правая половина DOF.
        #
        # Здесь мы намеренно воздействуем только
        # на первые суставы, чтобы проверить саму
        # цепочку "нейрон -> мотор".

        if len(angles) >= 6:

            angles[0] += steering
            angles[3] += steering

            angles[1] -= steering
            angles[4] -= steering


        # ----------------------------------------------------
        # 4. СОЗДАЁМ НОВЫЙ ACTION
        # ----------------------------------------------------

        new_action = LocomotionAction(
            joint_angles=angles,
            adhesion_onoff=action.adhesion_onoff,
        )


        # ----------------------------------------------------
        # 5. ACTION → MUSA
        # ----------------------------------------------------

        apply_locomotion_action(
            sim,
            fly.name,
            new_action,
        )

        sim.step()


        # ----------------------------------------------------
        # 6. ЖЁЛТАЯ ТОЧКА
        # ----------------------------------------------------

        viewer.user_scn.ngeom = 0

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([
                0.35,
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


        # ----------------------------------------------------
        # 7. МОНИТОРИНГ
        # ----------------------------------------------------

        left, center, right = sensors()

        print(
            "\r"
            f"L={left:.2f} "
            f"C={center:.2f} "
            f"R={right:.2f} "
            f"| 🧠 TURN={turn:+.2f} "
            f"| MOTOR={steering:+.3f}",
            end="",
            flush=True,
        )

        viewer.sync()

        time.sleep(0.005)
