"""Муза целиком: тело, её мозг с настоящими нейронами и управление.

Тело — модель NeuroMechFly (flygym 2.1), ходьба на свет.
Мозг — скелеты нейронов из коннектома maleCNS v1.0, привязаны к груди
и едут вместе с Музой. Привязка приблизительная: официальной сшивки
между ЭМ-объёмом коннектома и моделью flygym не существует.

Запуск:  ./run.sh
"""

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import (
    HybridTurningController,
    HybridControllerObservation,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)
from flygym.utils.math import Rotation3D

import neurons3d


# ============================================================
# НАСТРОЙКИ
# ============================================================

FPS = 60  # частота обновления картинки

LIGHT_START = np.array([2.0, 0.0, 1.5])  # мм
LIGHT_STEP = 1.0
LIGHT_LIMIT = 10.0

TURN = 0.90  # сила поворота


# ============================================================
# MUSA
# ============================================================

if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
    sys.exit(
        "На macOS интерактивный вьювер работает только под mjpython.\n"
        "Запускай так:  ./run.sh"
    )

fly = make_locomotion_fly(
    name="musa",
    add_adhesion=True,
    colorize=True,
)

world = FlatGroundWorld()

world.add_fly(
    fly,
    [0, 0, 0.8],
    Rotation3D("quat", [1, 0, 0, 0]),
)

sim = Simulation(world)

model = sim.mj_model
data = sim.mj_data

thorax_id = model.body("musa/c_thorax").id


# ============================================================
# CONTROLLER
# ============================================================

controller = HybridTurningController(
    timestep=sim.timestep,
    preprogrammed_steps=PreprogrammedSteps(),
    output_dof_order=fly.get_actuated_jointdofs_order("position"),
    enable_adhesion=True,
)

controller.reset(seed=0)
sim.reset()


# ============================================================
# МОЗГ: НАСТОЯЩИЕ НЕЙРОНЫ
# ============================================================

starts, ends, rgba, labels = neurons3d.load()


# ============================================================
# STATE
# ============================================================

light = LIGHT_START.copy()
walking = False
show_brain = True


# ============================================================
# ЗРИТЕЛЬНЫЙ НЕЙРОН
# ============================================================

def brain():

    thorax_pos = data.xpos[thorax_id].copy()
    thorax_rot = data.xmat[thorax_id].reshape(3, 3)

    # Свет из мировых координат в систему координат корпуса.
    body = thorax_rot.T @ (light - thorax_pos)

    forward = body[0]
    side = body[1]

    # Сильная боковая чувствительность.
    steering = np.tanh(side / 1.2)

    # Свет впереди усиливает реакцию.
    forward_gain = np.clip((forward + 1.0) / 2.5, 0.5, 1.0)

    steering *= forward_gain

    left_signal = np.clip(1.0 - TURN * steering, 0.20, 1.80)
    right_signal = np.clip(1.0 + TURN * steering, 0.20, 1.80)

    return np.array([left_signal, right_signal], dtype=float)


# ============================================================
# KEYBOARD
# ============================================================

def key_callback(key):

    global walking, show_brain

    # SPACE — ходьба
    if key == 32:
        walking = not walking

    # W / ↑ — свет вперёд
    elif key in (87, 265):
        light[0] += LIGHT_STEP

    # S / ↓ — свет назад
    elif key in (83, 264):
        light[0] -= LIGHT_STEP

    # A / ← — свет влево
    elif key in (65, 263):
        light[1] += LIGHT_STEP

    # D / → — свет вправо
    elif key in (68, 262):
        light[1] -= LIGHT_STEP

    # B — мозг показать / спрятать
    elif key == 66:
        show_brain = not show_brain
        # Сквозь прозрачное тело видно нейроны.
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = show_brain

    # R — свет в начальную точку
    elif key == 82:
        light[:] = LIGHT_START

    # X — поднять Музу и начать заново
    elif key == 88:
        controller.reset(seed=0)
        sim.reset()
        walking = False

    np.clip(light[:2], -LIGHT_LIMIT, LIGHT_LIMIT, out=light[:2])


# ============================================================
# START
# ============================================================

print()
print("🪰 МУЗА — тело и мозг")
print("=" * 50)
print("SPACE      ходьба вкл/выкл")
print("W A S D    свет вперёд / влево / назад / вправо")
print("↑ ← ↓ →    то же самое")
print("B          мозг показать / спрятать")
print("R          свет в начало")
print("X          поднять Музу заново")
print("=" * 50)
print(f"нейронов в голове: {len(labels)}, отрезков {len(starts)}")
for label in labels:
    print("  ", label)
print()


with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = show_brain

    wall_mark = time.perf_counter()
    sim_mark = data.time
    speed = 0.0

    while viewer.is_running():

        # ----------------------------------------------------
        # ФИЗИКА: столько шагов, сколько успеваем за кадр
        # ----------------------------------------------------

        frame_deadline = time.perf_counter() + 1.0 / FPS

        while time.perf_counter() < frame_deadline:

            if walking:

                obs = HybridControllerObservation.from_sim(sim, fly.name)
                action = controller.step(brain(), obs)
                apply_locomotion_action(sim, fly.name, action)

            sim.step()

        # ----------------------------------------------------
        # СВЕТ
        # ----------------------------------------------------

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.30, 0.0, 0.0]),
            pos=light,
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 1.0, 0.1, 1.0]),
        )

        ngeom = 1

        # ----------------------------------------------------
        # МОЗГ
        # ----------------------------------------------------

        if show_brain:

            ngeom = neurons3d.draw(
                viewer.user_scn,
                starts,
                ends,
                rgba,
                data.xpos[thorax_id].copy(),
                data.xmat[thorax_id].reshape(3, 3),
                first=ngeom,
            )

        viewer.user_scn.ngeom = ngeom

        # ----------------------------------------------------
        # МОНИТОР
        # ----------------------------------------------------

        now = time.perf_counter()

        if now - wall_mark > 0.5:
            speed = (data.time - sim_mark) / (now - wall_mark)
            wall_mark, sim_mark = now, data.time

        signal = brain()

        viewer.set_texts(
            (
                None,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                "ходьба\nмозг\nсвет X\nсвет Y\nмозг L / R\nскорость",
                f"{'ON' if walking else 'OFF'}\n"
                f"{'ON' if show_brain else 'OFF'}\n"
                f"{light[0]:+.1f}\n"
                f"{light[1]:+.1f}\n"
                f"{signal[0]:.2f} / {signal[1]:.2f}\n"
                f"{speed:.2f}x",
            )
        )

        print(
            f"\rWALK={'ON ' if walking else 'OFF'} "
            f"| BRAIN={'ON ' if show_brain else 'OFF'} "
            f"| L={signal[0]:.2f} R={signal[1]:.2f} "
            f"| X={light[0]:+.1f} Y={light[1]:+.1f} "
            f"| {speed:.2f}x realtime",
            end="",
            flush=True,
        )

        viewer.sync()

print()
