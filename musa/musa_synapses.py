"""Муза, которой рулит настоящая схема связей. Синапсы можно крутить руками.

Цепочка: свет -> LPLC2 (зрительные нейроны) -> нисходящие нейроны -> ноги.
Веса связей — настоящие, из коннектома maleCNS v1.0 (число синапсов).
В коннектоме эти связи строго на своей стороне тела: левый LPLC2 идёт
только к левым нисходящим, правый — только к правым.

Упрощение: настоящих фоторецепторов и оптических долей в симуляции нет,
свет подаётся прямо на LPLC2 слева и справа.

Одно окно на два экрана, чужих горячих клавиш нет:
  * слева 3D Муза — musa/window.py;
  * справа карта мозга, список синапсов с ползунками и кнопки — musa/panel.py.

Запуск:  ./run.sh musa/musa_synapses.py
"""

import time
from pathlib import Path

import glfw
import mujoco
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

import brainshell
import neurons3d
from brainmap import BrainMap
from panel import Panel
from window import Window


# ============================================================
# НАСТРОЙКИ
# ============================================================

FPS = 60

LIGHT_START = np.array([2.0, 0.0, 1.5])  # мм
LIGHT_SPEED = 6.0  # мм в секунду, пока клавиша зажата

TURN = 0.90  # насколько сильно перекос нейронов крутит ноги
SYNAPSE_SCALE = 2000.0  # столько синапсов = насыщение нейрона
GAIN_STEP = 0.25
GAIN_MAX = 3.0

DETAIL = 2  # каждый второй отрезок скелета: в теле разницы не видно, а кадр вдвое легче

CIRCUIT = Path(__file__).resolve().parent.parent / "brain" / "data" / "circuit.npz"


# ============================================================
# MUSA
# ============================================================

if not CIRCUIT.exists():
    raise SystemExit(
        "нет схемы связей, сначала выполни:\n  ./run-brain.sh brain/export_circuit.py"
    )

fly = make_locomotion_fly(name="musa", add_adhesion=True, colorize=True)

world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]))

# Оболочка мозга из коннектома — до компиляции модели.
brainshell.attach(world.mjcf_root)

sim = Simulation(world)

model = sim.mj_model
data = sim.mj_data

thorax_id = model.body("musa/c_thorax").id

controller = HybridTurningController(
    timestep=sim.timestep,
    preprogrammed_steps=PreprogrammedSteps(),
    output_dof_order=fly.get_actuated_jointdofs_order("position"),
    enable_adhesion=True,
)

controller.reset(seed=0)
sim.reset()


# ============================================================
# СХЕМА СВЯЗЕЙ
# ============================================================

circuit = np.load(CIRCUIT)

targets = [str(name) for name in circuit["target_types"]]
functions = [str(work) for work in circuit["functions"]]
details = [str(text) for text in circuit["details"]]

weights = {
    "L": np.asarray(circuit["weights_left"], dtype=float),
    "R": np.asarray(circuit["weights_right"], dtype=float),
}

# Скелеты нейронов для 3D: какому нейрону принадлежит каждый отрезок.
starts = circuit["starts"][::DETAIL] + neurons3d.OFFSET
ends = circuit["ends"][::DETAIL] + neurons3d.OFFSET
group = circuit["group"][::DETAIL]
labels = [str(name) for name in circuit["labels"]]
base_colors = circuit["colors"]
types = [str(t) for t in circuit["types"]]
sides = [str(s) for s in circuit["sides"]]
source_type = str(circuit["source_type"])

rgba = np.ones((len(starts), 4))


# ============================================================
# ОБЩЕЕ СОСТОЯНИЕ ДВУХ ОКОН
# ============================================================

state = {
    "weights": weights,
    # Во сколько раз усилена каждая связь: 1.0 — как в мозге.
    "gains": {"L": np.ones(len(targets)), "R": np.ones(len(targets))},
    "activity": np.zeros(len(labels)),
    "targets": targets,
    "functions": functions,
    "selected": 0,
    "side": None,  # None — обе стороны сразу
    "walking": False,
    "brain": True,
    "follow": True,
}

gains = state["gains"]
activity = state["activity"]

light = LIGHT_START.copy()


# ============================================================
# МОЗГ
# ============================================================

def circuit_step():
    """Свет -> LPLC2 -> нисходящие. Возвращает сигнал на левые и правые ноги."""

    thorax_pos = data.xpos[thorax_id].copy()
    thorax_rot = data.xmat[thorax_id].reshape(3, 3)

    body = thorax_rot.T @ (light - thorax_pos)

    forward = body[0]
    side = body[1]

    # Свет впереди видно лучше, чем сзади.
    forward_gain = np.clip((forward + 1.0) / 2.5, 0.2, 1.0)

    # Зрительный вход: слева и справа.
    eye = {
        "L": float(np.clip(side / 2.0, 0.0, 1.0) * forward_gain),
        "R": float(np.clip(-side / 2.0, 0.0, 1.0) * forward_gain),
    }

    drive = {}

    for body_side in ("L", "R"):

        synapses = weights[body_side] * gains[body_side]
        drive[body_side] = np.tanh(synapses * eye[body_side] / SYNAPSE_SCALE)

    # Какая сторона мозга возбуждена сильнее, туда Муза и поворачивает.
    steering = float(drive["L"].mean() - drive["R"].mean())

    left_signal = np.clip(1.0 - TURN * steering, 0.20, 1.80)
    right_signal = np.clip(1.0 + TURN * steering, 0.20, 1.80)

    # Активность для подсветки нейронов и карты.
    for i, (neuron_type, body_side) in enumerate(zip(types, sides)):

        if neuron_type == source_type:
            activity[i] = eye[body_side]
        else:
            activity[i] = drive[body_side][targets.index(neuron_type)]

    return np.array([left_signal, right_signal], dtype=float), eye


# ============================================================
# ДЕЙСТВИЯ: КНОПКИ В ПАНЕЛИ И КЛАВИШИ ДЕЛАЮТ ОДНО И ТО ЖЕ
# ============================================================

def toggle_walk():
    state["walking"] = not state["walking"]


def reset_weights():
    gains["L"][:] = 1.0
    gains["R"][:] = 1.0


def reset_musa():
    controller.reset(seed=0)
    sim.reset()
    state["walking"] = False


def toggle_brain():
    state["brain"] = not state["brain"]
    window.option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = state["brain"]
    window.option.geomgroup[brainshell.GROUP] = state["brain"]


def change_gain(step):
    """Меняет силу выбранного синапса: одной стороны или сразу обеих."""

    chosen = state["selected"]
    body_sides = ("L", "R") if state["side"] is None else (state["side"],)

    for body_side in body_sides:
        gains[body_side][chosen] = np.clip(
            gains[body_side][chosen] + step, 0.0, GAIN_MAX
        )


actions = {
    "walk": toggle_walk,
    "weights": reset_weights,
    "musa": reset_musa,
    "brain": toggle_brain,
}


# ============================================================
# KEYBOARD
# ============================================================

def key_callback(key):

    # SPACE — ходьба
    if key == 32:
        toggle_walk()

    # 1..N — выбрать связь
    elif 49 <= key < 49 + len(targets):
        state["selected"] = key - 49

    # = и - — сила выбранной связи
    elif key in (61, 45):
        change_gain(GAIN_STEP if key == 61 else -GAIN_STEP)

    # Q / E — перекос выбранной связи влево / вправо
    elif key in (81, 69):
        step = GAIN_STEP if key == 81 else -GAIN_STEP
        chosen = state["selected"]
        gains["L"][chosen] = np.clip(gains["L"][chosen] + step, 0.0, GAIN_MAX)
        gains["R"][chosen] = np.clip(gains["R"][chosen] - step, 0.0, GAIN_MAX)

    # 0 — вернуть всё как в мозге
    elif key == 48:
        reset_weights()

    # B — мозг показать / спрятать
    elif key == 66:
        toggle_brain()

    # F — камера следит за Музой или стоит на месте
    elif key == 70:
        state["follow"] = not state["follow"]

    # R — свет в начало
    elif key == 82:
        light[:] = LIGHT_START

    # X — поднять Музу заново
    elif key == 88:
        reset_musa()


def move_light(seconds):
    """Свет едет, пока клавиша зажата. Границ нет: уводи куда хочешь."""

    step = LIGHT_SPEED * seconds

    if window.key_held(glfw.KEY_W) or window.key_held(glfw.KEY_UP):
        light[0] += step
    if window.key_held(glfw.KEY_S) or window.key_held(glfw.KEY_DOWN):
        light[0] -= step
    if window.key_held(glfw.KEY_A) or window.key_held(glfw.KEY_LEFT):
        light[1] += step
    if window.key_held(glfw.KEY_D) or window.key_held(glfw.KEY_RIGHT):
        light[1] -= step


# ============================================================
# ЧТО КЛАДЁМ В 3D СЦЕНУ
# ============================================================

def fly_geoms(scene, ngeom):
    """Свет и нейроны поверх модели Музы."""

    mujoco.mjv_initGeom(
        scene.geoms[ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.30, 0.0, 0.0]),
        pos=light,
        mat=np.eye(3).flatten(),
        rgba=np.array([1.0, 1.0, 0.1, 1.0]),
    )

    ngeom += 1

    if state["brain"]:

        brightness = 0.25 + 0.75 * np.clip(activity, 0.0, 1.0)
        rgba[:, :3] = base_colors[group] * brightness[group, None]

        ngeom = neurons3d.draw(
            scene,
            starts,
            ends,
            rgba,
            data.xpos[thorax_id].copy(),
            data.xmat[thorax_id].reshape(3, 3),
            first=ngeom,
        )

    return ngeom


# ============================================================
# START
# ============================================================

print()
print("🪰 МУЗА — настоящие синапсы")
print("=" * 58)
print("Слева Муза, справа мозг: карта, список связей и ползунки.")
print()
print("SPACE        ходьба вкл/выкл")
print("W A S D      свет едет, пока держишь клавишу (границ нет)")
print(f"1..{len(targets)}         выбрать связь")
print("= и -        сила выбранной связи")
print("Q / E        перекос связи влево / вправо")
print("0            вернуть веса как в мозге")
print("B            мозг показать / спрятать")
print("F            камера следит за Музой")
print("X            поднять Музу заново")
print("ESC          выйти")
print("мышь         слева камера, справа выбор синапса и ползунки")
print("=" * 58)
print(f"{source_type}: {str(circuit['source_detail'])}")
print()
print("связи и чем управляют:")
for i, name in enumerate(targets):
    print(f"  {i + 1}. {name:7} L {weights['L'][i]:5.0f}  R {weights['R'][i]:5.0f}")
    print(f"     {details[i]}")
print()


window = Window(model, data, "Musa - brain and synapses", key_callback)
window.option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = state["brain"]
window.option.geomgroup[brainshell.GROUP] = state["brain"]

brain_map = BrainMap.from_circuit(circuit, model, data)
panel = Panel(brain_map, state, actions, model, data, gain_max=GAIN_MAX)

window.panel = panel

wall_mark = time.perf_counter()
sim_mark = data.time
speed = 0.0
frame_mark = time.perf_counter()

while window.is_running():

    # --------------------------------------------------------
    # ФИЗИКА: столько шагов, сколько успеваем за кадр
    # --------------------------------------------------------

    frame_deadline = time.perf_counter() + 1.0 / FPS

    while time.perf_counter() < frame_deadline:

        if state["walking"]:

            signal, eye = circuit_step()

            obs = HybridControllerObservation.from_sim(sim, fly.name)
            apply_locomotion_action(sim, fly.name, controller.step(signal, obs))

        sim.step()

    now = time.perf_counter()

    move_light(now - frame_mark)
    frame_mark = now

    signal, eye = circuit_step()

    if now - wall_mark > 0.5:
        speed = (data.time - sim_mark) / (now - wall_mark)
        wall_mark, sim_mark = now, data.time

    fly_rect, panel_rect = window.panes()

    # --------------------------------------------------------
    # ЛЕВЫЙ ЭКРАН: МУЗА
    # --------------------------------------------------------

    if state["follow"]:
        window.look_at(data.xpos[thorax_id])

    window.draw(fly_rect, window.camera, window.option, fly_geoms)

    window.overlay(
        fly_rect,
        "walk\neyes L/R\nlegs L/R\nlight x/y\nspeed",
        f"{'ON' if state['walking'] else 'OFF'}\n"
        f"{eye['L']:.2f} / {eye['R']:.2f}\n"
        f"{signal[0]:.2f} / {signal[1]:.2f}\n"
        f"{light[0]:+.1f} / {light[1]:+.1f}\n"
        f"{speed:.2f}x",
    )

    # --------------------------------------------------------
    # ПРАВЫЙ ЭКРАН: МОЗГ
    # --------------------------------------------------------

    panel.draw(window, panel_rect)

    window.finish()

    chosen = targets[state["selected"]]
    chosen_side = state["side"] or "L+R"

    print(
        f"\rWALK={'ON ' if state['walking'] else 'OFF'} "
        f"| глаза {eye['L']:.2f}/{eye['R']:.2f} "
        f"| ноги {signal[0]:.2f}/{signal[1]:.2f} "
        f"| свет {light[0]:+.1f}/{light[1]:+.1f} "
        f"| связь {chosen} {chosen_side} "
        f"x{gains['L'][state['selected']]:.2f}/{gains['R'][state['selected']]:.2f} "
        f"| {speed:.2f}x",
        end="",
        flush=True,
    )

window.close()
print()
