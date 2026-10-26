import json
from pathlib import Path

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

# Исходное состояние
initial_qpos = data.qpos.copy()

# Запоминаем позу
saved_qpos = data.qpos.copy()

# Какие суставы редактируем
JOINTS = {
    "1": "musa/c_thorax-lh_coxa-pitch",
    "2": "musa/lh_coxa-lh_trochanterfemur-pitch",
    "3": "musa/lh_trochanterfemur-lh_tibia-pitch",

    "4": "musa/c_thorax-rh_coxa-pitch",
    "5": "musa/rh_coxa-rh_trochanterfemur-pitch",
    "6": "musa/rh_trochanterfemur-rh_tibia-pitch",
}

selected = 1
step = 0.08


def joint_qpos_index(name):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )
    return model.jnt_qposadr[jid]


def apply_pose():
    # Полностью останавливаем динамику
    data.qvel[:] = 0
    data.qacc[:] = 0

    # Пересчитываем геометрию
    mujoco.mj_forward(model, data)


def save_pose():
    global saved_qpos

    saved_qpos = data.qpos.copy()

    Path.home().joinpath("musa_rear_pose.json").write_text(
        json.dumps(
            {
                "qpos": saved_qpos.tolist(),
            },
            indent=2,
        )
    )

    print("✓ Поза сохранена: ~/musa_rear_pose.json")


def restore_pose():
    data.qpos[:] = saved_qpos
    apply_pose()
    print("↶ Последняя поза восстановлена")


def reset_pose():
    data.qpos[:] = initial_qpos
    apply_pose()
    print("↻ Исходная поза")


def change_joint(amount):
    global selected

    name = JOINTS[str(selected)]
    idx = joint_qpos_index(name)

    data.qpos[idx] += amount

    # Не даём суставу улететь в бесконечность
    data.qpos[idx] = max(
        -2.5,
        min(2.5, data.qpos[idx]),
    )

    apply_pose()

    print(
        f"{selected}: {name.split('/')[-1]} "
        f"= {data.qpos[idx]:+.3f}"
    )


def key_callback(keycode):
    global selected

    try:
        key = chr(keycode).lower()
    except ValueError:
        return

    if key in "123456":
        selected = int(key)

        name = JOINTS[key]
        idx = joint_qpos_index(name)

        print(
            f"Выбран сустав {selected}: "
            f"{name.split('/')[-1]} "
            f"({data.qpos[idx]:+.3f})"
        )

    elif key == "a":
        change_joint(-step)

    elif key == "d":
        change_joint(+step)

    elif key == "p":
        save_pose()

    elif key == "r":
        restore_pose()

    elif key == "0":
        reset_pose()


print()
print("🪰 MUSA — EDITOR ЗАДНИХ НОГ")
print()
print("1-3  левая задняя нога")
print("4-6  правая задняя нога")
print()
print("A    уменьшить угол")
print("D    увеличить угол")
print()
print("P    сохранить позу")
print("R    восстановить сохранённую")
print("0    исходная поза")
print()
print("Корпус не редактируем.")
print()

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():
        # Не запускаем физическую динамику.
        # Viewer используется как редактор позы.
        viewer.sync()

print("Готово.")
