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

poses = {}
initial_qpos = data.qpos.copy()


def save_pose(number):
    poses[f"pose_{number}"] = data.qpos.copy().tolist()

    output = {
        name: values
        for name, values in poses.items()
    }

    Path.home().joinpath("musa_poses.json").write_text(
        json.dumps(output, indent=2)
    )

    print(f"✓ pose_{number} сохранена")


def key_callback(keycode):
    key = chr(keycode).lower()

    if key == "1":
        save_pose(1)

    elif key == "2":
        save_pose(2)

    elif key == "r":
        data.qpos[:] = initial_qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        print("↻ Исходная поза")

    elif key == "\x1b":
        print("Выход")


print()
print("🪰 MUSA POSE BUILDER")
print()
print("1 — сохранить pose_1")
print("2 — сохранить pose_2")
print("R — сбросить")
print("Esc — выход")
print()
print("Позы сохраняются в ~/musa_poses.json")


with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

print("Готово.")
