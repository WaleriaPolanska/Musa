import numpy as np
from tqdm import trange

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

# --- МУЗА ---
fly = make_locomotion_fly(
    name="musa",
    add_adhesion=True,
    colorize=True,
)

body_cam = fly.add_tracking_camera(
    name="body_cam",
    pos_offset=(-0.5, -7.5, 0.0),
    rotation=Rotation3D("euler", (1.57, 0.0, 0.0)),
    fovy=30.0,
)

world = FlatGroundWorld()
world.add_fly(
    fly,
    [0, 0, 0.5],
    Rotation3D("quat", [1, 0, 0, 0]),
)

sim = Simulation(world)
renderer = sim.set_renderer([body_cam])

# --- CPG ---
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

print()
print("🪰 МУЗА запущена")
print("CPG работает.")
print("Сейчас запускаем короткий танцевальный ролик...")

# --- ТАНЕЦ ---
run_time = 5.0
nsteps_sim = int(run_time / sim.timestep)

for step_idx in trange(nsteps_sim, desc="MUSA DANCE"):
    action = controller.step()

    apply_locomotion_action(sim, fly.name, action)
    sim.step()

    sim.render_as_needed_with_profile()

print()
print("💃 Танец закончен.")
print("Закрой программу Ctrl+C.")

from pathlib import Path

output = Path("musa_dance.mp4")
renderer.save_video(output)

print(f"Видео сохранено: {output.resolve()}")
