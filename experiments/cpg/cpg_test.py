# --- Google Colab setup ---
# If you opened this notebook in Google Colab, run this cell first to
# install FlyGym and enable headless (EGL) rendering. It is a no-op when
# the notebook is run anywhere else.
import os
import sys

if "google.colab" in sys.modules:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from flygym import Simulation
from flygym.anatomy import BodySegment
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
from pathlib import Path

output_dir = Path("demo_output/cpg_controller")
output_dir.mkdir(parents=True, exist_ok=True)

fly = make_locomotion_fly(
    name="cpg_demo",
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
spawn_pos = [0, 0, 0.5]  # FlyGym v1 default spawn z
spawn_rot = Rotation3D("quat", [1, 0, 0, 0])
world.add_fly(fly, spawn_pos, spawn_rot)

sim = Simulation(world)
renderer = sim.set_renderer([body_cam])

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

print("Actuated DoFs:", len(dof_order))
print("Simulation timestep:", sim.timestep)
print(
    "CPG intrinsic frequency (network, Hz):",
    float(cpg_network.intrinsic_freqs[0]),
)
print(
    "Kinematic spline cycle rate (Hz, informational):",
    round(preprogrammed_steps.step_cycle_frequency_hz, 4),
)
print("CPG phases:", np.round(cpg_network.curr_phases, 3))

sim.reset()

initial_action = LocomotionAction(
    joint_angles=preprogrammed_steps.default_pose_by_dof_order(dof_order),
    adhesion_onoff=np.ones(6, dtype=bool),
)
apply_locomotion_action(sim, fly.name, initial_action)
sim.warmup()

run_time = 2.0
nsteps_sim = int(run_time / sim.timestep)
time_grid = np.arange(nsteps_sim) * sim.timestep

thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
thorax_positions = np.full((nsteps_sim, 3), np.nan, dtype=np.float32)
cpg_phases = np.full((nsteps_sim, 6), np.nan, dtype=np.float32)
adhesion_onoff = np.zeros((nsteps_sim, 6), dtype=bool)

for step_idx in trange(nsteps_sim, desc="Running CPG"):
    action = controller.step()
    apply_locomotion_action(sim, fly.name, action)
    sim.step_with_profile()

    thorax_positions[step_idx] = sim.get_body_positions(fly.name)[thorax_idx]
    cpg_phases[step_idx] = controller.cpg_network.curr_phases % (2 * np.pi)
    adhesion_onoff[step_idx] = action.adhesion_onoff

    sim.render_as_needed_with_profile()

print(
    "Rendered frames (body_cam):",
    len(renderer.frames[body_cam.name]),
)
print(
    "Thorax forward displacement (first to last sample, sim):",
    f"{thorax_positions[-1, 0] - thorax_positions[0, 0]:.2f} mm",
)
sim.print_performance_report()

sim.renderer.show_in_notebook()
sim.renderer.save_video(output_dir / "cpg_controller_mixed_terrain.mp4")

fig, axes = plt.subplots(2, 1, figsize=(7, 4), tight_layout=True, sharex=True)

axes[0].plot(time_grid, thorax_positions[:, 0] - thorax_positions[0, 0])
axes[0].set_ylabel("Forward\nposition (mm)")
axes[0].set_title("Thorax displacement")

axes[1].imshow(
    adhesion_onoff.T,
    aspect="auto",
    interpolation="nearest",
    extent=(time_grid[0], time_grid[-1], 5.5, -0.5),
    cmap="gray",
)
axes[1].set_yticks(np.arange(6))
axes[1].set_yticklabels([leg.upper() for leg in preprogrammed_steps.legs])
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Leg")
axes[1].set_title("Adhesion state")

fig.savefig(output_dir / "cpg_controller_diagnostics.png")

input("Нажми Enter для выхода...")
