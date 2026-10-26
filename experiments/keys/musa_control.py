import time
import threading

import mujoco.viewer
from pynput import keyboard

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


running = True
paused = False
dance = False


def on_press(key):
    global running, paused, dance

    try:
        if key.char == "w":
            paused = False
            dance = False
            print("W: ходьба")

        elif key.char == "t":
            paused = False
            dance = not dance
            print("T:", "ТАНЕЦ" if dance else "обычная ходьба")

    except AttributeError:
        if key == keyboard.Key.space:
            paused = not paused
            print("PAUSE:", paused)

        elif key == keyboard.Key.esc:
            running = False
            return False


listener = keyboard.Listener(on_press=on_press)
listener.start()


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
print("🪰 МУЗА")
print("W     — ходьба")
print("Space — пауза")
print("T     — танец")
print("Esc   — выход")
print()


with mujoco.viewer.launch_passive(
    sim.mj_model,
    sim.mj_data,
) as viewer:

    while viewer.is_running() and running:

        if not paused:
            action = controller.step()

            # В танцевальном режиме немного меняем темп CPG.
            if dance:
                cpg_network.intrinsic_freqs[:] = 2.0
            else:
                cpg_network.intrinsic_freqs[:] = 1.0

            apply_locomotion_action(
                sim,
                fly.name,
                action,
            )

            sim.step()

        viewer.sync()
        time.sleep(sim.mj_model.opt.timestep)

listener.stop()
print("Муза остановлена.")
