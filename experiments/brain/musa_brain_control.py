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


# Сенсорные входы Музы
inputs = {
    "food": 0.0,
    "light": 0.0,
    "danger": 0.0,
    "sound": 0.0,
    "touch": 0.0,
    "obstacle": 0.0,
}

mode = "WALK"


def print_brain():
    print("\n🧠 MUSA BRAIN")
    print("=" * 40)
    print("MODE:", mode)

    for name, value in inputs.items():
        bar = "█" * int(value * 10)
        print(f"{name:10s} {bar}")


def stimulate(name):
    inputs[name] = 1.0
    print(f"\n⚡ STIMULUS: {name.upper()}")
    print_brain()


def clear_inputs():
    for name in inputs:
        inputs[name] = 0.0

    print("\n🧹 INPUTS CLEARED")
    print_brain()


def key_callback(keycode):
    global mode

    # GLFW key codes
    keys = {
        49: "food",       # 1
        50: "light",      # 2
        51: "danger",     # 3
        52: "sound",      # 4
        53: "touch",      # 5
        54: "obstacle",   # 6
    }

    if keycode in keys:
        stimulate(keys[keycode])

    elif keycode == 55:
        mode = "REAR_LEGS"
        print("\n🦵 MODE: REAR_LEGS")
        print_brain()

    elif keycode == 56:
        mode = "BREAKDANCE"
        print("\n🕺 MODE: BREAKDANCE")
        print_brain()

    elif keycode == 48:
        clear_inputs()

    elif keycode == 27:
        print("\nMUSA STOP")


print()
print("🪰 MUSA — BRAIN CONTROL")
print("=" * 40)
print()
print("1  FOOD")
print("2  LIGHT")
print("3  DANGER")
print("4  SOUND")
print("5  TOUCH")
print("6  OBSTACLE")
print()
print("7  REAR LEGS")
print("8  BREAKDANCE")
print("0  CLEAR")
print("ESC STOP")
print()

print_brain()


with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():
        # Пока физика работает штатно.
        mujoco.mj_step(model, data)
        viewer.sync()
