"""Оболочка мозга из коннектома — меш, приклеенный к груди Музы.

Тот же слой brain-shell, что виден в неуроглансере male-cns v1.0.
Рисует его сам MuJoCo, поэтому на каждый кадр это ничего не стоит.
Готовит файл ./run-brain.sh brain/export_shell.py
"""

from pathlib import Path

import mujoco
import numpy as np

import neurons3d

OBJ = Path(__file__).resolve().parent.parent / "brain" / "data" / "brain_shell.obj"

GROUP = 3  # отдельная группа geom, чтобы включать и выключать одной кнопкой
RGBA = (0.40, 0.80, 1.0, 0.60)


def attach(spec, body_name="musa/c_thorax"):
    """Добавляет оболочку в модель. Вызывать до Simulation(world)."""

    if not OBJ.exists():
        print("оболочки мозга нет, запусти ./run-brain.sh brain/export_shell.py")
        return False

    mesh = spec.add_mesh()
    mesh.name = "brain_shell"
    mesh.file = str(OBJ)

    geom = spec.body(body_name).add_geom()
    geom.name = "brain_shell"
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = "brain_shell"

    # Только картинка: без столкновений и без массы.
    geom.contype = 0
    geom.conaffinity = 0
    geom.density = 0.0
    geom.group = GROUP
    geom.rgba = np.array(RGBA)
    geom.pos = np.asarray(neurons3d.OFFSET)

    return True
