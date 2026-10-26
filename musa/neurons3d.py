"""Скелеты нейронов maleCNS в сцене MuJoCo.

Данные готовит ./run-brain.sh brain/export_for_sim.py — там своё окружение,
здесь нужен только numpy.
"""

from pathlib import Path

import mujoco
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "brain" / "data" / "neurons_for_sim.npz"

WIDTH = 3.0  # толщина линии в пикселях

# Куда сдвинуть облако нейронов относительно точки привязки, мм.
OFFSET = np.array([0.0, 0.0, 0.10])


def load():
    """starts, ends, rgba, подписи нейронов."""

    if not DATA.exists():
        raise SystemExit(
            "нет данных по нейронам, сначала выполни:\n"
            "  ./run-brain.sh brain/export_for_sim.py"
        )

    data = np.load(DATA)

    colors = data["colors"]
    rgba = np.hstack([colors, np.ones((len(colors), 1))])

    return data["starts"] + OFFSET, data["ends"] + OFFSET, rgba, list(data["labels"])


def draw(scene, starts, ends, rgba, pos, rot, first=0):
    """Кладёт нейроны в сцену, привязав их к телу с позицией pos и поворотом rot."""

    world_starts = pos + starts @ rot.T
    world_ends = pos + ends @ rot.T

    identity = np.eye(3).flatten()
    size = np.array([WIDTH, 0.0, 0.0])

    # Линии рисуются заметно быстрее капсул: их тут три тысячи.
    line = mujoco.mjtGeom.mjGEOM_LINE

    for i in range(len(world_starts)):

        geom = scene.geoms[first + i]

        mujoco.mjv_initGeom(
            geom,
            line,
            size,
            world_starts[i],
            identity,
            rgba[i],
        )

        mujoco.mjv_connector(
            geom,
            line,
            WIDTH,
            world_starts[i],
            world_ends[i],
        )

    return first + len(world_starts)
