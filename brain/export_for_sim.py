"""Готовит нейроны maleCNS для показа внутри симуляции Музы.

Скелеты прорежаются, переводятся в миллиметры и в систему координат MuJoCo,
результат — один npz, который читает musa/musa_full.py (там своё окружение,
navis туда не ставится).

    ./run-brain.sh brain/export_for_sim.py
"""

from pathlib import Path

import navis
import numpy as np

import malecns

# тип нейрона -> (цвет RGB, сколько штук взять)
SETS = {
    "LPLC2": ((1.00, 0.35, 0.15), 2),  # зрительные: реагируют на приближение
    "DNa02": ((0.25, 0.80, 1.00), 2),  # нисходящие: рулят поворотом
    "DNp01": ((1.00, 0.85, 0.20), 2),  # гигантское волокно: реакция бегства
}

VOXEL_MM = 8e-6  # вокселы коннектома -> миллиметры

# Коннектом: x — поперёк тела, y — вниз, z — назад.
# MuJoCo у flygym: x — вперёд, y — влево, z — вверх.
TO_MUJOCO = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)

PRUNE_TWIGS = 1250  # выкидываем веточки короче 10 мкм
TARGET_NODES = 300  # столько узлов оставляем от нейрона

OUT = Path(__file__).parent / "data" / "neurons_for_sim.npz"


def segments(neuron):
    """Отрезки скелета: массивы начал и концов по рёбрам дерева."""

    nodes = neuron.nodes.set_index("node_id")
    xyz = nodes[["x", "y", "z"]].to_numpy(dtype=float)

    has_parent = nodes["parent_id"].to_numpy() >= 0
    parent_rows = nodes.index.get_indexer(nodes["parent_id"].to_numpy()[has_parent])

    return xyz[has_parent], xyz[parent_rows]


def main():

    df = malecns.annotations()

    starts, ends, colors, labels = [], [], [], []

    for neuron_type, (color, count) in SETS.items():

        rows = df.loc[df["type"].fillna("") == neuron_type].head(count)

        if rows.empty:
            raise SystemExit(f"тип {neuron_type} не найден в аннотациях")

        for body_id in rows["bodyId"]:

            neuron = navis.read_swc(str(malecns.swc_path(int(body_id))))

            pruned = navis.prune_twigs(neuron, PRUNE_TWIGS)
            factor = max(1, pruned.n_nodes // TARGET_NODES)
            small = navis.downsample_neuron(pruned, factor)

            head, tail = segments(small)

            starts.append(head)
            ends.append(tail)
            colors.append(np.tile(color, (len(head), 1)))
            labels.append(f"{neuron_type} {body_id}")

            print(f"{neuron_type:7} {body_id:>9}  {neuron.n_nodes:5} -> {len(head)} отрезков")

    starts = np.concatenate(starts) * VOXEL_MM @ TO_MUJOCO.T
    ends = np.concatenate(ends) * VOXEL_MM @ TO_MUJOCO.T
    colors = np.concatenate(colors)

    # Центрируем на середине облака: в симуляции привяжем к голове Музы.
    center = (np.vstack([starts, ends]).min(0) + np.vstack([starts, ends]).max(0)) / 2
    starts -= center
    ends -= center

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, starts=starts, ends=ends, colors=colors, labels=labels)

    size = np.vstack([starts, ends])
    print(f"\nвсего {len(starts)} отрезков")
    print(f"габариты мозга: {(size.max(0) - size.min(0)).round(3)} мм")
    print("сохранил:", OUT)


if __name__ == "__main__":
    main()
