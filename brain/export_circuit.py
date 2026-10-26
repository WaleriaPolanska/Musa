"""Готовит настоящую цепочку «зрение → нисходящие нейроны» для симуляции.

Берёт из коннектома maleCNS v1.0:
  * веса связей LPLC2 -> нисходящие нейроны, с разбивкой по сторонам тела;
  * скелеты этих нейронов, чтобы рисовать их внутри Музы.

Результат — brain/data/circuit.npz, его читает musa/musa_synapses.py.

    ./run-brain.sh brain/export_circuit.py

Нужен файл весов (502 МБ), если его нет:
    curl -o brain/data/connectome-weights.feather \\
      https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather
"""

from pathlib import Path

import navis
import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as feather

import malecns
from export_for_sim import TO_MUJOCO, VOXEL_MM, segments

SOURCE = "LPLC2"  # зрительные нейроны на приближение объекта
SOURCE_FUNCTION = "looming detector (ACh)"
SOURCE_DETAIL = (
    "детектор приближения; медиатор ацетилхолин, значит все пять связей ниже "
    "возбуждающие"
)

# Нисходящий нейрон -> цвет, короткая подпись в окне и что за ней стоит.
# Подписи латиницей: шрифт MuJoCo кириллицу не рисует.
# Цифры посчитаны по коннектому: ./run-brain.sh brain/what_it_drives.py
TARGETS = {
    "DNp103": (
        (0.30, 0.85, 1.00),
        "wing + front-leg circuits",
        "ведёт к схемам крыла и передних ног: партнёры в T2 60% и T1 35%, "
        "прямых мотонейронов почти нет (11 синапсов)",
    ),
    "DNp01": (
        (1.00, 0.85, 0.20),
        "Giant Fiber: jump, wing muscle",
        "гигантское волокно: 94 синапса прямо на мотонейроны крыла, включая "
        "прыжковую мышцу — та самая реакция бегства",
    ),
    "DNp04": (
        (0.50, 1.00, 0.45),
        "backward escape, T1-T2",
        "бегство задним ходом: почти всё через вставочные нейроны, "
        "партнёры в T2 56% и T1 44%",
    ),
    "DNg40": (
        (1.00, 0.45, 0.85),
        "head and neck muscles (GNG)",
        "72 синапса на мотонейроны головы и шеи (GNG), партнёры в T2 74%; "
        "медиатор глутамат — такой нейрон чаще тормозит цель",
    ),
    "DNp06": (
        (0.95, 0.55, 0.25),
        "takeoff turn: wing, neck, legs",
        "взлёт и уклоняющийся поворот: 179 прямых синапсов на мотонейроны — "
        "крыло 56, голова и шея 47, брюшко 36, средние ноги 26",
    ),
}

SOURCE_COLOR = (1.00, 0.35, 0.15)

WEIGHTS = Path(__file__).parent / "data" / "connectome-weights.feather"
OUT = Path(__file__).parent / "data" / "circuit.npz"

PRUNE_TWIGS = 3000
TARGET_NODES = 120


def weights(df_ann):
    """Сумма синапсов LPLC2 -> цель, отдельно для левой и правой стороны."""

    if not WEIGHTS.exists():
        raise SystemExit(f"нет файла весов: {WEIGHTS}\nкак скачать — в шапке скрипта")

    table = feather.read_table(WEIGHTS, memory_map=True)
    edges = table.filter(pc.equal(table["type_pre"], SOURCE)).to_pandas()

    side = df_ann.set_index("bodyId")["somaSide"]
    edges["side_pre"] = edges["body_pre"].map(side)
    edges["side_post"] = edges["body_post"].map(side)

    # Связи в коннектоме строго на своей стороне, но проверим это на данных.
    same_side = edges["side_pre"] == edges["side_post"]

    result = {}

    for name in TARGETS:

        rows = edges.loc[(edges["type_post"] == name) & same_side]
        by_side = rows.groupby("side_pre")["weight"].sum()

        crossed = edges.loc[(edges["type_post"] == name) & ~same_side, "weight"].sum()

        result[name] = (
            float(by_side.get("L", 0.0)),
            float(by_side.get("R", 0.0)),
            float(crossed),
        )

        print(
            f"{SOURCE} -> {name:8} L: {result[name][0]:7.0f}  "
            f"R: {result[name][1]:7.0f}  крест-накрест: {crossed:.0f}"
        )

    return result


def skeleton(body_id):
    """Скелет нейрона в миллиметрах и в осях MuJoCo, прорежённый."""

    neuron = navis.read_swc(str(malecns.swc_path(int(body_id))))
    small = navis.downsample_neuron(
        navis.prune_twigs(neuron, PRUNE_TWIGS),
        max(1, navis.prune_twigs(neuron, PRUNE_TWIGS).n_nodes // TARGET_NODES),
    )

    head, tail = segments(small)

    return head * VOXEL_MM @ TO_MUJOCO.T, tail * VOXEL_MM @ TO_MUJOCO.T


def main():

    ann = malecns.annotations()

    synapses = weights(ann)

    starts, ends, group, labels, colors, sides, types = [], [], [], [], [], [], []

    wanted = [(SOURCE, SOURCE_COLOR)] + [(name, TARGETS[name][0]) for name in TARGETS]

    for neuron_type, color in wanted:

        rows = ann.loc[ann["type"].fillna("") == neuron_type]

        for body_side in ("L", "R"):

            side_rows = rows.loc[rows["somaSide"] == body_side]

            if side_rows.empty:
                print(f"нет {neuron_type} со стороной {body_side}, пропускаю")
                continue

            body_id = int(side_rows["bodyId"].iloc[0])

            head, tail = skeleton(body_id)

            starts.append(head)
            ends.append(tail)
            group.append(np.full(len(head), len(labels)))

            labels.append(f"{neuron_type}_{body_side} {body_id}")
            colors.append(color)
            sides.append(body_side)
            types.append(neuron_type)

            print(f"скелет {neuron_type}_{body_side} {body_id}: {len(head)} отрезков")

    starts = np.concatenate(starts)
    ends = np.concatenate(ends)

    center = (np.vstack([starts, ends]).min(0) + np.vstack([starts, ends]).max(0)) / 2
    starts -= center
    ends -= center

    np.savez_compressed(
        OUT,
        center=center,
        starts=starts,
        ends=ends,
        group=np.concatenate(group),
        labels=labels,
        colors=np.array(colors),
        sides=sides,
        types=types,
        target_types=list(TARGETS),
        functions=[TARGETS[name][1] for name in TARGETS],
        details=[TARGETS[name][2] for name in TARGETS],
        source_function=SOURCE_FUNCTION,
        source_detail=SOURCE_DETAIL,
        weights_left=[synapses[name][0] for name in TARGETS],
        weights_right=[synapses[name][1] for name in TARGETS],
        source_type=SOURCE,
    )

    print(f"\nвсего {len(starts)} отрезков, {len(labels)} нейронов")
    print("сохранил:", OUT)


if __name__ == "__main__":
    main()
