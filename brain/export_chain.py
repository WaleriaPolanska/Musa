"""Три этажа мозга: чувство -> связь -> движение. Настоящие цепочки коннектома.

Свет — путь зрительного компаса, которым муха держит курс на ориентир:

    R7/R8 -> MeTu -> TuBu -> ER -> EPG -> PFL3 -> DNa02 -> IN08A -> мышцы ног

Еда — обонятельный путь, которым муха идёт на запах:

    ORN -> PN -> LH -> DNp32 -> IN12B -> мышцы ног

Для каждого перехода считаем матрицу 2x2: сколько синапсов идёт с левой
стороны на левую и правую и наоборот. Заодно берём по одному нейрону
каждого семейства с каждой стороны, чтобы рисовать их в симуляции.

    ./run-brain.sh brain/export_chain.py          обе цепочки
    ./run-brain.sh brain/export_chain.py food     только еду
"""

import sys
from pathlib import Path

import navis
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

import malecns
from export_circuit import WEIGHTS
from export_for_sim import TO_MUJOCO, VOXEL_MM, segments

# Этаж -> что это делает, понятным языком.
FLOORS = {
    "sense": "ЧУВСТВО",
    "link": "СВЯЗЬ",
    "motor": "ДВИЖЕНИЕ",
}

# Семейство нейронов: как его выбрать, цвет и чем оно занято.
LIGHT_FAMILIES = [
    ("R7/R8", ("R7", "R8"), (1.00, 0.85, 0.25), "photoreceptors: catch the light"),
    ("MeTu", ("MeTu",), (1.00, 0.55, 0.20), "medulla: where the light is"),
    ("TuBu", ("TuBu",), (0.95, 0.35, 0.45), "tubercle: bearing to the landmark"),
    ("ER", ("ER",), (0.60, 0.45, 1.00), "ring: feeds the compass"),
    ("EPG", ("EPG",), (0.35, 0.65, 1.00), "compass: current heading"),
    ("PFL3", ("PFL3",), (0.30, 0.90, 0.75), "decision: how much to turn"),
    ("DNa02", ("DNa02",), (0.45, 1.00, 0.40), "steering command down the neck"),
    ("IN08A", ("IN08A",), (0.75, 0.95, 0.35),
     "premotor: 74k synapses onto all six legs"),
    ("legMN", {"superclass": ["vnc_motor"], "subclass": ["fl", "ml", "hl"]},
     (1.00, 0.75, 0.30), "leg motor neurons: the muscles themselves"),
]

LIGHT_FLOORS = ["sense", "sense", "sense", "link", "link", "motor", "motor", "motor"]

LIGHT_DETAILS = [
    "фоторецепторы отдают свет в медуллу: связь строго внутри своего глаза",
    "медулла передаёт в бугорок, тоже только на свою сторону",
    "бугорок заводит сигнал в кольцо эллипсоидного тела, своя сторона",
    "кольцо кормит компас: здесь стороны перемешиваются почти поровну, "
    "это и есть кольцо",
    "компас говорит решающему нейрону, куда смотрит муха; связь в основном своя",
    "решение уходит на рулевой нейрон КРЕСТ-НАКРЕСТ: левый PFL3 командует правым "
    "DNa02. Так в коннектоме, это и есть механизм поворота",
    "рулевой нейрон спускается в нервную цепочку и отдаёт команду премоторным "
    "вставочным IN08A — самой сильной своей цели (321 нейрон)",
    "IN08A сидят прямо на мотонейронах ног: 74 тысячи синапсов — средняя нога "
    "25564, задняя 23735, передняя 19766, крыло 4695. Медиатор глутамат, "
    "то есть на деле они цель скорее тормозят",
]

FOOD_FAMILIES = [
    ("ORN", ("ORN",), (0.95, 0.80, 0.35), "antenna: smells the food"),
    ("PN", {"type_contains": "PN", "type_not_startswith": "ORN"},
     (0.90, 0.50, 0.25), "antennal lobe: which smell it is"),
    ("LH", ("LH",), (0.85, 0.35, 0.55), "lateral horn: is it worth going"),
    ("DNp32", ("DNp32",), (0.45, 0.85, 0.95), "decision: go to the smell"),
    ("IN12B", ("IN12B",), (0.65, 0.95, 0.45),
     "premotor: 30k synapses onto all six legs"),
    ("legMN", {"superclass": ["vnc_motor"], "subclass": ["fl", "ml", "hl"]},
     (1.00, 0.75, 0.30), "leg motor neurons: the muscles themselves"),
]

FOOD_FLOORS = ["sense", "sense", "link", "motor", "motor"]

FOOD_DETAILS = [
    "рецепторы антенны отдают запах в антеннальную долю: связь и на свою "
    "сторону, и на чужую — нюхают обе антенны сразу. Слева и справа их "
    "восстановлено неровно (883 против 1343), поэтому и веса неровные",
    "проекционные нейроны несут запах в боковой рог, почти целиком по своей "
    "стороне",
    "боковой рог решает, стоит ли идти, и говорит нисходящему нейрону",
    "нисходящий спускается в нервную цепочку к премоторным вставочным IN12B "
    "(501 нейрон), причём почти целиком КРЕСТ-НАКРЕСТ",
    "IN12B сидят на мотонейронах ног: 30 тысяч синапсов — средняя нога 9928, "
    "задняя 8443, передняя 6272, крыло 3151. Медиатор ГАМК, то есть они "
    "тормозят: в живой мухе такая связь ногу придерживает, а не дёргает",
]

MEMORY_FAMILIES = [
    ("PN", {"type_contains": "PN", "type_not_startswith": "ORN"},
     (0.90, 0.50, 0.25), "antennal lobe: which smell it is"),
    ("KC", ("KC",), (0.55, 0.85, 1.00), "Kenyon cells: the smell as a code"),
    ("MBON", ("MBON",), (1.00, 0.60, 0.85), "output: approach or avoid"),
    ("DNp32", ("DNp32",), (0.45, 0.85, 0.95), "decision: go to the smell"),
]

MEMORY_FLOORS = ["sense", "link", "motor"]

MEMORY_DETAILS = [
    "запах уходит из антеннальной доли в грибовидное тело, к клеткам Кеньона: "
    "каждый запах зажигает свой набор клеток",
    "КЛЮЧЕВОЙ СИНАПС: клетки Кеньона сидят на выходных нейронах. Именно он "
    "слабеет, когда запах совпал с дофамином — здесь и живёт память",
    "выходные нейроны говорят нисходящему, идти к запаху или прочь",
]

CHAINS = {
    "light": (LIGHT_FAMILIES, LIGHT_FLOORS, LIGHT_DETAILS),
    "food": (FOOD_FAMILIES, FOOD_FLOORS, FOOD_DETAILS),
    "memory": (MEMORY_FAMILIES, MEMORY_FLOORS, MEMORY_DETAILS),
}

# Медиатор -> что он делает с целью. Коннектом даёт только счётчики синапсов,
# знак приходится брать отсюда.
TRANSMITTER_SIGN = {
    "acetylcholine": 1.0,
    "glutamate": -1.0,
    "gaba": -1.0,
    "histamine": -1.0,
    "dopamine": 0.0,
    "octopamine": 0.0,
    "serotonin": 0.0,
}

NT_FILE = Path(__file__).parent / "data" / "body-neurotransmitters.feather"

PRUNE_TWIGS = 3000
TARGET_NODES = 120

OUT = Path(__file__).parent / "data" / "chain_{name}.npz"


def side_lookup(annotations):
    """Сторона тела: где сома, а у рецепторов — из какого они глаза или антенны.

    У фоторецепторов и обонятельных рецепторов somaSide пустой, зато есть
    rootSide (он же суффикс в instance: ORN_VA1v_R). Это точная сторона, а не
    догадка по целям — рецепторы шлют синапсы на обе стороны сразу.
    """

    soma = annotations.set_index("bodyId")["somaSide"]
    root = annotations.set_index("bodyId")["rootSide"]

    root = root.where(root.isin(["L", "R"]))

    return soma.fillna(root)


def transmitters(ids_by_family):
    """Какой медиатор у каждого семейства и какой у него знак."""

    if not NT_FILE.exists():
        raise SystemExit(
            "нет предсказаний медиаторов, скачай:\n"
            "  curl -o brain/data/body-neurotransmitters.feather \\\n"
            "    https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
            "flat-connectome/body-neurotransmitters-male-cns-v1.0.feather"
        )

    table = pd.read_feather(NT_FILE, columns=["body", "consensus_nt"])
    table = table.set_index("body")["consensus_nt"]

    result = {}

    for family, ids in ids_by_family.items():

        known = table.loc[table.index.intersection(ids)]
        known = known[known != "unclear"]

        name = known.mode().iloc[0] if len(known) else "unclear"
        result[family] = (str(name), TRANSMITTER_SIGN.get(str(name), 1.0))

    return result


def family_ids(annotations, selector):
    """Семейство задаётся либо началом имени типа, либо полями аннотации."""

    types = annotations["type"].fillna("")

    if isinstance(selector, dict):

        mask = np.ones(len(annotations), dtype=bool)

        for column, values in selector.items():

            if column == "type_contains":
                mask &= types.str.contains(values, na=False).to_numpy()
            elif column == "type_not_startswith":
                mask &= ~types.str.startswith(values).to_numpy()
            else:
                mask &= annotations[column].isin(values).to_numpy()

        return annotations.loc[mask, "bodyId"].tolist()

    return annotations.loc[types.str.startswith(selector), "bodyId"].tolist()


def skeleton(body_id):
    """Скелет нейрона в миллиметрах и осях MuJoCo, прорежённый."""

    neuron = navis.read_swc(str(malecns.swc_path(int(body_id))))
    pruned = navis.prune_twigs(neuron, PRUNE_TWIGS)

    factor = pruned.n_nodes // TARGET_NODES

    # Мелкие нейроны прорежать нечего.
    if factor > 1:
        pruned = navis.downsample_neuron(pruned, factor)

    head, tail = segments(pruned)

    return head * VOXEL_MM @ TO_MUJOCO.T, tail * VOXEL_MM @ TO_MUJOCO.T


def build(name, annotations, table, sides):
    """Считает матрицы и собирает скелеты для одной цепочки."""

    families, floors, details = CHAINS[name]

    ids = {label: family_ids(annotations, selector) for label, selector, _, _ in families}

    names = [label for label, _, _, _ in families]

    side_of = sides.dropna()

    for label in names:
        known = len([b for b in ids[label] if b in side_of.index])
        print(f"   {label:7} {len(ids[label]):5} нейронов, сторона известна у {known}")

    nt = transmitters(ids)

    matrices = []

    print(f"\n=== {name}")

    for label, (medium, sign) in nt.items():
        mark = "возбуждает" if sign > 0 else ("тормозит" if sign < 0 else "модулирует")
        print(f"   {label:7} медиатор {medium:14} -> {mark}")

    for one, two in zip(names, names[1:]):

        edges = table.filter(
            pc.is_in(table["body_pre"], value_set=pa.array(ids[one]))
        ).to_pandas()
        edges = edges[edges["body_post"].isin(set(ids[two]))].copy()

        edges["s_pre"] = edges["body_pre"].map(side_of)
        edges["s_post"] = edges["body_post"].map(side_of)
        edges = edges.dropna(subset=["s_pre", "s_post"])

        matrix = np.zeros((2, 2))

        for row, pre in enumerate(("L", "R")):
            for column, post in enumerate(("L", "R")):
                mask = (edges["s_pre"] == pre) & (edges["s_post"] == post)
                matrix[row, column] = edges.loc[mask, "weight"].sum()

        matrices.append(matrix)

        print(
            f"{one:6} -> {two:6}  L->L {matrix[0,0]:8.0f}  L->R {matrix[0,1]:8.0f}"
            f"  R->L {matrix[1,0]:8.0f}  R->R {matrix[1,1]:8.0f}"
        )

    starts, ends, group, labels, colors, family_of, side_list = [], [], [], [], [], [], []

    for label, _, color, _ in families:

        for side in ("L", "R"):

            candidates = [b for b in ids[label] if side_of.get(b) == side]

            if not candidates:
                print(f"нет {label} со стороной {side}, пропускаю")
                continue

            body_id = int(candidates[0])
            head_points, tail_points = skeleton(body_id)

            starts.append(head_points)
            ends.append(tail_points)
            group.append(np.full(len(head_points), len(labels)))

            labels.append(f"{label}_{side} {body_id}")
            colors.append(color)
            family_of.append(label)
            side_list.append(side)

            print(f"скелет {label}_{side} {body_id}: {len(head_points)} отрезков")

    starts = np.concatenate(starts)
    ends = np.concatenate(ends)

    center = (np.vstack([starts, ends]).min(0) + np.vstack([starts, ends]).max(0)) / 2
    starts -= center
    ends -= center

    out = Path(str(OUT).format(name=name))

    np.savez_compressed(
        out,
        center=center,
        starts=starts,
        ends=ends,
        group=np.concatenate(group),
        labels=labels,
        colors=np.array(colors),
        families=family_of,
        sides=side_list,
        stage_from=names[:-1],
        stage_to=names[1:],
        stage_floors=floors,
        stage_details=details,
        stage_matrices=np.array(matrices),
        stage_signs=np.array([nt[label][1] for label in names[:-1]]),
        family_transmitters=[nt[label][0] for label in names],
        family_names=names,
        family_functions=[work for _, _, _, work in families],
        family_colors=np.array([color for _, _, color, _ in families]),
    )

    print(f"всего {len(starts)} отрезков, {len(labels)} нейронов, {len(matrices)} переходов")
    print("сохранил:", out)


def main():

    wanted = sys.argv[1:] or list(CHAINS)

    annotations = malecns.annotations()
    sides = side_lookup(annotations)
    table = feather.read_table(WEIGHTS, memory_map=True)

    for name in wanted:

        if name not in CHAINS:
            raise SystemExit(f"не знаю цепочку {name}, есть: {', '.join(CHAINS)}")

        build(name, annotations, table, sides)


if __name__ == "__main__":
    main()
