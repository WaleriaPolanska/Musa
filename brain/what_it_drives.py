"""Чем управляет нейрон: считаем по коннектому, а не по памяти.

Для каждого типа печатает:
  * сколько синапсов он шлёт вниз и сколько из них прямо на мотонейроны;
  * на какие мышцы эти мотонейроны работают;
  * в каких нейромерах сидят его партнёры в брюшной нервной цепочке;
  * какой у него медиатор — возбуждает он цель или тормозит.

    ./run-brain.sh brain/what_it_drives.py DNp01 DNp06
    ./run-brain.sh brain/what_it_drives.py            (типы из export_circuit)
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

import malecns
from export_circuit import SOURCE, TARGETS, WEIGHTS

# Мотонейроны в коннектоме помечены подклассом: какая это мышца.
MUSCLE = {
    "fl": "передние ноги",
    "ml": "средние ноги",
    "hl": "задние ноги",
    "wm": "крыло (в том числе прыжковая мышца)",
    "nm": "шея",
    "hm": "жужжальца",
    "ad": "брюшко",
    "pm": "ротовой аппарат",
    "am": "голова и шея (GNG)",
    "rm": "голова и шея (GNG)",
}

NEUROMERE = {
    "T1": "T1 — передние ноги и шея",
    "T2": "T2 — крыло и средние ноги",
    "T3": "T3 — задние ноги",
    "A1": "A1 — брюшко",
}

TRANSMITTER = {
    "acetylcholine": "ацетилхолин — возбуждает",
    "glutamate": "глутамат — чаще тормозит",
    "gaba": "ГАМК — тормозит",
}

NT_FILE = Path(__file__).parent / "data" / "body-neurotransmitters.feather"


def transmitter(name):

    if not NT_FILE.exists():
        return "медиатор неизвестен: нет файла body-neurotransmitters.feather"

    nt = pd.read_feather(
        NT_FILE,
        columns=["cell_type", "consensus_nt", "celltype_predicted_nt_confidence"],
    )
    rows = nt.loc[nt["cell_type"].fillna("") == name]

    if rows.empty:
        rows = nt.loc[nt["cell_type"].fillna("").str.startswith(name)]

    if rows.empty:
        return "медиатор не предсказан"

    row = rows.iloc[0]
    kind = TRANSMITTER.get(str(row["consensus_nt"]), str(row["consensus_nt"]))

    return f"{kind} (уверенность {row['celltype_predicted_nt_confidence']:.2f})"


def main():

    names = sys.argv[1:] or [SOURCE] + list(TARGETS)

    annotations = malecns.annotations()
    by_id = annotations.set_index("bodyId")

    motor = annotations[annotations["superclass"].isin(["vnc_motor", "cb_motor"])]
    motor_subclass = motor.set_index("bodyId")["subclass"]

    table = feather.read_table(WEIGHTS, memory_map=True)

    for name in names:

        types = annotations["type"].fillna("")

        ids = annotations.loc[types == name, "bodyId"].tolist()

        # Семейства вроде IN12B — это не один тип, а IN12B001, IN12B002 и так далее.
        if not ids:
            ids = annotations.loc[types.str.startswith(name), "bodyId"].tolist()

        if not ids:
            print(f"{name}: такого типа нет в аннотациях")
            continue

        out = table.filter(pc.is_in(table["body_pre"], value_set=pa.array(ids))).to_pandas()

        print(f"\n=== {name} ({len(ids)} нейронов)")
        print(f"    медиатор: {transmitter(name)}")
        print(f"    всего синапсов на выходе: {out['weight'].sum():.0f}")

        direct = out[out["body_post"].isin(motor_subclass.index)].copy()

        if direct.empty:
            print("    прямых синапсов на мотонейроны нет, работает через вставочные")
        else:
            direct["muscle"] = direct["body_post"].map(motor_subclass).map(MUSCLE)
            muscles = direct.groupby("muscle")["weight"].sum().sort_values(ascending=False)
            print(f"    прямо на мотонейроны: {direct['weight'].sum():.0f}")
            for muscle, weight in muscles.head(4).items():
                print(f"        {muscle}: {weight:.0f}")

        partners = out.copy()
        partners["neuromere"] = partners["body_post"].map(by_id["somaNeuromere"])
        partners["superclass"] = partners["body_post"].map(by_id["superclass"])

        vnc = partners[partners["superclass"].fillna("").str.startswith("vnc")]

        if not vnc.empty:
            share = vnc.groupby("neuromere")["weight"].sum()
            share = (share / share.sum() * 100).sort_values(ascending=False)
            print("    партнёры в нервной цепочке:")
            for neuromere, part in share.head(3).items():
                print(f"        {NEUROMERE.get(neuromere, neuromere)}: {part:.0f}%")


if __name__ == "__main__":
    main()
