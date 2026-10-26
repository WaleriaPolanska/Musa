"""Поиск нейронов в maleCNS v1.0 по названию типа.

    ./run-brain.sh brain/find.py LPLC2      конкретный тип
    ./run-brain.sh brain/find.py LC         все типы, где встречается LC
    ./run-brain.sh brain/find.py            самые крупные классы нейронов
"""

import sys

import malecns


def overview(df):

    print(f"maleCNS {malecns.VERSION}: {len(df)} тел\n")
    print("крупнейшие классы:\n")

    counts = df["superclass"].value_counts().head(15)

    for name, count in counts.items():
        print(f"  {count:7d}  {name}")

    print("\nИщи типы так:  ./run-brain.sh brain/find.py LPLC2")


def search(df, pattern):

    found = malecns.find(pattern, df)

    if found.empty:
        print(f"ничего не найдено по «{pattern}»")
        return

    print(f"«{pattern}»: {len(found)} нейронов, {found['type'].nunique()} типов\n")

    grouped = found.groupby("type", dropna=False)

    for name, rows in sorted(grouped, key=lambda kv: -len(kv[1]))[:20]:

        sides = rows["somaSide"].value_counts().to_dict()
        sides = " ".join(f"{k}:{v}" for k, v in sorted(sides.items()))
        superclass = rows["superclass"].dropna().unique()
        superclass = superclass[0] if len(superclass) else "—"

        body_ids = " ".join(str(b) for b in rows["bodyId"].head(3))

        print(f"  {name:<18} {len(rows):5d}  {superclass:<20} {sides}")
        print(f"  {'':<18} bodyId: {body_ids} ...")

    print("\nПосмотреть в 3D:  ./run-brain.sh brain/view.py --type <тип> --limit 5")


def main():

    df = malecns.annotations()

    if len(sys.argv) > 1:
        search(df, sys.argv[1])
    else:
        overview(df)


if __name__ == "__main__":
    main()
