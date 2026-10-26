"""3D-просмотр нейронов maleCNS v1.0. Скелеты качаются из публичного бакета.

    ./run-brain.sh brain/view.py 11498 12384
    ./run-brain.sh brain/view.py --type LPLC2 --limit 5

Открывается в браузере: левая кнопка — вращать, колесо — зум,
правая кнопка — двигать, двойной клик — вернуть камеру.
"""

import argparse
import webbrowser
from pathlib import Path

import navis
import plotly.io as pio

import malecns

OUT = Path(__file__).parent / "out"


def load(body_ids, names):

    neurons = []

    for body_id, name in zip(body_ids, names):

        neuron = navis.read_swc(str(malecns.swc_path(body_id)), units="8 nm")
        neuron.id = body_id
        neuron.name = f"{name} {body_id}"

        neurons.append(neuron)

    return navis.NeuronList(neurons)


def main():

    parser = argparse.ArgumentParser(description="показать нейроны maleCNS в 3D")
    parser.add_argument("body_ids", nargs="*", type=int, help="bodyId нейронов")
    parser.add_argument("--type", help="показать нейроны этого типа")
    parser.add_argument("--limit", type=int, default=5, help="сколько штук для --type")

    args = parser.parse_args()

    df = malecns.annotations()

    if args.type:

        rows = df.loc[df["type"].str.fullmatch(args.type, case=False, na=False)]

        if rows.empty:
            rows = malecns.find(args.type, df)

        if rows.empty:
            raise SystemExit(f"тип «{args.type}» не найден, поищи через brain/find.py")

        rows = rows.head(args.limit)
        title = f"{args.type} ({len(rows)})"

    elif args.body_ids:

        rows = df.loc[df["bodyId"].isin(args.body_ids)]
        title = " ".join(str(b) for b in args.body_ids)

    else:
        raise SystemExit("дай bodyId или --type, примеры в шапке файла")

    print(f"нейронов: {len(rows)}")

    neurons = load(rows["bodyId"].tolist(), rows["type"].fillna("?").tolist())

    figure = navis.plot3d(neurons, backend="plotly", inline=False)
    figure.update_layout(title=f"maleCNS {malecns.VERSION} — {title}")

    OUT.mkdir(exist_ok=True)
    path = OUT / f"{(args.type or 'neurons')}.html"

    pio.write_html(figure, path, include_plotlyjs="cdn", auto_open=False)

    print("открываю", path)

    webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
