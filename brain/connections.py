"""С кем связан тип нейронов — запрос к neuPrint (male-cns:v1.0).

Нужен токен: https://neuprint.janelia.org → войти → Account → скопировать токен, потом

    export NEUPRINT_APPLICATION_CREDENTIALS=<токен>
    ./run-brain.sh brain/connections.py LPLC2

Скелеты и аннотации (find.py, view.py) токена не требуют — они из публичного бакета.
"""

import os
import sys

import malecns

TOKEN_VAR = "NEUPRINT_APPLICATION_CREDENTIALS"


def main():

    if len(sys.argv) < 2:
        raise SystemExit("укажи тип нейрона, например: brain/connections.py LPLC2")

    neuron_type = sys.argv[1]

    if not os.environ.get(TOKEN_VAR):
        raise SystemExit(
            f"нет токена в {TOKEN_VAR}.\n"
            "Возьми его на https://neuprint.janelia.org (Account) и выполни:\n"
            f"  export {TOKEN_VAR}=<токен>"
        )

    from neuprint import Client, NeuronCriteria, fetch_simple_connections

    client = Client("neuprint.janelia.org", dataset=f"male-cns:{malecns.VERSION}")

    print("neuPrint:", client.fetch_version())

    connections = fetch_simple_connections(
        NeuronCriteria(type=neuron_type),
        None,
        client=client,
    )

    if connections.empty:
        raise SystemExit(f"связей для «{neuron_type}» не найдено")

    downstream = (
        connections.groupby("type_post")["weight"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )

    print(f"\n{neuron_type} → кому (сумма синапсов):\n")

    for name, weight in downstream.items():
        print(f"  {int(weight):7d}  {name}")


if __name__ == "__main__":
    main()
