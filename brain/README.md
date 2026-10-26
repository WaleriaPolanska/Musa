# Коннектом maleCNS — оцифрованный мозг мухи

Полная нервная система взрослого самца дрозофилы: 166 700 нейронов,
125 млн синапсов. Janelia FlyEM + Кембридж + MRC LMB + Google Research,
статья вышла в Cell 3 сентября 2026. Лицензия CC-BY.

## Версии

| версия | дата | что это |
|---|---|---|
| v0.9 | 5 октября 2025 | самый первый релиз |
| **v1.0** | **8 июня 2026** | **последняя**: вычитка и уточнённые аннотации нейронов |

Скрипты здесь работают с **v1.0**. Если где-то осталось `v0.9` — это старая версия.

## Вьюверы в браузере

- **Neuroglancer** — 3D + ЭМ-срезы, без регистрации:
  <https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json>
  Контроллы: левая кнопка — вращать, колесо — зум, правая — двигать срез, `s` — скрыть/показать слой.
- **neuPrint** — поиск нейронов и связей:
  <https://neuprint.janelia.org/?dataset=male-cns%3Av1.0&qt=findneurons> (нужен вход через Google)
- **Cell Type Explorer** — типы клеток и связи тип-к-типу:
  <https://reiserlab.github.io/celltype-explorer-drosophila-male-cns>
- Всё остальное: <https://male-cns.janelia.org/explore/>

## Скрипты

```bash
./run-brain.sh brain/find.py              # крупнейшие классы нейронов
./run-brain.sh brain/find.py LPLC         # найти типы по названию
./run-brain.sh brain/view.py --type LPLC2 --limit 5   # 3D в браузере
./run-brain.sh brain/view.py 11498 12384              # по bodyId
```

`find.py` и `view.py` **токен не требуют** — данные качаются из публичного
бакета `gs://flyem-male-cns/v1.0` и кэшируются в `brain/data/`
(аннотации 14 МБ один раз + скелеты по ~10–70 КБ на нейрон).

Контроллы в `view.py`: левая кнопка — вращать, колесо — зум,
правая кнопка — двигать, двойной клик — вернуть камеру.

### Нейроны внутрь симуляции

```bash
./run-brain.sh brain/export_for_sim.py
```

Готовит `brain/data/neurons_for_sim.npz` для `musa/musa_full.py`: берёт по два
нейрона типов LPLC2, DNa02, DNp01, прорежает ветки, переводит в миллиметры и в
оси MuJoCo. Какие типы брать — список `SETS` в начале скрипта.

### Схема связей для симуляции

```bash
./run-brain.sh brain/export_circuit.py
```

Считает настоящие веса LPLC2 → нисходящие нейроны (с разбивкой по сторонам)
и складывает их вместе со скелетами в `brain/data/circuit.npz` для
`musa/musa_synapses.py`. Нужен файл весов всего коннектома, 502 МБ:

```bash
curl -o brain/data/connectome-weights.feather \
  https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather
```

В нём 25 568 639 связей с колонками body_pre, body_post, weight, type_pre,
type_post. Токен не нужен.

### Оболочка мозга

```bash
./run-brain.sh brain/export_shell.py
```

Качает слой `brain-shell` (53 МБ) из того же неуроглансера, пересобирает
поверхность по сетке 6 мкм и прорежает до 23 656 треугольников,
кладёт в `brain/data/brain_shell.obj` в общих с нейронами координатах.

### Связи через neuPrint (нужен токен)

```bash
export NEUPRINT_APPLICATION_CREDENTIALS=<токен>
./run-brain.sh brain/connections.py LPLC2
```

Токен: <https://neuprint.janelia.org> → войти → **Account** → скопировать.
Это единственный шаг, который надо сделать руками.

## Зачем это Музе

`LPLC2` и `LC` — зрительные проекционные нейроны, которые реагируют на
движение и приближение объекта. В `musa/brain_light.py` зрительный нейрон
написан от руки; здесь лежит настоящая схема связей, на которую его можно
опереть.

## Окружение

`brainvenv/` — python 3.12, navis 1.12.0, neuprint-python 0.6.3, plotly 7.1.0.
К окружению симуляции (`flygym21/`) отношения не имеет, они независимы.

## Цепочки для Музы (главное)

```bash
./run-brain.sh brain/export_chain.py
```

Выгружает три цепочки, по которым живёт главная программа
(`musa/musa_chain.py`): зрительный компас, обонятельную и память
(грибовидное тело). Для каждого перехода — матрица 2×2 «сторона → сторона»
по числу синапсов, знак по медиатору отправителя и описание, что делает
переход. Результат: `brain/data/chain_light.npz`, `chain_food.npz`,
`chain_memory.npz`. Сторона нейрона берётся из `somaSide`, а если его нет —
из `rootSide`.

Нужен `brain/data/connectome-weights.feather` (481 МБ) и аннотации — они уже
лежат в `brain/data`. Не удалять: без них цепочки не пересобрать.
