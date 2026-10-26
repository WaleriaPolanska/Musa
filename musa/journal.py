"""Журнал: что делала ты и что делала Муза.

Пишет две вещи. События — когда ты включила ходьбу, покрутила синапс, сбросила
веса. Замеры — где была Муза, где были свет и еда, куда она поворачивала. На
выходе складывает всё в таблицу CSV и печатает короткий итог.

Это не память Музы: она пока ничего не запоминает, все веса берутся из
коннектома и меняются только твоей рукой. Настоящая память у мухи живёт в
грибовидном теле, его в этих данных тоже видно.
"""

import csv
import time
from pathlib import Path

import numpy as np

SAMPLE_EVERY = 0.5  # секунд между замерами


class Journal:
    """Копит события и замеры, потом складывает в таблицу."""

    def __init__(self, folder):

        self.folder = Path(folder)
        self.events = []
        self.samples = []

        self.started = time.time()
        self._last_sample = 0.0

        self.walked = 0.0  # сколько миллиметров прошла
        self._previous = None

    def _moment(self):
        return time.time() - self.started

    # --------------------------------------------------------
    # ЗАПИСЬ
    # --------------------------------------------------------

    def event(self, kind, what, detail=""):
        """Что-то произошло по твоей команде."""

        self.events.append(
            {
                "time": round(self._moment(), 2),
                "kind": kind,
                "what": what,
                "detail": detail,
            }
        )

    def sample(self, position, sources, turns, legs, walking):
        """Замер раз в полсекунды: где все и куда Муза поворачивает."""

        moment = self._moment()

        if walking and self._previous is not None:
            self.walked += float(np.linalg.norm(position[:2] - self._previous[:2]))

        self._previous = np.asarray(position, dtype=float).copy()

        if moment - self._last_sample < SAMPLE_EVERY:
            return

        self._last_sample = moment

        row = {
            "time": round(moment, 2),
            "walking": int(walking),
            "musa_x": round(float(position[0]), 2),
            "musa_y": round(float(position[1]), 2),
            "walked_mm": round(self.walked, 2),
            "legs_L": round(float(legs[0]), 2),
            "legs_R": round(float(legs[1]), 2),
        }

        for name, place in sources.items():
            row[f"{name}_x"] = round(float(place[0]), 2)
            row[f"{name}_y"] = round(float(place[1]), 2)

        for name, turn in turns.items():
            row[f"turn_{name}"] = round(float(turn), 3)

        self.samples.append(row)

    # --------------------------------------------------------
    # ИТОГ
    # --------------------------------------------------------

    def save(self):
        """Кладёт обе таблицы в outputs и возвращает пути."""

        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.folder.mkdir(parents=True, exist_ok=True)

        written = []

        for name, rows in (("events", self.events), ("track", self.samples)):

            if not rows:
                continue

            path = self.folder / f"journal-{stamp}-{name}.csv"

            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            written.append(path)

        return written

    def summary(self, feelings):
        """Короткий итог строками: что было и что ты накрутила."""

        lines = []

        minutes = self._moment() / 60.0
        walking = sum(row["walking"] for row in self.samples) * SAMPLE_EVERY

        lines.append(f"время в окне: {minutes:.1f} мин, из них ходьба {walking:.0f} с")
        lines.append(f"прошла: {self.walked:.1f} мм")
        lines.append(f"событий записано: {len(self.events)}, замеров: {len(self.samples)}")

        for feeling in feelings:

            chain = feeling["chain"]
            gains = feeling["gains"]

            changed = [
                (
                    f"{chain.stage_from[i]}>{chain.stage_to[i]}",
                    gains["L"][i],
                    gains["R"][i],
                )
                for i in range(len(chain.stage_from))
                if abs(gains["L"][i] - 1.0) > 1e-6 or abs(gains["R"][i] - 1.0) > 1e-6
            ]

            if not changed:
                lines.append(f"{feeling['stimulus']}: синапсы как в мозге, ты их не трогала")
                continue

            lines.append(f"{feeling['stimulus']}: изменённые синапсы")

            for name, left, right in changed:
                lines.append(f"    {name:16} слева x{left:.2f}   справа x{right:.2f}")

        return lines
