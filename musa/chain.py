"""Три этажа мозга как расчёт: зрение -> связь -> движение.

Каждый переход — матрица 2x2 из коннектома: сколько синапсов идёт с левой
стороны на левую и на правую и то же самое справа.

По цепочке идут две величины: общий уровень возбуждения и перекос между
сторонами. Перекос решает, куда поворачивать, и на каждом этаже он нормирован
на свой собственный вес — иначе на кольце эллипсоидного тела, где стороны
почти симметричны, он бы просто растворился. Общий уровень нужен, чтобы
перекошенный ползунок тянул Музу в свою сторону даже при ровном свете:
усиленный синапс подаёт на свою сторону больше, чем ослабленный.

ЗАПАС оставлен нарочно: при x1.00 перекос идёт вполсилы, так что ползунок
заметно работает в обе стороны, а не упирается в потолок сразу.

Ползунок меняет силу перехода относительно того, что в мозге: x1.00 — как
в коннектоме.
"""

import numpy as np

SIDES = ("L", "R")

HEADROOM = 1.0  # перекос проходит этаж без затухания; запас делаем на ногах


class Chain:
    """Цепочка переходов и активность каждого семейства нейронов."""

    def __init__(self, data):

        self.families = [str(name) for name in data["family_names"]]
        self.family_functions = [str(work) for work in data["family_functions"]]

        self.stage_from = [str(name) for name in data["stage_from"]]
        self.stage_to = [str(name) for name in data["stage_to"]]
        self.floors = [str(name) for name in data["stage_floors"]]
        self.details = [str(text) for text in data["stage_details"]]

        self.matrices = np.asarray(data["stage_matrices"], dtype=float)

        # Коннектом считает синапсы, но не говорит, возбуждают они или тормозят.
        # Знак берём из медиатора того, кто шлёт: ацетилхолин возбуждает,
        # ГАМК, глутамат и гистамин тормозят.
        self.signs = (
            np.asarray(data["stage_signs"], dtype=float)
            if "stage_signs" in data
            else np.ones(len(self.matrices))
        )

        self.transmitters = (
            [str(name) for name in data["family_transmitters"]]
            if "family_transmitters" in data
            else ["unclear"] * (len(self.matrices) + 1)
        )

        # Сколько сигнала идёт — по числу синапсов; в какую сторону он толкает
        # цель — по медиатору. Поэтому уровень считаем по модулю, а перекос
        # по знаковой матрице.
        self.signed = self.matrices * self.signs[:, None, None]

        # Никотин цепляется к никотиновым рецепторам ацетилхолина, поэтому
        # действует только на холиновые этажи. Остальные его не замечают.
        self.cholinergic = np.array(
            [self.transmitters[i] == "acetylcholine" for i in range(len(self.matrices))]
        )

        # Сколько синапсов уходит с каждой стороны — это и показываем в списке.
        self.weights = {
            "L": self.matrices[:, 0, :].sum(axis=1),
            "R": self.matrices[:, 1, :].sum(axis=1),
        }

        # Насколько переход пропускает перекос, если веса как в мозге.
        # Тормозной этаж переворачивает знак, поэтому берём модуль.
        self.norms = np.array(
            [self._contrast(matrix) for matrix in self.signed], dtype=float
        )

        # И сколько синапсов переход несёт вообще — для общего уровня.
        self.levels = np.array(
            [max(matrix.sum() / 2.0, 1e-6) for matrix in self.matrices], dtype=float
        )

        # Сколько раз цепочка переворачивает знак: нечётное число тормозных
        # этажей разворачивает поворот целиком.
        self.polarity = float(np.prod(self.signs)) or 1.0

        # Активность каждого семейства с каждой стороны.
        self.activity = {side: np.zeros(len(self.families)) for side in SIDES}

        # Перекос, который цепочка даёт сама по себе: слева и справа синапсов
        # не поровну. Его вычитаем, иначе Муза кривила бы и в ровном свете.
        self._diffs = np.zeros(len(self.matrices))
        self._means = np.zeros(len(self.matrices))

        # Кривизна цепочки: сколько перекоса она даёт сама, в покое и на
        # каждую единицу возбуждения сверх покоя.
        self.rest_diffs = np.zeros(len(self.matrices))
        self.rest_means = np.zeros(len(self.matrices))
        self.driven_diffs = np.zeros(len(self.matrices))

        unit = {side: np.ones(len(self.matrices)) for side in SIDES}

        # Поправка ищется в несколько проходов: этажи стоят друг за другом,
        # и правка верхнего меняет то, что видит нижний. Меряем в покое,
        # когда ни один источник не чувствуется.
        for _ in range(30):
            self.step({"L": 0.0, "R": 0.0}, unit)
            self.rest_diffs = self._diffs.copy()
            self.rest_means = self._means.copy()

        # Второй замер — на ровном раздражителе с обеих сторон. Разница
        # показывает, насколько кривизна растёт вместе с возбуждением.
        for _ in range(30):
            self.step({"L": 0.5, "R": 0.5}, unit)
            driven = np.maximum(self._means - self.rest_means, 1e-6)
            self.driven_diffs = (self._diffs - self.rest_diffs) / driven

        # Насколько сильно цепочка отвечает на полный перекос входа: по этому
        # числу два чувства приводятся к одной силе, иначе одно задавит другое.
        self.response = max(abs(self.step({"L": 1.0, "R": 0.0}, unit)), 1e-3)

    @staticmethod
    def _contrast(matrix):
        """Во сколько раз переход усиливает разницу левый-правый."""

        left = matrix[0, 0] - matrix[1, 0]
        right = matrix[0, 1] - matrix[1, 1]

        return max(abs(left - right) / 2.0, 1e-6)

    def step(self, eye, gains, drug=0.0):
        """Прогоняет сигнал по цепочке. eye — вход на рецепторы.

        drug — доза никотина: во столько раз сильнее работают холиновые этажи.
        """

        level = np.array([eye["L"], eye["R"]], dtype=float)

        self.activity["L"][0] = level[0]
        self.activity["R"][0] = level[1]

        for stage, matrix in enumerate(self.matrices):

            dose = 1.0 + drug if self.cholinergic[stage] else 1.0

            gain_column = (
                np.array([[gains["L"][stage]], [gains["R"][stage]]]) * dose
            )

            scaled = matrix * gain_column
            scaled_signed = self.signed[stage] * gain_column

            # Насколько этаж вообще возбуждён на входе: поправка на кривизну
            # цепочки пропорциональна этому уровню, иначе в темноте Муза
            # начала бы крутиться сама по себе.
            self._means[stage] = float(level.mean())

            raw = level @ scaled
            raw_signed = level @ scaled_signed

            # Общий уровень: сколько возбуждения пришло на этаж вообще.
            common = np.tanh(raw.sum() / self.levels[stage])

            # Перекос: он и решает, куда поворачивать.
            self._diffs[stage] = (raw_signed[0] - raw_signed[1]) / self.norms[stage]

            driven = self._means[stage] - self.rest_means[stage]

            crooked = self.rest_diffs[stage] + self.driven_diffs[stage] * driven

            contrast = np.tanh((self._diffs[stage] - crooked) * HEADROOM)

            level = np.array([common + contrast, common - contrast]) * 0.5

            self.activity["L"][stage + 1] = level[0]
            self.activity["R"][stage + 1] = level[1]

        # Последнее семейство — мотонейроны ног. Их перекос и есть поворот.
        return float(level[0] - level[1])
