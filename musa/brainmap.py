"""Мозг в 3D на правом экране: нейроны, оболочка и синапсы между нейронами.

Рисуется отдельной сценой MuJoCo в своём прямоугольнике окна. Мозг стоит
далеко от арены, чтобы его камера не цепляла Музу и пол. Камеру крутят
мышкой, по шарику-синапсу можно кликнуть.

Мозг собирается из двух разных наборов данных — цепочки из трёх этажей
(chain.npz) и схемы бегства (circuit.npz), — поэтому конструкторов два.
"""

import mujoco
import numpy as np

# Где стоит мозг: далеко от арены и высоко, Муза туда не заходит.
MAP_AT = np.array([0.0, 0.0, 50.0])

CAMERA_DISTANCE = 1.35
CAMERA_AZIMUTH = 90.0
CAMERA_ELEVATION = -25.0

SKELETON_STEP = 2  # каждый второй отрезок: хватает и быстрее
LINE_WIDTH = 2.0

MARKER_MM = 0.02  # базовый радиус шарика-синапса
PICK_RADIUS_PX = 45


class BrainMap:
    """Трёхмерный мозг: скелеты нейронов, оболочка и синапсы."""

    def __init__(self, model, data, starts, ends, group, colors, keys, links):

        self.model = model
        self.data = data

        self.colors = np.asarray(colors)
        self.keys = keys

        starts = np.asarray(starts) + MAP_AT
        ends = np.asarray(ends) + MAP_AT
        group = np.asarray(group)

        # Центр каждого нейрона — чтобы было откуда и куда вести синапс.
        self.centers = {}

        for number, key in enumerate(keys):
            points = np.vstack([starts[group == number], ends[group == number]])
            self.centers[key] = points.mean(axis=0)

        keep = slice(None, None, SKELETON_STEP)

        self.starts = starts[keep]
        self.ends = ends[keep]
        self.group = group[keep]

        self.markers = []

        for link in links:

            source = self.centers.get(link["from"])
            target = self.centers.get(link["to"])

            if source is None or target is None:
                continue

            # Разносим синапсы веером поперёк линии источник -> цель.
            direction = target - source
            across = np.cross(direction, [0.0, 0.0, 1.0])
            length = np.linalg.norm(across) or 1.0

            marker = dict(link)
            marker["pos"] = (
                source + direction * 0.45 + across / length * link.get("spread", 0.0)
            )
            marker["source"] = source
            marker["target"] = target

            self.markers.append(marker)

        self._lines_at = 0
        self._markers_at = 0

    # --------------------------------------------------------
    # ОТКУДА БЕРЁМ ДАННЫЕ
    # --------------------------------------------------------

    @classmethod
    def from_chain(cls, chain_data, model, data):
        """Цепочка из трёх этажей: зрение -> связь -> движение."""

        families = [str(name) for name in chain_data["family_names"]]

        keys = [
            (str(family), str(side))
            for family, side in zip(chain_data["families"], chain_data["sides"])
        ]

        stage_from = [str(name) for name in chain_data["stage_from"]]
        stage_to = [str(name) for name in chain_data["stage_to"]]
        floors = [str(name) for name in chain_data["stage_floors"]]

        family_colors = np.asarray(chain_data["family_colors"])

        links = []

        for stage, (first, second) in enumerate(zip(stage_from, stage_to)):
            for side in ("L", "R"):

                if (first, side) not in keys or (second, side) not in keys:
                    continue

                links.append(
                    {
                        "index": stage,
                        "side": side,
                        "name": f"{first}>{second}",
                        "floor": floors[stage],
                        "from": (first, side),
                        "to": (second, side),
                        "color": family_colors[families.index(first)],
                        "neuron": keys.index((first, side)),
                        "spread": 0.05 * (stage - (len(stage_from) - 1) / 2),
                    }
                )

        return cls(
            model,
            data,
            chain_data["starts"],
            chain_data["ends"],
            chain_data["group"],
            chain_data["colors"],
            keys,
            links,
        )

    @classmethod
    def from_circuit(cls, circuit, model, data):
        """Схема бегства: LPLC2 -> нисходящие нейроны."""

        targets = [str(name) for name in circuit["target_types"]]
        source_type = str(circuit["source_type"])

        keys = [
            (str(kind), str(side))
            for kind, side in zip(circuit["types"], circuit["sides"])
        ]

        colors = np.asarray(circuit["colors"])

        links = []

        for index, name in enumerate(targets):
            for side in ("L", "R"):

                if (name, side) not in keys or (source_type, side) not in keys:
                    continue

                links.append(
                    {
                        "index": index,
                        "side": side,
                        "name": name,
                        "floor": "",
                        "from": (source_type, side),
                        "to": (name, side),
                        "color": colors[keys.index((name, side))],
                        "neuron": keys.index((name, side)),
                        "spread": 0.06 * (index - (len(targets) - 1) / 2),
                    }
                )

        return cls(
            model,
            data,
            circuit["starts"],
            circuit["ends"],
            circuit["group"],
            colors,
            keys,
            links,
        )

    # --------------------------------------------------------
    # КАМЕРА
    # --------------------------------------------------------

    def camera(self):

        camera = mujoco.MjvCamera()
        # Смотрим чуть вперёд от середины: там мозг, позади — нервная цепочка.
        camera.lookat[:] = MAP_AT + np.array([0.12, 0.0, 0.0])
        camera.distance = CAMERA_DISTANCE
        camera.azimuth = CAMERA_AZIMUTH
        camera.elevation = CAMERA_ELEVATION

        return camera

    @staticmethod
    def option():
        """Модель в панели не рисуем — только наши геометрии."""

        option = mujoco.MjvOption()

        for group in range(6):
            option.geomgroup[group] = 0

        return option

    # --------------------------------------------------------
    # ТОЧКА МОЗГА -> ПИКСЕЛИ ЭКРАНА
    # --------------------------------------------------------

    @staticmethod
    def _axes(camera):
        """Куда смотрит камера, где у неё право и верх."""

        azimuth = np.radians(camera.azimuth)
        elevation = np.radians(camera.elevation)

        forward = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )

        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right) or 1.0

        up = np.cross(right, forward)

        return forward, right, up

    def to_pixels(self, rect, point, camera, fovy_deg=45.0):
        """Точка мозга -> пиксели кадра. None, если точка за спиной камеры."""

        forward, right, up = self._axes(camera)

        eye = np.asarray(camera.lookat) - forward * camera.distance
        offset = np.asarray(point) - eye

        depth = float(offset @ forward)

        if depth <= 1e-6:
            return None

        scale = (rect.height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)

        x = rect.left + rect.width / 2.0 + scale * float(offset @ right) / depth
        y = rect.bottom + rect.height / 2.0 + scale * float(offset @ up) / depth

        return x, y

    def pick(self, px, py, rect, camera):
        """Какой синапс под курсором. Возвращает (номер связи, сторона)."""

        best = None
        best_distance = PICK_RADIUS_PX

        for marker in self.markers:

            spot = self.to_pixels(rect, marker["pos"], camera)

            if spot is None:
                continue

            distance = np.hypot(px - spot[0], py - spot[1])

            if distance < best_distance:
                best = (marker["index"], marker["side"])
                best_distance = distance

        return best

    def label_positions(self, rect, camera):
        """Подписи синапсов: доли панели для mjr_text и сам маркер."""

        for marker in self.markers:

            spot = self.to_pixels(rect, marker["pos"], camera)

            if spot is None:
                continue

            yield (
                (spot[0] - rect.left) / rect.width + 0.012,
                (spot[1] - rect.bottom) / rect.height + 0.008,
                marker,
            )

    # --------------------------------------------------------
    # РИСОВАНИЕ
    # --------------------------------------------------------

    def build(self, scene, first):
        """Скелеты кладём один раз: мозг в панели стоит на месте."""

        identity = np.eye(3).flatten()
        ngeom = first

        size = np.array([LINE_WIDTH, 0.0, 0.0])

        self._lines_at = ngeom

        for i in range(len(self.starts)):

            geom = scene.geoms[ngeom]

            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                size,
                self.starts[i],
                identity,
                np.ones(4),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                LINE_WIDTH,
                self.starts[i],
                self.ends[i],
            )

            ngeom += 1

        self._markers_at = ngeom

        return ngeom

    def refresh(self, scene, weights, gains, activity, selected):
        """Каждый кадр: перекрашиваем скелеты и пересобираем шарики синапсов."""

        identity = np.eye(3).flatten()

        brightness = 0.25 + 0.75 * np.clip(activity, 0.0, 1.0)
        rgba = np.ones((len(self.starts), 4))
        rgba[:, :3] = self.colors[self.group] * brightness[self.group, None]
        rgba[:, 3] = 0.9

        for i in range(len(self.starts)):
            scene.geoms[self._lines_at + i].rgba[:] = rgba[i]

        ngeom = self._markers_at

        strongest = max(
            float(np.max(weights["L"])), float(np.max(weights["R"])), 1.0
        )

        for marker in self.markers:

            index = marker["index"]
            side = marker["side"]

            strength = weights[side][index] * gains[side][index]
            radius = MARKER_MM * (0.35 + 0.65 * np.sqrt(strength / strongest))

            position = marker["pos"]

            chosen_index, chosen_side = selected

            # Выбранный синапс — белое кольцо вокруг шарика.
            if chosen_index == index and chosen_side in (None, side):

                mujoco.mjv_initGeom(
                    scene.geoms[ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([radius * 1.45, 0.0, 0.0]),
                    position,
                    identity,
                    np.array([1.0, 1.0, 1.0, 0.55]),
                )

                ngeom += 1

            glow = 0.35 + 0.65 * float(np.clip(activity[marker["neuron"]], 0.0, 1.0))

            mujoco.mjv_initGeom(
                scene.geoms[ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, 0.0, 0.0]),
                position,
                identity,
                np.array([*(np.asarray(marker["color"]) * glow), 1.0]),
            )

            ngeom += 1

            # Линия от источника к цели, чтобы было видно, что с чем связано.
            line = scene.geoms[ngeom]

            mujoco.mjv_initGeom(
                line,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.array([1.0, 0.0, 0.0]),
                position,
                identity,
                np.array([1.0, 1.0, 1.0, 0.30]),
            )
            mujoco.mjv_connector(
                line,
                mujoco.mjtGeom.mjGEOM_LINE,
                1.0,
                marker["source"],
                marker["target"],
            )

            ngeom += 1

        return ngeom
