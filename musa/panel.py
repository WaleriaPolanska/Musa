"""Правый экран окна: карта мозга, список синапсов с ползунками и кнопки.

Рисуется примитивами MuJoCo в том же окне, что и Муза, поэтому никаких
лишних библиотек не нужно. Строку можно выбрать мышкой, силу связи тянуть
ползунком, внизу кнопки.
"""

import mujoco
import numpy as np

from brainmap import BrainMap

MAP_PART = 0.40  # какую долю правого экрана занимает карта мозга

ROW_HEIGHT = 20
ROW_GAP = 3
NAME_PART = 0.56  # доля ширины под название связи и её работу
BUTTON_HEIGHT = 34

BACKGROUND = (0.10, 0.11, 0.13)
ACTIVE_MIX = 0.75  # насколько строка перекрашивается в цвет нейрона, когда он работает
ROW_COLOR = (0.17, 0.18, 0.22)
ROW_CHOSEN = (0.30, 0.34, 0.45)
TRACK_COLOR = (0.13, 0.14, 0.17)
BUTTON_COLOR = (0.20, 0.24, 0.30)
BUTTON_ON = (0.25, 0.45, 0.35)

PUNISH_COLOR = (0.30, 0.33, 0.78)  # ток — синий
REWARD_COLOR = (0.20, 0.50, 0.28)
FORGET_COLOR = (0.32, 0.32, 0.36)
MEMORY_TAB = (0.24, 0.26, 0.40)
MEMORY_TAB_ON = (0.45, 0.48, 0.72)
FLOOR_COLOR = (0.13, 0.15, 0.20)
MUSIC_ON = (0.50, 0.32, 0.58)

MENU_COLOR = (0.18, 0.13, 0.22)

# Список треков в выпадающем меню MUSIC: имя в player.py (None — все по
# очереди) и надпись.
TRACKS = ((None, "ALL"), ("dream", "DREAM"), ("steps", "STEPS"), ("miami", "MIAMI"))

FLOORS = {"sense": "SENSE", "link": "LINK", "motor": "MOTOR"}

MEMORY_MAX = 1.5  # предел ползунка памяти: вес синапса KC -> MBON

# У каждой цепочки своё чувство на первом этаже.
SENSES = {"light": "VISION", "food": "SMELL"}

FONT = mujoco.mjtFont.mjFONT_NORMAL


class Panel:
    """Карта мозга, список синапсов и кнопки на правой половине окна."""

    MAP_GEOM = 20000

    def __init__(self, brain_map, state, actions, model, data, gain_max=3.0):

        self.brain_map = brain_map
        self.state = state
        self.actions = actions
        self.gain_max = gain_max

        self.camera = brain_map.camera()
        self.option = BrainMap.option()

        # Своя сцена: карта стоит на месте, её геометрию собираем один раз.
        self.scene = mujoco.MjvScene(model, self.MAP_GEOM)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0

        mujoco.mjv_updateScene(
            model,
            data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

        self._rebuild()

        self._rows = []
        self._sliders = []
        self._buttons = []
        self._map_rect = None
        self._drag = None
        self._music_button = None
        self._menu = []  # пункты открытого списка треков: (прямоугольник, трек)

    def _rebuild(self):
        """Пересобирает неподвижную часть сцены под текущую цепочку."""

        mujoco.mjv_updateScene(
            self.brain_map.model,
            self.brain_map.data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

        self.brain_map.build(self.scene, self.scene.ngeom)

    def use(self, brain_map):
        """Переключиться на другую цепочку: свет или еда."""

        self.brain_map = brain_map
        self._rebuild()

    # --------------------------------------------------------
    # МЫШЬ
    # --------------------------------------------------------

    @staticmethod
    def _inside(rect, x, y):
        return (
            rect.left <= x <= rect.left + rect.width
            and rect.bottom <= y <= rect.bottom + rect.height
        )

    def _set_value(self, kind, index, side, x, rect):
        """Ползунок: положение курсора -> сила связи или вес памяти."""

        part = (x - rect.left) / max(rect.width, 1)

        if kind == "volume":
            self.state["volume"] = float(np.clip(part, 0.0, 1.0))
            return

        if kind == "memory":
            self.state["sources"][index]["memory"] = float(
                np.clip(part * MEMORY_MAX, 0.0, MEMORY_MAX)
            )
            return

        self.state["gains"][side][index] = float(
            np.clip(part * self.gain_max, 0.0, self.gain_max)
        )

    def on_press(self, x, y, rect):

        # Открытый список треков — поверх всего: пункт выбирает трек, клик
        # мимо закрывает список.
        if self.state.get("music_open"):
            self.state["music_open"] = False
            for item_rect, track in self._menu:
                if self._inside(item_rect, x, y):
                    self.state["track"] = track
                    return
            if self._music_button is not None and self._inside(self._music_button, x, y):
                return

        if self._music_button is not None and self._inside(self._music_button, x, y):
            self.state["music_open"] = True
            return

        # С конца: что нарисовано позже, то сверху (строка MUSIC лежит
        # поверх нижних строк списка синапсов).
        for slider_rect, kind, index, side in reversed(self._sliders):
            if self._inside(slider_rect, x, y):
                self._choose(kind, index, side)
                self._set_value(kind, index, side, x, slider_rect)
                self._drag = ("slider", kind, index, side, slider_rect)
                return

        for row_rect, kind, index, side in self._rows:
            if self._inside(row_rect, x, y):
                self._choose(kind, index, side)
                return

        for button_rect, name in self._buttons:
            if self._inside(button_rect, x, y):
                self.actions[name]()
                return

        if self._map_rect is not None and self._inside(self._map_rect, x, y):

            hit = self.brain_map.pick(x, y, self._map_rect, self.camera)

            if hit is not None:
                self.state["selected"], self.state["side"] = hit

            # Тянешь по мозгу — крутишь его.
            self._drag = ("map", x, y)

    def on_drag(self, x, y, rect):

        if self._drag is None:
            return

        if self._drag[0] == "map":

            _, last_x, last_y = self._drag

            self.camera.azimuth -= (x - last_x) * 0.25
            self.camera.elevation = float(
                np.clip(self.camera.elevation + (y - last_y) * 0.25, -89.0, 89.0)
            )

            self._drag = ("map", x, y)
            return

        _, kind, index, side, slider_rect = self._drag
        self._set_value(kind, index, side, x, slider_rect)

    def _choose(self, kind, index, side):
        """Клик по строке: выбрать синапс или источник."""

        if kind == "memory":
            self.state["chosen_source"] = index
            return

        if kind == "volume":
            return

        self.state["selected"], self.state["side"] = index, side

    def on_release(self):
        self._drag = None

    def on_scroll(self, amount, x=None, y=None, rect=None):
        """Колесо: над мозгом зум, над списком — сила выбранного.

        В закладке памяти список другой, там колесо не трогаем.
        """

        # Над мозгом колесо приближает и отдаляет.
        if (
            x is not None
            and self._map_rect is not None
            and self._inside(self._map_rect, x, y)
        ):
            self.camera.distance = float(
                np.clip(self.camera.distance * (1.0 - 0.1 * amount), 0.4, 8.0)
            )
            return

        index = self.state["selected"]
        sides = ("L", "R") if self.state["side"] is None else (self.state["side"],)
        step = 0.1 if amount > 0 else -0.1

        for side in sides:
            self.state["gains"][side][index] = float(
                np.clip(self.state["gains"][side][index] + step, 0.0, self.gain_max)
            )

    # --------------------------------------------------------
    # РИСОВАНИЕ
    # --------------------------------------------------------

    def _memory_table(self, window, rect, top, row_height, row_gap, margin,
                      name_width, slider_left, slider_width):
        """Таблица: что Муза запомнила про каждый источник."""

        header = mujoco.MjrRect(rect.left + margin, top, rect.width - 2 * margin, row_height)

        mujoco.mjr_label(
            header, FONT, "WHAT MUSA REMEMBERS", *FLOOR_COLOR,
            1.0, 0.75, 0.80, 0.95, window.context,
        )

        top -= row_height + row_gap

        for index, spot in enumerate(self.state.get("sources", [])):

            chosen = index == self.state.get("chosen_source", 0)

            name_rect = mujoco.MjrRect(rect.left + margin, top, name_width, row_height)
            slider_rect = mujoco.MjrRect(slider_left, top, slider_width, row_height)

            weight = float(spot["memory"])
            pull = 2.0 * weight - 1.0

            # Строка загорается цветом источника, когда он сейчас чувствуется.
            felt = float(np.clip(spot["drive"]["L"] + spot["drive"]["R"], 0.0, 1.0))

            base = np.asarray(ROW_CHOSEN if chosen else ROW_COLOR)
            row_color = base + (np.asarray(spot["color"]) - base) * (ACTIVE_MIX * felt)

            mujoco.mjr_label(
                name_rect,
                FONT,
                f"{'>' if chosen else ' '} {spot['latin']}   pull {pull:+.2f}"
                f"   lessons {spot['taught']}",
                *row_color, 1.0, 1.0, 1.0, 1.0, window.context,
            )

            mujoco.mjr_rectangle(slider_rect, *TRACK_COLOR, 1.0)

            filled = int(slider_width * min(weight / MEMORY_MAX, 1.0))

            if filled > 0:
                color = np.asarray(spot["color"]) * (0.5 + 0.5 * felt)
                mujoco.mjr_rectangle(
                    mujoco.MjrRect(slider_left, top, filled, row_height),
                    *np.clip(color if chosen else color * 0.75, 0.0, 1.0), 1.0,
                )

            window.overlay(
                slider_rect,
                "KC>MBON",
                f"x{weight:.2f}",
                corner=mujoco.mjtGridPos.mjGRID_TOPLEFT,
            )

            self._rows.append((name_rect, "memory", index, "L"))
            self._sliders.append((slider_rect, "memory", index, "L"))

            top -= row_height + row_gap

        # Подсказка, что делают цветные кнопки.
        note = mujoco.MjrRect(rect.left + margin, top - row_height, rect.width - 2 * margin, row_height)

        mujoco.mjr_label(
            note, FONT,
            "click a row to steer it with arrows; PUNISH and REWARD teach what is sensed",
            *BACKGROUND, 1.0, 0.7, 0.7, 0.75, window.context,
        )

    def _synapse_list(self, window, rect, top, row_height, row_gap, margin,
                      name_width, slider_left, slider_width):
        """Список синапсов выбранной цепочки."""

        floor = None

        for marker in self.brain_map.markers:

            index, side = marker["index"], marker["side"]

            # Заголовок этажа: чувство, связь, движение.
            if marker.get("floor") and marker["floor"] != floor:

                floor = marker["floor"]

                mujoco.mjr_label(
                    mujoco.MjrRect(rect.left + margin, top, rect.width - 2 * margin, row_height),
                    FONT,
                    SENSES.get(self.state.get("chain_name"), FLOORS.get(floor, floor))
                    if floor == "sense"
                    else FLOORS.get(floor, floor),
                    *FLOOR_COLOR, 1.0, 0.75, 0.80, 0.95, window.context,
                )

                top -= row_height + row_gap

            chosen = (index, side) == (self.state["selected"], self.state["side"])

            name_rect = mujoco.MjrRect(rect.left + margin, top, name_width, row_height)
            slider_rect = mujoco.MjrRect(slider_left, top, slider_width, row_height)

            weight = self.state["weights"][side][index]
            gain = self.state["gains"][side][index]
            work = self.state["functions"][index]

            glow = float(np.clip(self.state["activity"][marker["neuron"]], 0.0, 1.0))

            base = np.asarray(ROW_CHOSEN if chosen else ROW_COLOR)
            row_color = base + (np.asarray(marker["color"]) - base) * (ACTIVE_MIX * glow)

            mujoco.mjr_label(
                name_rect, FONT, f"{marker['name']} {side} - {work}",
                *row_color, 1.0, 1.0, 1.0, 1.0, window.context,
            )

            mujoco.mjr_rectangle(slider_rect, *TRACK_COLOR, 1.0)

            filled = int(slider_width * min(gain / self.gain_max, 1.0))

            if filled > 0:
                color = np.asarray(marker["color"]) * (0.45 + 0.55 * glow)
                mujoco.mjr_rectangle(
                    mujoco.MjrRect(slider_left, top, filled, row_height),
                    *np.clip(color if chosen else color * 0.75, 0.0, 1.0), 1.0,
                )

            window.overlay(
                slider_rect, f"{weight * gain:.0f}", f"x{gain:.2f}  act {glow:.2f}",
                corner=mujoco.mjtGridPos.mjGRID_TOPLEFT,
            )

            self._rows.append((name_rect, "gain", index, side))
            self._sliders.append((slider_rect, "gain", index, side))

            top -= row_height + row_gap

    def draw(self, window, rect):

        scale = window.pixel_scale()

        row_height = int(ROW_HEIGHT * scale)
        row_gap = int(ROW_GAP * scale)
        button_height = int(BUTTON_HEIGHT * scale)
        margin = int(8 * scale)

        mujoco.mjr_rectangle(rect, *BACKGROUND, 1.0)

        # --- карта мозга ---

        map_height = int(rect.height * MAP_PART)
        self._map_rect = mujoco.MjrRect(
            rect.left, rect.bottom + rect.height - map_height, rect.width, map_height
        )

        # Камера в сцене живёт отдельно от геометрии: её обновляем каждый кадр.
        mujoco.mjv_updateCamera(
            self.brain_map.model, self.brain_map.data, self.camera, self.scene
        )

        self.scene.ngeom = self.brain_map.refresh(
            self.scene,
            self.state["weights"],
            self.state["gains"],
            self.state["activity"],
            (self.state["selected"], self.state["side"]),
        )

        mujoco.mjr_render(self._map_rect, self.scene, window.context)

        # Подписи синапсов на карте: mjr_text считает доли последнего
        # нарисованного прямоугольника, то есть карты.
        for text_x, text_y, marker in self.brain_map.label_positions(
            self._map_rect, self.camera
        ):

            chosen = (marker["index"], marker["side"]) == (
                self.state["selected"],
                self.state["side"],
            )

            mujoco.mjr_text(
                FONT,
                f"{marker['name']} {marker['side']}",
                window.context,
                text_x,
                text_y,
                *((1.0, 1.0, 1.0) if chosen else (0.72, 0.72, 0.76)),
            )

        window.overlay(
            self._map_rect,
            "brain",
            "drag = turn, wheel = zoom",
            corner=mujoco.mjtGridPos.mjGRID_TOPLEFT,
        )

        # --- список: синапсы или таблица памяти ---

        self._rows = []
        self._sliders = []

        name_width = int(rect.width * NAME_PART)
        slider_left = rect.left + margin + name_width
        slider_width = rect.left + rect.width - slider_left - margin

        top = self._map_rect.bottom - margin - row_height

        self._buttons = []

        if self.state.get("mode") == "memory":
            self._memory_table(
                window, rect, top, row_height, row_gap, margin, name_width,
                slider_left, slider_width,
            )
        else:
            self._synapse_list(
                window, rect, top, row_height, row_gap, margin, name_width,
                slider_left, slider_width,
            )

        # --- кнопки ---

        sources = self.state.get("sources", [])
        chosen = self.state.get("chosen_source", 0)

        # Верхний ряд — сразу все закладки: щёлкаешь и ты там.
        open_tab = self.state.get("active", 0)

        tabs = [
            (
                f"source{index}",
                ("> " if index == chosen else "") + spot["latin"].upper(),
                np.clip(
                    np.asarray(spot["color"])
                    * (0.75 if index == chosen else 0.30),
                    0.0,
                    1.0,
                ),
            )
            for index, spot in enumerate(sources)
        ]

        tabs.append(
            (
                "memory",
                "MEMORY",
                MEMORY_TAB_ON if self.state.get("mode") == "memory" else MEMORY_TAB,
            )
        )

        rows_of_buttons = [
            tabs,
            [
                ("punish", f"SHOCK {self.state.get('teach_name', '')}", PUNISH_COLOR),
                ("reward", f"REWARD {self.state.get('teach_name', '')}", REWARD_COLOR),
                ("forget", "FORGET", FORGET_COLOR),
            ],
            [
                (
                    "walk",
                    "STOP" if self.state["walking"] else "RUN",
                    BUTTON_ON if self.state["walking"] else BUTTON_COLOR,
                ),
                ("weights", "WEIGHTS = BRAIN", BUTTON_COLOR),
                ("musa", "RESET MUSA", BUTTON_COLOR),
                ("flip", "FLIP MUSA", BUTTON_COLOR),
                (
                    "warp",
                    f"TIME x{self.state.get('warp', 1)}",
                    BUTTON_ON if self.state.get("warp", 1) > 1 else BUTTON_COLOR,
                ),
                (
                    "out",
                    "OUT",
                    BUTTON_ON if self.state.get("errand") in ("out", "home") else BUTTON_COLOR,
                ),
                (
                    "cafe",
                    "CAFE",
                    BUTTON_ON if self.state.get("errand") == "cafe" else BUTTON_COLOR,
                ),
            ],
        ]

        bottom = rect.bottom + margin

        for line in reversed(rows_of_buttons):

            line = [item for item in line if item[0] in self.actions]

            if not line:
                continue

            button_width = (rect.width - margin * (len(line) + 1)) // len(line)
            left = rect.left + margin

            for name, text, color in line:

                button_rect = mujoco.MjrRect(left, bottom, button_width, button_height)

                mujoco.mjr_label(
                    button_rect,
                    FONT,
                    text,
                    *color,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    window.context,
                )

                self._buttons.append((button_rect, name))
                left += button_width + margin

            bottom += button_height + margin

        # --- своя жизнь Музы: над кнопками, всегда видно ---
        self._life_strip(window, rect, bottom, button_height, margin)
        self._music_menu(window, button_height, margin)

    def _music_menu(self, window, height, margin):
        """Выпадающий список треков над кнопкой MUSIC."""

        self._menu = []
        if not self.state.get("music_open") or self._music_button is None:
            return
        button = self._music_button
        step = height + margin // 2
        # Сплошной фон под списком: сквозь щели не видно строк под ним.
        mujoco.mjr_rectangle(
            mujoco.MjrRect(button.left, button.bottom + height, button.width, len(TRACKS) * step + margin // 2),
            *BACKGROUND, 1.0,
        )
        for place, (track, text) in enumerate(TRACKS):
            # Сверху вниз по порядку: ALL, потом треки.
            item = mujoco.MjrRect(button.left, button.bottom + (len(TRACKS) - place) * step, button.width, height)
            chosen = self.state.get("track") == track
            mujoco.mjr_label(
                item, FONT, ("> " if chosen else "") + text, *(MUSIC_ON if chosen else MENU_COLOR),
                1.0, 1.0, 1.0, 1.0, window.context,
            )
            self._menu.append((item, track))

    def _life_strip(self, window, rect, bottom, height, margin):
        """Своя жизнь Музы шкалами: туалет и усталость, а ниже — айфон:
        удовольствие от ленты и усталость от экрана. Сверху — музыка: выбор
        трека и громкость."""

        state = self.state
        strip = mujoco.MjrRect(rect.left, bottom, rect.width, 4 * (height + margin))
        mujoco.mjr_rectangle(strip, *BACKGROUND, 1.0)

        # Строки списка, закрытые шкалами, клики не ловят.
        top = strip.bottom + strip.height
        self._sliders = [item for item in self._sliders if item[0].bottom >= top]
        self._rows = [item for item in self._rows if item[0].bottom >= top]

        rows = (
            (
                "MUSIC",
                (0.25, 0.16, 0.28),
                (("volume", state.get("volume", 0.15), (0.70, 0.45, 0.85)),),
            ),
            (
                f"POOPS: {state.get('poop_count', 0)}   {state.get('doing', '')}",
                (0.22, 0.17, 0.12),
                (
                    ("to poop", state.get("poop", 0.0), (0.55, 0.36, 0.16)),
                    ("energy", state.get("energy", 1.0), (0.45, 0.85, 0.15)),
                ),
            ),
            (
                "FOOD & DRINK",
                (0.20, 0.22, 0.12),
                (
                    ("full", state.get("full", 0.0), (0.80, 0.45, 0.15)),
                    ("water", state.get("water", 0.0), (0.25, 0.55, 0.90)),
                ),
            ),
            (
                "JOY & WORK",
                (0.13, 0.18, 0.30),
                (
                    ("joy", state.get("joy", 0.0), (0.85, 0.35, 0.55)),
                    ("work due", state.get("work", 0.0), (0.30, 0.65, 0.65)),
                ),
            ),
        )

        # Нижняя строка — айфон, над ней — туалет и усталость.
        for level, (title, title_color, bars) in enumerate(reversed(rows)):

            row_bottom = bottom + level * (height + margin)
            left = rect.left + margin

            # Заголовок и шкалы делят строку поровну.
            third = (rect.width - (len(bars) + 2) * margin) // (len(bars) + 1)

            if title == "MUSIC":
                # Кнопка списка треков: что выбрано; щёлкнешь — список.
                self._music_button = mujoco.MjrRect(left, row_bottom, third, height)
                chosen = dict(TRACKS)[state.get("track")]
                mujoco.mjr_label(
                    self._music_button, FONT, f"MUSIC: {chosen} v",
                    *(MUSIC_ON if state.get("music_open") else title_color),
                    1.0, 1.0, 1.0, 1.0, window.context,
                )
            else:
                mujoco.mjr_label(
                    mujoco.MjrRect(left, row_bottom, third, height), FONT, title,
                    *title_color, 1.0, 1.0, 0.95, 0.85, window.context,
                )

            for place, (label, value, color) in enumerate(bars):

                bar_left = left + (third + margin) * (place + 1)
                track = mujoco.MjrRect(bar_left, row_bottom, third, height)
                mujoco.mjr_rectangle(track, *TRACK_COLOR, 1.0)
                if label == "volume":
                    self._sliders.append((track, "volume", 0, None))

                filled = int(third * min(max(value, 0.0), 1.0))
                if filled > 0:
                    mujoco.mjr_rectangle(
                        mujoco.MjrRect(bar_left, row_bottom, filled, height), *color, 1.0
                    )

                # mjr_text заметно быстрее mjr_overlay: координаты — доли
                # последнего нарисованного прямоугольника, то есть шкалы.
                mujoco.mjr_text(
                    FONT, f" {label} {min(value, 1.0):.0%}", window.context,
                    0.02, 0.3, 1.0, 1.0, 1.0,
                )
