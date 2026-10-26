"""Дом Музы: пять этажей, квартира Музы на пятом.

За входной дверью квартиры — коридор: напротив дверь соседей, рядом проход на
лестницу — перегорожен цепью с табличкой OUT OF STOCK (покрашено, не ходить),
в торце лифт с кнопкой вызова. На первом этаже такой же холл, под табличкой
банки краски, прямо по коридору от лифта — стеклянные двери на улицу.

Муза ходит сюда по кнопке OUT (musa_chain.py): коридор, кнопка, лифт, первый
этаж. Упор в стены коридора и кабины — room.walls(), кабина и двери лифта —
room.LIFT.

Рисуется только этаж, где Муза (storey): 5-й или 1-й. Изнутри видны только
квартира Музы, коридор, холл, лифт и лестница — квартиры соседей глухие.
Пока кабина едет (storey=None), дом виден снаружи, закрытый: камера
спускается вдоль окон подъезда. Стены между камерой и Музой прозрачные.
"""

import mujoco
import numpy as np

import room

T2 = 2 * room.WALL_THICK
X0, X1, Y0, BACK = room.HOUSE  # наружный контур дома
INNER = (X0 + T2, X1 - T2, Y0 + T2, BACK - T2)  # внутри наружных стен
C0, C1 = room.CORRIDOR_LOW, room.CORRIDOR_HIGH  # коридор по y
LANE = room.LANE  # середина коридора
LIFT_X = room.LIFT_X  # передняя стена шахты лифта (сторона коридора)
SHAFT = (room.CAB[0], X1 - T2)  # шахта по x
LIFT_HEIGHT = 3.8  # высота проёма лифта
NEIGHBOR_DOORS = (-4.8,)  # квартира соседей напротив квартиры Музы
STAIR_WALL = 8.4  # западная стена лестничной клетки
LANDING = 10.4  # площадка этажа: от STAIR_WALL до сюда, здесь проход из коридора
TURN_X = 14.0  # отсюда до восточной стены — площадка между этажами
STAIR_Y = (C1 + T2, C1 + T2 + 5.0)  # лестничная клетка по y
STAIR_MID = (STAIR_Y[0] + STAIR_Y[1]) / 2  # вниз по южной половине, дальше по северной
STEPS = 8  # ступенек в марше, два марша на этаж
PAINT = (0.95, 0.42, 0.08, 1.0)  # оранжевая краска в банках и в луже
# Буквы 3×5 для таблички: по строке на ряд, 1 — закрашено.
FONT = {
    "O": ("111", "101", "101", "101", "111"),
    "U": ("101", "101", "101", "101", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "F": ("111", "100", "110", "100", "100"),
    "S": ("111", "100", "111", "001", "111"),
    "C": ("111", "100", "100", "100", "111"),
    "K": ("101", "110", "100", "110", "101"),
    " ": ("000", "000", "000", "000", "000"),
}

# Рольставни в квартирах соседей: у каждого окна свой цвет. У Музы их нет.
CURTAINS = (
    (0.78, 0.22, 0.22), (0.22, 0.45, 0.75), (0.88, 0.72, 0.22),
    (0.40, 0.66, 0.36), (0.62, 0.36, 0.62), (0.92, 0.56, 0.30), (0.95, 0.95, 0.90),
)
WALL_TOP = room.WALL_HEIGHT

FACADE = (0.60, 0.61, 0.62)  # крупная серая плитка
GROUT = (0.46, 0.47, 0.48, 1.0)  # швы между плитками
TILE = (1.3, 0.9)  # плитка: ширина и высота, мм
CAR = (4.6, 1.95)  # машина во дворе: половина длины и половина ширины
STEEL = (0.62, 0.64, 0.67, 1.0)
CONCRETE = (0.60, 0.59, 0.57, 1.0)
GLASS = (0.70, 0.82, 0.90, 0.35)

level = room.level


def draw(scene, ngeom, storey=5):
    """Этаж storey (5 или 1) изнутри, или дом снаружи (None), пока едет лифт."""

    eye = room.LAST_EYE["eye"]
    box = mujoco.mjtGeom.mjGEOM_BOX
    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER

    def put(kind, size, pos, color, glow=0.0, shiny=0.3, rotation=np.eye(3)):
        nonlocal ngeom
        geom = scene.geoms[ngeom]
        mujoco.mjv_initGeom(
            geom, kind, np.array(size, dtype=float), np.array(pos, dtype=float),
            np.asarray(rotation, dtype=float).flatten(), np.array(color, dtype=float),
        )
        geom.emission = glow
        geom.specular = shiny
        ngeom += 1

    def block(ax, bx, ay, by, az, bz, color, shiny=0.3):
        """Коробка по углам."""
        if bx - ax > 1e-6 and by - ay > 1e-6 and bz - az > 1e-6:
            put(box, ((bx - ax) / 2, (by - ay) / 2, (bz - az) / 2), ((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2), color, shiny=shiny)

    def hides(axis, face, outward):
        """Камера за этой стеной — стену делаем прозрачной. axis — вдоль чего
        лежит стена, значит сравниваем другую координату."""
        across = eye[1] if axis == "x" else eye[0]
        return (across - face) * outward > 0.0

    def door(axis, face, outward, middle, floor_z, half=1.1, height=3.6, leaf=(0.42, 0.26, 0.15, 1.0)):
        """Дверь на стене: коробка, полотно, филёнка и ручка. axis — вдоль
        какой оси лежит стена, face — плоскость стены, outward — куда
        смотрит эта сторона."""

        def at(along, off, z):
            return (along, face + outward * off, z) if axis == "x" else (face + outward * off, along, z)

        def size(along, off, z):
            return (along, off, z) if axis == "x" else (off, along, z)

        panel = tuple(0.8 * c for c in leaf[:3]) + (1.0,)
        put(box, size(half + 0.12, 0.02, height / 2 + 0.06), at(middle, 0.02, floor_z + height / 2), (0.93, 0.92, 0.88, 1.0))
        put(box, size(half, 0.03, height / 2), at(middle, 0.035, floor_z + height / 2), leaf)
        put(box, size(half - 0.3, 0.035, height / 2 - 0.5), at(middle, 0.04, floor_z + height / 2 + 0.1), panel)
        put(mujoco.mjtGeom.mjGEOM_SPHERE, (0.09, 0, 0), at(middle + half - 0.3, 0.14, floor_z + 1.5), (0.85, 0.75, 0.35, 1.0), shiny=1.0)

    def street_leaf(floor_z, faded):
        """Дверь на улицу: стеклянная створка в раме на петле (открывается на
        улицу) и рядом глухая стеклянная вставка. faded — стена прозрачная,
        рама тоже бледнее."""
        frame = (0.25, 0.26, 0.28, room.SEE_THROUGH if faded else 1.0)
        x = X0 + room.WALL_THICK
        low = room.STREET_LOW
        leaf_end = low + room.STREET_LEAF
        end = leaf_end + room.STREET_GLASS
        top = floor_z + room.STREET_HEIGHT

        # Коробка: косяки, перекладина, стойка между створкой и вставкой.
        # Коробка чуть толще стены: вровень с ней грани мигали.
        for y in (low, end):
            block(X0 - 0.03, X0 + T2 + 0.03, y - 0.08, y + 0.08, floor_z, top, frame)
        block(X0 - 0.03, X0 + T2 + 0.03, low - 0.08, end + 0.08, top - 0.14, top + 0.02, frame)
        block(x - 0.06, x + 0.06, leaf_end - 0.07, leaf_end + 0.07, floor_z, top, frame)

        # Глухая вставка.
        block(x - 0.02, x + 0.02, leaf_end, end, floor_z + 0.25, top - 0.14, GLASS)
        block(x - 0.05, x + 0.05, leaf_end, end, floor_z, floor_z + 0.25, frame)

        # Створка на петле: стекло в раме, ручки-скобы с двух сторон.
        hinge, free = room.street_panel()
        along = (free - hinge) / np.linalg.norm(free - hinge)
        turn = np.array([[along[0], -along[1], 0.0], [along[1], along[0], 0.0], [0.0, 0.0, 1.0]])
        middle = (hinge + free) / 2
        height = room.STREET_HEIGHT - 0.16
        put(box, (room.STREET_LEAF / 2 - 0.08, 0.02, height / 2 - 0.25), (middle[0], middle[1], floor_z + height / 2 + 0.1), GLASS, rotation=turn)
        for off in (-1.0, 1.0):
            post = middle + along * off * (room.STREET_LEAF / 2 - 0.05)
            put(box, (0.05, 0.05, height / 2), (post[0], post[1], floor_z + height / 2), frame, rotation=turn)
        for z in (floor_z + 0.12, floor_z + height - 0.06):
            put(box, (room.STREET_LEAF / 2, 0.05, 0.08), (middle[0], middle[1], z), frame, rotation=turn)
        for side in (-1.0, 1.0):
            knob = room.street_handle(side=side)
            put(box, (0.03, 0.03, 0.55), (knob[0], knob[1], floor_z + 1.1), frame, shiny=1.0)

    def tiles(axis, face, outward, lo, hi, low, high):
        """Швы крупной плитки на наружной стороне куска фасада."""
        def at(along, z):
            return (along, face + outward * 0.004, z) if axis == "x" else (face + outward * 0.004, along, z)

        def size(along, z):
            return (along, 0.004, z) if axis == "x" else (0.004, along, z)

        # Сетка общая для всего дома (от земли и от нуля по оси), а не от
        # края куска: иначе кусок над дверью выделялся полосой.
        first_z = room.GROUND + np.ceil((low - room.GROUND) / TILE[1] + 1e-6) * TILE[1]
        for z in np.arange(first_z, high - 0.02, TILE[1]):
            put(box, size((hi - lo) / 2, 0.02), at((lo + hi) / 2, z), GROUT, shiny=0.1)
        for along in np.arange(np.ceil(lo / TILE[0] + 1e-6) * TILE[0], hi - 0.02, TILE[0]):
            put(box, size(0.02, (high - low) / 2), at(along, (low + high) / 2), GROUT, shiny=0.1)

    def curtained(axis, face, outward, middle, number, index, bare=False):
        """Окно как у Музы на этом месте фасада, с рольставней изнутри — у
        каждого окна своя. bare — окно самой Музы: без штор."""

        def at(along, off, z):
            return (along, face + outward * off, z) if axis == "x" else (face + outward * off, along, z)

        def size(along, off, z):
            return (along, off, z) if axis == "x" else (off, along, z)

        half, height = room.WINDOW_HALF, 3.0
        bottom = level(number) + room.SILL_TOP
        white = (0.97, 0.97, 0.95, 1.0)
        put(box, size(half, 0.02, height / 2), at(middle, 0.02, bottom + height / 2), (0.25, 0.28, 0.32, 1.0), shiny=0.9)
        for side in (-1.0, 1.0):
            put(box, size(0.08, 0.03, height / 2 + 0.08), at(middle + side * half, 0.03, bottom + height / 2), white)
        put(box, size(half + 0.08, 0.03, 0.08), at(middle, 0.03, bottom + height), white)
        put(box, size(half + 0.2, 0.2, 0.05), at(middle, 0.2, bottom - 0.05), white)
        if bare:
            return
        # Горизонтальная рольставня изнутри: короб сверху, полотно из
        # ламелей опущено у каждого окна по-своему.
        color = CURTAINS[(number * 3 + index) % len(CURTAINS)]
        dark = (*(0.75 * c for c in color), 1.0)
        drop = (0.3, 0.55, 0.8, 0.45, 0.65)[(number + 2 * index) % 5] * height
        top = bottom + height - 0.1
        put(box, size(half - 0.02, 0.03, drop / 2), at(middle, 0.03, top - drop / 2), (*color, 1.0))
        for z in np.arange(top - 0.18, top - drop, -0.18):
            put(box, size(half - 0.02, 0.035, 0.012), at(middle, 0.03, z), dark)
        put(box, size(half - 0.02, 0.04, 0.03), at(middle, 0.03, top - drop), dark)
        put(box, size(half, 0.05, 0.12), at(middle, 0.04, bottom + height - 0.12), (0.90, 0.90, 0.88, 1.0))

    def letters(lines, middle_x, face_y, middle_z, pixel):
        """Надпись красными буквами на табличке (лицом к коридору): буквы
        из квадратиков 3×5, строки по центру. Подряд идущие квадратики строки
        — одной коробкой."""
        red = (0.85, 0.08, 0.08, 1.0)
        height = len(lines) * 6 - 1
        for number, line in enumerate(lines):
            width = len(line) * 4 - 1
            left = middle_x - width * pixel / 2
            top = middle_z + height * pixel / 2 - number * 6 * pixel
            for row_index in range(5):
                bits = "0".join(FONT[char][row_index] for char in line)
                z = top - (row_index + 0.5) * pixel
                start = None
                for column, bit in enumerate(bits + "0"):
                    if bit == "1" and start is None:
                        start = column
                    elif bit != "1" and start is not None:
                        block(left + start * pixel, left + column * pixel, face_y - 0.01, face_y, z - pixel / 2, z + pixel / 2, red)
                        start = None

    def row(axis, face, outward, start, end, number):
        """Окна этажа на фасаде — как у Музы: на её стороне на тех же местах,
        на стороне соседей — зеркально через коридор. У всех шторы, только у
        самой Музы (5-й этаж, её сторона) — без штор."""
        mirror = lambda y: 2 * LANE - y
        copies = {
            ("x", Y0): [(room.WINDOW_X, 0, True)],
            ("x", BACK): [(room.WINDOW_X, 3, False)],
            ("y", X0): [(room.PHONE_WINDOW, 1, True), (mirror(room.PHONE_WINDOW), 4, False)],
            ("y", X1): [(room.KITCHEN_WINDOW, 2, True), (mirror(room.KITCHEN_WINDOW), 5, False)],
        }
        for middle, index, musa_side in copies[(axis, face)]:
            if start <= middle <= end:
                curtained(axis, face, outward, middle, number, index, bare=musa_side and number == 5)
        # В торце коридора напротив лифта — окно подъезда, без штор, снаружи
        # и изнутри. На первом этаже там стеклянные двери на улицу.
        if axis == "y" and face == X0 and number >= 2 and start <= LANE <= end:
            curtained(axis, face, outward, LANE, number, 0, bare=True)
            curtained(axis, X0 + T2, 1.0, LANE, number, 0, bare=True)

    sides = (
        # ось, плоскость, наружу, от, до, толщина стены (от, до)
        ("x", Y0, -1.0, X0, X1, (Y0, Y0 + T2)),
        ("x", BACK, 1.0, X0, X1, (BACK - T2, BACK)),
        ("y", X0, -1.0, Y0, BACK, (X0, X0 + T2)),
        ("y", X1, 1.0, Y0, BACK, (X1 - T2, X1)),
    )

    def outer_wall(axis, face, outward, a, b, lo, hi, low, high, color):
        """Кусок фасада из серой плитки. В западной стене первого этажа —
        проём двери на улицу."""
        pieces = [(lo, hi, low, high)]
        door_lo = room.STREET_LOW
        door_hi = door_lo + room.STREET_LEAF + room.STREET_GLASS
        if axis == "y" and face == X0 and low <= room.GROUND < high and lo < door_lo and door_hi < hi:
            door_top = room.GROUND + room.STREET_HEIGHT
            # Над дверью — одна плита во всю ширину, без отдельного столба.
            pieces = [(lo, door_lo, low, door_top), (door_hi, hi, low, door_top), (lo, hi, door_top, high)]
        for lo, hi, low, high in pieces:
            if axis == "x":
                block(lo, hi, a, b, low, high, color)
            else:
                block(a, b, lo, hi, low, high, color)
            if color[3] >= 1.0:
                tiles(axis, face, outward, lo, hi, low, high)

    def parked_car(x, y, facing, color):
        """Электромобиль во дворе, в размер Музы — в салон она поместится.
        Гладкий низкий кузов с закруглённым носом, тонированная стеклянная
        крыша, сиденья видно сквозь стёкла, светодиодные полосы через весь
        перед и зад, колёса с гладкими колпаками, огонёк зарядного разъёма.
        facing — куда нос: +1 на восток, −1 на запад."""
        g = room.GROUND
        half, width = CAR
        body = (*color, 1.0)
        glass = (0.35, 0.45, 0.55, 0.35)
        seat = (0.18, 0.18, 0.20, 1.0)
        # Кузов обтекаемый: середина, боковины-капсулы (скруглены сверху и
        # снизу), нос и корма закруглены и в плане.
        along_x = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        edge = 0.45
        block(x - half + 0.6, x + half - 0.6, y - width + edge, y + width - edge, g + 0.4, g + 1.3, body, shiny=1.0)
        for side in (-1.0, 1.0):
            put(mujoco.mjtGeom.mjGEOM_CAPSULE, (edge, half - 0.6, 0), (x, y + side * (width - edge), g + 0.85), body, shiny=1.0, rotation=along_x)
        for end in (-1.0, 1.0):
            put(mujoco.mjtGeom.mjGEOM_ELLIPSOID, (0.9, width, 0.45), (x + end * (half - 0.6), y, g + 0.85), body, shiny=1.0)
        # Салон: боковые стёкла, наклонные лобовое и заднее, плоская
        # прямоугольная крыша цвета кузова.
        middle = x - facing * 0.4
        roof = g + 2.45
        block(middle - 1.9, middle + 1.9, y - width + 0.25, y + width - 0.25, g + 1.3, roof, glass, shiny=1.0)
        for lean in (1.0, -1.0):
            tilt = np.radians(40.0) * lean
            slope = np.array([[np.cos(tilt), 0.0, np.sin(tilt)], [0.0, 1.0, 0.0], [-np.sin(tilt), 0.0, np.cos(tilt)]])
            put(box, (0.04, width - 0.3, 0.72), (middle + lean * 2.33, y, g + 1.85), glass, shiny=1.0, rotation=slope)
        block(middle - 2.0, middle + 2.0, y - width + 0.2, y + width - 0.2, roof, roof + 0.12, body, shiny=1.0)
        # Сиденья: спинки к корме.
        front = x - facing * 0.1
        for side in (-0.95, 0.95):
            block(front - 0.6, front + 0.6, y + side - 0.6, y + side + 0.6, g + 1.3, g + 1.6, seat)
            back = front - facing * 0.65
            block(back - 0.12, back + 0.12, y + side - 0.5, y + side + 0.5, g + 1.3, g + 2.0, seat)
        rear = x - facing * 1.55
        block(rear - 0.45, rear + 0.45, y - width + 0.6, y + width - 0.6, g + 1.3, g + 1.6, seat)
        block(rear - facing * 0.5 - 0.1, rear - facing * 0.5 + 0.1, y - width + 0.7, y + width - 0.7, g + 1.3, g + 1.85, seat)
        # Колёса с гладкими колпаками.
        wheel = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        for along in (-half + 1.5, half - 1.5):
            for side in (-1.0, 1.0):
                put(cylinder, (0.55, 0.2, 0), (x + along, y + side * (width - 0.1), g + 0.55), (0.08, 0.08, 0.09, 1.0), rotation=wheel)
                put(cylinder, (0.38, 0.03, 0), (x + along, y + side * (width + 0.12), g + 0.55), (0.78, 0.80, 0.83, 1.0), shiny=1.0, rotation=wheel)
        # Светодиодные полосы: белая спереди, красная сзади — через весь кузов.
        nose, tail = x + facing * (half + 0.28), x - facing * (half + 0.28)
        block(nose - 0.04, nose + 0.04, y - width + 0.3, y + width - 0.3, g + 1.0, g + 1.08, (0.95, 0.97, 1.0, 1.0))
        block(tail - 0.04, tail + 0.04, y - width + 0.3, y + width - 0.3, g + 1.05, g + 1.13, (0.95, 0.1, 0.12, 1.0))
        for index in range(ngeom - 2, ngeom):
            scene.geoms[index].emission = 0.8
        # Огонёк зарядного разъёма на заднем крыле.
        put(mujoco.mjtGeom.mjGEOM_SPHERE, (0.09, 0, 0), (x - facing * (half - 1.0), y - width - 0.02, g + 1.05), (0.2, 0.8, 1.0, 1.0), glow=1.0)

    def yard():
        """Двор перед подъездом: мощение, дорожка от двери вправо вдоль дома
        и к воротам в углу, стоянка на 4 машины вдоль дорожки, забор с туями,
        раздвижные ворота с кнопкой на столбике."""
        g = room.GROUND
        fence_x, house_x = room.FENCE_X, X0
        s0, s1 = room.SPINE
        b0, b1 = room.BOTTOM
        a0, a1 = room.APRON
        block(fence_x, house_x, Y0, BACK, g, g + 0.01, (0.40, 0.40, 0.42, 1.0), shiny=0.1)
        path = (0.58, 0.55, 0.50, 1.0)
        block(s0, house_x, a0, a1, g + 0.01, g + 0.02, path, shiny=0.1)
        block(s0, s1, b0, a1, g + 0.01, g + 0.02, path, shiny=0.1)
        block(fence_x, s1, b0, b1, g + 0.01, g + 0.02, path, shiny=0.1)

        # Стоянка как во дворе: две машины справа от двери носом к дому, две
        # через дорожку носом к забору — выезжают задом на дорожку. Дорожка
        # к ним продолжается на север.
        block(s0, s1, a1, a1 + 12.0, g + 0.01, g + 0.02, path, shiny=0.1)
        white = (0.92, 0.92, 0.88, 1.0)
        spots = (
            (house_x - 0.4 - CAR[0], a0 - 2.9, 1.0, (0.75, 0.75, 0.78)),
            (house_x - 0.4 - CAR[0], a0 - 7.9, 1.0, (0.15, 0.25, 0.45)),
            (s0 - 0.3 - CAR[0], a1 + 3.4, -1.0, (0.55, 0.10, 0.12)),
            (s0 - 0.3 - CAR[0], a1 + 8.4, -1.0, (0.20, 0.45, 0.30)),
        )
        for x, y, facing, color in spots:
            for side in (-2.5, 2.5):
                block(x - CAR[0] - 0.3, x + CAR[0] + 0.3, y + side - 0.05, y + side + 0.05, g + 0.01, g + 0.025, white)
            parked_car(x, y, facing, color)

        # Забор: столбы, две перекладины и сетка; в западном — ворота.
        green = (0.18, 0.26, 0.22, 1.0)
        mesh = (0.18, 0.26, 0.22, 0.35)
        height = 2.6
        runs = [((fence_x, Y0), (house_x, Y0)), ((fence_x, BACK), (house_x, BACK)),
                ((fence_x, Y0), (fence_x, b0)), ((fence_x, b1), (fence_x, BACK))]
        for (ax, ay), (bx, by) in runs:
            length = abs(bx - ax) + abs(by - ay)
            if ay == by:
                for x in np.linspace(ax, bx, max(2, int(length / 2.0) + 1)):
                    block(x - 0.06, x + 0.06, ay - 0.06, ay + 0.06, g, g + height, green)
                for z in (g + 0.8, g + height - 0.1):
                    block(ax, bx, ay - 0.03, ay + 0.03, z - 0.04, z + 0.04, green)
                block(ax, bx, ay - 0.01, ay + 0.01, g + 0.1, g + height - 0.1, mesh)
            else:
                for y in np.linspace(ay, by, max(2, int(length / 2.0) + 1)):
                    block(ax - 0.06, ax + 0.06, y - 0.06, y + 0.06, g, g + height, green)
                for z in (g + 0.8, g + height - 0.1):
                    block(ax - 0.03, ax + 0.03, ay, by, z - 0.04, z + 0.04, green)
                block(ax - 0.01, ax + 0.01, ay, by, g + 0.1, g + height - 0.1, mesh)

        # Туи вдоль всего забора изнутри (кроме ворот): ярусами, как живые,
        # чуть разной высоты и оттенка.
        rng = np.random.default_rng(11)

        def thuja(x, y):
            tall = rng.uniform(1.6, 2.3)
            shade = rng.uniform(0.85, 1.15)
            block(x - 0.07, x + 0.07, y - 0.07, y + 0.07, g, g + 0.3, (0.35, 0.24, 0.14, 1.0))
            for tier in range(4):
                part = tier / 4
                radius = 0.62 * (1.0 - 0.72 * part)
                z = g + 0.35 + tall * (0.2 + 0.72 * part)
                green_tone = (0.09 * shade, (0.27 + 0.05 * tier) * shade, 0.13 * shade, 1.0)
                put(mujoco.mjtGeom.mjGEOM_ELLIPSOID, (radius, radius, 0.42 * tall * (1.0 - 0.3 * part)), (x, y, z), green_tone, shiny=0.05)

        for x in np.arange(fence_x + 0.9, house_x - 0.5, 1.3):
            thuja(x, Y0 + 0.7)
            thuja(x, BACK - 0.7)
        for y in np.arange(Y0 + 0.7, BACK - 0.5, 1.3):
            if not b0 - 0.6 < y < b1 + 2 * room.YARD_GATE + 0.6:  # у ворот и там, куда они отъезжают, туй нет
                thuja(fence_x + 0.7, y)

        # Клумба слева от входа вдоль всей стены дома: бортик, земля, кусты
        # в два ряда с цветами.
        bed = (house_x - 4.8, house_x - 0.2, a1 + 0.3, BACK - 1.6)
        block(bed[0], bed[1], bed[2], bed[3], g, g + 0.3, (0.62, 0.62, 0.60, 1.0))
        block(bed[0] + 0.12, bed[1] - 0.12, bed[2] + 0.12, bed[3] - 0.12, g + 0.3, g + 0.34, (0.28, 0.18, 0.10, 1.0))
        for row_x, offset in ((bed[0] + 1.25, 0.0), (bed[1] - 1.25, 1.1)):
            for number, y in enumerate(np.arange(bed[2] + 1.2 + offset, bed[3] - 1.0, 2.2)):
                size = rng.uniform(1.0, 1.4)
                put(mujoco.mjtGeom.mjGEOM_ELLIPSOID, (size, size, size * 0.8), (row_x, y, g + 0.34 + size * 0.7), (0.16, 0.42 * rng.uniform(0.9, 1.1), 0.18, 1.0), shiny=0.05)
                for petal in range(6):
                    angle = rng.uniform(0, 2 * np.pi)
                    bloom = ((0.95, 0.25, 0.35), (0.98, 0.85, 0.2), (0.85, 0.45, 0.9), (1.0, 1.0, 1.0))[(number + petal) % 4]
                    put(mujoco.mjtGeom.mjGEOM_SPHERE, (0.16, 0, 0),
                        (row_x + np.cos(angle) * size * 0.8, y + np.sin(angle) * size * 0.8, g + 0.34 + size * 1.1), (*bloom, 1.0), shiny=0.2)

        # Раздвижные ворота: решётка на рельсе, отъезжает вдоль забора на
        # север. Кнопка на столбике у ворот.
        gate = room.YARD_GATE
        shift = room.GATE["open"] * 2 * gate
        block(fence_x - 0.1, fence_x + 0.1, b0, b1 + 2 * gate, g, g + 0.03, (0.30, 0.30, 0.32, 1.0))
        g0, g1 = b0 + shift, b1 + shift
        block(fence_x - 0.18, fence_x - 0.08, g0, g1, g + 0.15, g + 0.25, green)
        block(fence_x - 0.18, fence_x - 0.08, g0, g1, g + height - 0.2, g + height - 0.1, green)
        for y in np.arange(g0 + 0.2, g1, 0.35):
            block(fence_x - 0.16, fence_x - 0.1, y - 0.03, y + 0.03, g + 0.15, g + height - 0.1, green)
        bx, by, bz = room.GATE_BUTTON
        block(bx - 0.08, bx + 0.08, by - 0.08, by + 0.08, g, g + bz + 0.35, (0.30, 0.31, 0.33, 1.0))
        block(bx + 0.08, bx + 0.1, by - 0.12, by + 0.12, g + bz - 0.15, g + bz + 0.15, (0.20, 0.21, 0.23, 1.0))
        put(cylinder, (0.08, 0.02, 0), (bx + 0.12, by, g + bz), (0.2, 0.9, 0.3, 1.0), glow=1.0 if room.GATE["called"] else 0.3,
            rotation=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]))

    def street_door(floor_z, faded):
        street_leaf(floor_z, faded)
        if not faded:
            put(mujoco.mjtGeom.mjGEOM_SPHERE, (0.36, 0, 0), (X0 - 0.4, LANE, floor_z + room.STREET_HEIGHT + 0.8), (1.0, 0.92, 0.70, 1.0), glow=0.7)

    # --- Лифт едет: дом снаружи, закрытый, с крышей — камера спускается
    # вдоль окон подъезда.
    if storey is None:
        for axis, face, outward, start, end, (a, b) in sides:
            outer_wall(axis, face, outward, a, b, start, end, room.GROUND, WALL_TOP, (*FACADE, 1.0))
            for number in range(1, 6):
                row(axis, face, outward, start, end, number)
        block(X0, X1, Y0, BACK, WALL_TOP, WALL_TOP + 0.2, CONCRETE)
        street_door(level(1), False)
        yard()
        return ngeom

    floor_z = level(storey)
    top_z = floor_z + WALL_TOP

    # --- Фасад.
    for axis, face, outward, start, end, (a, b) in sides:
        behind = hides(axis, face, outward)
        color = (*FACADE, room.SEE_THROUGH if behind else 1.0)
        if storey == 5:
            # Тело дома под квартирой — глухое, с окнами четырёх этажей и
            # стеклянными дверями внизу.
            outer_wall(axis, face, outward, a, b, start, end, room.GROUND, 0.0, (*FACADE, 1.0))
            for lower in range(1, 5):
                row(axis, face, outward, start, end, lower)
            if axis == "y" and face == X0:
                street_door(level(1), False)
            # Пятый этаж: у квартиры свои стены, здесь — коридор и соседи.
            if axis == "x" and face == Y0:
                continue
            lo = start if axis == "x" else C0
            outer_wall(axis, face, outward, a, b, lo, end, 0.0, WALL_TOP, color)
            if not behind:
                row(axis, face, outward, lo, end, 5)
        else:
            # Этажи выше — глухой фасад с окнами, как дом под квартирой на
            # пятом. Первый этаж: стена со стороны камеры прозрачна только
            # напротив холла, остальное — плитка с окнами.
            upper = level(2)
            outer_wall(axis, face, outward, a, b, start, end, upper, WALL_TOP, (*FACADE, 1.0))
            for number in range(2, 6):
                row(axis, face, outward, start, end, number)
            if face == Y0:
                block(X0, X1, Y0, BACK, WALL_TOP, WALL_TOP + 0.2, CONCRETE)  # крыша
            # Первый этаж — сплошной: пока Муза в холле, камера внутри.
            outer_wall(axis, face, outward, a, b, start, end, floor_z, upper, (*FACADE, 1.0))
            row(axis, face, outward, start, end, storey)

    # --- Пол этажа. На 5-м в нём дыра лестницы вниз, под ней пол 4-го.
    if storey == 5:
        for ax, bx, ay, by in (
            (INNER[0], INNER[1], INNER[2], STAIR_Y[0]),
            (INNER[0], LANDING, STAIR_Y[0], STAIR_Y[1]),
            (INNER[0], INNER[1], STAIR_Y[1], INNER[3]),
        ):
            block(ax, bx, ay, by, -0.51, -0.01, CONCRETE)
        pit = level(4)
        block(LANDING, INNER[1], STAIR_Y[0], STAIR_Y[1], pit - 0.5, pit, CONCRETE)
        # Стенки колодца лестницы до 4-го этажа.
        block(STAIR_WALL - room.WALL_THICK, STAIR_WALL, STAIR_Y[0], STAIR_Y[1], pit, 0.0, (*room.WALL_COLOR, 1.0))
        block(STAIR_WALL, INNER[1], STAIR_Y[1], STAIR_Y[1] + room.WALL_THICK, pit, 0.0, (*room.WALL_COLOR, 1.0))
        block(LANDING, INNER[1], C1, STAIR_Y[0], pit, -0.51, (*room.WALL_COLOR, 1.0))
    else:
        block(INNER[0], INNER[1], INNER[2], INNER[3], floor_z - 0.51, floor_z - 0.01, CONCRETE)

    # --- Глухие объёмы: квартиры соседей, на первом — ещё квартира под
    # квартирой Музы. Внутрь не заглянуть, сверху крыша. Лестница открыта.
    closed = (*room.WALL_COLOR, 1.0)
    for ax, bx, ay, by in (
        (INNER[0], STAIR_WALL, C1 + T2, INNER[3]),
        (STAIR_WALL, INNER[1], STAIR_Y[1], INNER[3]),
    ):
        block(ax, bx, ay, by, floor_z, top_z, closed)
        block(ax, bx, ay, by, top_z, top_z + 0.15, CONCRETE)
    if storey != 5:
        block(INNER[0], INNER[1], INNER[2], C0 - T2, floor_z, top_z, closed)
        block(INNER[0], INNER[1], INNER[2], C0 - T2, top_z, top_z + 0.15, CONCRETE)

    # --- Стены коридора этого этажа.
    wall = (*room.WALL_COLOR, 1.0)
    block(INNER[0], STAIR_WALL, C1, C1 + T2, floor_z, top_z, wall)
    block(LANDING, LIFT_X, C1, C1 + T2, floor_z, top_z, wall)
    block(STAIR_WALL, LANDING, C1, C1 + T2, floor_z + 4.2, top_z, wall)

    # Южная стена коридора: на 5-м у квартиры Музы своя, здесь — вдоль кухни.
    if storey == 5:
        south_seen = hides("x", C0 - room.WALL_THICK, -1.0)
        block(room.HALF + T2, INNER[1], C0 - T2, C0, floor_z, top_z, (*room.WALL_COLOR, room.SEE_THROUGH if south_seen else 1.0))
    else:
        block(INNER[0], INNER[1], C0 - T2, C0, floor_z, top_z, wall)

    # Шахта лифта: передняя стена с проёмом, северная стенка.
    lift_seen = hides("y", LIFT_X + room.WALL_THICK / 2, 1.0)
    lift_wall = (*room.WALL_COLOR, room.SEE_THROUGH if lift_seen else 1.0)
    block(LIFT_X, SHAFT[0], C0, LANE - room.LIFT_DOOR, floor_z, top_z, lift_wall)
    block(LIFT_X, SHAFT[0], LANE + room.LIFT_DOOR, C1 + T2, floor_z, top_z, lift_wall)
    block(LIFT_X, SHAFT[0], LANE - room.LIFT_DOOR, LANE + room.LIFT_DOOR, floor_z + LIFT_HEIGHT, top_z, lift_wall)
    block(SHAFT[0], SHAFT[1], C1, C1 + T2, floor_z, top_z, wall)

    # Пол коридора: линолеум полосами.
    for number, y in enumerate(np.linspace(C0 + 0.5, C1 - 0.5, 5)):
        tone = 0.62 if number % 2 else 0.57
        put(box, ((LIFT_X - INNER[0]) / 2, 0.49, 0.0008), ((INNER[0] + LIFT_X) / 2, y, floor_z + 0.004), (tone, tone * 0.97, tone * 0.9, 1.0))

    # Двери: соседи напротив; внизу ещё квартира под Музой.
    for x in NEIGHBOR_DOORS:
        door("x", C1, -1.0, x, floor_z, half=1.2)
    if storey != 5:
        door("x", C0, 1.0, room.FRONT_DOOR_X, floor_z)

    # Лифт: две стальные створки разъезжаются, когда кабина на этом этаже и
    # двери открыты.
    cab_here = abs(room.LIFT["z"] - floor_z) < 1e-6
    opened = room.LIFT["doors"] if cab_here else 0.0
    leaf = room.LIFT_DOOR / 2
    for side in (-1.0, 1.0):
        middle = LANE + side * (leaf + opened * (room.LIFT_DOOR - 0.1))
        # Створки чуть шире и одна перед другой: закрытые заходят друг за
        # друга, без щели, сквозь которую видно кабину.
        put(box, (0.03, leaf + 0.05, LIFT_HEIGHT / 2), (LIFT_X + room.WALL_THICK + 0.02 * side, middle, floor_z + LIFT_HEIGHT / 2), STEEL, shiny=1.0)
    if not lift_seen:
        face = LIFT_X
        for side in (-1.0, 1.0):
            put(box, (0.03, 0.06, LIFT_HEIGHT / 2), (face - 0.03, LANE + side * (room.LIFT_DOOR + 0.06), floor_z + LIFT_HEIGHT / 2), (0.45, 0.47, 0.50, 1.0), shiny=0.8)
        put(box, (0.03, room.LIFT_DOOR + 0.12, 0.06), (face - 0.03, LANE, floor_z + LIFT_HEIGHT + 0.06), (0.45, 0.47, 0.50, 1.0), shiny=0.8)
        # Кнопка вызова на высоте поднятой лапки: горит ярче, пока лифт едет сюда.
        button = room.lift_button(storey)
        called = room.LIFT["called"] is not None and abs(room.LIFT["called"] - floor_z) < 1e-6
        put(box, (0.02, 0.22, 0.3), (face - 0.02, button[1], button[2]), (0.30, 0.31, 0.33, 1.0), shiny=0.9)
        put(cylinder, (0.09, 0.03, 0), (face - 0.05, button[1], button[2]), (1.0, 0.55, 0.15, 1.0), glow=1.0 if called else 0.3)
        put(box, (0.02, 0.4, 0.12), (face - 0.03, LANE, floor_z + LIFT_HEIGHT + 0.4), (0.08, 0.08, 0.09, 1.0), shiny=0.9)
        put(box, (0.02, 0.08, 0.08), (face - 0.05, LANE, floor_z + LIFT_HEIGHT + 0.4), (1.0, 0.35, 0.1, 1.0), glow=1.0 if cab_here else 0.2)

    # Кабина, когда она на этом этаже: пол, стенки, потолок, тросы;
    # направляющие в шахте.
    if cab_here:
        cab = (SHAFT[0] + 0.05, room.CAB[1] + 0.06, room.CAB[2] - 0.06, room.CAB[3] + 0.06)
        block(cab[0], cab[1], cab[2], cab[3], floor_z, floor_z + 0.1, (0.35, 0.33, 0.30, 1.0))
        block(cab[1] - 0.06, cab[1], cab[2], cab[3], floor_z, floor_z + 4.2, STEEL)
        block(cab[0], cab[1], cab[2], cab[2] + 0.06, floor_z, floor_z + 4.2, STEEL)
        block(cab[0], cab[1], cab[3] - 0.06, cab[3], floor_z, floor_z + 4.2, STEEL)
        block(cab[0], cab[1], cab[2], cab[3], floor_z + 4.2, floor_z + 4.3, STEEL)
        for side in (-0.3, 0.3):
            put(cylinder, (0.03, (WALL_TOP - 4.3) / 2, 0), ((cab[0] + cab[1]) / 2, LANE + side, floor_z + (4.3 + WALL_TOP) / 2), (0.15, 0.15, 0.16, 1.0))
    for y in (room.CAB[2] - 0.14, room.CAB[3] + 0.14):
        block(SHAFT[1] - 0.25, SHAFT[1] - 0.1, y - 0.05, y + 0.05, floor_z, top_z, (0.30, 0.30, 0.32, 1.0))

    # --- Лестница: на 5-м — вниз до 4-го, на 1-м — вверх до 2-го.
    flight_top = floor_z if storey > 1 else level(2)
    rise = room.STOREY / 2 / STEPS
    run = (TURN_X - LANDING) / STEPS
    for step in range(STEPS):
        top = flight_top - (step + 1) * rise
        block(LANDING + step * run, LANDING + (step + 1) * run, STAIR_Y[0], STAIR_MID, top - 1.5 * rise, top, CONCRETE)
        top2 = flight_top - room.STOREY / 2 - (step + 1) * rise
        block(TURN_X - (step + 1) * run, TURN_X - step * run, STAIR_MID, STAIR_Y[1], top2 - 1.5 * rise, top2, CONCRETE)
    middle = flight_top - room.STOREY / 2
    block(TURN_X, INNER[1], STAIR_Y[0], STAIR_Y[1], middle - 0.3, middle, CONCRETE)
    block(LANDING, TURN_X, STAIR_MID - 0.04, STAIR_MID + 0.04, flight_top - room.STOREY, flight_top + 1.0, (0.20, 0.20, 0.22, 1.0))
    if storey == 1:
        # Стенки лестницы до 2-го этажа: выше стен холла.
        block(STAIR_WALL - room.WALL_THICK, STAIR_WALL, STAIR_Y[0], STAIR_Y[1], top_z, level(2) + 1.0, closed)
        block(STAIR_WALL, INNER[1], STAIR_Y[1], STAIR_Y[1] + room.WALL_THICK, top_z, level(2) + 1.0, closed)

    # Проход на лестницу перегорожен цепью, на ней табличка.
    chain_y = C1 - 0.05
    for number in range(13):
        part = number / 12
        x = STAIR_WALL + (LANDING - STAIR_WALL) * part
        sag = 0.5 * (1.0 - (2 * part - 1.0) ** 2)
        put(mujoco.mjtGeom.mjGEOM_SPHERE, (0.05, 0, 0), (x, chain_y, floor_z + 2.2 - sag), (0.72, 0.62, 0.30, 1.0), shiny=1.0)
    middle_x, sign_z = (STAIR_WALL + LANDING) / 2, floor_z + 1.2
    for side in (-0.6, 0.6):
        block(middle_x + side - 0.02, middle_x + side + 0.02, chain_y - 0.02, chain_y + 0.02, sign_z + 0.45, floor_z + 2.2 - 0.5 * (1 - side ** 2), (0.72, 0.62, 0.30, 1.0))
    put(box, (0.9, 0.02, 0.45), (middle_x, chain_y - 0.03, sign_z), (0.92, 0.18, 0.15, 1.0), shiny=0.5)
    put(box, (0.84, 0.022, 0.39), (middle_x, chain_y - 0.04, sign_z), (0.97, 0.95, 0.90, 1.0))
    letters(("OUT OF", "STOCK"), middle_x, chain_y - 0.063, sign_z, 0.065)

    # Стеклянные двери на улицу на первом этаже: прямо по коридору от лифта.
    # Под табличкой — две банки краски: одна опрокинута и разлилась, на
    # другой лежит кисточка.
    if storey == 1:
        street_door(floor_z, False)
        can_x, can_y = STAIR_WALL + 0.55, C1 + 0.3
        put(cylinder, (0.26, 0.28, 0), (can_x, can_y, floor_z + 0.28), (0.72, 0.74, 0.78, 1.0), shiny=0.9)
        put(cylinder, (0.27, 0.02, 0), (can_x, can_y, floor_z + 0.57), (0.55, 0.57, 0.60, 1.0), shiny=0.9)
        put(cylinder, (0.20, 0.281, 0), (can_x, can_y, floor_z + 0.28), PAINT)  # этикетка цвета краски
        # Кисточка поперёк банки: ручка и щетина в краске.
        block(can_x - 0.45, can_x + 0.05, can_y - 0.04, can_y + 0.04, floor_z + 0.59, floor_z + 0.65, (0.55, 0.35, 0.18, 1.0))
        block(can_x + 0.05, can_x + 0.3, can_y - 0.09, can_y + 0.09, floor_z + 0.58, floor_z + 0.66, PAINT)
        # Опрокинутая банка и лужа краски от неё в коридор.
        lying = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        tipped = (can_x + 0.95, can_y + 0.05, floor_z + 0.26)
        mujoco.mjv_initGeom(scene.geoms[ngeom], cylinder, np.array([0.26, 0.28, 0.0]), np.array(tipped), lying.flatten(), np.array([0.72, 0.74, 0.78, 1.0]))
        scene.geoms[ngeom].specular = 0.9
        ngeom += 1
        put(mujoco.mjtGeom.mjGEOM_ELLIPSOID, (0.75, 0.55, 0.012), (can_x + 1.35, can_y - 0.35, floor_z + 0.012), PAINT, shiny=1.0)

    yard()
    return ngeom
