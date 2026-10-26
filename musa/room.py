"""Комната Музы: стены, ковёр и матрас в углу.

Стены держит упор (hold_inside), а сворачивать от них Муза начинает заранее
(obstacle_push). Матрас мягкий: лапки чуть в нём утопают. Ковёр, подушка, унитаз
и какашки — только картинка.

Стены, которые загораживают камеру, становятся почти прозрачными, как в
кукольном домике: иначе, стоит Музе подойти к стене, её было бы не видно.
"""

import mujoco
import numpy as np

HALF = 9.0  # мм: от центра до стены

# Кухня — ниша в северной стене: проём шириной 2 * NOOK_HALF, глубина до NOOK_X.
NOOK_HALF = 7.0
NOOK_X = 16.0
WALL_HEIGHT = 6.0
WALL_THICK = 0.3

MATTRESS_LOW = np.array([-8.4, 3.4])  # угол матраса ближе к центру комнаты
MATTRESS_HIGH = np.array([-3.2, 8.4])  # от стен отступ: у стены не застревает
MATTRESS_TOP = 0.08  # высота матраса: мягкий, лапки чуть утопают

KEEP = 1.6  # мм: ближе этого грудь к стене не подходит — голова как раз у стены

CARPET_BASE = 0.006  # ковёр лежит поверх досок
CARPET_STEP = 0.006  # мм между слоями ковра (точности глубины хватает, см. znear)
CARPET = (1.6, 0.0)  # середина ковра: не заходит ни в ванную, ни под матрас

# Туалет у южной стены, смотрит в комнату (по +x).
TOILET = np.array([-7.7, -6.25])
SEAT_HEIGHT = 1.45  # мм: верх сиденья
TOILET_BODY = TOILET + np.array([0.1, 0.0])  # середина унитаза с чашей, если смотреть сверху
TOILET_RADIUS = 1.2  # до переднего края чаши
# Куда Муза подходит перед тем, как сесть: снаружи упора унитаза (радиус
# плюс лоб 1.2), иначе туда не дойти — она съезжала вдоль унитаза и застревала.
TOILET_FRONT = TOILET_BODY + np.array([TOILET_RADIUS + 1.2 + 0.2, 0.0])

# Ванная: угловая комнатка в углу (-HALF, -HALF). Две стены наружные,
# две — перегородки: северная (y = BATH_HIGH) и восточная с дверью (x = BATH_X).
BATH_X = -4.0  # перегородка с дверью
BATH_LOW, BATH_HIGH = -HALF, -3.5
BATH_WEST = -HALF  # западная стена ванной
DOOR_WIDTH = 2.4
DOOR_LOW, DOOR_HIGH = TOILET[1] - DOOR_WIDTH / 2, TOILET[1] + DOOR_WIDTH / 2  # проём напротив унитаза
HINGE = np.array([BATH_X, DOOR_LOW])  # петля у южного косяка, дверь открывается наружу
JAMB_KEEP = 0.8  # мм: у косяков и у полотна двери упор меньше, иначе в узкий проём не пройти
PART_HEIGHT = 4.0  # высота перегородок
DOOR_HEIGHT = 3.4
DOOR_OUTSIDE = np.array([BATH_X + 1.8, TOILET[1]])  # перед проёмом снаружи
# Открывает сбоку — вне круга, который заметает дверь, передней лапкой к ручке.
DOOR_OPEN_SPOT = np.array([BATH_X + 1.8, DOOR_HIGH + 0.75])
# Закрывает, стоя за краем открытой двери спиной к ней — пинает задней лапкой.
# Не вплотную к стене: там отворот от стены мешает дойти, и она крутилась.
DOOR_KICK_SPOT = np.array([BATH_X + DOOR_WIDTH + 1.8, DOOR_LOW + 1.2])
DOOR_AWAY = np.array([BATH_X + 4.5, TOILET[1] + 1.5])  # сюда уходит из ванной

DOOR = {"open": 0.0}

# Кухня: холодильник и раковина у задней стены, стол посередине, три ведра
# для мусора вдоль боковой стены. (где, радиус сверху) — во всё это Муза
# упирается, как в остальные вещи.
# Длинный стол посередине, ближе ко входу; холодильник и раковина у задней
# стены, вёдра вдоль боковой. Вокруг стола с обеих сторон проход к каждой вещи.
FRIDGE_HALF = np.array([0.9, 1.0, 1.9])  # холодильник побольше
FRIDGE = np.array([NOOK_X - FRIDGE_HALF[0] - 0.05, NOOK_HALF - FRIDGE_HALF[1] - 0.05])  # левый задний угол
SINK = np.array([NOOK_X - 0.75, -3.0])
TABLE = np.array([HALF + 1.3, 0.0])
TABLE_HALF = np.array([0.7, 2.6])
TABLE_TOP = 1.35
BINS = [np.array([HALF + 1.5 + 0.9 * i, -NOOK_HALF + 0.55]) for i in range(3)]
BIN_COLORS = ((0.20, 0.40, 0.85), (0.95, 0.80, 0.15), (0.25, 0.65, 0.30))  # бумага, пластик, стекло
# Длинный стол для упора — цепочкой кругов вдоль него.
TABLE_CIRCLES = [TABLE + np.array([0.0, y]) for y in (-1.8, -0.6, 0.6, 1.8)]
# Невидимый упор в углу за раковиной: там карман между раковиной, стенами
# и вёдрами — Муза заползала и не могла развернуться.
SINK_CORNER = np.array([NOOK_X - 1.0, -NOOK_HALF + 1.0])
KITCHEN = (
    [(FRIDGE, 1.15), (SINK, 0.85), (SINK_CORNER, 1.0)]
    + [(spot, 0.8) for spot in TABLE_CIRCLES]
    + [(spot, 0.4) for spot in BINS]
)


# Окно в восточной стене (y = -HALF), под ним подоконник с пепельницей.
WINDOW_X = 6.5
WINDOW_HALF = 1.3  # полуширина окна
SILL_TOP = 1.3  # высота подоконника
SILL_DEPTH = 0.6  # насколько подоконник выступает в комнату
ASHTRAY = np.array([WINDOW_X + 0.5, -HALF + 0.32])

# Ещё два окна у Музы, такие же, но без подоконника с пепельницей:
# на кухне в задней стене и на стене с телефоном (между ванной и матрасом).
KITCHEN_WINDOW = 2.5  # y середины окна в задней стене кухни
PHONE_WINDOW = 0.0  # y середины окна в западной стене комнаты


def surface(position):
    """На какой высоте лежит вещь в этой точке: на столе, на подоконнике
    или на полу."""

    position = np.asarray(position[:2])

    if np.all(np.abs(position - TABLE) < TABLE_HALF):
        return TABLE_TOP
    if abs(position[0] - WINDOW_X) < WINDOW_HALF + 0.25 and position[1] < -HALF + SILL_DEPTH:
        return SILL_TOP
    return 0.0


def window(scene, ngeom):
    """Окно с рамой и переплётом, подоконник и пепельница на нём."""

    def put(kind, size, pos, color, glow=0.0, shiny=0.4):
        nonlocal ngeom
        geom = scene.geoms[ngeom]
        mujoco.mjv_initGeom(
            geom, kind, np.array(size, dtype=float), np.array(pos, dtype=float),
            np.eye(3).flatten(), np.array(color, dtype=float),
        )
        geom.emission = glow
        geom.specular = shiny
        ngeom += 1

    box = mujoco.mjtGeom.mjGEOM_BOX
    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    face = -HALF + 0.02  # чуть перед стеной
    bottom, top = SILL_TOP, SILL_TOP + 3.0
    middle = (bottom + top) / 2
    white = (0.97, 0.97, 0.95, 1.0)

    # Стекло прозрачное: за ним настоящая улица (street).
    put(box, (WINDOW_HALF, 0.01, (top - bottom) / 2), (WINDOW_X, face, middle), (0.70, 0.80, 0.90, 0.18), shiny=1.0)

    rain = np.array([[np.cos(0.25), 0, np.sin(0.25)], [0, 1, 0], [-np.sin(0.25), 0, np.cos(0.25)]])
    rng = np.random.default_rng(3)
    for _ in range(18):
        x = WINDOW_X + rng.uniform(-WINDOW_HALF + 0.1, WINDOW_HALF - 0.1)
        z = rng.uniform(bottom + 0.3, top - 0.3)
        mujoco.mjv_initGeom(
            scene.geoms[ngeom], box, np.array([0.006, 0.014, 0.18]), np.array([x, face + 0.010, z]),
            rain.flatten(), np.array([0.80, 0.84, 0.90, 0.6]),
        )
        scene.geoms[ngeom].emission = 0.3
        ngeom += 1
    for x in (WINDOW_X - WINDOW_HALF, WINDOW_X + WINDOW_HALF):
        put(box, (0.08, 0.05, (top - bottom) / 2 + 0.08), (x, face + 0.02, middle), white)
    put(box, (WINDOW_HALF + 0.08, 0.05, 0.08), (WINDOW_X, face + 0.02, top), white)
    put(box, (0.035, 0.035, (top - bottom) / 2), (WINDOW_X, face + 0.02, middle), white)
    put(box, (WINDOW_HALF, 0.035, 0.035), (WINDOW_X, face + 0.02, middle), white)

    # Подоконник.
    put(box, (WINDOW_HALF + 0.25, SILL_DEPTH / 2, 0.06), (WINDOW_X, -HALF + SILL_DEPTH / 2, SILL_TOP - 0.06), (0.93, 0.93, 0.90, 1.0), shiny=0.6)

    # Пепельница: стеклянное донышко, тёмный бортик, пепел внутри.
    ax, ay = ASHTRAY
    put(cylinder, (0.24, 0.03, 0), (ax, ay, SILL_TOP + 0.03), (0.35, 0.40, 0.45, 1.0), shiny=1.0)
    put(cylinder, (0.26, 0.015, 0), (ax, ay, SILL_TOP + 0.075), (0.22, 0.25, 0.28, 1.0), shiny=1.0)
    put(cylinder, (0.19, 0.012, 0), (ax, ay, SILL_TOP + 0.08), (0.45, 0.45, 0.45, 1.0))

    return ngeom  # 0 — закрыта, 1 — открыта настежь

def side_windows(scene, ngeom):
    """Окна на кухне и у телефона: стекло, рама с переплётом, подоконник.
    Штор нет — видна улица."""

    def put(size, pos, color, shiny=0.4):
        nonlocal ngeom
        geom = scene.geoms[ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_BOX, np.array(size, dtype=float), np.array(pos, dtype=float),
            np.eye(3).flatten(), np.array(color, dtype=float),
        )
        geom.specular = shiny
        ngeom += 1

    bottom, top = SILL_TOP, SILL_TOP + 3.0
    middle, half_h = (bottom + top) / 2, (top - bottom) / 2
    white = (0.97, 0.97, 0.95, 1.0)

    # (плоскость стены изнутри, куда в комнату, середина окна)
    for face, inward, y in ((NOOK_X - 0.02, -1.0, KITCHEN_WINDOW), (-HALF + 0.02, 1.0, PHONE_WINDOW)):
        put((0.01, WINDOW_HALF, half_h), (face, y, middle), (0.70, 0.80, 0.90, 0.18), shiny=1.0)
        for side in (-1.0, 1.0):
            put((0.05, 0.08, half_h + 0.08), (face + inward * 0.02, y + side * WINDOW_HALF, middle), white)
        put((0.05, WINDOW_HALF + 0.08, 0.08), (face + inward * 0.02, y, top), white)
        put((0.035, 0.035, half_h), (face + inward * 0.02, y, middle), white)
        put((0.035, WINDOW_HALF, 0.035), (face + inward * 0.02, y, middle), white)
        put((SILL_DEPTH / 2, WINDOW_HALF + 0.25, 0.06), (face + inward * SILL_DEPTH / 2, y, SILL_TOP - 0.06), (0.93, 0.93, 0.90, 1.0), shiny=0.6)

    return ngeom


LAST_EYE = {"eye": np.zeros(3), "focus": np.zeros(3)}

# Входная дверь — в стене напротив окна, открывается в коридор. Петля у
# восточного косяка посередине толщины стены.
FRONT_DOOR_X = WINDOW_X
FRONT_DOOR_HALF = 1.2
FRONT_HEIGHT = 3.7
FRONT_HINGE = np.array([FRONT_DOOR_X + FRONT_DOOR_HALF, HALF + WALL_THICK])
FRONT = {"open": 0.0}

# Дом: квартира Музы на 5 этаже, улица на четыре этажа ниже.
STOREY = 7.0  # мм от пола до пола
GROUND = -4 * STOREY  # уровень улицы
CORRIDOR_LOW = HALF + 2 * WALL_THICK  # стена квартиры с входной дверью
CORRIDOR_HIGH = CORRIDOR_LOW + 5.0
LANE = (CORRIDOR_LOW + CORRIDOR_HIGH) / 2  # середина коридора

# Лифт в восточном торце коридора. Кабина такая, чтобы Муза в ней
# помещалась вместе с упором от стенок.
LIFT_X = 11.2  # стена шахты со стороны коридора
LIFT_DOOR = 1.3  # полуширина проёма
CAB = (LIFT_X + 2 * WALL_THICK, 15.9, LANE - 2.2, LANE + 2.2)  # внутри кабины: x0, x1, y0, y1
LIFT_SPEED = 7.0  # мм в секунду: этаж за секунду
LIFT = {"z": 0.0, "doors": 0.0, "called": None, "idle": 0.0}
WHERE = {"floor": 5}  # на каком этаже Муза


def level(storey):
    """Высота пола этажа: 5-й — пол квартиры (0), 1-й — улица."""
    return GROUND + (storey - 1) * STOREY

WALL_COLOR = (0.93, 0.88, 0.80)
SEE_THROUGH = 0.12  # прозрачность стены между камерой и Музой

WALLS = {
    # имя: (центр, половинные размеры, наружу — куда смотрит стена)
    # Северная стена разрезана проёмом в кухню.
    "north": (
        (HALF + WALL_THICK, -(HALF + NOOK_HALF) / 2 - WALL_THICK),
        (WALL_THICK, (HALF - NOOK_HALF) / 2 + WALL_THICK),
        (1.0, 0.0),
    ),
    "north_b": (
        (HALF + WALL_THICK, (HALF + NOOK_HALF) / 2 + WALL_THICK),
        (WALL_THICK, (HALF - NOOK_HALF) / 2 + WALL_THICK),
        (1.0, 0.0),
    ),
    # Три стены кухни: задняя и две боковые.
    # Задняя стена кухни и стена с телефоном — по четыре куска вокруг окна.
    "kitchen_back": (((NOOK_X + WALL_THICK), (-(NOOK_HALF + 2 * WALL_THICK) + KITCHEN_WINDOW - WINDOW_HALF) / 2), (WALL_THICK, (KITCHEN_WINDOW - WINDOW_HALF - (-(NOOK_HALF + 2 * WALL_THICK))) / 2), (1.0, 0.0)),
    "kitchen_back_b": (((NOOK_X + WALL_THICK), (KITCHEN_WINDOW + WINDOW_HALF + (NOOK_HALF + 2 * WALL_THICK)) / 2), (WALL_THICK, ((NOOK_HALF + 2 * WALL_THICK) - (KITCHEN_WINDOW + WINDOW_HALF)) / 2), (1.0, 0.0)),
    "kitchen_back_sill": (((NOOK_X + WALL_THICK), KITCHEN_WINDOW), (WALL_THICK, WINDOW_HALF), (1.0, 0.0)),
    "kitchen_back_top": (((NOOK_X + WALL_THICK), KITCHEN_WINDOW), (WALL_THICK, WINDOW_HALF), (1.0, 0.0)),
    "kitchen_left": (((HALF + NOOK_X) / 2 + WALL_THICK, -NOOK_HALF - WALL_THICK), ((NOOK_X - HALF) / 2 + WALL_THICK, WALL_THICK), (0.0, -1.0)),
    "kitchen_right": (((HALF + NOOK_X) / 2 + WALL_THICK, NOOK_HALF + WALL_THICK), ((NOOK_X - HALF) / 2 + WALL_THICK, WALL_THICK), (0.0, 1.0)),
    "south": (((-HALF - WALL_THICK), (-(HALF + 2 * WALL_THICK) + PHONE_WINDOW - WINDOW_HALF) / 2), (WALL_THICK, (PHONE_WINDOW - WINDOW_HALF - (-(HALF + 2 * WALL_THICK))) / 2), (-1.0, 0.0)),
    "south_b": (((-HALF - WALL_THICK), (PHONE_WINDOW + WINDOW_HALF + (HALF + 2 * WALL_THICK)) / 2), (WALL_THICK, ((HALF + 2 * WALL_THICK) - (PHONE_WINDOW + WINDOW_HALF)) / 2), (-1.0, 0.0)),
    "south_sill": (((-HALF - WALL_THICK), PHONE_WINDOW), (WALL_THICK, WINDOW_HALF), (-1.0, 0.0)),
    "south_top": (((-HALF - WALL_THICK), PHONE_WINDOW), (WALL_THICK, WINDOW_HALF), (-1.0, 0.0)),
    # Стена с входной дверью — три куска вокруг проёма.
    "west": (
        ((-HALF - 2 * WALL_THICK + FRONT_DOOR_X - FRONT_DOOR_HALF) / 2, HALF + WALL_THICK),
        ((FRONT_DOOR_X - FRONT_DOOR_HALF + HALF + 2 * WALL_THICK) / 2, WALL_THICK),
        (0.0, 1.0),
    ),
    "west_b": (
        ((FRONT_DOOR_X + FRONT_DOOR_HALF + HALF + 2 * WALL_THICK) / 2, HALF + WALL_THICK),
        ((HALF + 2 * WALL_THICK - FRONT_DOOR_X - FRONT_DOOR_HALF) / 2, WALL_THICK),
        (0.0, 1.0),
    ),
    "west_top": ((FRONT_DOOR_X, HALF + WALL_THICK), (FRONT_DOOR_HALF, WALL_THICK), (0.0, 1.0)),
    # Стена с окном — четыре куска вокруг настоящего проёма: через стекло
    # видно улицу.
    "east": (
        ((-HALF - 2 * WALL_THICK + WINDOW_X - WINDOW_HALF) / 2, -HALF - WALL_THICK),
        ((WINDOW_X - WINDOW_HALF + HALF + 2 * WALL_THICK) / 2, WALL_THICK),
        (0.0, -1.0),
    ),
    "east_b": (
        ((WINDOW_X + WINDOW_HALF + HALF + 2 * WALL_THICK) / 2, -HALF - WALL_THICK),
        ((HALF + 2 * WALL_THICK - WINDOW_X - WINDOW_HALF) / 2, WALL_THICK),
        (0.0, -1.0),
    ),
    "east_sill": ((WINDOW_X, -HALF - WALL_THICK), (WINDOW_HALF, WALL_THICK), (0.0, -1.0)),
    "east_top": ((WINDOW_X, -HALF - WALL_THICK), (WINDOW_HALF, WALL_THICK), (0.0, -1.0)),
}

# Куски стен не на всю высоту: (низ, верх). Остальные — от пола до потолка.
WALL_Z = {
    "east_sill": (0.0, SILL_TOP),
    "east_top": (SILL_TOP + 3.0, WALL_HEIGHT),
    "west_top": (FRONT_HEIGHT, WALL_HEIGHT),
    "kitchen_back_sill": (0.0, SILL_TOP),
    "kitchen_back_top": (SILL_TOP + 3.0, WALL_HEIGHT),
    "south_sill": (0.0, SILL_TOP),
    "south_top": (SILL_TOP + 3.0, WALL_HEIGHT),
}


def wall_z(name):
    return WALL_Z.get(name, (0.0, WALL_HEIGHT))


def door_panel():
    """Полотно двери отрезком: от петли до свободного края."""

    angle = DOOR["open"] * np.pi / 2
    direction = np.array([np.sin(angle), np.cos(angle)])
    return HINGE.copy(), HINGE + direction * DOOR_WIDTH


HANDLE_HEIGHT = 0.9  # на высоте поднятой передней лапки


def door_handle(open_part=None):
    """Ручка двери со стороны комнаты (где её берёт Муза)."""

    part = DOOR["open"] if open_part is None else open_part
    angle = part * np.pi / 2
    direction = np.array([np.sin(angle), np.cos(angle)])
    outward = np.array([np.cos(angle), -np.sin(angle)])  # от двери в комнату
    spot = HINGE + direction * (DOOR_WIDTH - 0.35) + outward * 0.13
    return np.array([spot[0], spot[1], HANDLE_HEIGHT])


def front_panel():
    """Полотно входной двери отрезком: закрыта — вдоль стены на запад,
    открыта — стоит в коридоре на север."""

    angle = FRONT["open"] * np.pi / 2
    direction = np.array([-np.cos(angle), np.sin(angle)])
    return FRONT_HINGE.copy(), FRONT_HINGE + direction * 2 * FRONT_DOOR_HALF


def front_handle(open_part=None, side=-1.0):
    """Ручка входной двери: side −1 — из квартиры, +1 — из коридора."""

    part = FRONT["open"] if open_part is None else open_part
    angle = part * np.pi / 2
    direction = np.array([-np.cos(angle), np.sin(angle)])
    inward = np.array([-np.sin(angle), -np.cos(angle)])  # в квартиру, пока закрыта
    spot = FRONT_HINGE + direction * (2 * FRONT_DOOR_HALF - 0.35) + inward * 0.13 * -side
    return np.array([spot[0], spot[1], HANDLE_HEIGHT])


def lift_button(storey):
    """Кнопка вызова лифта на стене коридора, у проёма."""
    return np.array([LIFT_X - 0.05, LANE + LIFT_DOOR + 0.45, level(storey) + HANDLE_HEIGHT])


def lift_here():
    """Кабина стоит на том же этаже, что и Муза."""
    return abs(LIFT["z"] - level(WHERE["floor"])) < 1e-6


def in_cab(position):
    return bool(CAB[0] < position[0] < CAB[1] and CAB[2] < position[1] < CAB[3])


def in_corridor(position):
    return bool(position[1] > CORRIDOR_LOW - WALL_THICK)


def walls():
    """Все стены отрезками на полу, у каждой свой упор: (начало, конец,
    упор). Дверные полотна и косяки — с малым упором, иначе в проём не пройти."""

    wide, narrow = [], []
    door_lo, door_hi = FRONT_DOOR_X - FRONT_DOOR_HALF, FRONT_DOOR_X + FRONT_DOOR_HALF
    lo, hi = CORRIDOR_LOW, CORRIDOR_HIGH

    def seg(a, b, into=wide):
        into.append((np.array(a, dtype=float), np.array(b, dtype=float)))

    if WHERE["floor"] == 5:
        for first, second in segments():
            seg(first, second)
        narrow.append(door_panel())
        # Проём входной двери: косяки и полотно.
        seg((door_lo, HALF), (door_lo, lo), narrow)
        seg((door_hi, HALF), (door_hi, lo), narrow)
        narrow.append(front_panel())
        seg((-HALF, lo), (door_lo, lo))
        seg((door_hi, lo), (LIFT_X, lo))
        seg((-HALF, lo), (-HALF, hi))
    else:
        seg((-HALF, lo), (LIFT_X, lo))
        # Западный торец холла с проёмом двери на улицу: косяки и створка.
        street_lo, street_hi = STREET_LOW, STREET_LOW + STREET_LEAF
        inside, outside = -HALF, HOUSE[0]
        seg((inside, lo), (inside, street_lo))
        seg((inside, street_hi), (inside, hi))
        seg((inside, street_lo), (outside, street_lo), narrow)
        seg((inside, street_hi), (outside, street_hi), narrow)
        narrow.append(street_panel())
        # Двор: ходит только по дорожке — площадка у двери, полоса вдоль
        # дома, полоса к воротам. Края дорожки с малым упором.
        a0, a1 = APRON
        s0, s1 = SPINE
        b0, b1 = BOTTOM
        path = [
            ((outside, a0), (outside, street_lo)),
            ((outside, street_hi), (outside, a1)),
            ((outside, a1), (s0, a1)),
            ((s0, a1), (s0, b1)),
            ((s0, b1), (FENCE_X, b1)),
            ((FENCE_X, b0), (s1, b0)),
            ((s1, b0), (s1, a0)),
            ((s1, a0), (outside, a0)),
        ]
        if GATE["open"] < 0.9:
            path.append(((FENCE_X, b1), (FENCE_X, b0)))
        # Тротуар вокруг участка: изнутри — край участка (у ворот проход),
        # снаружи — бордюр.
        x0, x1, y0, y1 = LOT
        far = SIDEWALK - 0.8
        path += [
            ((x0, y0), (x0, b0)), ((x0, b1), (x0, y1)),
            ((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x0, y1), (x1, y1)),
            ((x0 - far, y0 - far), (x1 + far, y0 - far)), ((x1 + far, y0 - far), (x1 + far, y1 + far)),
            ((x1 + far, y1 + far), (x0 - far, y1 + far)),
            # Западный бордюр с проходом на зебру, зебра до двери кафе.
            ((x0 - far, y1 + far), (x0 - far, CAFE_Y + CROSS_HALF)), ((x0 - far, CAFE_Y - CROSS_HALF), (x0 - far, y0 - far)),
            ((x0 - far, CAFE_Y + CROSS_HALF), (CAFE_FACE, CAFE_Y + CROSS_HALF)),
            ((x0 - far, CAFE_Y - CROSS_HALF), (CAFE_FACE, CAFE_Y - CROSS_HALF)),
        ]
        # Кафе: витрина с дверью, стены, стойка, столики.
        c0, c1, back = CAFE_Y - CAFE_HALF, CAFE_Y + CAFE_HALF, CAFE_FACE - CAFE_DEPTH
        path += [
            ((CAFE_FACE, c0), (CAFE_FACE, CAFE_Y - CAFE_DOOR)), ((CAFE_FACE, CAFE_Y + CAFE_DOOR), (CAFE_FACE, c1)),
            ((back, c0), (back, c1)), ((back, c0), (CAFE_FACE, c0)), ((back, c1), (CAFE_FACE, c1)),
            # Стойка и проходы за неё — закрыты с обоих концов: туда нельзя.
            ((COUNTER_EDGE, COUNTER_Y[0]), (COUNTER_EDGE, COUNTER_Y[1])),
            ((back, COUNTER_Y[0]), (COUNTER_EDGE, COUNTER_Y[0])),
            ((back, COUNTER_Y[1]), (COUNTER_EDGE, COUNTER_Y[1])),
        ]
        if CAFE["open"] < 0.9:
            path.append(((CAFE_FACE, CAFE_Y - CAFE_DOOR), (CAFE_FACE, CAFE_Y + CAFE_DOOR)))
        # Туалет кафе — копия ванной квартиры в юго-западном углу.
        with bathroom_as("cafe"):
            path += [(tuple(a), tuple(b)) for a, b in bath_pieces()]
            path.append(tuple(tuple(p) for p in door_panel()))
        for tx, ty in CAFE_TABLES:
            rx, ry = CAFE_TABLE_RADIUS + 0.3, CAFE_TABLE_RADIUS + 0.2  # столешница; стулья Муза обходит сама
            path += [((tx - rx, ty - ry), (tx + rx, ty - ry)), ((tx + rx, ty - ry), (tx + rx, ty + ry)),
                     ((tx + rx, ty + ry), (tx - rx, ty + ry)), ((tx - rx, ty + ry), (tx - rx, ty - ry))]
        for a, b in path:
            seg(a, b, narrow)

    # Коридор: северная стена, стена шахты с проёмом лифта.
    seg((-HALF, hi), (LIFT_X, hi))
    seg((LIFT_X, lo), (LIFT_X, LANE - LIFT_DOOR))
    seg((LIFT_X, LANE + LIFT_DOOR), (LIFT_X, hi))
    seg((LIFT_X, LANE - LIFT_DOOR), (CAB[0], LANE - LIFT_DOOR), narrow)
    seg((LIFT_X, LANE + LIFT_DOOR), (CAB[0], LANE + LIFT_DOOR), narrow)
    if LIFT["doors"] < 0.8 or not lift_here():
        seg((LIFT_X + WALL_THICK, LANE - LIFT_DOOR), (LIFT_X + WALL_THICK, LANE + LIFT_DOOR), narrow)

    # Кабина изнутри.
    seg((CAB[0], CAB[2]), (CAB[0], LANE - LIFT_DOOR))
    seg((CAB[0], LANE + LIFT_DOOR), (CAB[0], CAB[3]))
    seg((CAB[0], CAB[2]), (CAB[1], CAB[2]))
    seg((CAB[0], CAB[3]), (CAB[1], CAB[3]))
    seg((CAB[1], CAB[2]), (CAB[1], CAB[3]))

    return [(a, b, KEEP) for a, b in wide] + [(a, b, JAMB_KEEP) for a, b in narrow]


def jamb_points():
    """Углы проёмов: у них упор малый, как у косяков."""

    door_lo, door_hi = FRONT_DOOR_X - FRONT_DOOR_HALF, FRONT_DOOR_X + FRONT_DOOR_HALF
    points = [
        (BATH_X, DOOR_LOW), (BATH_X, DOOR_HIGH),
        (door_lo, HALF), (door_hi, HALF), (door_lo, CORRIDOR_LOW), (door_hi, CORRIDOR_LOW),
        (LIFT_X, LANE - LIFT_DOOR), (LIFT_X, LANE + LIFT_DOOR),
        (CAB[0], LANE - LIFT_DOOR), (CAB[0], LANE + LIFT_DOOR),
        (-HALF, STREET_LOW), (-HALF, STREET_LOW + STREET_LEAF),
        (HOUSE[0], STREET_LOW), (HOUSE[0], STREET_LOW + STREET_LEAF),
    ]
    return np.array(points, dtype=float)


def segments():
    """Все стены отрезками на полу: наружные, перегородки ванной и дверь,
    пока она не открыта."""

    # Контур комнаты вместе с нишей кухни.
    corners = [
        (-HALF, -HALF), (HALF, -HALF), (HALF, -NOOK_HALF), (NOOK_X, -NOOK_HALF),
        (NOOK_X, NOOK_HALF), (HALF, NOOK_HALF), (HALF, HALF), (-HALF, HALF),
    ]
    count = len(corners)
    out = [(np.array(corners[i]), np.array(corners[(i + 1) % count])) for i in range(count)]

    # Северная стена прервана проёмом входной двери (косяки — в walls()).
    out = [(a, b) for a, b in out if not (a[1] == HALF and b[1] == HALF)]
    out += [
        (np.array([HALF, HALF]), np.array([FRONT_DOOR_X + FRONT_DOOR_HALF, HALF])),
        (np.array([FRONT_DOOR_X - FRONT_DOOR_HALF, HALF]), np.array([-HALF, HALF])),
    ]

    out += [
        (np.array([-HALF, BATH_HIGH]), np.array([BATH_X, BATH_HIGH])),
        (np.array([BATH_X, BATH_LOW]), np.array([BATH_X, DOOR_LOW])),
        (np.array([BATH_X, DOOR_HIGH]), np.array([BATH_X, BATH_HIGH])),
    ]

    # Дверь — препятствие в любом положении: тело сквозь неё не проходит.
    out.append(door_panel())

    return out


def keep_for(close):
    """Упор: у косяков и дверных полотен меньше, у стен — обычный."""

    if np.min(np.linalg.norm(jamb_points() - close, axis=1)) < 1e-6:
        return JAMB_KEEP

    for first, second, keep in walls():
        if keep == KEEP:
            continue
        along = second - first
        offset = close - first
        part = float(offset @ along) / max(float(along @ along), 1e-9)
        if 0.0 <= part <= 1.0 and float(np.linalg.norm(offset - along * part)) < 1e-6:
            return JAMB_KEEP

    return KEEP


def nearest(point, first, second):
    """Ближайшая к точке точка отрезка."""

    along = second - first
    part = np.clip((point - first) @ along / max(along @ along, 1e-9), 0.0, 1.0)
    return first + along * part


def keep_in_room(position, margin=1.8, before=None):
    """Предмет не может оказаться за стеной: иначе Муза тянется к нему и
    упирается в стену или в угол."""

    # К окну — на подоконник: вещь запрыгивает на него, а не упирается в
    # запрет «не ближе 1.8 мм к стене». На подоконнике можно у самого стекла.
    # Двигаешь от окна — спрыгивает с подоконника на пол.
    near_window = abs(position[0] - WINDOW_X) < WINDOW_HALF + 0.25
    if near_window and position[1] < -HALF + margin:
        was_on_sill = before is not None and before[1] < -HALF + SILL_DEPTH
        leaving = before is not None and position[1] > before[1]
        if was_on_sill and leaving and position[1] > -HALF + SILL_DEPTH - 0.1:
            position[1] = -HALF + margin
            return
        position[1] = float(np.clip(position[1], -HALF + 0.15, -HALF + SILL_DEPTH - 0.1))
        return

    limit = HALF - margin

    # В нишу кухни можно глубже.
    if abs(position[1]) < NOOK_HALF - margin and position[0] > HALF - margin:
        position[0] = float(np.clip(position[0], -limit, NOOK_X - margin))
        return

    position[0] = float(np.clip(position[0], -limit, limit))
    position[1] = float(np.clip(position[1], -limit, limit))


def in_bathroom(position):
    return bool(position[0] < BATH_X and BATH_LOW < position[1] < BATH_HIGH)


def mattress_center():
    return (MATTRESS_LOW + MATTRESS_HIGH) / 2.0


def on_mattress(position, margin=0.4):
    """Стоит ли точка над матрасом (с отступом от края)."""

    return bool(
        np.all(position[:2] > MATTRESS_LOW + margin)
        and np.all(position[:2] < MATTRESS_HIGH - margin)
    )


def build(world):
    """Добавляет стены и матрас в мир. Вызывать после world.add_fly."""

    spec = world.mjcf_root

    for name, (center, half, _) in WALLS.items():
        low, high = wall_z(name)
        spec.worldbody.add_geom(
            name=f"room_wall_{name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(center[0], center[1], (low + high) / 2),
            size=(half[0], half[1], (high - low) / 2),
            rgba=(*WALL_COLOR, 1.0),
            contype=0,
            conaffinity=0,
        )

    middle = mattress_center()
    half = (MATTRESS_HIGH - MATTRESS_LOW) / 2.0

    spec.worldbody.add_geom(
        name="room_mattress",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(middle[0], middle[1], MATTRESS_TOP / 2),
        size=(half[0], half[1], MATTRESS_TOP / 2),
        rgba=(0.92, 0.93, 0.97, 1.0),
        contype=0,
        conaffinity=0,
    )

    # Пар контакта нет нарочно.
    #
    # Стены держит упор (hold_inside): с настоящим контактом Муза на полном
    # ходу проваливалась в стену, её выталкивало, и она летела кувырком.
    #
    # Матрас мягкий: только картинка, лапки чуть утопают. С твёрдой
    # ступенькой присоски цеплялись за его бок, и на краю она опрокидывалась.


def wall_ids(model):
    return {name: model.geom(f"room_wall_{name}").id for name in WALLS}


def cut_away(model, ids, camera):
    """Стены между камерой и комнатой делает почти прозрачными."""

    azimuth = np.radians(camera.azimuth)
    elevation = np.radians(camera.elevation)

    look = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    eye = np.asarray(camera.lookat) - look * camera.distance
    LAST_EYE["eye"] = eye
    LAST_EYE["focus"] = np.asarray(camera.lookat).copy()

    for name, (center, half, outward) in WALLS.items():

        outward = np.asarray(outward)
        outside = float(eye[:2] @ outward) > float(np.asarray(center) @ outward)

        model.geom_rgba[ids[name]][3] = SEE_THROUGH if outside else 1.0


def wall_table():
    """Стены одной таблицей для упора: считается раз в кадр, а не на
    каждом шаге физики (тогда упор съедал больше времени, чем сама физика)."""

    segs = walls()
    first = np.array([a for a, _, _ in segs])
    second = np.array([b for _, b, _ in segs])
    keep = np.array([k for _, _, k in segs])
    return first, second, keep, jamb_points()


def _nearest_all(point, table):
    """Ближайшие точки всех стен к точке и расстояния до них — разом."""

    first, second, _, _ = table
    along = second - first
    part = np.clip(
        np.einsum("ij,ij->i", point - first, along) / np.maximum(np.einsum("ij,ij->i", along, along), 1e-9),
        0.0, 1.0,
    )
    close = first + along * part[:, None]
    return close, np.linalg.norm(point - close, axis=1)


def hold_inside(data, circles=(), parts=(), table=None):
    """Упор в стены и предметы: грудь не заходит за границу, скорость в
    стену или предмет гасится — стукнулась и стоит.

    circles — предметы на полу: (где, на сколько к нему можно подойти грудью).
    parts — другие части тела (id тела, толщина): брюшко тоже не должно
    заходить в дверь и стены, иначе Муза застревала в двери задом.
    table — стены из wall_table(); если нет, соберём сами.
    qpos[0:2] — положение груди: у Музы свободный сустав стоит на ней.
    """

    table = wall_table() if table is None else table
    _, _, keeps, jambs = table
    stopped = False

    def shove(amount, away):
        nonlocal stopped
        data.qpos[0:2] += away * amount
        speed = float(data.qvel[0:2] @ away)
        if speed < 0.0:
            data.qvel[0:2] -= away * speed
        stopped = True

    for body, radius in parts:
        spot = data.xpos[body][:2].copy()
        close, gaps = _nearest_all(spot, table)
        for index in np.nonzero(gaps < radius)[0]:
            gap = gaps[index]
            away = (spot - close[index]) / gap if gap > 1e-6 else np.array([1.0, 0.0])
            shove(radius - gap, away)

    for center, keep in circles:

        spot = data.qpos[0:2]
        offset = spot - center
        gap = float(np.linalg.norm(offset))

        if gap >= keep:
            continue

        away = offset / gap if gap > 1e-6 else np.array([1.0, 0.0])
        data.qpos[0:2] = center + away * keep

        speed = float(data.qvel[0:2] @ away)
        if speed < 0.0:
            data.qvel[0:2] -= away * speed

        stopped = True

    spot = data.qpos[0:2].copy()
    close, gaps = _nearest_all(spot, table)

    # У косяков упор меньше, иначе в узкий проём не пройти.
    near_jamb = np.min(np.linalg.norm(close[:, None, :] - jambs[None, :, :], axis=2), axis=1) < 1e-6
    limits = np.where(near_jamb, JAMB_KEEP, keeps)

    for index in np.nonzero(gaps < limits)[0]:
        gap = gaps[index]
        away = (spot - close[index]) / gap if gap > 1e-6 else np.array([1.0, 0.0])
        data.qpos[0:2] = close[index] + away * limits[index]
        speed = float(data.qvel[0:2] @ away)
        if speed < 0.0:
            data.qvel[0:2] -= away * speed
        stopped = True
        spot = data.qpos[0:2].copy()

    if not stopped:
        return

    # Упёрлась: лапки толкают дальше, а грудь стоит — тело встало бы на
    # дыбы и опрокинулось. Держим её ровно: крен и тангаж гасим, курс её.
    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    data.qpos[3:7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    data.qvel[3:5] = 0.0
    data.qvel[2] = min(data.qvel[2], 0.0)


def obstacle_push(position, heading, circles, reach=4.0, item_reach=2.0, ignore_door=False):
    """Отворот сразу от стен и от предметов: все отталкивания складываются
    в одно «прочь». По отдельности стена говорила «налево», ноутбук у стены —
    «направо», и Муза стояла, уткнувшись между ними."""

    push = np.zeros(2)
    strongest = 0.0
    door = door_panel()

    for first, second, _ in walls():

        # По делам у двери от самой двери не шарахается — иначе, открыв её,
        # делала крюк по комнате. Упор (hold_inside) всё равно не пустит насквозь.
        if ignore_door and np.allclose(first, door[0]) and np.allclose(second, door[1]):
            continue

        close = nearest(position[:2], first, second)
        gap = float(np.linalg.norm(close - position[:2]))

        if gap > reach or gap < 1e-6:
            continue

        outward = (close - position[:2]) / gap
        facing = float(heading[:2] @ outward)

        if facing <= 0.0:
            continue

        keep = keep_for(close)
        strength = min((1.0 - (gap - keep) / (reach - keep)) * (0.3 + facing), 1.0)
        push -= outward * strength
        strongest = max(strongest, strength)

    for center, keep in circles:

        offset = np.asarray(center) - position[:2]
        distance = float(np.linalg.norm(offset))
        gap = distance - keep

        if gap > item_reach or distance < 1e-6:
            continue

        outward = offset / distance
        facing = float(heading[:2] @ outward)

        if facing <= 0.0:
            continue

        strength = min((1.0 - max(gap, 0.0) / item_reach) * (0.3 + facing), 1.0)
        push -= outward * strength
        strongest = max(strongest, strength)

    if strongest <= 0.0:
        return 0.0

    course = heading[:2] / max(np.linalg.norm(heading[:2]), 1e-9)
    wanted = course + push * 2.0
    angle = np.arctan2(course[0] * wanted[1] - course[1] * wanted[0], float(course @ wanted))

    if abs(angle) > np.pi - 0.05:
        angle = np.pi

    return float(np.sign(angle) * strongest * 6.0)


def blocked_by_bathroom(start, end):
    """Проходит ли прямая от start до end сквозь ванную (с запасом)."""

    for part in np.linspace(0.0, 1.0, 40):
        point = start + (end - start) * part
        if point[0] < BATH_X + 1.0 and point[1] < BATH_HIGH + 1.0:
            return True
    return False


BATH_CORNER = np.array([BATH_X + 2.0, BATH_HIGH + 2.0])  # обход ванной снаружи


def toilet(scene, ngeom, flushing=0.0, lift=0.0):
    """Унитаз: подставка, чаша, сиденье с дыркой и водой, поднятая крышка,
    бачок с кнопкой и труба в стену. flushing — вода крутится при смыве."""

    identity = np.eye(3).flatten()
    white = (0.97, 0.97, 0.98, 1.0)
    shade = (0.85, 0.86, 0.90, 1.0)

    def put(kind, size, offset, color, rotation=identity):
        nonlocal ngeom
        mujoco.mjv_initGeom(
            scene.geoms[ngeom], kind, np.array(size, dtype=float),
            np.array([TOILET[0] + offset[0], TOILET[1] + offset[1], offset[2] + lift]),
            np.asarray(rotation, dtype=float).flatten(), np.array(color, dtype=float),
        )
        scene.geoms[ngeom].specular = 0.8
        scene.geoms[ngeom].shininess = 0.8
        ngeom += 1

    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    ellipsoid = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    box = mujoco.mjtGeom.mjGEOM_BOX

    # Подставка и чаша.
    put(ellipsoid, (0.55, 0.45, 0.50), (0.10, 0.0, 0.45), shade)
    put(ellipsoid, (0.98, 0.78, 0.50), (0.28, 0.0, 0.88), white)

    # Сиденье: белое кольцо, внутри вода.
    put(ellipsoid, (1.05, 0.85, 0.07), (0.30, 0.0, SEAT_HEIGHT - 0.03), white)
    swirl = 0.15 * np.sin(flushing * 12.0)
    put(ellipsoid, (0.70, 0.50, 0.05), (0.35, 0.0, SEAT_HEIGHT + 0.01), (0.55 + swirl, 0.75 + swirl, 0.90, 1.0))

    # Крышка поднята и чуть откинута к бачку.
    tilt = np.radians(12.0)
    lean = np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]])
    put(box, (0.05, 0.80, 0.85), (-0.75, 0.0, SEAT_HEIGHT + 0.80), white, lean)

    # Бачок, его крышка, кнопка смыва.
    put(box, (0.32, 0.90, 0.75), (-1.05, 0.0, SEAT_HEIGHT + 0.55), white)
    put(box, (0.36, 0.95, 0.05), (-1.05, 0.0, SEAT_HEIGHT + 1.33), shade)
    put(cylinder, (0.14, 0.03, 0), (-1.05, 0.0, SEAT_HEIGHT + 1.40), (0.75, 0.77, 0.80, 1.0))

    # Труба от бачка в стену.
    put(cylinder, (0.08, 0.15, 0), (-1.45, 0.0, 0.55), (0.75, 0.77, 0.80, 1.0),
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))

    return ngeom


def bath_pieces():
    """Перегородки ванной отрезками: северная и восточная с проёмом двери."""
    return [
        (np.array([BATH_WEST, BATH_HIGH]), np.array([BATH_X, BATH_HIGH])),
        (np.array([BATH_X, BATH_LOW]), np.array([BATH_X, DOOR_LOW])),
        (np.array([BATH_X, DOOR_HIGH]), np.array([BATH_X, BATH_HIGH])),
    ]


def bathroom(scene, ngeom, lift=0.0):
    """Перегородки ванной, дверная коробка и дверь с ручкой. Перегородка
    между камерой и Музой становится прозрачной, как наружные стены."""

    eye, focus = LAST_EYE["eye"], LAST_EYE["focus"]

    def put(kind, size, pos, color, rotation=np.eye(3)):
        nonlocal ngeom
        mujoco.mjv_initGeom(
            scene.geoms[ngeom], kind, np.array(size, dtype=float), np.array(pos, dtype=float) + np.array([0.0, 0.0, lift]),
            np.asarray(rotation, dtype=float).flatten(), np.array(color, dtype=float),
        )
        ngeom += 1

    box = mujoco.mjtGeom.mjGEOM_BOX
    sphere = mujoco.mjtGeom.mjGEOM_SPHERE

    def blocks_view(first, second):
        """Камера и Муза по разные стороны перегородки."""
        along = second - first
        normal = np.array([-along[1], along[0]])
        return float((eye[:2] - first) @ normal) * float((focus[:2] - first) @ normal) < 0.0

    wall = (*WALL_COLOR, 1.0)
    thick = 0.08

    for first, second in bath_pieces():
        middle = (first + second) / 2.0
        half = np.abs(second - first) / 2.0 + thick
        alpha = SEE_THROUGH if blocks_view(first, second) else 1.0
        put(box, (half[0], half[1], PART_HEIGHT / 2), (middle[0], middle[1], PART_HEIGHT / 2), (*WALL_COLOR, alpha))
        put(box, (half[0] + 0.02, half[1] + 0.02, 0.12), (middle[0], middle[1], 0.12), (0.45, 0.33, 0.22, 1.0))

    # Над дверью — перемычка.
    door_width = DOOR_HIGH - DOOR_LOW
    lintel = (PART_HEIGHT - DOOR_HEIGHT) / 2
    alpha = SEE_THROUGH if blocks_view(np.array([BATH_X, DOOR_LOW]), np.array([BATH_X, DOOR_HIGH])) else 1.0
    put(box, (thick, door_width / 2, lintel), (BATH_X, (DOOR_LOW + DOOR_HIGH) / 2, DOOR_HEIGHT + lintel), (*WALL_COLOR, alpha))

    # Дверь: петля у южного косяка, открывается наружу, в комнату.
    hinge, free = door_panel()
    direction = (free - hinge) / DOOR_WIDTH
    center = hinge + direction * door_width / 2
    turn = np.array([[direction[0], -direction[1], 0], [direction[1], direction[0], 0], [0, 0, 1]])

    put(box, (door_width / 2 - 0.02, 0.06, DOOR_HEIGHT / 2 - 0.01), (center[0], center[1], DOOR_HEIGHT / 2), (0.55, 0.36, 0.20, 1.0), turn)
    put(box, (door_width / 2 - 0.35, 0.065, DOOR_HEIGHT / 2 - 0.45), (center[0], center[1], DOOR_HEIGHT / 2), (0.62, 0.42, 0.25, 1.0), turn)

    handle = hinge + direction * (door_width - 0.35)
    normal = np.array([-direction[1], direction[0]])
    for side in (1.0, -1.0):
        spot = handle + normal * 0.13 * side
        put(sphere, (0.09, 0, 0), (spot[0], spot[1], HANDLE_HEIGHT), (0.85, 0.75, 0.35, 1.0))

    return ngeom


def floor(scene, ngeom):
    """Деревянный пол в комнате: доски разного оттенка со швами и
    сдвинутыми стыками. В ванной — светлая плитка."""

    def put(size, pos, color):
        nonlocal ngeom
        mujoco.mjv_initGeom(
            scene.geoms[ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array(size, dtype=float),
            np.array(pos, dtype=float), np.eye(3).flatten(), np.array(color, dtype=float),
        )
        ngeom += 1

    # Тёмная подложка — она и видна в швах между досками.
    put((HALF, HALF, 0.0008), (0.0, 0.0, 0.0020), (0.30, 0.19, 0.10, 1.0))

    width, length, seam = 0.8, 4.0, 0.03
    rng = np.random.default_rng(7)
    rows = int(round(2 * HALF / width))

    for row in range(rows):

        y = -HALF + (row + 0.5) * width
        shift = (row * 1.7) % length  # стыки в соседних рядах не совпадают
        x = -HALF - shift

        while x < HALF:
            left, right = max(x, -HALF), min(x + length, HALF)
            x += length

            if right - left < 0.05:
                continue

            # В ванной вместо досок плитка.
            if right <= BATH_X and y < BATH_HIGH:
                continue
            if left < BATH_X and y < BATH_HIGH:
                left = BATH_X

            tone = rng.uniform(0.85, 1.12)
            put(
                ((right - left) / 2 - seam, width / 2 - seam, 0.0008),
                ((left + right) / 2, y, 0.0060),
                (0.66 * tone, 0.45 * tone, 0.25 * tone, 1.0),
            )

    tile = 0.9
    for i in range(int((BATH_X + HALF) / tile) + 1):
        for j in range(int((BATH_HIGH + HALF) / tile) + 1):
            x0, y0 = -HALF + i * tile, -HALF + j * tile
            x1, y1 = min(x0 + tile, BATH_X), min(y0 + tile, BATH_HIGH)
            if x1 - x0 < 0.05 or y1 - y0 < 0.05:
                continue
            light = 0.93 if (i + j) % 2 else 0.84
            put(((x1 - x0) / 2 - 0.02, (y1 - y0) / 2 - 0.02, 0.0008), ((x0 + x1) / 2, (y0 + y1) / 2, 0.0060), (light, light, light * 1.02, 1.0))

    return ngeom


def kitchen(scene, ngeom):
    """Холодильник, раковина с краном, стол и три ведра для сортировки."""

    def put(kind, size, pos, color, rotation=np.eye(3), shiny=0.4):
        nonlocal ngeom
        geom = scene.geoms[ngeom]
        mujoco.mjv_initGeom(
            geom, kind, np.array(size, dtype=float), np.array(pos, dtype=float),
            np.asarray(rotation, dtype=float).flatten(), np.array(color, dtype=float),
        )
        geom.specular = shiny
        ngeom += 1

    box = mujoco.mjtGeom.mjGEOM_BOX
    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    white = (0.95, 0.95, 0.96, 1.0)
    steel = (0.72, 0.74, 0.78, 1.0)
    across = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # ось цилиндра вдоль x

    # Плитка на полу кухни, чёрно-белая.
    tile = 1.0
    for i in range(int(NOOK_X - HALF)):
        for j in range(int(2 * NOOK_HALF)):
            x, y = HALF + (i + 0.5) * tile, -NOOK_HALF + (j + 0.5) * tile
            tone = 0.92 if (i + j) % 2 else 0.18
            put(box, (tile / 2 - 0.02, tile / 2 - 0.02, 0.0008), (x, y, 0.0060), (tone, tone, tone, 1.0), shiny=0.8)

    # Холодильник: корпус, дверь морозилки сверху, большая дверь, ручки.
    fx, fy = FRIDGE
    hx, hy, hz = FRIDGE_HALF
    door = fx - hx - 0.02
    put(box, (hx, hy, hz), (fx, fy, hz), white, shiny=0.8)
    put(box, (0.02, hy - 0.04, hz * 0.3), (door, fy, hz * 1.66), (0.98, 0.98, 0.99, 1.0), shiny=0.9)
    put(box, (0.02, hy - 0.04, hz * 0.62), (door, fy, hz * 0.68), (0.98, 0.98, 0.99, 1.0), shiny=0.9)
    put(box, (0.04, 0.05, hz * 0.17), (door - 0.06, fy - hy + 0.2, hz * 1.66), steel, shiny=1.0)
    put(box, (0.04, 0.05, hz * 0.32), (door - 0.06, fy - hy + 0.2, hz * 0.9), steel, shiny=1.0)

    # Раковина: тумба, столешница, чаша, кран с изгибом, ручки дверок.
    sx, sy = SINK
    put(box, (0.70, 0.80, 0.70), (sx, sy, 0.70), (0.55, 0.38, 0.24, 1.0))
    put(box, (0.75, 0.85, 0.05), (sx, sy, 1.43), (0.85, 0.85, 0.82, 1.0), shiny=0.7)
    put(box, (0.42, 0.52, 0.02), (sx - 0.05, sy, 1.47), (0.35, 0.37, 0.40, 1.0), shiny=1.0)
    put(cylinder, (0.05, 0.25, 0), (sx + 0.52, sy, 1.72), steel, shiny=1.0)
    put(cylinder, (0.04, 0.22, 0), (sx + 0.32, sy, 1.95), steel, across, shiny=1.0)
    put(cylinder, (0.035, 0.07, 0), (sx + 0.12, sy, 1.88), steel, shiny=1.0)
    for side in (-0.35, 0.35):
        put(box, (0.02, 0.03, 0.12), (sx - 0.72, sy + side, 1.05), steel, shiny=1.0)

    # Стол: столешница на четырёх ножках.
    tx, ty = TABLE
    hx, hy = TABLE_HALF
    put(box, (hx, hy, 0.05), (tx, ty, TABLE_TOP - 0.05), (0.62, 0.40, 0.22, 1.0), shiny=0.5)
    for dx in (-hx + 0.12, hx - 0.12):
        for dy in (-hy + 0.12, 0.0, hy - 0.12):
            put(box, (0.05, 0.05, (TABLE_TOP - 0.1) / 2), (tx + dx, ty + dy, (TABLE_TOP - 0.1) / 2), (0.50, 0.32, 0.18, 1.0))

    # Три ведра: бумага, пластик, стекло — с крышками и значком на боку.
    for spot, color in zip(BINS, BIN_COLORS):
        put(cylinder, (0.36, 0.45, 0), (spot[0], spot[1], 0.45), (*color, 1.0), shiny=0.5)
        put(cylinder, (0.39, 0.04, 0), (spot[0], spot[1], 0.93), tuple(np.array(color[:3]) * 0.7) + (1.0,), shiny=0.6)
        put(box, (0.14, 0.06, 0.02), (spot[0], spot[1], 0.99), tuple(np.array(color[:3]) * 0.5) + (1.0,))
        put(box, (0.12, 0.005, 0.12), (spot[0], spot[1] + 0.365, 0.50), (1.0, 1.0, 1.0, 1.0))

    return ngeom


# Улицы со всех сторон дома. Масштаб как у двери: 1 мм ≈ 0.6 м.
HOUSE = (
    -HALF - 2 * WALL_THICK,
    NOOK_X + 2 * WALL_THICK,
    -HALF - 2 * WALL_THICK,
    CORRIDOR_HIGH + 4 * WALL_THICK + 16.0,  # за коридором квартиры соседей, почти как у Музы
)
# Перед подъездом (на запад) — двор: стоянка на 4 машины, забор с туями,
# раздвижные ворота. Улица начинается за двором.
YARD = 26.0  # глубина двора от фасада до забора
LOT = (HOUSE[0] - YARD, HOUSE[1], HOUSE[2], HOUSE[3])  # дом со двором — вокруг него улицы
FENCE_X = HOUSE[0] - YARD  # западный забор
# Дорожка: площадка у двери, от неё вправо (на юг) вдоль дома, внизу — на
# запад к воротам в углу двора.
APRON = (CORRIDOR_LOW - 0.2, LANE + 2.6)  # площадка у двери по y: от чуть ниже двери подъезда
SPINE = (HOUSE[0] - 14.8, HOUSE[0] - 10.8)  # дорожка вдоль дома по x: между ней и домом стоят машины
BOTTOM = (HOUSE[2] + 1.2, HOUSE[2] + 5.2)  # полоса к воротам по y
GATE_Y = (BOTTOM[0] + BOTTOM[1]) / 2  # середина ворот
YARD_GATE = (BOTTOM[1] - BOTTOM[0]) / 2  # полуширина ворот
GATE = {"open": 0.0, "called": False}  # раздвижные ворота
GATE_BUTTON = np.array([FENCE_X + 0.5, BOTTOM[1] + 0.4, HANDLE_HEIGHT])  # кнопка на столбике у ворот
SIDEWALK = 6.0  # ширина тротуара: фонари у бордюра, Муза проходит свободно

# Дверь дома на улицу: на первом этаже в западной стене, прямо по коридору от
# лифта. Стеклянная створка на петле у южного косяка открывается на улицу,
# рядом глухая стеклянная вставка.
STREET_LEAF = 3.6  # ширина створки: Муза проходит свободно
STREET_GLASS = 0.9  # ширина глухой вставки
STREET_LOW = CORRIDOR_LOW + 0.2  # южный косяк, здесь петля
STREET_HEIGHT = 4.6
STREET_HINGE = np.array([HOUSE[0] + WALL_THICK, STREET_LOW])
STREET = {"open": 0.0}


def street_panel():
    """Створка двери на улицу отрезком: закрыта — вдоль стены на север,
    открыта — стоит на тротуаре на запад."""
    angle = STREET["open"] * np.pi / 2
    direction = np.array([-np.sin(angle), np.cos(angle)])
    return STREET_HINGE.copy(), STREET_HINGE + direction * STREET_LEAF


def street_handle(open_part=None, side=-1.0):
    """Ручка двери на улицу: side −1 — из холла, +1 — с улицы."""
    part = STREET["open"] if open_part is None else open_part
    angle = part * np.pi / 2
    direction = np.array([-np.sin(angle), np.cos(angle)])
    inward = np.array([np.cos(angle), np.sin(angle)])  # в холл, пока закрыта
    spot = STREET_HINGE + direction * (STREET_LEAF - 0.35) + inward * 0.13 * -side
    return np.array([spot[0], spot[1], HANDLE_HEIGHT])


def in_street(position):
    """Муза на улице — во дворе перед подъездом."""
    return bool(WHERE["floor"] == 1 and position[0] < HOUSE[0])
ROAD = 12.0  # дорога на две полосы
CAR_SPEED = 15.0  # мм в секунду
CAR_HALF = 3.75  # половина длины машины на улице

# Кафе в доме напротив ворот, через дорогу на запад: стеклянная витрина,
# стеклянная дверь разъезжается сама, внутри столики и барная стойка с
# кофемашиной, за стойкой — бариста. Больше квартиры Музы. К нему от
# тротуара — зебра через дорогу.
CAFE_FACE = LOT[0] - 2 * SIDEWALK - ROAD - 0.4  # витрина: плоскость x
CAFE_Y = 0.0  # середина витрины вдоль улицы
CAFE_HALF = 12.0  # полуширина кафе вдоль улицы
CAFE_DEPTH = 18.0  # вглубь дома
CAFE_HEIGHT = 5.2  # потолок кафе; выше — этажи дома
CAFE_DOOR = 1.6  # полуширина двери
CROSS_HALF = 2.2  # полуширина зебры
COUNTER_X = CAFE_FACE - 12.0  # барная стойка: середина по x
COUNTER_Y = (CAFE_Y - 5.0, CAFE_Y + 3.0)  # стойка вдоль улицы; южнее — туалет
COUNTER_TOP = 1.2  # высота стойки: муха чуть выше неё
# Столики и стулья в размер Музы — она садится на стул, как на унитаз:
# сиденье на той же высоте, стол ей по грудь.
CAFE_TABLE_TOP = 2.4
CAFE_TABLE_RADIUS = 1.1
UMBRELLA_RADIUS = 2.6  # зонтики на террасе: широкие, восьмиугольные
CHAIR_SEAT = 1.45
CHAIR_TUCK = 1.5  # стул задвинут: от середины стола
CHAIR_PULL = 0.8  # насколько Муза его отодвигает
SEAT_SIDE = 1.8  # стоит сбоку от стула: отодвигает его и забирается вбок
BARISTA_STEP = 0.5  # подставка баристы за стойкой, во всю длину стойки
MACHINE_Y = COUNTER_Y[0] + 1.6  # кофемашина на стойке
COUNTER_EDGE = COUNTER_X + 0.95  # край столешницы со стороны зала
COUNTER_SERVE = (COUNTER_X + 0.55, COUNTER_Y[1] - 1.8)  # сюда бариста ставит кофе
CUP_SCALE = 1.5  # стаканчик крупнее
# Столики: ряд у северной стены и один у южной; между ними — проход к стойке.
CAFE_TABLES = ((CAFE_FACE - 4.0, CAFE_Y + 5.0), (CAFE_FACE - 8.0, CAFE_Y + 5.0), (CAFE_FACE - 4.0, CAFE_Y - 8.0))
# Дверь, Муза внутри; стаканчик Музы: None, "brew" — наливается под
# кофемашиной (fill), "counter" — на стойке, "held" — в лапках, "table" — на
# столике table; turn — бариста повернулась к кофемашине (0..1).
CAFE = {"open": 0.0, "inside": False, "cup": None, "fill": 0.0,
        "seat": None, "pull": 0.0, "sip": 0.0, "beat": 0.0, "order": None}  # order — сколько секунд бариста варит  # seat — столик Музы, pull — стул отодвинут, sip — подносит ко рту


def aisle_side(table):
    """С какой стороны столика проход (там стул Музы): +1 или −1 по y."""
    return -1.0 if table[1] > CAFE_Y else 1.0


def beside_chair(table):
    """Где Муза стоит сбоку от своего (отодвинутого) стула."""
    return np.array([table[0] + SEAT_SIDE, table[1] + aisle_side(table) * (CHAIR_TUCK + CHAIR_PULL)])


def cup_on_table(table):
    """Где стоит стаканчик Музы на её столике (x, y, высота)."""
    return (table[0], table[1] + aisle_side(table) * 0.55, CAFE_TABLE_TOP)
CROSSING = {"on": False, "paused": 0.0, "last": None, "car": None}  # Муза на зебре — машина ждёт; car — где машина вдоль улицы


def zebra_clear():
    """Машина на западной улице не на зебре и не въезжает на неё."""
    car = CROSSING["car"]
    return car is None or abs(car - CAFE_Y) > CROSS_HALF + CAR_HALF + 1.5


def in_cafe(position):
    return bool(WHERE["floor"] == 1 and position[0] < CAFE_FACE and abs(position[1] - CAFE_Y) < CAFE_HALF)



def street(scene, ngeom, moment):
    """Кольцо улиц вокруг дома: тротуар с бордюром, дорога с разметкой,
    тротуар напротив, серые дома по всем сторонам, фонари и машины."""

    def put(kind, size, pos, color, rotation=np.eye(3), glow=0.0, shiny=0.3):
        nonlocal ngeom
        geom = scene.geoms[ngeom]
        # Улица внизу, у подножия дома: всё опускаем на GROUND.
        mujoco.mjv_initGeom(
            geom, kind, np.array(size, dtype=float), np.array(pos, dtype=float) + np.array([0.0, 0.0, GROUND]),
            np.asarray(rotation, dtype=float).flatten(), np.array(color, dtype=float),
        )
        geom.emission = glow
        geom.specular = shiny
        ngeom += 1

    box = mujoco.mjtGeom.mjGEOM_BOX
    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    sphere = mujoco.mjtGeom.mjGEOM_SPHERE

    x0, x1, y0, y1 = LOT

    def rect(ax, bx, ay, by, height, color, z0=0.0, shiny=0.3):
        if bx - ax > 1e-6 and by - ay > 1e-6:
            put(box, ((bx - ax) / 2, (by - ay) / 2, height / 2), ((ax + bx) / 2, (ay + by) / 2, z0 + height / 2), color, shiny=shiny)

    def ring(near, far, height, color, z0=0.0, shiny=0.3):
        """Кольцо вокруг дома от расстояния near до far — четыре полосы без
        перекрытий, чтобы ничего не мигало."""
        rect(x0 - far, x1 + far, y0 - far, y0 - near, height, color, z0, shiny)
        rect(x0 - far, x1 + far, y1 + near, y1 + far, height, color, z0, shiny)
        rect(x0 - far, x0 - near, y0 - near, y1 + near, height, color, z0, shiny)
        rect(x1 + near, x1 + far, y0 - near, y1 + near, height, color, z0, shiny)

    grey = (0.62, 0.62, 0.60, 1.0)
    curb = (0.50, 0.50, 0.50, 1.0)
    asphalt = (0.18, 0.18, 0.19, 1.0)
    paint = (0.92, 0.92, 0.88, 1.0)
    d1, d2 = SIDEWALK, SIDEWALK + ROAD
    d3 = d2 + SIDEWALK

    ring(0.0, d1 - 0.12, 0.15, grey)
    ring(d1 - 0.12, d1 + 0.12, 0.20, curb)
    ring(d1 + 0.12, d2 - 0.12, 0.02, asphalt, shiny=0.1)
    ring(d2 - 0.12, d2 + 0.12, 0.20, curb)
    ring(d2 + 0.12, d3, 0.15, grey)

    # Разметка на каждой стороне — только между перекрёстками.
    middle = (d1 + d2) / 2
    sides = (
        ("x", y0 - middle, x0 - d1, x1 + d1, -1.0),
        ("x", y1 + middle, x0 - d1, x1 + d1, 1.0),
        ("y", x0 - middle, y0 - d1, y1 + d1, -1.0),
        ("y", x1 + middle, y0 - d1, y1 + d1, 1.0),
    )
    for axis, line, start, end, _ in sides:
        for edge in (line - (ROAD / 2 - 0.45), line + (ROAD / 2 - 0.45)):
            if axis == "x":
                put(box, ((end - start) / 2, 0.06, 0.004), ((start + end) / 2, edge, 0.024), paint)
            else:
                put(box, (0.06, (end - start) / 2, 0.004), (edge, (start + end) / 2, 0.024), paint)
        for along in np.arange(start + 0.75, end - 0.75, 3.0):
            if axis == "x":
                put(box, (0.75, 0.06, 0.004), (along, line, 0.024), paint)
            else:
                put(box, (0.06, 0.75, 0.004), (line, along, 0.024), paint)

    # Фонари вдоль тротуара у дома.
    for axis, line, start, end, outward in sides:
        spot_line = line - outward * (ROAD / 2 + 0.6)
        for along in np.linspace(start + 4, end - 4, 3):
            px, py = (along, spot_line) if axis == "x" else (spot_line, along)
            put(cylinder, (0.08, 3.0, 0), (px, py, 3.0), (0.25, 0.26, 0.28, 1.0), shiny=0.6)
            put(sphere, (0.30, 0, 0), (px, py, 6.1), (1.0, 0.92, 0.70, 1.0), glow=0.6)

    # Серые дома напротив: шесть этажей одинаковых тёмных окон — выше дома
    # Музы, из её окна видно их верхние этажи.
    depth, height = 11.0, 6 * STOREY + 2.0
    eye = LAST_EYE["eye"]
    for axis, line, start, end, outward in sides:
        face = line + outward * (ROAD / 2 + SIDEWALK + 0.4)
        length = end - start
        center = face + outward * depth / 2

        # Камера над этой улицей или за домом — дом полупрозрачный, чтобы
        # не загораживать комнату (как стены в кукольном домике).
        across = eye[1] if axis == "x" else eye[0]
        alpha = SEE_THROUGH if (across - line) * outward > 0.0 else 1.0
        cafe = axis == "y" and outward < 0  # на западе, напротив ворот двора — кафе
        if cafe and CAFE["inside"]:
            alpha = SEE_THROUGH  # Муза в кафе: дом над ним прозрачный, как стены квартиры
        wall = (0.50, 0.51, 0.53, alpha)

        if axis == "x":
            put(box, (length / 2, depth / 2, height / 2), ((start + end) / 2, center, height / 2), wall, shiny=0.1)
        elif cafe:
            _cafe_house(put, face, start, end, height, wall)
            _cafe(put, face)
        else:
            put(box, (depth / 2, length / 2, height / 2), (center, (start + end) / 2, height / 2), wall, shiny=0.1)
        if alpha < 1.0:
            continue
        for floor_number in range(6):
            z = 3.4 + floor_number * STOREY
            for along in np.arange(start + 3, end - 2, 4.0):
                if cafe and floor_number == 0 and abs(along - CAFE_Y) < CAFE_HALF + 1.0:
                    continue  # на первом этаже там витрина кафе
                wx, wy = (along, face - outward * 0.02) if axis == "x" else (face - outward * 0.02, along)
                size = (0.8, 0.02, 1.1) if axis == "x" else (0.02, 0.8, 1.1)
                put(box, size, (wx, wy, z), (0.25, 0.27, 0.30, 1.0), shiny=0.8)

    # Машины: по одной на улице, на соседних улицах в разные стороны.
    colors = ((0.70, 0.12, 0.12), (0.20, 0.35, 0.65), (0.85, 0.75, 0.20), (0.25, 0.25, 0.28))
    for number, (axis, line, start, end, _) in enumerate(sides):
        span = end - start
        forward = number % 2 == 0
        for lane, direction, color in (
            (line + ROAD / 4, 1.0, colors[number]) if forward else (line - ROAD / 4, -1.0, colors[number]),
        ):
            clock = moment - (CROSSING["paused"] if number == 2 else 0.0)
            run = (CAR_SPEED * clock + 17.0 * number + (0 if direction > 0 else span / 2)) % span
            along = start + (run if direction > 0 else span - run)
            if number == 2:
                # Муза на зебре или подходит к ней: машина встаёт перед
                # стоп-линией и ждёт; уже въехавшая на зебру проезжает.
                last = CROSSING["last"]
                front = along + direction * CAR_HALF
                stop = CAFE_Y - direction * (CROSS_HALF + 1.0)
                waiting = 0.0 <= direction * (stop - front) < 1.5
                if CROSSING["on"] and last is not None and waiting:
                    CROSSING["paused"] += moment - last
                CROSSING["last"] = moment
                CROSSING["car"] = along
            _car(put, axis, along, lane, direction, color)

    # Зебра через западную улицу к кафе.
    west = x0 - (SIDEWALK + ROAD / 2)
    for x in np.arange(west - ROAD / 2 + 0.6, west + ROAD / 2 - 0.3, 1.0):
        put(box, (0.3, CROSS_HALF, 0.004), (x, CAFE_Y, 0.026), paint)

    return ngeom


CAFE_FONT = {
    "C": ("111", "100", "100", "100", "111"),
    "A": ("010", "101", "111", "101", "101"),
    "F": ("111", "100", "110", "100", "100"),
    "E": ("111", "100", "110", "100", "111"),
}


def _cafe_house(put, face, start, end, height, wall):
    """Дом над кафе и по бокам: первый этаж вырезан под кафе."""

    box = mujoco.mjtGeom.mjGEOM_BOX
    depth = CAFE_DEPTH + 4.0
    back = face - depth
    c0, c1 = CAFE_Y - CAFE_HALF, CAFE_Y + CAFE_HALF

    def block(x0, x1, y0, y1, z0, z1):
        put(box, ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2), ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), wall, shiny=0.1)

    block(back, face, start, end, CAFE_HEIGHT, height)
    block(back, face, start, c0, 0.0, CAFE_HEIGHT)
    block(back, face, c1, end, 0.0, CAFE_HEIGHT)
    block(back, face - CAFE_DEPTH, c0, c1, 0.0, CAFE_HEIGHT)


def takeaway(put, bottom, fill, rotation=np.eye(3)):
    """Бумажный стаканчик с картонным пояском и крышкой; fill — сколько
    налито (пока наливается, крышки нет)."""

    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    x, y, z = bottom
    up = np.asarray(rotation)[:, 2]
    k = CUP_SCALE
    at = lambda h: (x + up[0] * h * k, y + up[1] * h * k, z + up[2] * h * k)
    put(cylinder, (0.13 * k, 0.2 * k, 0), at(0.2), (0.97, 0.96, 0.93, 1.0), rotation=rotation, shiny=0.5)
    put(cylinder, (0.135 * k, 0.07 * k, 0), at(0.2), (0.62, 0.44, 0.26, 1.0), rotation=rotation)
    if fill < 1.0:
        put(cylinder, (0.12 * k, 0.01 * k, 0), at(0.02 + 0.36 * fill), (0.30, 0.17, 0.08, 1.0), rotation=rotation)
        return
    put(cylinder, (0.14 * k, 0.025 * k, 0), at(0.42), (0.95, 0.95, 0.94, 1.0), rotation=rotation, shiny=0.6)


def _cafe(put, face):
    """Кафе на первом этаже дома напротив: фасад в плоскости x = face,
    смотрит на восток, к дороге. Стеклянная витрина и дверь, внутри —
    деревянный пол, лампы, столики, барная стойка с кофемашиной."""

    box = mujoco.mjtGeom.mjGEOM_BOX
    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    sphere = mujoco.mjtGeom.mjGEOM_SPHERE
    ellipsoid = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    wood = (0.32, 0.19, 0.11, 1.0)
    light_wood = (0.62, 0.44, 0.28, 1.0)
    warm = (1.0, 0.82, 0.55, 1.0)
    cream = (0.96, 0.93, 0.85, 1.0)
    red = (0.78, 0.16, 0.16, 1.0)
    glass = (0.72, 0.84, 0.90, 0.22)
    steel = (0.70, 0.72, 0.75, 1.0)
    dark = (0.12, 0.12, 0.13, 1.0)
    c0, c1 = CAFE_Y - CAFE_HALF, CAFE_Y + CAFE_HALF
    back = face - CAFE_DEPTH
    mid = face - CAFE_DEPTH / 2

    # Пол, потолок, стены изнутри.
    put(box, (CAFE_DEPTH / 2, CAFE_HALF, 0.02), (mid, CAFE_Y, 0.02), light_wood, shiny=0.5)
    if not CAFE["inside"]:
        put(box, (CAFE_DEPTH / 2, CAFE_HALF, 0.03), (mid, CAFE_Y, CAFE_HEIGHT - 0.03), cream)  # потолок: внутри камера смотрит сверху
    put(box, (0.03, CAFE_HALF, CAFE_HEIGHT / 2), (back + 0.03, CAFE_Y, CAFE_HEIGHT / 2), (0.86, 0.74, 0.58, 1.0))
    for y in (c0 + 0.03, c1 - 0.03):
        put(box, (CAFE_DEPTH / 2, 0.03, CAFE_HEIGHT / 2), (mid, y, CAFE_HEIGHT / 2), (0.86, 0.74, 0.58, 1.0))

    # Туалет кафе (в юго-западном углу): светлая плитка на полу.
    wc = BATHROOMS["cafe"]
    put(box, ((wc["BATH_X"] - wc["BATH_WEST"]) / 2, (wc["BATH_HIGH"] - wc["BATH_LOW"]) / 2, 0.01),
        ((wc["BATH_X"] + wc["BATH_WEST"]) / 2, (wc["BATH_HIGH"] + wc["BATH_LOW"]) / 2, 0.05), (0.88, 0.90, 0.92, 1.0))

    # Стеклянная витрина в деревянных переплётах, стеклянная дверь из двух
    # створок — разъезжаются, когда Муза подходит.
    for lo, hi in ((c0, CAFE_Y - CAFE_DOOR), (CAFE_Y + CAFE_DOOR, c1)):
        put(box, (0.02, (hi - lo) / 2, 2.0), (face, (lo + hi) / 2, 2.3), glass, shiny=0.9)
    for y in list(np.linspace(c0, CAFE_Y - CAFE_DOOR, 4)) + list(np.linspace(CAFE_Y + CAFE_DOOR, c1, 4)):
        put(box, (0.06, 0.08, 2.1), (face, y, 2.3), wood)
    for z in (0.25, 4.35):
        put(box, (0.07, CAFE_HALF, 0.1), (face, CAFE_Y, z), wood)
    shift = CAFE["open"] * (CAFE_DOOR - 0.1)
    for side in (-1.0, 1.0):
        y = CAFE_Y + side * (CAFE_DOOR / 2 + shift)
        put(box, (0.03, CAFE_DOOR / 2, 2.0), (face + 0.08, y, 2.3), glass, shiny=0.9)
        put(box, (0.04, 0.05, 2.0), (face + 0.08, y - side * (CAFE_DOOR / 2 - 0.05), 2.3), wood)

    # Полосатая маркиза и вывеска. Муза внутри — их нет: камера смотрит
    # сверху, и они закрывали бы её.
    outside = not CAFE["inside"]
    tilt = np.radians(25.0)
    slope = np.array([[np.cos(tilt), 0.0, np.sin(tilt)], [0.0, 1.0, 0.0], [-np.sin(tilt), 0.0, np.cos(tilt)]])
    stripes = 24
    width = 2 * (CAFE_HALF + 0.3) / stripes
    for number in range(stripes if outside else 0):
        y = c0 - 0.3 + width * (number + 0.5)
        color = red if number % 2 == 0 else cream
        put(box, (0.9, width / 2, 0.03), (face + 0.9 * np.cos(tilt), y, 4.95 - 0.9 * np.sin(tilt)), color, rotation=slope)
        put(box, (0.02, width / 2, 0.15), (face + 1.8 * np.cos(tilt), y, 4.95 - 1.8 * np.sin(tilt) - 0.12), color)

    # Вывеска над маркизой: тёмно-зелёная доска, светлые буквы CAFE.
    if outside:
        put(box, (0.05, 2.9, 0.95), (face + 0.06, CAFE_Y, 6.3), (0.10, 0.28, 0.20, 1.0))
    pixel = 0.13
    text = "CAFE" if outside else ""
    advance = 4 * pixel * 2
    first = CAFE_Y - advance * (len(text) - 1) / 2 - pixel * 2
    for place, letter in enumerate(text):
        for row, bits in enumerate(CAFE_FONT[letter]):
            for column, bit in enumerate(bits):
                if bit == "1":
                    y = first + place * advance + column * pixel * 2 + pixel
                    z = 6.3 + (2 - row) * pixel * 2
                    put(box, (0.02, pixel, pixel), (face + 0.12, y, z), cream, glow=0.5)

    # Фонарики у двери и цветы в горшках.
    for side in (-1.0, 1.0):
        put(sphere, (0.18, 0, 0), (face + 0.3, CAFE_Y + side * (CAFE_DOOR + 0.5), 3.4), warm, glow=0.8)
        put(cylinder, (0.3, 0.3, 0), (face + 0.6, CAFE_Y + side * (CROSS_HALF + 0.8), 0.3), (0.62, 0.35, 0.22, 1.0))
        put(sphere, (0.45, 0, 0), (face + 0.6, CAFE_Y + side * (CROSS_HALF + 0.8), 0.95), (0.20, 0.50, 0.24, 1.0))

    def chair(x, y, facing):
        """Стул: сиденье, спинка с той стороны, что дальше от стола, ножки."""
        put(box, (0.45, 0.45, 0.06), (x, y, CHAIR_SEAT), wood, shiny=0.5)
        put(box, (0.45, 0.06, 0.6), (x, y - facing * 0.42, CHAIR_SEAT + 0.66), wood)
        for dx in (-0.38, 0.38):
            for dy in (-0.38, 0.38):
                put(cylinder, (0.04, CHAIR_SEAT / 2, 0), (x + dx, y + dy, CHAIR_SEAT / 2), dark)

    def table(x, y, top_color, umbrella=None, cups=True):
        """Столик в размер Музы и два стула по сторонам; стул Музы, если это
        её столик, отодвинут."""
        put(cylinder, (0.08, CAFE_TABLE_TOP / 2, 0), (x, y, CAFE_TABLE_TOP / 2), dark)
        put(cylinder, (0.5, 0.03, 0), (x, y, 0.03), dark)
        put(cylinder, (CAFE_TABLE_RADIUS, 0.04, 0), (x, y, CAFE_TABLE_TOP - 0.04), top_color, shiny=0.6)
        if cups:
            put(cylinder, (0.12, 0.1, 0), (x + 0.3, y - aisle_side((x, y)) * 0.5, CAFE_TABLE_TOP + 0.1), (0.98, 0.98, 0.96, 1.0), shiny=0.8)
        for side in (-1.0, 1.0):
            pulled = CAFE["pull"] if CAFE["seat"] == (x, y) and side == aisle_side((x, y)) else 0.0
            chair(x, y + side * (CHAIR_TUCK + CHAIR_PULL * pulled), -side)
        if umbrella is not None and not CAFE["inside"]:
            # Зонтик: восьмиугольный купол из восьми скатов, полосами двух
            # цветов; стойка чуть выше купола, сверху маленький шарик.
            apex = CAFE_TABLE_TOP + 2.6
            put(cylinder, (0.04, (apex + 0.15) / 2, 0), (x, y, (apex + 0.15) / 2), (0.85, 0.85, 0.82, 1.0))
            put(sphere, (0.08, 0, 0), (x, y, apex + 0.2), (0.85, 0.85, 0.82, 1.0))
            span, tilt = UMBRELLA_RADIUS, np.radians(18.0)
            slope = np.array([[np.cos(tilt), 0.0, np.sin(tilt)], [0.0, 1.0, 0.0], [-np.sin(tilt), 0.0, np.cos(tilt)]])
            for panel in range(8):
                turn = panel * np.pi / 4
                spin = np.array([[np.cos(turn), -np.sin(turn), 0.0], [np.sin(turn), np.cos(turn), 0.0], [0.0, 0.0, 1.0]])
                reach = span / 2 * np.cos(tilt)
                put(box, (span / 2, span * np.tan(np.pi / 8), 0.02),
                    (x + np.cos(turn) * reach, y + np.sin(turn) * reach, apex - span / 2 * np.sin(tilt)),
                    umbrella if panel % 2 == 0 else cream, rotation=spin @ slope)

    # Столики на тротуаре с зонтиками и внутри.
    for number, y in enumerate((CAFE_Y - 7.0, CAFE_Y + 7.0)):
        table(face + 2.6, y, cream, red if number == 0 else cream)
    for x, y in CAFE_TABLES:
        table(x, y, cream, cups=CAFE["seat"] != (x, y))

    # Лампы под потолком.
    for x in (face - 4.5, face - 9.0, face - 13.5):
        for y in (CAFE_Y - 6.0, CAFE_Y + 6.0):
            put(cylinder, (0.01, 0.5, 0), (x, y, CAFE_HEIGHT - 0.5), dark)
            put(sphere, (0.3, 0, 0), (x, y, CAFE_HEIGHT - 1.05), warm, glow=0.9)

    # Барная стойка: тёмное дерево, светлая столешница; на ней кофемашина,
    # чашки, кофе Музы, когда она у стойки.
    y0, y1 = COUNTER_Y
    top = COUNTER_TOP
    # За стойкой — подставка для баристы, чтобы её было видно над стойкой.
    put(box, (0.85, (COUNTER_Y[1] - COUNTER_Y[0]) / 2, BARISTA_STEP / 2), (COUNTER_X - 1.45, (COUNTER_Y[0] + COUNTER_Y[1]) / 2, BARISTA_STEP / 2), wood)
    put(box, (0.6, (y1 - y0) / 2, top / 2), (COUNTER_X, (y0 + y1) / 2, top / 2), wood)
    put(box, ((COUNTER_EDGE - COUNTER_X + 0.6) / 2, (y1 - y0) / 2 + 0.1, 0.05), ((COUNTER_EDGE + COUNTER_X - 0.6) / 2, (y0 + y1) / 2, top + 0.05), light_wood, shiny=0.7)
    surface = top + 0.1  # верх столешницы
    machine_y = MACHINE_Y
    # Кофемашина стоит у края стойки к залу, краником — к баристе.
    put(box, (0.45, 0.8, 0.55), (COUNTER_X + 0.35, machine_y, top + 0.65), steel, shiny=1.0)
    put(box, (0.46, 0.82, 0.08), (COUNTER_X + 0.35, machine_y, top + 1.25), dark, shiny=0.8)
    # Носик на кронштейне со стороны баристы: стаканчик под ним стоит на
    # столешнице между машиной и баристой, а не в машине.
    put(box, (0.2, 0.12, 0.06), (COUNTER_X - 0.25, machine_y, top + 0.98), steel, shiny=1.0)
    put(cylinder, (0.12, 0.08, 0), (COUNTER_X - 0.38, machine_y, top + 0.9), dark, shiny=0.9)
    put(sphere, (0.06, 0, 0), (COUNTER_X - 0.11, machine_y, top + 0.95), (0.3, 1.0, 0.4, 1.0), glow=1.0)
    for number in range(4):
        put(cylinder, (0.09, 0.08, 0), (COUNTER_X + 0.5, machine_y + 1.2 + number * 0.25, top + 0.18), cream)
    # Стаканчик Музы: под кофемашиной (наливается), на стойке, на столике.
    if CAFE["cup"] == "brew":
        takeaway(put, (COUNTER_X - 0.38, machine_y, surface), CAFE["fill"])
    elif CAFE["cup"] == "counter":
        takeaway(put, (COUNTER_SERVE[0], COUNTER_SERVE[1], surface), 1.0)
    elif CAFE["cup"] == "table" and CAFE["seat"] is not None:
        takeaway(put, cup_on_table(CAFE["seat"]), 1.0)
    # К кофе — круассан на тарелке.
    if CAFE["seat"] is not None and CAFE["cup"] in ("sip", "table"):
        tx, ty = CAFE["seat"]
        near = aisle_side(CAFE["seat"]) * 0.45
        put(cylinder, (0.4, 0.015, 0), (tx + 0.55, ty + near, CAFE_TABLE_TOP + 0.02), (0.98, 0.98, 0.97, 1.0), shiny=0.7)
        if CAFE["cup"] == "sip":
            bend = np.array([[np.cos(0.4), -np.sin(0.4), 0.0], [np.sin(0.4), np.cos(0.4), 0.0], [0.0, 0.0, 1.0]])
            put(mujoco.mjtGeom.mjGEOM_ELLIPSOID, (0.32, 0.14, 0.1), (tx + 0.55, ty + near, CAFE_TABLE_TOP + 0.12), (0.86, 0.58, 0.25, 1.0), rotation=bend, shiny=0.4)

    # Полки с банками на задней стене и меню.
    for z in (2.4, 3.3):
        put(box, (0.25, (y1 - y0) / 2, 0.04), (back + 0.3, (y0 + y1) / 2, z), wood)
        for number, y in enumerate(np.arange(y0 + 0.5, y1 - 0.3, 0.8)):
            colors = ((0.55, 0.35, 0.20), (0.85, 0.75, 0.40), (0.35, 0.50, 0.30))
            put(cylinder, (0.15, 0.25, 0), (back + 0.3, y, z + 0.29), (*colors[number % 3], 1.0), shiny=0.8)
    put(box, (0.04, 2.0, 0.7), (back + 0.08, CAFE_Y + 3.5, 3.4), (0.10, 0.11, 0.12, 1.0))
    for line in range(4):
        put(box, (0.02, 1.4 - 0.2 * (line % 2), 0.04), (back + 0.12, CAFE_Y + 3.5, 3.85 - line * 0.3), cream, glow=0.3)


def _car(put, axis, along, lane, direction, color):
    """Машина: кузов, стёкла, четыре колеса, фары и стопы. put сам ведёт
    счёт фигур в сцене."""

    box = mujoco.mjtGeom.mjGEOM_BOX
    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    sphere = mujoco.mjtGeom.mjGEOM_SPHERE

    def at(forward, side, z):
        """Точка в системе машины → в мир."""
        if axis == "x":
            return (along + forward * direction, lane + side, z)
        return (lane + side, along + forward * direction, z)

    def size(long_half, side_half, z_half):
        return (long_half, side_half, z_half) if axis == "x" else (side_half, long_half, z_half)

    wheel = (
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float) if axis == "x"
        else np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    )

    put(box, size(3.75, 1.5, 0.6), at(0, 0, 0.95), (*color, 1.0), shiny=0.9)
    put(box, size(2.0, 1.35, 0.55), at(-0.4, 0, 2.1), (0.15, 0.19, 0.24, 1.0), shiny=1.0)
    for forward in (-2.4, 2.4):
        for side in (-1.45, 1.45):
            put(cylinder, (0.45, 0.18, 0), at(forward, side, 0.45), (0.08, 0.08, 0.08, 1.0), wheel)
    for side in (-1.0, 1.0):
        put(sphere, (0.18, 0, 0), at(3.72, side, 1.05), (1.0, 0.95, 0.75, 1.0), glow=0.8)
        put(sphere, (0.15, 0, 0), at(-3.72, side, 1.05), (0.9, 0.1, 0.1, 1.0), glow=0.6)


def front_door(scene, ngeom):
    """Входная дверь: коробка с двух сторон стены и полотно, которое
    поворачивается на петле в коридор. Ручки с обеих сторон — на высоте
    поднятой лапки."""

    def put(size, pos, color, rotation=np.eye(3), kind=mujoco.mjtGeom.mjGEOM_BOX, shiny=0.4):
        nonlocal ngeom
        geom = scene.geoms[ngeom]
        mujoco.mjv_initGeom(
            geom, kind, np.array(size, dtype=float), np.array(pos, dtype=float),
            np.asarray(rotation, dtype=float).flatten(), np.array(color, dtype=float),
        )
        geom.specular = shiny
        ngeom += 1

    wood = (0.42, 0.26, 0.15, 1.0)
    frame = (0.93, 0.92, 0.88, 1.0)
    brass = (0.85, 0.75, 0.35, 1.0)
    x = FRONT_DOOR_X
    half = FRONT_DOOR_HALF

    # Коробка по обе стороны стены: два косяка и перекладина.
    for face in (HALF - 0.02, HALF + 2 * WALL_THICK + 0.02):
        for side in (-1.0, 1.0):
            put((0.08, 0.02, FRONT_HEIGHT / 2), (x + side * (half + 0.08), face, FRONT_HEIGHT / 2), frame)
        put((half + 0.16, 0.02, 0.08), (x, face, FRONT_HEIGHT + 0.08), frame)

    # Полотно на петле.
    hinge, free = front_panel()
    along = (free - hinge) / np.linalg.norm(free - hinge)
    turn = np.array([[along[0], -along[1], 0.0], [along[1], along[0], 0.0], [0.0, 0.0, 1.0]])
    middle = (hinge + free) / 2
    height = FRONT_HEIGHT - 0.05
    put((half, 0.06, height / 2), (middle[0], middle[1], height / 2), wood, turn)
    for side in (-1.0, 1.0):
        normal = np.array([-along[1], along[0]]) * side
        panel = middle + normal * 0.065
        put((half - 0.3, 0.01, height / 2 - 0.5), (panel[0], panel[1], height / 2 + 0.1), (0.48, 0.30, 0.18, 1.0), turn)

    for side in (-1.0, 1.0):
        knob = front_handle(side=side)
        put((0.09, 0, 0), knob, brass, kind=mujoco.mjtGeom.mjGEOM_SPHERE, shiny=1.0)

    return ngeom


def draw(scene, ngeom, poops, moment=0.0, home=True):
    """Пол, кухня, улицы, входная дверь, ковёр, подушка на матрасе и какашки.
    home=False — Муза не на своём этаже: от квартиры ничего, только улица."""

    ngeom = street(scene, ngeom, moment)
    if not home:
        # Туалет кафе: тот же унитаз, те же перегородки и дверь; какашки —
        # если Муза сейчас в туалете кафе.
        in_cafe_wc = BATH_NOW["name"] == "cafe"
        with bathroom_as("cafe"):
            ngeom = toilet(scene, ngeom, lift=GROUND)
            ngeom = bathroom(scene, ngeom, lift=GROUND)
        for spot in poops if in_cafe_wc else ():
            mujoco.mjv_initGeom(
                scene.geoms[ngeom], mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.array([0.10, 0.08, 0.05]),
                np.array(spot, dtype=float) + np.array([0.0, 0.0, GROUND]), np.eye(3).flatten(),
                np.array([0.30, 0.18, 0.08, 1.0]),
            )
            ngeom += 1
        return ngeom
    ngeom = floor(scene, ngeom)
    ngeom = front_door(scene, ngeom)
    ngeom = kitchen(scene, ngeom)
    ngeom = window(scene, ngeom)
    ngeom = side_windows(scene, ngeom)

    identity = np.eye(3).flatten()

    def put(kind, size, pos, color, rotation=identity):
        nonlocal ngeom
        mujoco.mjv_initGeom(
            scene.geoms[ngeom], kind, np.array(size, dtype=float),
            np.array(pos, dtype=float), rotation, np.array(color, dtype=float),
        )
        ngeom += 1

    box = mujoco.mjtGeom.mjGEOM_BOX
    ellipsoid = mujoco.mjtGeom.mjGEOM_ELLIPSOID

    # Ковёр: бордовая кайма, узор полосами, середина.
    # Слои ковра разведены по высоте с запасом: если они касаются друг
    # друга или пола, видеокарта путает, какой ближе, и узор мигает.
    thin = 0.0015
    cx, cy = CARPET
    put(box, (5.3, 5.3, thin), (cx, cy, CARPET_BASE + CARPET_STEP * 1), (0.45, 0.10, 0.12, 1.0))
    put(box, (4.9, 4.9, thin), (cx, cy, CARPET_BASE + CARPET_STEP * 2), (0.80, 0.62, 0.35, 1.0))

    for shift in np.linspace(-3.9, 3.9, 7):
        put(box, (0.12, 4.6, thin), (cx + shift, cy, CARPET_BASE + CARPET_STEP * 3), (0.55, 0.20, 0.18, 1.0))

    put(box, (1.4, 1.4, thin), (cx, cy, CARPET_BASE + CARPET_STEP * 4), (0.25, 0.35, 0.55, 1.0))

    # Матрас: полосатый чехол и подушка.
    middle = mattress_center()
    half = (MATTRESS_HIGH - MATTRESS_LOW) / 2.0

    for shift in np.linspace(-half[0] + 0.3, half[0] - 0.3, 6):
        put(box, (0.05, half[1] - 0.05, 0.002), (middle[0] + shift, middle[1], MATTRESS_TOP + CARPET_STEP), (0.62, 0.72, 0.90, 1.0))

    # Подушка: прямоугольная, объёмная — пухлая середина выше краёв, шов по краю.
    pillow = np.array([middle[0], MATTRESS_HIGH[1] - 0.75])
    put(box, (1.2, 0.5, 0.09), (pillow[0], pillow[1], MATTRESS_TOP + 0.09), (0.97, 0.97, 0.99, 1.0))
    put(box, (1.08, 0.4, 0.07), (pillow[0], pillow[1], MATTRESS_TOP + 0.2), (0.99, 0.99, 1.0, 1.0))
    put(box, (1.22, 0.52, 0.012), (pillow[0], pillow[1], MATTRESS_TOP + 0.14), (0.88, 0.88, 0.92, 1.0))

    # Плинтусы — чтобы стена «стояла» на полу, даже когда прозрачная.
    for name, (center, half_wall, _) in WALLS.items():
        if wall_z(name)[0] > 0.0:
            continue
        put(box, (half_wall[0] + 0.02, half_wall[1] + 0.02, 0.12), (center[0], center[1], 0.12), (0.45, 0.33, 0.22, 1.0))

    ngeom = toilet(scene, ngeom)
    ngeom = bathroom(scene, ngeom)

    for spot in poops:
        put(ellipsoid, (0.10, 0.08, 0.05), spot, (0.30, 0.18, 0.08, 1.0))

    return ngeom


# --- Ванные: дома и в кафе -------------------------------------------------
# Туалет в кафе — та же ванная, что дома, сдвинутая в юго-западный угол
# кафе: те же унитаз, дверь, точки подхода. Пока Муза ходит в туалет в кафе,
# эти имена модуля указывают на кафе — и вся механика (дверь лапкой, сесть,
# смыть, закрыть задней лапкой) работает без изменений.
from contextlib import contextmanager

_BATH_NAMES = (
    "TOILET", "TOILET_BODY", "TOILET_FRONT", "BATH_X", "BATH_WEST", "BATH_LOW", "BATH_HIGH",
    "DOOR_LOW", "DOOR_HIGH", "HINGE", "DOOR_OUTSIDE", "DOOR_OPEN_SPOT", "DOOR_KICK_SPOT",
    "DOOR_AWAY", "BATH_CORNER", "DOOR",
)
BATHROOMS = {"home": {name: globals()[name] for name in _BATH_NAMES}}
_shift = np.array([CAFE_FACE - CAFE_DEPTH - BATH_WEST, CAFE_Y - CAFE_HALF - BATH_LOW])
BATHROOMS["cafe"] = {}
for _name, _value in BATHROOMS["home"].items():
    if _name == "DOOR":
        BATHROOMS["cafe"][_name] = {"open": 0.0}
    elif _name in ("BATH_X", "BATH_WEST"):
        BATHROOMS["cafe"][_name] = _value + _shift[0]
    elif _name in ("BATH_LOW", "BATH_HIGH", "DOOR_LOW", "DOOR_HIGH"):
        BATHROOMS["cafe"][_name] = _value + _shift[1]
    else:
        BATHROOMS["cafe"][_name] = _value + _shift
BATH_NOW = {"name": "home"}


def use_bathroom(name):
    """Какой туалет сейчас «тот самый»: дома или в кафе."""
    globals().update(BATHROOMS[name])
    BATH_NOW["name"] = name


@contextmanager
def bathroom_as(name):
    """На время — другой туалет (нарисовать, построить стены), потом обратно."""
    before = BATH_NOW["name"]
    use_bathroom(name)
    try:
        yield
    finally:
        use_bathroom(before)
