"""Предметы в арене, нарисованные примитивами MuJoCo.

Пока здесь одна сигарета: белая палочка с фильтром, уголёк и дым.
"""

import mujoco
import numpy as np


LYING = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0])


def along(direction):
    """Матрица поворота: своя ось x сигареты ложится вдоль direction."""

    forward = np.asarray(direction, dtype=float)
    forward = forward / (np.linalg.norm(forward) or 1.0)

    side = np.cross([0.0, 0.0, 1.0], forward)
    length = np.linalg.norm(side)

    if length < 1e-6:
        return LYING

    side /= length
    up = np.cross(forward, side)

    # Цилиндр стоит вдоль своей z, поэтому собственная z должна смотреть вдоль x.
    # -side: без него матрица зеркальная, и у банки просвечивало дно.
    return np.column_stack([up, -side, forward]).flatten()


def cigarette(scene, ngeom, position, lit, moment, direction=None, scale=1.0, smoke=True):
    """Сигарета: белая палочка с фильтром, уголёк и дым от уголька.
    smoke=False — без дымка от уголька (в лапках дым идёт выдохами)."""

    lying = LYING if direction is None else along(direction)

    def place(offset):
        """Смещение вдоль сигареты: её длинная ось — это собственная z."""
        turn = np.asarray(lying, dtype=float).reshape(3, 3)
        return position + turn @ np.array([0.0, 0.0, offset * scale])

    mujoco.mjv_initGeom(
        scene.geoms[ngeom],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([0.11 * scale, 0.45 * scale, 0.0]),
        pos=place(-0.10),
        mat=lying,
        rgba=np.array([0.95, 0.95, 0.92, 1.0]),
    )

    ngeom += 1

    mujoco.mjv_initGeom(
        scene.geoms[ngeom],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([0.115 * scale, 0.22 * scale, 0.0]),
        pos=place(0.57),
        mat=lying,
        rgba=np.array([0.80, 0.62, 0.32, 1.0]),
    )

    ngeom += 1

    if not lit:
        return ngeom

    # Уголёк.
    mujoco.mjv_initGeom(
        scene.geoms[ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.12 * scale, 0.0, 0.0]),
        pos=place(-0.56),
        mat=np.eye(3).flatten(),
        rgba=np.array([1.0, 0.35, 0.08, 1.0]),
    )

    ngeom += 1

    # Дым: три клуба, поднимаются и тают.
    for puff in range(3 if smoke else 0):

        rise = (moment * 0.6 + puff * 0.4) % 1.2

        mujoco.mjv_initGeom(
            scene.geoms[ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([(0.06 + rise * 0.07) * scale, 0.0, 0.0]),
            pos=place(-0.56) + np.array([0.0, 0.0, (0.2 + rise) * scale]),
            mat=np.eye(3).flatten(),
            rgba=np.array([0.85, 0.85, 0.88, 0.30 * (1.0 - rise / 1.2)]),
        )

        ngeom += 1

    return ngeom




def beam(scene, ngeom, position, color, width=3.0):
    """Столб до пола: чтобы источник было видно издалека."""

    line = scene.geoms[ngeom]

    mujoco.mjv_initGeom(
        line,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.array([width, 0.0, 0.0]),
        np.asarray(position, dtype=float),
        np.eye(3).flatten(),
        np.array([*color, 0.7]),
    )
    mujoco.mjv_connector(
        line,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.array([position[0], position[1], 0.0]),
        np.asarray(position, dtype=float),
    )

    return ngeom + 1


# Ореол вокруг горящей лампочки: радиус и прозрачность каждого слоя.
GLOW = ((0.38, 0.16), (0.55, 0.09), (0.78, 0.05))


def bulb(scene, ngeom, position, lit, shade=1.0):
    """Лампочка: стеклянная колба, цоколь, нить накала и свечение вокруг."""

    position = np.asarray(position, dtype=float)
    identity = np.eye(3).flatten()

    # Свечение рисуем первым, чтобы стекло и цоколь легли поверх него.
    if lit:

        for radius, alpha in GLOW:

            mujoco.mjv_initGeom(
                scene.geoms[ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, 0.0, 0.0]),
                position,
                identity,
                np.array([1.0, 0.93, 0.55, alpha * shade]),
            )

            ngeom += 1

    glass = (
        np.array([1.0, 0.95, 0.55, 0.55])
        if lit
        else np.array([0.75, 0.78, 0.80, 0.45])
    )

    mujoco.mjv_initGeom(
        scene.geoms[ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.26, 0.0, 0.0]),
        position,
        identity,
        glass * np.array([shade, shade, shade, 1.0]),
    )

    ngeom += 1

    # Нить накала: маленький яркий шарик внутри колбы.
    if lit:

        mujoco.mjv_initGeom(
            scene.geoms[ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.09, 0.0, 0.0]),
            position,
            identity,
            np.array([1.0, 0.95, 0.45, 1.0]),
        )

        ngeom += 1

    # Шейка и цоколь сверху: лампочка как будто висит.
    mujoco.mjv_initGeom(
        scene.geoms[ngeom],
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([0.11, 0.07, 0.0]),
        position + np.array([0.0, 0.0, 0.26]),
        identity,
        np.array([0.55, 0.55, 0.58, 1.0]),
    )

    ngeom += 1

    mujoco.mjv_initGeom(
        scene.geoms[ngeom],
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([0.13, 0.10, 0.0]),
        position + np.array([0.0, 0.0, 0.42]),
        identity,
        np.array([0.72, 0.70, 0.62, 1.0]),
    )

    return ngeom + 1


def burger(scene, ngeom, position, shade=1.0, scale=1.0):
    """Бургер: булка, котлета, сыр, салат и крышка."""

    position = np.asarray(position, dtype=float)
    identity = np.eye(3).flatten()

    layers = [
        # (полуразмеры, высота над точкой, цвет)
        ((0.34, 0.34, 0.12), 0.00, (0.80, 0.58, 0.28)),  # нижняя булка
        ((0.36, 0.36, 0.05), 0.13, (0.35, 0.65, 0.30)),  # салат
        ((0.33, 0.33, 0.09), 0.23, (0.42, 0.24, 0.13)),  # котлета
        ((0.35, 0.35, 0.03), 0.33, (0.95, 0.75, 0.20)),  # сыр
    ]

    for size, height, color in layers:

        mujoco.mjv_initGeom(
            scene.geoms[ngeom],
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            np.array(size) * scale,
            position + np.array([0.0, 0.0, height * scale]),
            identity,
            np.array([*(np.asarray(color) * shade), 1.0]),
        )

        ngeom += 1

    # Верхняя булка — купол.
    mujoco.mjv_initGeom(
        scene.geoms[ngeom],
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        np.array([0.34, 0.34, 0.24]) * scale,
        position + np.array([0.0, 0.0, 0.42 * scale]),
        identity,
        np.array([*(np.array([0.85, 0.62, 0.30]) * shade), 1.0]),
    )

    return ngeom + 1


def _painter(scene, position, turn, shade, scale=1.0):
    """Кисть для предметов: кладёт фигуру в сцену в системе предмета."""

    def put(ngeom, kind, size, offset, color, rotation=np.eye(3), shiny=0.5, glow=0.0):
        geom = scene.geoms[ngeom]
        mujoco.mjv_initGeom(
            geom,
            kind,
            np.array(size, dtype=float) * scale,
            position + turn @ np.array(offset, dtype=float) * scale,
            (turn @ rotation).flatten(),
            np.array([*(np.asarray(color[:3]) * shade), color[3] if len(color) > 3 else 1.0]),
        )
        geom.specular = shiny
        geom.shininess = shiny
        geom.emission = glow
        return ngeom + 1

    return put


def _spin(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _lean(angle):
    """Наклон вокруг оси x предмета."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


MONSTER_HEIGHT = 0.88  # от донышка до крышки


def monster(scene, ngeom, position, shade=1.0, up=None):
    """Банка энергетика Monster: чёрная жесть с блеском, серебряные донышко,
    плечико и крышка с ключом, три рваных зелёных когтя с двух сторон.
    position — середина донышка, up — куда смотрит её верх."""

    position = np.asarray(position, dtype=float)
    turn = np.eye(3) if up is None else np.asarray(along(up), dtype=float).reshape(3, 3)
    put = _painter(scene, position, turn, shade)

    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    box = mujoco.mjtGeom.mjGEOM_BOX

    black = (0.05, 0.05, 0.06)
    silver = (0.72, 0.74, 0.78)
    green = (0.42, 1.0, 0.08)

    # Донышко, сама банка, плечико, горлышко и ободок крышки.
    ngeom = put(ngeom, cylinder, (0.225, 0.025, 0), (0, 0, 0.025), silver, shiny=0.9)
    ngeom = put(ngeom, cylinder, (0.250, 0.340, 0), (0, 0, 0.390), black, shiny=0.9)
    ngeom = put(ngeom, cylinder, (0.238, 0.020, 0), (0, 0, 0.745), black, shiny=0.9)
    ngeom = put(ngeom, cylinder, (0.222, 0.020, 0), (0, 0, 0.785), silver, shiny=0.9)
    ngeom = put(ngeom, cylinder, (0.205, 0.025, 0), (0, 0, 0.830), silver, shiny=0.9)
    ngeom = put(ngeom, cylinder, (0.212, 0.010, 0), (0, 0, 0.862), silver, shiny=1.0)

    # Крышка чуть темнее, на ней ключ и дырочка.
    ngeom = put(ngeom, cylinder, (0.180, 0.004, 0), (0, 0, 0.871), (0.55, 0.57, 0.60), shiny=1.0)
    ngeom = put(ngeom, box, (0.035, 0.070, 0.006), (0.0, -0.04, 0.879), silver, shiny=1.0)
    ngeom = put(ngeom, cylinder, (0.040, 0.004, 0), (0.0, 0.09, 0.874), (0.10, 0.10, 0.10))

    # Тонкая серебряная полоска внизу — там на банке надпись.
    ngeom = put(ngeom, cylinder, (0.2515, 0.018, 0), (0, 0, 0.13), (0.55, 0.57, 0.60), shiny=0.9)

    # Когти: каждый из трёх кусков, чуть вкривь, как рваный след.
    claw = ((-0.10, 0.10), (0.0, 0.0), (0.10, -0.10))
    pieces = ((0.60, 0.090, 0.10), (0.44, 0.075, -0.12), (0.29, 0.065, 0.14))

    for angle in (0.0, np.pi):

        face = _spin(angle)

        for shift, tilt in claw:
            for height, half, jag in pieces:

                offset = face @ np.array([0.252, shift + np.sin(tilt) * (height - 0.44), height])

                ngeom = put(
                    ngeom, box, (0.008, 0.016, half), offset, green,
                    face @ _lean(tilt + jag), shiny=0.3, glow=0.25,
                )

    return ngeom


IPHONE_DEPTH = 0.045  # половина толщины айфона
IPHONE_SCREEN = 0.024  # насколько лицо экрана выступает над корпусом
POST = 0.86  # высота одного поста в ленте


def iphone(scene, ngeom, position, facing, moment, shade=1.0, scale=1.0, feed=None):
    """Айфон стоит на ребре, экраном к Музе. Экран светится и мерцает.
    position — середина нижнего края, facing — куда смотрит экран.
    feed — насколько прокручена лента инстаграма; None — домашний экран."""

    position = np.asarray(position, dtype=float)

    facing = np.array([facing[0], facing[1], 0.0])
    facing /= max(np.linalg.norm(facing), 1e-9)
    side = np.cross([0.0, 0.0, 1.0], facing)
    turn = np.column_stack([facing, side, [0.0, 0.0, 1.0]])

    put = _painter(scene, position, turn, shade, scale)
    box = mujoco.mjtGeom.mjGEOM_BOX
    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER

    width, height, depth = 0.36, 0.76, IPHONE_DEPTH
    flicker = 0.85 + 0.15 * np.sin(moment * 3.0)

    # Корпус, рамка, экран, «чёлка» и полоска снизу.
    ngeom = put(ngeom, box, (depth, width, height), (0, 0, height), (0.20, 0.21, 0.24), shiny=0.9)
    ngeom = put(ngeom, box, (0.004, width - 0.02, height - 0.02), (depth + 0.010, 0, height), (0.02, 0.02, 0.03), shiny=1.0)
    ngeom = put(
        ngeom, box, (0.004, width - 0.035, height - 0.05), (depth + 0.020, 0, height),
        (0.30 * flicker, 0.55 * flicker, 1.0 * flicker), shiny=0.2, glow=0.9,
    )
    ngeom = put(ngeom, box, (0.004, 0.08, 0.022), (depth + 0.065, 0, 2 * height - 0.07), (0.02, 0.02, 0.02))
    ngeom = put(ngeom, box, (0.003, 0.10, 0.006), (depth + 0.065, 0, 0.07), (0.9, 0.9, 0.95), glow=0.6)

    if feed is not None:
        ngeom = _instagram(put, ngeom, depth, width, height, feed)

    # Пара «иконок» на экране.
    for row in range(3 if feed is None else 0):
        for column in range(3):
            ngeom = put(
                ngeom, box, (0.003, 0.045, 0.045),
                (depth + 0.045, -0.18 + 0.18 * column, 2 * height - 0.25 - 0.18 * row),
                (0.95, 0.55 + 0.15 * column, 0.30 + 0.2 * row), glow=0.7,
            )

    # Сзади — блок камер.
    ngeom = put(ngeom, box, (0.006, 0.12, 0.12), (-depth - 0.004, -0.19, 2 * height - 0.17), (0.28, 0.29, 0.32), shiny=0.9)
    for dy, dz in ((-0.05, -0.05), (0.05, -0.05), (-0.05, 0.05)):
        ngeom = put(
            ngeom, cylinder, (0.035, 0.008, 0), (-depth - 0.012, -0.19 + dy, 2 * height - 0.17 + dz),
            (0.05, 0.05, 0.08), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float), shiny=1.0,
        )

    return ngeom


# Фото в ленте: небо, земля, солнце и кто-то в кадре — у каждого поста своё.
PHOTOS = (
    ((0.45, 0.75, 1.00), (0.25, 0.65, 0.30), (1.00, 0.85, 0.20), (0.95, 0.60, 0.50)),
    ((0.98, 0.60, 0.45), (0.20, 0.35, 0.75), (1.00, 0.95, 0.70), (0.15, 0.15, 0.20)),
    ((0.75, 0.85, 0.95), (0.90, 0.85, 0.70), (1.00, 1.00, 1.00), (0.95, 0.45, 0.15)),
    ((0.15, 0.10, 0.30), (0.30, 0.20, 0.45), (0.95, 0.95, 0.80), (0.85, 0.30, 0.60)),
    ((0.60, 0.90, 0.85), (0.95, 0.80, 0.55), (1.00, 0.80, 0.30), (0.30, 0.55, 0.95)),
    ((0.95, 0.80, 0.85), (0.55, 0.75, 0.40), (1.00, 1.00, 0.85), (0.60, 0.35, 0.20)),
)


def _instagram(put, ngeom, depth, width, height, feed):
    """Лента как в инстаграме: шапка, посты с фото, сердечком и подписью,
    внизу панель. feed — сколько прокручено; посты уезжают вверх и
    обрезаются по краю экрана."""

    box = mujoco.mjtGeom.mjGEOM_BOX
    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    ellipsoid = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    face = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # ось кружков — наружу

    x_back = depth + 0.035
    x_front = depth + 0.050
    half = width - 0.035

    top, bottom = 2 * height - 0.05, 0.05
    view_top, view_bottom = top - 0.14, bottom + 0.12

    white = (0.98, 0.98, 0.98)
    ink = (0.10, 0.10, 0.12)
    grey = (0.75, 0.75, 0.78)

    # Белый фон приложения.
    ngeom = put(ngeom, box, (0.003, half, (top - bottom) / 2), (x_back - 0.010, 0, (top + bottom) / 2), white, glow=0.8)

    def strip(ngeom, y, half_y, z, half_z, color, x=x_back, glow=0.8):
        """Прямоугольник ленты, обрезанный по окну просмотра."""
        low, high = max(z - half_z, view_bottom), min(z + half_z, view_top)
        if high <= low:
            return ngeom
        return put(ngeom, box, (0.003, half_y, (high - low) / 2), (x, y, (high + low) / 2), color, glow=glow)

    def dot(ngeom, kind, size, y, z, color):
        """Кружок или фигура — только если целиком в окне."""
        reach = size[0] if kind == cylinder else size[2]
        if z - reach < view_bottom or z + reach > view_top:
            return ngeom
        rotation = face if kind == cylinder else np.eye(3)
        return put(ngeom, kind, size, (x_front, y, z), color, rotation, glow=0.35)

    first = int(np.floor(feed / POST))

    for number in range(first, first + 3):

        post_top = view_top + feed - number * POST
        photo = PHOTOS[number % len(PHOTOS)]

        # Шапка поста: кружок аватарки и имя.
        z = post_top - 0.05
        ngeom = dot(ngeom, cylinder, (0.032, 0.003, 0), -half + 0.06, z, photo[3])
        ngeom = strip(ngeom, -half + 0.20, 0.09, z, 0.012, ink, x_front)

        # Фото: небо, земля, солнце и фигура.
        photo_top, photo_bottom = post_top - 0.10, post_top - 0.72
        horizon = photo_bottom + 0.22
        ngeom = strip(ngeom, 0, half, (photo_top + horizon) / 2, (photo_top - horizon) / 2, photo[0], glow=0.3)
        ngeom = strip(ngeom, 0, half, (horizon + photo_bottom) / 2, (horizon - photo_bottom) / 2, photo[1], glow=0.3)
        ngeom = dot(ngeom, cylinder, (0.06, 0.003, 0), half - 0.12, photo_top - 0.12, photo[2])
        ngeom = dot(ngeom, ellipsoid, (0.004, 0.07, 0.12), -0.05, horizon + 0.08, photo[3])

        # Сердечко, комментарий, отправить и две строки подписи.
        z = photo_bottom - 0.045
        ngeom = dot(ngeom, cylinder, (0.022, 0.003, 0), -half + 0.05, z, (0.93, 0.20, 0.30))
        ngeom = dot(ngeom, cylinder, (0.022, 0.003, 0), -half + 0.12, z, ink)
        ngeom = dot(ngeom, cylinder, (0.022, 0.003, 0), -half + 0.19, z, ink)
        ngeom = strip(ngeom, -half + 0.22, 0.18, z - 0.055, 0.010, grey, x_front)
        ngeom = strip(ngeom, -half + 0.16, 0.12, z - 0.085, 0.010, grey, x_front)

    # Шапка приложения и нижняя панель стоят на месте.
    ngeom = put(ngeom, box, (0.003, half, 0.07), (x_back, 0, top - 0.07), white, glow=0.8)
    ngeom = put(ngeom, box, (0.003, 0.11, 0.018), (x_front, -half + 0.16, top - 0.07), ink)
    ngeom = put(ngeom, cylinder, (0.022, 0.003, 0), (x_front, half - 0.06, top - 0.07), ink, face)
    ngeom = put(ngeom, box, (0.003, half, 0.06), (x_back, 0, bottom + 0.06), white, glow=0.8)
    for y in np.linspace(-half + 0.07, half - 0.07, 5):
        ngeom = put(ngeom, box, (0.003, 0.018, 0.018), (x_front, y, bottom + 0.07), ink)

    return ngeom


LAPTOP_KEYBOARD = -0.18  # где середина клавиатуры от середины основания (к экрану)


def _code_line(number):
    """Строка кода: отступ и длины слов. Одна и та же строка всегда одинаковая."""

    rng = np.random.default_rng(number * 7919 + 13)
    indent = int(rng.integers(0, 4)) * 2 if number % 6 else 0
    words = [int(rng.integers(2, 9)) for _ in range(int(rng.integers(1, 5)))]
    return indent, words


def _terminal(put, ngeom, on_lid, lid_turn, x_front, width, lid_height, typed, moment):
    """Терминал: чёрный фон, зелёные строки кода, мигающий курсор.
    typed — сколько знаков напечатано; строки уезжают вверх."""

    box = mujoco.mjtGeom.mjGEOM_BOX
    green = (0.25, 1.0, 0.35)
    dim = (0.12, 0.55, 0.18)

    rows = 11
    char = 0.052
    row_height = (lid_height - 0.22) / rows
    left = -(width - 0.12)

    # Чёрный фон поверх обычного экрана.
    ngeom = put(ngeom, box, (0.003, width - 0.07, lid_height / 2 - 0.08),
                on_lid(x_front - 0.010, lid_height / 2 + 0.01), (0.02, 0.03, 0.02), lid_turn, glow=0.3)

    # Какая строка сейчас печатается и сколько в ней уже знаков.
    number, left_chars = 0, float(typed)
    while True:
        indent, words = _code_line(number)
        length = indent + sum(words) + len(words)
        if left_chars < length:
            break
        left_chars -= length
        number += 1

    first = max(0, number - rows + 1)

    for row, line in enumerate(range(first, number + 1)):

        indent, words = _code_line(line)
        z = lid_height - 0.15 - row * row_height
        column = indent
        shown = left_chars if line == number else 1e9

        # Приглашение "$" в начале строки без отступа.
        if indent == 0:
            ngeom = put(ngeom, box, (0.003, char * 0.4, row_height * 0.28),
                        on_lid(x_front, z, left + char * 0.4), dim, lid_turn, glow=0.9)
            column = 1.5

        for size in words:
            visible = min(size, shown - column)
            if visible <= 0:
                break
            y = left + (column + visible / 2) * char
            ngeom = put(ngeom, box, (0.003, visible * char / 2 * 0.92, row_height * 0.28),
                        on_lid(x_front, z, y), green, lid_turn, glow=0.9)
            column += size + 1

        # Мигающий курсор в конце печатаемой строки.
        if line == number and (moment * 2.0) % 1.0 < 0.6:
            y = left + (min(column, indent + left_chars) + 0.5) * char
            ngeom = put(ngeom, box, (0.003, char * 0.45, row_height * 0.38),
                        on_lid(x_front, z, y), green, lid_turn, glow=1.0)

    return ngeom


def laptop(scene, ngeom, position, facing, moment, shade=1.0, typed=None):
    """Ноутбук: основание с клавиатурой и открытая крышка с горящим экраном.
    position — середина основания на полу, facing — куда смотрит экран.
    typed — если Муза печатает: сколько знаков, и на экране терминал."""

    position = np.asarray(position, dtype=float)

    facing = np.array([facing[0], facing[1], 0.0])
    facing /= max(np.linalg.norm(facing), 1e-9)
    side = np.cross([0.0, 0.0, 1.0], facing)
    turn = np.column_stack([facing, side, [0.0, 0.0, 1.0]])

    put = _painter(scene, position, turn, shade)
    box = mujoco.mjtGeom.mjGEOM_BOX

    aluminium = (0.70, 0.72, 0.76)
    depth, width, thick = 0.70, 1.05, 0.035
    flicker = 0.88 + 0.12 * np.sin(moment * 2.3)

    # Основание, клавиатура, тачпад.
    ngeom = put(ngeom, box, (depth, width, thick), (0, 0, thick), aluminium, shiny=0.8)
    ngeom = put(ngeom, box, (0.30, 0.90, 0.004), (LAPTOP_KEYBOARD, 0, 2 * thick + 0.008), (0.08, 0.08, 0.09))
    for row in range(5):
        for column in range(12):
            ngeom = put(
                ngeom, box, (0.022, 0.030, 0.004),
                (-0.40 + 0.11 * row, -0.80 + 0.145 * column, 2 * thick + 0.020),
                (0.14, 0.14, 0.15),
            )
    ngeom = put(ngeom, box, (0.14, 0.22, 0.003), (0.36, 0, 2 * thick + 0.008), (0.60, 0.62, 0.66), shiny=0.9)

    # Крышка открыта назад на 105 градусов: петля у дальнего края основания.
    hinge = np.array([-depth, 0.0, 2 * thick])
    lid_turn = np.array(
        [[np.cos(0.26), 0, np.sin(0.26)], [0, 1, 0], [-np.sin(0.26), 0, np.cos(0.26)]]
    )
    lid_height = 1.30

    def on_lid(x, z, y=0.0):
        return hinge + lid_turn @ np.array([x, y, z])

    ngeom = put(ngeom, box, (thick * 0.7, width, lid_height / 2), on_lid(0, lid_height / 2), aluminium, lid_turn, shiny=0.8)
    ngeom = put(ngeom, box, (0.004, width - 0.03, lid_height / 2 - 0.03), on_lid(thick * 0.7 + 0.010, lid_height / 2), (0.02, 0.02, 0.03), lid_turn, shiny=1.0)
    ngeom = put(
        ngeom, box, (0.004, width - 0.07, lid_height / 2 - 0.08), on_lid(thick * 0.7 + 0.020, lid_height / 2 + 0.01),
        (0.20 * flicker, 0.45 * flicker, 0.85 * flicker), lid_turn, shiny=0.2, glow=0.9,
    )

    if typed is not None:
        return _terminal(
            put, ngeom, on_lid, lid_turn, thick * 0.7 + 0.050, width, lid_height, typed, moment
        )

    # На экране — окна.
    for x0, z0, w, h, color in (
        (-0.55, 0.85, 0.35, 0.25, (0.95, 0.95, 0.98)),
        (0.25, 0.70, 0.40, 0.35, (0.35, 0.85, 0.55)),
        (-0.30, 0.35, 0.50, 0.18, (0.95, 0.75, 0.30)),
    ):
        ngeom = put(
            ngeom, box, (0.003, w, h / 2), on_lid(thick * 0.7 + 0.050, z0, x0),
            color, lid_turn, glow=0.7,
        )

    return ngeom
