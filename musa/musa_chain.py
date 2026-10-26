"""Муза и три этажа её мозга: чувство -> связь -> движение.

Два чувства, две настоящие цепочки из коннектома. Свет — зрительный компас,
еда — запах. Обе крутятся ползунками, обе рулят ногами одновременно.

Цепочка настоящая, из коннектома maleCNS v1.0 — тот самый зрительный компас,
которым живая муха держит курс на ориентир:

    R7/R8 -> MeTu -> TuBu -> ER -> EPG -> PFL3 -> DNa02

ЗРЕНИЕ    фоторецепторы отдают свет в медуллу и дальше в бугорок,
          всё строго внутри своего глаза;
СВЯЗЬ     бугорок заводит сигнал в кольцо, кольцо кормит компас,
          компас говорит решающему нейрону — здесь стороны смешиваются;
ДВИЖЕНИЕ  решение уходит на рулевой нейрон крест-накрест и дальше в ноги.

Что здесь упрощено: в симуляции нет настоящих фоторецепторов, свет подаётся
на них напрямую слева и справа; по цепочке передаётся только перекос между
сторонами, и каждый этаж нормирован (см. musa/chain.py).

Запуск:  ./run.sh musa/musa_chain.py
"""

import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import (
    HybridTurningController,
    HybridControllerObservation,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)
from flygym.utils.math import Rotation3D

import brainshell
import building
import props
import room
from brainmap import BrainMap
from chain import Chain
from journal import Journal
from player import Music
from panel import Panel
from window import Window


# ============================================================
# НАСТРОЙКИ
# ============================================================

FPS = 60

LIGHT_START = np.array([0.0, 0.0, 5.2])  # мм: под потолком посреди комнаты
LIGHT_STRENGTH = 0.5  # тяга к лампочке по умолчанию: вдвое слабее
LIGHT_SPEED = 6.0  # мм в секунду, пока клавиша зажата

TURN = 1.50  # насколько сильно перекос мотонейронов крутит ноги
GAIN_STEP = 0.25
GAIN_MAX = 3.0

DETAIL = 2  # каждый второй отрезок скелета

DATA = Path(__file__).resolve().parent.parent / "brain" / "data"

# Бургер и банка далеко друг от друга: между ними она застревала — тянут
# поровну с двух сторон, и она шла прямо в узкую щель.
# Бургер и энергетик стоят на длинном кухонном столе, на разных его
# половинах, с краю со стороны комнаты: оттуда Муза их берёт и туда же кладёт.
FOOD_A_START = np.array(
    [room.TABLE[0] - room.TABLE_HALF[0] + 0.3, room.TABLE[1] - 1.8, room.TABLE_TOP + 0.18]
)
FOOD_B_START = np.array(
    [room.TABLE[0] - room.TABLE_HALF[0] + 0.3, room.TABLE[1] + 1.8, room.TABLE_TOP + 0.02]
)

MEMORY_STEP = 0.25  # насколько дофамин двигает синапс KC -> MBON за раз

# Сигарета лежит в пепельнице на подоконнике — там её место.
CIGARETTE_START = np.array([room.ASHTRAY[0], room.ASHTRAY[1], room.SILL_TOP + 0.12])
# Место айфона — как ты его поставила 26.09: у стенки ванной, ближе к
# южной стене, экраном в комнату.
IPHONE_START = np.array([-7.2, -1.3448, 0.0])
IPHONE_FACING = np.array([0.695, 0.719])

# Курит у окна: встаёт на задние лапки и опирается брюшком на подоконник.
LEAN_PITCH = np.radians(60)  # насколько выпрямляется
LEAN_GAP = 1.2  # мм от края подоконника до груди: вся сигарета помещается до стекла
HELD_CIGARETTE = 0.6  # размер сигареты в лапках
FILTER_END = 0.79  # от середины модели сигареты до конца фильтра (в её размерах)
LEAN_UP = 0.30  # мм над подоконником
# Передние лапки у окна: левой опирается о подоконник (кончик на его
# поверхности), правой курит — держит сигарету перед мордой и на затяжке
# подаётся ближе. У окна
# выключено касание пола передним лапкам и телу: брюшко упиралось в пол, и это
# гнуло передние лапки вниз, в тело. Средние и задние стоят на полу как есть.
# Ноутбук вплотную к стене у матраса, экраном в комнату: щели между ним и
# стеной нет, застревать негде.
COMPUTER_START = np.array([-1.5, 9.0 - 1.0, 0.0])

NICOTINE_DOSE = 0.6  # во столько раз сильнее холиновые синапсы у самой сигареты
CRAVING = 0.04  # за секунду: насколько никотин сам тянет память вверх

GRAB_RANGE = 1.8  # мм: ближе этого Муза берёт сигарету передними лапками
REACH_SECONDS = 0.6  # сколько тянется к ней
GRAB_PAUSE = 2.0  # секунд после того, как отобрали: не хватает обратно
PORTION = {"burger": 12, "monster": 8}  # укусов / глотков, и всё съедено
FULL_PAUSE = 15.0  # секунд после еды: сыта и ничего не хватает

# Насколько довернуть суставы передних ног, чтобы лапки поднялись ко рту.
HOLD_POSE = {
    "coxa:PITCH": 1.2,
    "trochanterfemur:PITCH": -0.9,
    "tibia:PITCH": -1.1,
    "tarsus1:PITCH": -0.6,
}
NOD = {"monster": (-0.8, 0.6)}  # кивок лапок на такте: coxa, tibia
CAN_LIP_AHEAD = 0.45  # мм перед хоботком: сюда банка ставит край на глотке
CAN_LIP_UP = -0.10
CAN_POUR = np.radians(75)  # насколько опрокидывает банку на глотке

# Айфон в лапках: где стоит (вперёд, вбок, вверх от хоботка, мм) и свайп.
PHONE_SPOT = (1.05, -0.35, -0.55)
PHONE_SCALE = 0.8
SWIPE = 0.7  # насколько опускается бедро правой лапки перед свайпом
SWIPE_LIFT = -0.3  # плечо правой лапки: минус — лапка выше
SWIPE_AWAY = -0.8  # голень на обратном ходе: лапка отходит от экрана
SWIPE_SECONDS = 1.5  # один свайп: провела вверх, вернула лапку, пауза

# Предметы — препятствия: насколько они широкие, если смотреть сверху (мм).
ITEM_RADIUS = {
    "burger": 0.52,
    "monster": 0.25,
    "cigarette": 0.5,
    "iphone": 0.4,
    "computer": 1.1,
}
HEAD_REACH = 1.2  # мм от груди до лба: на столько грудь не подходит к предмету

GRABBABLE = ("burger", "monster", "cigarette", "iphone", "computer")

# За компьютером: лапки тянутся вниз к клавиатуре и стучат по очереди.
TYPE_REACH = 0.8  # насколько опущено бедро, чтобы лапки доставали клавиш
TYPE_TAP = 0.35  # насколько поднимается лапка между ударами
TYPE_RATE = 5.0  # ударов в секунду каждой лапкой
TYPE_CHARS = 9.0  # знаков в секунду на экране
KEYBOARD_AHEAD = 0.89  # мм перед хоботком: где кончики лапок у клавиатуры

# Своя жизнь Музы. Это не из коннектома: усталость и позыв — простые
# счётчики, а путь к матрасу — прямое наведение.
SIT_LEGS = {"coxa:PITCH": 0.5, "trochanterfemur:PITCH": -0.4}  # лапки свисают вперёд
LIE_POSE = {"trochanterfemur:PITCH": -0.8, "tibia:PITCH": -0.8}  # лечь
ENERGY_SECONDS = 120.0  # столько ходьбы — и энергия кончилась, хочет отдохнуть
POOP_SECONDS = 90.0  # столько времени — и хочет в туалет
POOP_PER_BITE = 0.08  # каждый укус или глоток ускоряет
LIE_SECONDS = 1.0  # сколько ложится и сколько встаёт
REST_SECONDS = 8.0  # сколько лежит
POOP_PAUSE = 3.0  # сколько сидит на унитазе
DOOR_SECONDS = 1.0  # встать, лапку к двери, толкнуть, опустить лапку
DOOR_LEG = {"coxa:PITCH": -0.6, "tibia:PITCH": -0.3}  # передняя лапка к ручке
# Задняя лапка вбок: назад она бьёт у самого пола, а брюшко выше и
# упиралось в дверь раньше лапки — Муза застревала в двери задом.
KICK_LEG = {"coxa:ROLL": -0.6, "trochanterfemur:PITCH": -0.5}

# Где кончик лапки от груди в этих позах (вперёд, вбок вправо — минус), мм.
# Измерено на стойке; для левой лапки вбок с другим знаком.
DOOR_TIP = np.array([0.92, 1.25])
KICK_TIP = np.array([-1.15, 1.59])
DOOR_PULL = 0.33  # до этой доли дверь тянет лапкой, дальше едет сама
LIFT_DOOR_SECONDS = 1.0  # створки лифта съезжаются или разъезжаются
GATE_SECONDS = 2.0  # ворота отъезжают или закрываются

# Поход из квартиры на первый этаж и обратно (кнопка OUT): точки маршрута.
FRONT_INSIDE = np.array([room.FRONT_DOOR_X, room.HALF - 2.0])  # перед входной дверью дома
FRONT_OUTSIDE = np.array([room.FRONT_DOOR_X, room.LANE + 0.8])  # в коридоре напротив двери
FRONT_KICK = np.array([room.FRONT_DOOR_X + 2.9, room.LANE + 0.8])  # за краем открытой двери
FRONT_PULL = np.array([room.FRONT_DOOR_X - 0.2, room.CORRIDOR_LOW + 1.8])  # у ручки из коридора
LIFT_CALL = np.array([room.LIFT_X - 1.7, room.LANE + 0.8])  # у кнопки вызова
LIFT_DOORWAY = np.array([room.LIFT_X - 0.9, room.LANE])  # перед дверями лифта
CAB_SPOT = np.array([room.CAB[1] - 1.7, room.LANE])  # в кабине
BOARD_SPEED = 4.0  # мм в секунду: заходит в кабину
CAFE_SPEED = 12.0  # по кафе — обычным шагом, не как в тесный проём
LIFT_OUT = np.array([room.LIFT_X - 2.4, room.LANE])  # вышла из лифта
LOBBY = [np.array([3.0, room.LANE])]  # проходит по холлу к двери на улицу
# Во двор и обратно: у двери в холле, за дверью, у ручки снаружи. Дверь
# закрывается сама.
STREET_DOOR_Y = room.STREET_LOW + room.STREET_LEAF / 2
STREET_INSIDE = np.array([-room.HALF + 2.0, STREET_DOOR_Y])
STREET_PULL = np.array([room.HOUSE[0] - 2.0, STREET_DOOR_Y + 0.8])
# Сквозь проём двери подъезда идёт по прямой, путь ведём сами: у проёма
# косяки, створка и упор, и она там застревала.
STREET_THROUGH_IN = np.array([-room.HALF + 2.2, STREET_DOOR_Y])
STREET_THROUGH_OUT = np.array([room.HOUSE[0] - 2.0, STREET_DOOR_Y])
STREET_DOORWAY_IN = np.array([-room.HALF + 0.8, STREET_DOOR_Y])  # в проёме со стороны холла
STREET_DOORWAY_OUT = np.array([room.HOUSE[0] - 1.0, STREET_DOOR_Y])  # в проёме со стороны двора
# Дорожка по двору: поворот вправо вдоль дома, угол, у кнопки ворот, за
# воротами на тротуаре, снова во дворе.
SPINE_X = (room.SPINE[0] + room.SPINE[1]) / 2
SPINE_TOP = np.array([SPINE_X, (room.APRON[0] + room.APRON[1]) / 2])  # середина площадки: есть где развернуться
SPINE_CORNER = np.array([SPINE_X, room.GATE_Y])
GATE_INSIDE = np.array([room.FENCE_X + 2.6, room.GATE_Y])
GATE_OUTSIDE = np.array([room.FENCE_X - 1.5, room.GATE_Y])
# Круг вокруг дома по середине тротуара: от ворот на юг, против часовой
# стрелки через все углы и обратно к воротам.
_x0, _x1, _y0, _y1 = room.LOT
_w = room.SIDEWALK / 2 - 0.2
AROUND = [np.array(p) for p in (
    (_x0 - _w, room.GATE_Y), (_x0 - _w, _y0 - _w), (_x1 + _w, _y0 - _w),
    (_x1 + _w, _y1 + _w), (_x0 - _w, _y1 + _w),
)]
AROUND_REACH = 2.5  # на тротуаре точку засчитываем издали: развернуться там негде
# Кафе через дорогу: по зебре, в стеклянную дверь, к стойке за кофе.
CROSS_NEAR = np.array([_x0 - _w, room.CAFE_Y])  # свой тротуар напротив зебры
CROSS_FAR = np.array([room.CAFE_FACE + room.SIDEWALK / 2, room.CAFE_Y])  # тротуар у кафе
CAFE_OUTSIDE = np.array([room.CAFE_FACE + 1.8, room.CAFE_Y])  # у двери снаружи
CAFE_INSIDE = np.array([room.CAFE_FACE - 2.0, room.CAFE_Y])  # за дверью внутри
CAFE_COUNTER = np.array([room.COUNTER_EDGE + 1.9, room.COUNTER_Y[1] - 1.8])  # у стойки
CAFE_AISLE_Y = room.CAFE_Y - 1.0  # проход между столиками и стойкой
# Бариста: к кофемашине, на задние лапки, передние на машину, кофе
# наливается, вниз, стаканчик в лапки, к стойке, ставит перед Музой.
BARISTA_STEPS = (
    ("turn", 0.25), ("walk", 1.0), ("turn", 0.25), ("rear", 0.4), ("brew", 1.6),
    ("down", 0.4), ("turn", 0.25), ("walk", 1.0), ("turn", 0.25), ("serve", 0.4),
)
BARISTA_TOTAL = sum(seconds for _, seconds in BARISTA_STEPS)
BARISTA_SERVE = np.array([room.COUNTER_X - 1.4, room.COUNTER_Y[1] - 1.8])  # напротив Музы
BARISTA_MACHINE = np.array([room.COUNTER_X - 1.4, room.MACHINE_Y])  # у кофемашины
BARISTA_REAR = np.radians(60)  # на задних лапках: насколько выпрямляется
ON_MACHINE = 0.35  # передние лапки вперёд ровно настолько, чтобы лечь на кофемашину
COFFEE_SECONDS = 6.0  # столько пьёт кофе за столиком
COFFEE_ENERGY = 0.5  # стаканчик кофе: столько энергии
COFFEE_WATER = 0.4  # и столько воды — прибывают, пока пьёт
PULL_SECONDS = 0.8  # отодвигает стул лапкой
LIFT_CUP = 0.5  # берёт стаканчик со стола ко рту и ставит обратно
# Сидя тело выпрямлено: поза «с банкой» подняла бы лапки над головой. Эта
# подносит кончики передних лапок к хоботку.
SEAT_HOLD = {"coxa:PITCH": 0.8, "trochanterfemur:PITCH": 0.4, "tibia:PITCH": -0.8, "tarsus1:PITCH": -0.8}
HOME_SPOT = np.array([room.FRONT_DOOR_X - 2.0, room.HALF - 4.0])  # зашла домой
SIT_SECONDS = 0.8  # сколько садится и сколько встаёт
SIT_PITCH = np.radians(65)  # насколько выпрямляется, сидя как человек
FLIP_SECONDS = 3.0  # сколько дрыгает лапками на спине
RIGHTING_SECONDS = 1.0  # сколько переворачивается обратно
# Тело: сытость от бургера, вода и энергия от энергетика. Сытость и вода
# спадают сами, энергию тратят ходьба и лента.
FULL_PER_BITE = 0.09  # 12 укусов — наелась
WATER_PER_SIP = 0.13  # 8 глотков — напилась
ENERGY_PER_SIP = 0.12  # столько энергии даёт один глоток
FULL_SECONDS = 150.0  # за столько сытость уходит целиком
WATER_SECONDS = 100.0
HUNGRY = 0.15  # сытость ниже этого — сама идёт есть бургер
THIRSTY = 0.15  # вода ниже этого — сама идёт пить энергетик
FULL_ENOUGH = 0.8  # сытость (или вода) выше этого — к еде (питью) не тянет
FOCUS_SECONDS = 6.0  # сколько держится выбранной цели
FOCUS_MIN = 0.02  # слабее этого — никуда не тянет
OTHERS_WEIGHT = 0.25  # остальное приятное, пока идёт к цели
JOY_PER_BITE = 0.10  # укус или глоток — средне радости
JOY_FADE = 0.05  # в секунду: радость тает, когда ничего не радует

# Лента в айфоне: сначала удовольствие, потом общая усталость его съедает.
PHONE_TIRES_SECONDS = 30.0  # за столько лента съедает всю энергию — быстрее ходьбы
JOY_RATE = 0.35  # как быстро удовольствие догоняет, куда тянет
JOY_FADES_BELOW = 0.65  # ниже этой энергии удовольствие начинает падать
PHONE_BREAK = 20.0  # секунд после того, как отложила: айфон не берёт

# Свободное время: нужд нет — через IDLE_SECONDS сама решает, чем заняться.
# Не случайно: у каждого занятия тяга = то, что оно даст её просевшим нуждам,
# умноженное на выученный вес синапса KC>MBON (память о похожем источнике).
# Только что сделанное на время слабее (синапс привыкает).
IDLE_SECONDS = 4.0
IDLE_THRESHOLD = 0.12  # слабее — просто стоит
HABIT = 0.2  # привыкание к только что сделанному
HABIT_SECONDS = 60.0  # за столько привыкание проходит
PHONE_SESSION = 6.0  # в свободное время листает ленту столько, потом откладывает
LOOK_SECONDS = 6.0  # смотрит в окно
WINDOW_SPOT = np.array([room.WINDOW_X, -room.HALF + room.SILL_DEPTH + 1.6])  # перед окном
STROLL = [np.array(p) for p in ((4.5, 3.0), (-1.5, 3.0), (-1.5, -2.5), (4.5, -2.5))]  # вокруг ковра
WALK_JOY = 0.12  # прогулка радует немного
WINDOW_JOY = 0.12
BARISTA_JOY = 0.2  # поболтала с баристой
COFFEE_FULL = 0.35  # в кафе к кофе — круассан

# Сигарета: радует, но сил отнимает больше, чем даёт радости.
SMOKE_JOY = 0.03  # удовольствия в секунду, пока курит
SMOKE_DRAIN = 0.045  # энергии в секунду — в полтора раза больше
SMOKE_PUFFS = 3  # меньше трёх затяжек не бросает
EXHALE_SECONDS = 2.0  # сколько живёт облачко выдоха (по тем же часам, что и затяжки)

# Компьютер — обязанность: раз в две минуты идёт работать. Радости чуть-чуть,
# а сил отнимает втрое больше.
WORK_EVERY = 120.0  # секунд между работами
WORK_SECONDS = 20.0  # сколько печатает за раз
WORK_JOY = 0.01
WORK_DRAIN = 0.03

SHOCK_SECONDS = 0.3  # один короткий удар током
SHOCK_TWITCH = 0.15  # слабая дрожь лапок: пугает, но не вредит

APPROACH = 4.0  # мм: ближе этого и впереди — Муза к нему подходит, учим его

SMELLED = 0.05  # слабее этого запах не чувствуется, учить нечему


# ============================================================
# MUSA
# ============================================================

for name in ("light", "food", "memory"):
    if not (DATA / f"chain_{name}.npz").exists():
        raise SystemExit(
            "нет цепочек, сначала выполни:\n  ./run-brain.sh brain/export_chain.py"
        )

fly = make_locomotion_fly(name="musa", add_adhesion=True, colorize=True)

world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]))

brainshell.attach(world.mjcf_root)
room.build(world)

sim = Simulation(world)

model = sim.mj_model
data = sim.mj_data

# Ближняя плоскость камеры была 0.0005 мм при дальней 250 мм: на такой
# разброс точности глубины не хватает, и ковёр с полом мигали.
model.vis.map.znear = 0.05 / model.stat.extent
model.vis.map.zfar = 120.0 / model.stat.extent

# Пол физики бесконечный, но рисовать его не надо: он закрыл бы улицу
# на четыре этажа ниже. Столкновения с ним от размера не зависят.
model.geom_size[model.geom("ground_plane").id, :2] = 0.01


def floor_z():
    """Высота пола там, где Муза: 0 — её квартира, ниже — первый этаж."""
    return room.level(room.WHERE["floor"])


def shown_z():
    """Физически Муза всегда ходит по полу квартиры (пол физики MuJoCo не
    сдвинуть — неподвижные геометрии он считает один раз). В лифте и на
    первом этаже её тело только рисуется ниже, на эту высоту."""
    return room.LIFT["z"] if state["chore"] == "ride" else floor_z()


def view_floor():
    """Какой этаж показываем: где Муза. В лифте: пока закрываются двери —
    откуда едет, пока кабина едет — дом снаружи (None), приехала — куда
    приехала."""

    if state["chore"] != "ride":
        return room.WHERE["floor"]
    if state["chore_time"] < LIFT_DOOR_SECONDS:
        return state["ride_from_storey"]
    if abs(room.LIFT["z"] - state["ride_to"]) > 1e-6:
        return None
    return state["ride_storey"]


# Начальный вид камеры: вся квартира. На другом этаже — тот же ракурс,
# только на высоте того этажа и над коридором.
START_CAMERA = ([2.5, -0.07, 1.0], 33.5, -0.1, -38.5)


def floor_lookat(storey):
    """Куда смотрит камера на этаже: на 5-м — на квартиру, на 1-м — на холл."""
    lookat = START_CAMERA[0]
    if storey == 5:
        return np.array(lookat, dtype=float)
    # На первом этаже левее: видно и холл, и двор с воротами.
    return np.array([lookat[0] - 17.5, room.LANE - 3.0, room.level(storey) + lookat[2]])


def camera_to(storey):
    """Камера на этаж: как при запуске, на высоте этого этажа."""
    _, distance, azimuth, elevation = START_CAMERA
    window.camera.lookat[:] = floor_lookat(storey)
    window.camera.distance = distance
    window.camera.azimuth = azimuth
    window.camera.elevation = elevation


def window_view(storey):
    """Камера в торце коридора у окна (на 1-м — у стеклянной двери), под
    верхом стен: сверху вдоль коридора на двери лифта, без крупного плана."""
    eye = np.array([-room.HALF + 0.6, room.LANE, room.level(storey) + 5.5])  # в коридоре у окна, сверху
    lookat = np.array([room.LIFT_X, room.LANE, room.level(storey) + 1.6])
    window.camera.lookat[:] = lookat
    window.camera.distance = float(np.linalg.norm(lookat - eye))
    window.camera.azimuth = 0.0
    window.camera.elevation = float(np.degrees(np.arcsin((lookat[2] - eye[2]) / window.camera.distance)))


def lift_camera():
    """Пока Муза заходит в лифт, камера смотрит от окна подъезда на двери
    лифта; пока кабина едет — снаружи спускается вдоль окон подъезда; на
    новом этаже снова от окна. Вышла из кабины — обычный вид этажа."""

    if state["chore"] in ("board", "ride"):
        state["lift_cam"] = True
        storey = view_floor()
        if storey is None:
            # Кабина едет: камера снаружи спускается с ней вдоль окон подъезда.
            window.camera.lookat[:] = [room.HOUSE[0], room.LANE, room.LIFT["z"] + 2.8]
            window.camera.distance = 14.0
            window.camera.azimuth = 0.0
            window.camera.elevation = -8.0
        else:
            window_view(storey)
        return
    # В холле первого этажа камера остаётся в коридоре у стеклянной двери:
    # снаружи холл закрыт сплошным фасадом.
    here = data.xpos[thorax_id]
    in_hall = -room.HALF + 3.0 < here[0] < room.CAB[1] and room.CORRIDOR_LOW - 1.0 < here[1] < room.CORRIDOR_HIGH + 1.0
    if state["errand"] is not None and room.WHERE["floor"] == 1 and in_hall:
        state["lift_cam"] = True
        window_view(1)
        return
    # Идёт вокруг дома по тротуару: камера за ней, снаружи от дома — с той
    # стороны, где она идёт, дом её не загораживает.
    x0, x1, y0, y1 = room.LOT
    beyond = {(-1.0, 0.0): x0 - here[0], (1.0, 0.0): here[0] - x1, (0.0, -1.0): y0 - here[1], (0.0, 1.0): here[1] - y1}
    side, gap = max(beyond.items(), key=lambda item: item[1])
    if state["errand"] is not None and room.WHERE["floor"] == 1 and (gap > 0.0 or room.in_street(here)):
        state["lift_cam"] = True
        # Куда хотим смотреть: как на квартиру, с той стороны дома, где она
        # идёт; на западе — через дорогу на кафе; в кафе — ближе; на зебре —
        # почти сверху, чтобы ни фонари, ни машины её не закрывали.
        _, distance, azimuth, elevation = START_CAMERA
        if gap > 0.0:
            azimuth = 180.0 if side == (-1.0, 0.0) else np.degrees(np.arctan2(-side[1], -side[0]))
        if room.in_cafe(here):
            distance = 18.0
        elif room.CROSSING["on"]:
            elevation, distance = -75.0, 26.0
        target = np.array([here[0], here[1], floor_lookat(1)[2]])

        # Камера не прыгает за каждым шагом и не крутится рывками: всё
        # плавно подтягивается к цели.
        camera = window.camera
        follow, turn_rate = 0.06, 0.04
        camera.lookat[:] = camera.lookat + (target - camera.lookat) * follow
        swing = (azimuth - camera.azimuth + 180.0) % 360.0 - 180.0
        camera.azimuth = camera.azimuth + swing * turn_rate
        camera.elevation = camera.elevation + (elevation - camera.elevation) * turn_rate
        camera.distance = camera.distance + (distance - camera.distance) * turn_rate
        return
    if state.get("lift_cam"):
        stepping_out = state["route"] and state["route"][0][0] == "walk" and np.allclose(state["route"][0][1], LIFT_OUT)
        if stepping_out and data.time - state.get("ride_done", 0.0) < 4.0:
            window_view(room.WHERE["floor"])
            return
        state["lift_cam"] = False
        camera_to(room.WHERE["floor"])


def lower_musa(scene, first, last, dz):
    """Сдвигает на экране тело Музы (и то, что рисуется у тела) на dz."""
    for index in range(first, last):
        geom = scene.geoms[index]
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and model.geom_bodyid[geom.objid] == 0:
            continue  # пол и стены мира остаются на месте
        geom.pos[2] += dz

thorax_id = model.body("musa/c_thorax").id
mouth_id = model.body("musa/c_haustellum").id

# Кончики передних лапок: в них и лежит то, что Муза держит.
abdomen_tip = model.body("musa/c_abdomen6").id

# Брюшко тоже упирается в стены и дверь: середина и кончик, их толщина (мм).
BODY_PARTS = [
    (model.body("musa/c_abdomen4").id, 0.45),
    (abdomen_tip, 0.30),
]
hand_ids = [model.body("musa/lf_tarsus5").id, model.body("musa/rf_tarsus5").id]

# Суставы передних ног: их и подменяем, когда Муза берёт сигарету.
dof_names = []

for dof in fly.get_actuated_jointdofs_order("position"):
    text = str(dof)
    child = text.split("child=BodySegment(name='")[1].split("'")[0]
    axis = text.split("RotationAxis.")[1].split(":")[0]
    dof_names.append(f"{child}:{axis}")

front_dofs = {
    i: HOLD_POSE.get(name.split("_", 1)[1], 0.0)
    for i, name in enumerate(dof_names)
    if name.startswith(("lf_", "rf_"))
}

# Контроллер ног считаем раз в CONTROL_EVERY шагов физики (2000 раз в секунду
# вместо 10 000): ногам хватает, а симуляция идёт в разы быстрее.
CONTROL_EVERY = 5

controller = HybridTurningController(
    timestep=sim.timestep * CONTROL_EVERY,
    preprogrammed_steps=PreprogrammedSteps(),
    output_dof_order=fly.get_actuated_jointdofs_order("position"),
    enable_adhesion=True,
)

# Шагает чаще: в полтора раза быстрее ходит. Больше — начинает спотыкаться.
STRIDE_RATE = 1.4
controller._base_intrinsic_freqs = controller._base_intrinsic_freqs * STRIDE_RATE

controller.reset(seed=0)
sim.reset()

stand_ctrl = data.ctrl.copy()  # обычная стойка: от неё ложится
stand_height = 1.02  # мм: на такой высоте грудь, когда она просто стоит


# ============================================================
# ТРИ ЭТАЖА
# ============================================================

def load_chain(name, stimulus, latin, color, position, share=None):
    """Собирает всё, что нужно одной закладке: цепочку, карту, состояние.

    share — уже загруженная закладка на той же цепочке: у двух запахов нейроны
    одни и те же, поэтому карту мозга они делят. Ползунки при этом у каждого
    свои: в жизни так не бывает, зато можно сравнить два запаха друг с другом.
    """

    data_file = np.load(DATA / f"chain_{name}.npz")
    chain = Chain(data_file)

    families = [str(family) for family in data_file["families"]]
    sides = [str(side) for side in data_file["sides"]]

    return {
        "name": name,
        "stimulus": stimulus,
        "latin": latin,  # для кнопки закладки: шрифт MuJoCo кириллицу не рисует
        "color": np.asarray(color),
        "position": np.asarray(position, dtype=float),
        "data": data_file,
        "chain": chain,
        "map": share["map"] if share else BrainMap.from_chain(data_file, model, data),
        "families": families,
        "sides": sides,
        # Нейрон -> его место в цепочке, чтобы раскрашивать по активности.
        "numbers": [chain.families.index(family) for family in families],
        "weights": chain.weights,
        "gains": {
            "L": np.ones(len(chain.stage_from)),
            "R": np.ones(len(chain.stage_from)),
        },
        "activity": np.zeros(len(data_file["labels"])),
        "targets": chain.stage_to,
        # Плюс или минус в начале строки — возбуждает этот синапс цель
        # или тормозит. Знак взят из медиатора того, кто шлёт.
        "functions": [
            f"{'+' if chain.signs[i] > 0 else '-'} "
            f"{chain.family_functions[chain.families.index(family)]}"
            for i, family in enumerate(chain.stage_to)
        ],
        "turn": 0.0,
        "drive": {"L": 0.0, "R": 0.0},
    }


# ============================================================
# МОЗГ
# ============================================================

def sense(position):
    """Насколько сильно источник виден или слышен слева и справа."""

    thorax_pos = data.xpos[thorax_id].copy()
    thorax_rot = data.xmat[thorax_id].reshape(3, 3)

    body = thorax_rot.T @ (position - thorax_pos)

    forward = body[0]
    side = body[1]

    # Впереди чувствуется лучше, чем сзади; дальше — слабее.
    ahead = np.clip((forward + 1.0) / 2.5, 0.2, 1.0)
    near = float(np.exp(-np.linalg.norm(body[:2]) / 12.0))

    return {
        "L": float(np.clip(side / 2.0, 0.0, 1.0) * ahead * near),
        "R": float(np.clip(-side / 2.0, 0.0, 1.0) * ahead * near),
    }


def cigarette():
    return sources[3]


def mouth_spot():
    """Куда Муза тянет сигарету: чуть впереди хоботка."""

    heading = data.xmat[thorax_id].reshape(3, 3)[:, 0]

    return data.xpos[mouth_id] + heading * 0.55 + np.array([0.0, 0.0, -0.03])


def held_item():
    """Что сейчас в лапках, или None."""

    return None if state["held"] is None else sources[state["held"]]


def drop_item():
    """Отпустить: вещь возвращается на своё место — как в Sims, Муза вещи
    не таскает. Место меняешь только ты (стрелки, R, G)."""

    spot = held_item()

    if spot is None:
        return

    spot["position"][:] = spot["home"]
    spot["facing"] = spot["home_facing"].copy()
    spot.pop("up", None)

    # Работала или курила у окна — отходит плавно туда, где встанет.
    if "release_to" in state:
        state["frozen"] = state.pop("release_to")

    state["held"] = None
    state["grab_block"] = GRAB_PAUSE

    journal.event(spot["name"], "выпустила из лапок")


def bite_beat(spot, moment):
    """Ритм еды: 0 — лапки опущены, 1 — поднесла ко рту.

    Двигаются именно лапки, а предмет едет вместе с ними.
    """

    if spot["latin"] == "burger":
        return float(0.5 + 0.5 * np.cos(2.0 * np.pi * (moment % 1.4) / 1.4))

    if spot["latin"] == "monster":
        return float(max(0.0, np.sin(np.pi * (moment % 2.2) / 2.2)))

    if spot["latin"] == "iphone":
        # Свайп: лапка внизу экрана -> быстро вверх -> обратно -> пауза.
        phase = (moment % SWIPE_SECONDS) / SWIPE_SECONDS
        if phase < 0.3:
            return float(0.5 - 0.5 * np.cos(np.pi * phase / 0.3))
        if phase < 0.7:
            return float(0.5 + 0.5 * np.cos(np.pi * (phase - 0.3) / 0.4))
        return 0.0

    if spot["latin"] == "cigarette":
        # Затяжка: поднесла ко рту, подержала, опустила, отдохнула.
        phase = moment % 4.0
        if phase < 1.0:
            return float(0.5 - 0.5 * np.cos(np.pi * phase))
        if phase < 2.0:
            return 1.0
        if phase < 3.0:
            return float(0.5 + 0.5 * np.cos(np.pi * (phase - 2.0)))
        return 0.0

    return 0.0


def exhale(heading, count=1):
    """Облачка дыма изо рта: летят вперёд и вверх, расширяются и тают."""

    rng = np.random.default_rng()
    for _ in range(count):
        start = data.xpos[mouth_id] + heading * 0.25 + rng.normal(0.0, 0.05, 3)
        drift = heading * 0.9 + np.array([0.0, 0.0, 0.35]) + rng.normal(0.0, 0.15, 3)
        state["smoke"].append((start, drift, time.perf_counter()))


def item_keep(spot):
    """Ближе этого грудь к предмету не подходит — упирается лбом."""

    return ITEM_RADIUS.get(spot["latin"], 0.3) + HEAD_REACH


def in_the_way():
    """Предметы, которые Муза сейчас не возьмёт: их она обходит заранее.
    То, за чем идёт, и то, что может взять, не обходит — к ним подходит."""

    target = None
    for kind, where in state["route"]:
        if kind == "item":
            target = where
        break

    can_grab = state.get("grab_block", 0.0) <= 0.0 and held_item() is None
    out = []

    for spot in sources:
        if spot is held_item() or spot["position"][2] > 2.0:
            continue
        if spot["latin"] == target:
            continue
        if can_grab and spot["latin"] in GRABBABLE and not state["errand"]:
            continue
        out.append((spot["position"][:2].copy(), item_keep(spot)))

    out.append((room.TOILET_BODY.copy(), room.TOILET_RADIUS + HEAD_REACH))
    for spot, radius in room.KITCHEN:
        out.append((spot.copy(), radius + HEAD_REACH))
    return out


def obstacles():
    """Всё, во что Муза упирается на полу: предметы и унитаз."""

    out = []

    thorax = data.xpos[thorax_id][:2]

    for spot in sources:
        if spot is held_item() or spot["position"][2] > 2.0:
            continue
        # Уже стоит вплотную (например, только что работала за ноутбуком) —
        # не выталкиваем рывком, пусть отойдёт сама. Внутрь снаружи не пустим.
        if float(np.linalg.norm(thorax - spot["position"][:2])) < item_keep(spot) - 0.05:
            continue
        out.append((spot["position"][:2].copy(), item_keep(spot)))

    # Кухонная мебель.
    for spot, radius in room.KITCHEN:
        out.append((spot.copy(), radius + HEAD_REACH))

    # На унитазе она сидит — там её ведём сами, упор не мешает.
    if state["chore"] != "poop":
        out.append((room.TOILET_BODY.copy(), room.TOILET_RADIUS + HEAD_REACH))

    return out


def lean_place():
    """Где и как стоять, куря у окна: мордой к окну, тело почти стоймя,
    брюшко на краю подоконника. И где встать на лапки, когда докурит."""

    yaw = -np.pi / 2  # окно в стене y = -HALF
    turn = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    rise = np.zeros(4)
    mujoco.mju_axisAngle2Quat(rise, np.array([0.0, 1.0, 0.0]), -LEAN_PITCH)
    lean = np.zeros(4)
    mujoco.mju_mulQuat(lean, turn, rise)

    place = data.qpos[:7].copy()
    place[0] = room.ASHTRAY[0]
    place[1] = -room.HALF + room.SILL_DEPTH + LEAN_GAP
    place[2] = room.SILL_TOP + LEAN_UP
    place[3:7] = lean

    back = data.qpos[:7].copy()
    back[0] = room.ASHTRAY[0]
    back[1] = -room.HALF + room.KEEP + 0.1
    back[2] = float(stand_height)
    back[3:7] = turn

    return place, back


def desk_place(laptop):
    """Где стоять за ноутбуком: мордой к экрану, лапки над клавиатурой."""

    facing = np.append(laptop["facing"], 0.0)
    keys = laptop["position"] + facing * props.LAPTOP_KEYBOARD

    rotation = data.xmat[thorax_id].reshape(3, 3)
    mouth_ahead = float((rotation.T @ (data.xpos[mouth_id] - data.xpos[thorax_id]))[0])

    place = data.qpos[:7].copy()
    place[0:2] = keys[:2] + facing[:2] * (KEYBOARD_AHEAD + mouth_ahead)

    yaw = np.arctan2(-facing[1], -facing[0])
    place[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    return place


def hold_can(spot, beat, heading):
    """Банка: в покое стоит в лапках ровно, на глотке лапки поднимают её
    вверх, и она опрокидывается краем к хоботку — донышком кверху.

    Край ставим перед хоботком, а не в него, поэтому в голову банка не
    заходит. Возвращает, где стоит донышко.
    """

    grip = hands_spot()
    upright = np.array([0.0, 0.0, 1.0])

    lip = data.xpos[mouth_id] + heading * CAN_LIP_AHEAD + upright * CAN_LIP_UP

    # Ведём край банки по прямой от «над лапками» к хоботку, а донышко
    # считаем от края: так банка всё время остаётся перед головой.
    rest_top = grip + heading * 0.22 + upright * 0.55 * props.MONSTER_HEIGHT
    top = rest_top + (lip - rest_top) * beat

    # Верх наклоняется назад, к Музе, донышко уходит вперёд и вверх.
    angle = CAN_POUR * beat * state["reach"]
    spot["up"] = upright * np.cos(angle) - heading * np.sin(angle)

    return top - spot["up"] * props.MONSTER_HEIGHT


def hold_phone(spot, moment, heading):
    """Айфон стоит перед Музой экраном к ней, лента едет на каждый свайп.

    Возвращает, где середина нижнего края айфона.
    """

    swipes = moment / SWIPE_SECONDS
    phase = swipes % 1.0

    # На обратном ходе лапка отходит от экрана: видно, что провела и оторвала.
    state["swipe_back"] = float(np.sin(np.pi * (phase - 0.3) / 0.4)) if 0.3 < phase < 0.7 else 0.0

    # Пока лапка идёт вверх, лента едет за ней на один пост, потом стоит.
    moved = 0.5 - 0.5 * np.cos(np.pi * min(phase / 0.3, 1.0))
    spot["scroll"] = (np.floor(swipes) + moved) * props.POST

    aside = data.xmat[thorax_id].reshape(3, 3)[:, 1]
    ahead, side, up = PHONE_SPOT

    return (
        data.xpos[mouth_id]
        + heading * ahead
        + aside * side
        + np.array([0.0, 0.0, up])
    )


def hands_spot():
    """Точка между кончиками передних лапок."""

    return (data.xpos[hand_ids[0]] + data.xpos[hand_ids[1]]) / 2.0


def take_item(seconds):
    """Если предмет рядом — Муза кладёт на него лапки и тянет ко рту."""

    state["grab_block"] = max(0.0, state.get("grab_block", 0.0) - seconds)

    spot = held_item()

    # Погасшую сигарету держать незачем: роняет сама.
    if spot is not None and spot is cigarette() and not spot["lit"]:
        drop_item()
        # Ты её погасила — сразу снова не берёт, иначе тут же зажгла бы.
        state["grab_block"] = FULL_PAUSE
        return

    # Стоит без дел, ходьба выключена: сама ничего не хватает — что взять,
    # решает в свободное время.
    idle = state["errand"] is None and not state["walking"]

    if spot is None and state["grab_block"] <= 0.0 and not idle:

        # По делам берёт только то, за чем идёт. Раньше по дороге в туалет
        # хватала айфон, залипала в ленту — и поход вставал.
        wanted = None
        if state["errand"] is not None:
            wanted = next((where for kind, where in state["route"] if kind == "item"), "")

        for index, near in enumerate(sources):

            if near["latin"] not in GRABBABLE:
                continue

            if wanted is not None and near["latin"] != wanted:
                continue

            # Сытая еду не берёт: раньше хватала, откусывала раз — «наелась» —
            # и тут же бросала.
            if appetite(near) <= 0.0:
                continue
            close = float(
                np.linalg.norm(near["position"][:2] - data.xpos[thorax_id][:2])
            )

            # Дотягивается, если подошла вплотную: упёрлась лбом — и берёт.
            if close > max(GRAB_RANGE, item_keep(near) + 0.5):
                continue

            state["held"] = index
            state["frozen"] = data.qpos[:7].copy()
            state.pop("pose_to", None)
            state.pop("release_to", None)

            # За компьютер встаёт сама: перед экраном, лапки над клавиатурой.
            # Ноутбук не двигается. Отходит оттуда же.
            if near["latin"] == "computer":
                state["pose_to"] = desk_place(near)
                state["release_to"] = state["pose_to"].copy()

            # Сигарету из пепельницы курит у окна: встаёт на задние лапки,
            # брюшком на подоконник. Потом плавно опускается.
            if near is cigarette() and room.surface(near["home"]) == room.SILL_TOP:
                state["pose_to"], state["release_to"] = lean_place()

            # Снимок команд суставам: позу строим от него, иначе сдвиг
            # прибавлялся бы каждый шаг и лапки просто крутились бы.
            state["base_ctrl"] = data.ctrl[: len(dof_names)].copy()

            near["taken_from"] = near["position"].copy()
            spot = near

            state["puffs"] = 0.0

            # Сигарета загорается, когда Муза берёт её в лапки.
            if near is cigarette() and not near["lit"]:
                near["lit"] = True
                journal.event("сигарета", "зажгла, как взяла в лапки")

            journal.event(near["name"], "взяла лапками", f"{close:.1f} мм до неё")
            break

    if spot is None:
        state["reach"] = max(0.0, state["reach"] - seconds / REACH_SECONDS)
        return

    state["reach"] = min(1.0, state["reach"] + seconds / REACH_SECONDS)

    moment = time.perf_counter()
    beat = bite_beat(spot, moment)

    state["beat"] = beat

    # Предмет лежит в лапках и ездит вместе с ними, а не сам по себе.
    heading = data.xmat[thorax_id].reshape(3, 3)[:, 0]
    # У каждого предмета свой посадочный уровень в лапках.
    drop = {"burger": -0.18}.get(spot["latin"], -0.05)

    # Бургер держит чуть дальше от морды, чем остальное.
    ahead = {"burger": 0.30}.get(spot["latin"], 0.10)
    target = hands_spot() + heading * ahead + np.array([0.0, 0.0, drop])

    if spot["latin"] == "monster":
        target = hold_can(spot, beat, heading)

    if spot["latin"] == "iphone":
        target = hold_phone(spot, moment, heading)

    # Курит у окна правой лапкой: сигарета зажата в кончике лапки, фильтром к
    # морде. Между затяжками она у лапки, в стороне от рта; на затяжке лапка
    # подаётся к морде и фильтр упирается в рот, потом отводит — выдох.
    if spot is cigarette() and "pose_to" in state:
        paw = data.xpos[hand_ids[1]]
        mouth = data.xpos[mouth_id] + np.array([0.0, -SMOKE_AHEAD, 0.0])  # чуть впереди хоботка
        # Горизонтально: от рта в сторону лапки, не задирая вверх.
        side = np.array([paw[0] - mouth[0], paw[1] - mouth[1], 0.0])
        side = side / max(np.linalg.norm(side), 1e-9)
        filter_end = mouth + side * (0.30 - 0.25 * beat)
        spot["hold_dir"] = -side  # фильтр смотрит к морде
        target = filter_end + side * FILTER_END * HELD_CIGARETTE

    # Затянулась и отводит сигарету — выдыхает клуб дыма, короткий: сначала
    # облако, потом маленькая струйка. Между выдохами — перерыв на затяжку.
    if spot is cigarette() and spot["lit"] and state["reach"] > 0.9:
        if 2.0 < moment % 4.0 < 2.4:
            flat = np.array([heading[0], heading[1], 0.0])
            if "pose_to" in state:
                flat = np.array([0.0, -1.0, 0.0])  # у окна — к стеклу
            flat = flat / max(np.linalg.norm(flat), 1e-9)
            puff = int(moment // 4.0)
            if state.get("puff_key") != puff:
                state["puff_key"] = puff
                exhale(flat, count=10)
            else:
                exhale(flat)

    if spot["latin"] == "computer":
        # Ноутбук стоит на месте — Муза сама встала перед ним и печатает.
        target = spot["position"].copy()
        spot["typed"] = spot.get("typed", 0.0) + TYPE_CHARS * seconds * state["reach"]

    spot["position"][:] = (
        spot["taken_from"] + (target - spot["taken_from"]) * state["reach"]
    )

    # Укус или глоток: считаем каждый такт, пока предмет у рта.
    period = {"burger": 1.4, "monster": 2.2}.get(spot["latin"])

    if period and state["reach"] > 0.9:

        bite = int(moment / period)

        if bite != state.get("last_bite"):
            state["last_bite"] = bite
            spot["eaten"] += 1
            state["poop"] += POOP_PER_BITE
            nourish(spot)

        if spot["eaten"] >= PORTION[spot["latin"]]:
            finish_item(spot)


def nourish(spot):
    """Укус насыщает, глоток поит и бодрит. Наелась или напилась — бросает."""

    state["joy"] = min(1.0, state["joy"] + JOY_PER_BITE)

    if spot["latin"] == "burger":
        state["full"] = min(1.0, state["full"] + FULL_PER_BITE)
        enough = state["full"] >= 1.0
    else:
        state["water"] = min(1.0, state["water"] + WATER_PER_SIP)
        state["energy"] = min(1.0, state["energy"] + ENERGY_PER_SIP)
        enough = state["water"] >= 1.0

    if enough:
        journal.event(spot["name"], "наелась" if spot["latin"] == "burger" else "напилась")
        drop_item()
        state["grab_block"] = FULL_PAUSE


def appetite(spot):
    """Голодную еда тянет, сытую — нет. Так и у живых мух: голод усиливает
    тягу к запаху еды."""

    if spot["latin"] == "burger":
        level = state["full"]
    elif spot["latin"] == "monster":
        level = state["water"]
    else:
        return 1.0

    # Сытую не тянет совсем; чем голоднее — тем сильнее.
    return float(np.clip((FULL_ENOUGH - level) / FULL_ENOUGH, 0.0, 1.0))


def finish_item(spot):
    """Доела или допила: предмет кончился, новый появляется на старом
    месте, а Муза какое-то время сыта и ничего не хватает."""

    state["held"] = None
    state["reach"] = 0.0
    state["grab_block"] = FULL_PAUSE

    spot["eaten"] = 0
    spot.pop("up", None)
    spot["position"][:] = spot["home"]

    journal.event(spot["name"], "доела", "новый на старом месте, сыта")


def hold_pose():
    """Пока держит — передние лапки идут в позу, тело придерживаем."""

    if state["reach"] <= 0.0 or "base_ctrl" not in state:
        if state["held"] is None:
            state.pop("pose_to", None)
        front_legs_on_floor(True)
        return

    base = state["base_ctrl"]
    beat = state.get("beat", 0.0)

    for index, offset in front_dofs.items():

        name = dof_names[index]

        # Кивок: на такте лапки подтягиваются ко рту и снова опускаются.
        # Банку поднимает иначе, чем бургер: она выше и держится снизу.
        latin = held_item()["latin"] if held_item() is not None else ""
        coxa_nod, tibia_nod = NOD.get(latin, (0.75, -0.55))

        nod = 0.0

        if latin == "computer":
            # Обе лапки внизу у клавиш, стучат по очереди: левая, правая.
            if name.endswith("trochanterfemur:PITCH"):
                tap = np.sin(2.0 * np.pi * TYPE_RATE * data.time)
                lift = max(0.0, tap) if name.startswith("lf_") else max(0.0, -tap)
                nod = TYPE_REACH - TYPE_TAP * lift

        elif latin == "iphone":
            # Левая держит, правая водит по экрану: бедро опущено — лапка
            # внизу экрана, выпрямилось — наверху.
            if name == "rf_trochanterfemur:PITCH":
                nod = SWIPE * (1.0 - beat)
            elif name == "rf_coxa:PITCH":
                nod = SWIPE_LIFT
            elif name == "rf_tibia:PITCH":
                nod = SWIPE_AWAY * state.get("swipe_back", 0.0)

        elif name.endswith("coxa:PITCH"):
            nod = coxa_nod * beat
        elif name.endswith("tibia:PITCH"):
            nod = tibia_nod * beat

        if latin in ("iphone",) and name.endswith("PITCH") and name.startswith("lf_"):
            nod = 0.0

        data.ctrl[index] = base[index] + (offset + nod) * state["reach"]

        # Курит у окна, стоя на задних: своя поза передних лапок — перед
        # брюхом. Обычная поза держания складывала их внутрь тела.
        if latin == "cigarette" and "pose_to" in state:
            joint = name.split("_", 1)[1]
            target = stand_ctrl[index]
            if name.startswith("rf_"):
                target += RIGHT_SMOKE.get(joint, 0.0) + RIGHT_PUFF.get(joint, 0.0) * beat
            else:
                target += LEAN_LEFT.get(joint, 0.0)
            data.ctrl[index] = base[index] + (target - base[index]) * state["reach"]

    front_legs_on_floor("cigarette" != latin or "pose_to" not in state)

    if "frozen" in state and (state["held"] is not None or "pose_to" in state):
        data.qpos[:7] = state["frozen"]

        # К компьютеру или к окну подходит плавно, пока тянет лапки, и так
        # же плавно отходит, когда отпустила.
        if "pose_to" in state:
            part = state["reach"]
            target = state["pose_to"]
            data.qpos[0:3] = state["frozen"][0:3] + (target[0:3] - state["frozen"][0:3]) * part
            data.qpos[3:7] = slerp(state["frozen"][3:7], target[3:7], part)

        data.qvel[:6] = 0.0


# Пары контакта с полом передних лапок и тела: у окна их выключаем.
FRONT_FLOOR = [
    i for i in range(model.npair)
    if model.pair(i).name.startswith(("musa/lf_", "musa/rf_", "musa/c_"))
]
FRONT_FLOOR_MARGIN = model.pair_margin[FRONT_FLOOR].copy()


LEAN_LEFT = {"coxa:PITCH": 0.4}  # левая у окна: кончик на подоконнике
# Правая у окна: кончик на высоте рта, сбоку от морды — держит сигарету
# горизонтально; на затяжке подаётся к морде.
RIGHT_SMOKE = {"coxa:PITCH": 1.0, "trochanterfemur:PITCH": -0.3, "tibia:PITCH": -0.5}
RIGHT_PUFF = {"tibia:PITCH": -0.2}
SMOKE_AHEAD = 0.35  # мм: фильтр чуть впереди хоботка, не в голове


def front_legs_on_floor(on):
    """Касаются ли пола передние лапки и тело. Выключено — пол их не гнёт."""

    if state.get("front_floor", True) == on:
        return
    state["front_floor"] = on
    model.pair_margin[FRONT_FLOOR] = FRONT_FLOOR_MARGIN if on else -10.0


def nicotine_dose():
    """Никотин действует тем сильнее, чем ближе горящая сигарета."""

    smoke = cigarette()

    if not smoke["lit"]:
        return 0.0

    felt = float(np.clip(smoke["drive"]["L"] + smoke["drive"]["R"], 0.0, 1.0))

    return NICOTINE_DOSE * felt


def run_chain(feeling, drive):
    """Гонит цепочку и приводит её к общей силе и правильной стороне."""

    chain = feeling["chain"]

    raw = chain.step(drive, feeling["gains"], drug=state["nicotine"])

    # Полярность: нечётное число тормозных этажей разворачивает поворот.
    # Какой стороне тормозить ноги — это уже наш выбор, а не данные.
    return raw * chain.polarity * (0.4 / chain.response)


def show_lead(feeling):
    """Подсветка: какая сторона сейчас ведёт на каждом этаже."""

    chain = feeling["chain"]

    for i, (number, body_side) in enumerate(zip(feeling["numbers"], feeling["sides"])):

        other = "R" if body_side == "L" else "L"
        lead = chain.activity[body_side][number] - chain.activity[other][number]

        feeling["activity"][i] = float(np.clip(0.5 + lead, 0.0, 1.0))


def brain_step():
    """Каждый источник гонит свою цепочку, повороты складываются."""

    steering = 0.0
    strongest = {}

    # Доза считается по прошлому кадру: сигарета должна быть уже учуяна.
    state["nicotine"] = nicotine_dose()

    for spot in sources:

        feeling = feelings[spot["chain"]]

        spot["drive"] = sense(spot["position"])

        raw = run_chain(feeling, spot["drive"])

        # Вес KC -> MBON: 1.00 тянет к источнику, 0.50 делает его безразличным,
        # 0.00 разворачивает. Так и работает выученное избегание.
        wish = (2.0 * spot["memory"] - 1.0) * appetite(spot) * spot.get("strength", 1.0)
        spot["wish"] = wish
        spot["turn"] = raw * wish

        # Насколько сильно её к нему тянет — независимо от того, прямо он
        # или сбоку: это и выбирает цель.
        spot["pull"] = (spot["drive"]["L"] + spot["drive"]["R"]) * wish

        # Подсветку в закладке показываем по выбранному источнику, а если
        # выбран другой — по тому, который сейчас слышно лучше.
        felt = spot["drive"]["L"] + spot["drive"]["R"]
        picked = sources.index(spot) == state["chosen_source"]

        if picked or felt > strongest.get(spot["chain"], -1.0):
            strongest[spot["chain"]] = np.inf if picked else felt
            show_lead(feeling)

    # Одна цель за раз: раньше бургер, банка и свет тянули одновременно, и
    # она кружила вокруг стола. Выбранную цель держит FOCUS_SECONDS, остальное
    # приятное тянет вчетверо слабее. От неприятного отворачивает в полную силу.
    focus = state.get("focus")
    best = max(sources, key=lambda x: x["pull"])
    # Цель держится своё время, даже если на миг оказалась сзади. Бросает
    # её, только если та стала не нужна: наелась, выучила «плохо», уже в лапках.
    if (
        focus is None
        or data.time > state.get("focus_until", 0.0)
        or focus["wish"] <= 0.0
        or focus is held_item()
    ):
        focus = best if best["pull"] >= FOCUS_MIN else None
        state["focus"] = focus
        state["focus_until"] = data.time + FOCUS_SECONDS

    for spot in sources:
        weight = 1.0 if spot is focus or spot["pull"] < 0.0 else OTHERS_WEIGHT
        steering += spot["turn"] * weight

    rotation = data.xmat[thorax_id].reshape(3, 3)
    thorax = data.xpos[thorax_id]

    # Сама идёт по делам: наводится прямо туда, остальное не слушает.
    if state["errand"] is not None and state["chore"] is None:
        body = rotation.T @ (np.append(errand_target(), 0.0) - thorax)
        steering = float(np.clip(1.5 * np.arctan2(body[1], body[0]), -1.5, 1.5))


    # Стена — препятствие: у стены её отворот главнее всего остального.
    # По делам в угол подпускаем стену ближе, иначе не дойдёт. В коридоре и
    # у лифта — ещё ближе: там узко, и отворот разворачивал её от кабины на
    # круг по всему коридору.
    reach = 4.0
    if state["errand"] in OUTINGS:
        reach = 1.4
    elif state["errand"]:
        reach = 2.2
    away = room.obstacle_push(
        thorax, rotation[:, 0], in_the_way(), reach=reach,
        ignore_door=state["errand"] in ("poop", "leave", "close"),
    )

    steering = float(np.clip(steering, -1.5, 1.5))

    if away:
        closeness = min(abs(away) / 6.0, 1.0)
        steering = steering * (1.0 - closeness) + np.clip(away, -1.5, 1.5) * closeness

    left_signal = np.clip(1.0 - TURN * steering, 0.20, 1.80)
    right_signal = np.clip(1.0 + TURN * steering, 0.20, 1.80)

    # Застряла — сначала отходит назад, потом разворачивается на месте
    # (левые лапки шагают назад, правые вперёд) и пробует снова. Одно
    # верчение на месте в тесном углу не спасало.
    if state.get("unstick", 0.0) > data.time:
        if state["unstick"] - data.time > 1.2:
            return np.array([-0.8, -0.8]), 0.0
        return np.array([-0.8, 0.8]), 1.5

    return np.array([left_signal, right_signal], dtype=float), steering


def crave(seconds):
    """Привыкание: под никотином то, что Муза чувствует, само становится милее.

    Это не из коннектома. Так ведут себя живые мухи: никотин и спирт они
    начинают искать сами, и метка ложится на то, что было рядом.
    """

    if state["nicotine"] <= 0.0:
        return

    felt = [(spot["drive"]["L"] + spot["drive"]["R"], spot) for spot in sources]
    strength, spot = max(felt, key=lambda pair: pair[0])

    if strength < SMELLED:
        return

    before = spot["memory"]
    spot["memory"] = float(
        np.clip(before + CRAVING * state["nicotine"] * seconds, 0.0, 1.5)
    )

    # В журнал пишем не каждый кадр, а когда набежала заметная доза.
    if int(before * 10) != int(spot["memory"] * 10):
        journal.event(
            "никотин",
            f"{spot['name']} стал милее",
            f"вес KC>MBON x{spot['memory']:.2f}",
        )


def teach_target():
    """Кого учим: что в лапках; иначе то, к чему Муза подходит (ближайшее
    впереди); иначе то, что выбрано кнопкой."""

    if held_item() is not None:
        return held_item()

    thorax = data.xpos[thorax_id]
    rotation = data.xmat[thorax_id].reshape(3, 3)

    best, best_distance = None, APPROACH

    for spot in sources:

        body = rotation.T @ (spot["position"] - thorax)
        distance = float(np.linalg.norm(body[:2]))

        if body[0] > 0.0 and distance < best_distance:
            best, best_distance = spot, distance

    return best or sources[state["chosen_source"]]


def teach(delta, label):
    """Дофамин двигает память того, кого сейчас учим (см. teach_target)."""

    spot = teach_target()

    spot["memory"] = float(np.clip(spot["memory"] + delta, 0.0, 1.5))
    spot["taught"] += 1

    journal.event(
        "память",
        f"{spot['name']} {label}",
        f"вес KC>MBON x{spot['memory']:.2f}",
    )


# ============================================================
# ОБЩЕЕ СОСТОЯНИЕ ОКНА
# ============================================================

state = {
    "selected": 0,
    "side": None,
    "walking": False,
    "follow": False,  # камера свободная; F — следить за Музой
    "active": 0,
    "chain_name": "light",
    "mode": "chain",
    "chosen_source": 0,
    "nicotine": 0.0,
    "energy": 1.0,  # 0 — сил нет, хочет отдохнуть
    "poop": 0.0,  # 1 — хочет в туалет
    "errand": None,  # куда идёт сама: "rest", "poop" или "leave"
    "route": [],  # шаги маршрута: ("walk", точка), ("door", "open"), ...
    "chore": None,  # что делает сейчас: "rest", "poop" или "flipped"
    "chore_time": 0.0,
    "walk_before": False,
    "poops": [],
    "smoke": [],  # облачка выдоха: где появилось, куда летит, когда
    "poop_count": 0,
    "full": 0.3,  # сытость: от бургера
    "water": 0.3,  # вода: от энергетика
    "joy": 0.0,  # удовольствие от ленты и сигареты
    "work": 0.0,  # 1 — пора к компьютеру
    "worked": 0.0,  # сколько уже напечатала за этот раз
    "doing": "",
    "shock": 0.0,
    "warp": 1,  # ускорение жизни: x1, x2, x4, x8
    "held": None,  # какой предмет сейчас в лапках
    "reach": 0.0,  # 0 — лапки внизу, 1 — предмет у рта
}


# ============================================================
# ДВА ЧУВСТВА
# ============================================================

burger = load_chain("food", "бургер", "burger", (0.95, 0.45, 0.15), FOOD_A_START)

feelings = [
    load_chain("light", "свет", "light", (1.0, 1.0, 0.1), LIGHT_START),
    burger,
    load_chain(
        "food", "энергетик", "monster", (0.45, 0.95, 0.20), FOOD_B_START, share=burger
    ),
    load_chain("memory", "память", "memory", (0.6, 0.85, 1.0), FOOD_A_START),
]

# Три источника: свет и два запаха. У каждого свой вес KC -> MBON — это и есть
# память. 1.00 — как в коннектоме, меньше — источник стал неприятен, 0.00 —
# Муза от него отворачивается.
def source(name, latin, chain_index, position, color):
    return {
        "name": name,
        "latin": latin,  # шрифт MuJoCo кириллицу не рисует
        "chain": chain_index,
        "position": np.asarray(position, dtype=float).copy(),
        "color": np.asarray(color, dtype=float),
        "memory": 1.0,
        "taught": 0,
        "drive": {"L": 0.0, "R": 0.0},
        "turn": 0.0,
        "eaten": 0,  # сколько укусов или глотков уже сделано
        "start": np.asarray(position, dtype=float).copy(),
        "home": np.asarray(position, dtype=float).copy(),  # своё место, как в Sims
    }


sources = [
    source("свет", "light", 0, LIGHT_START, (1.0, 1.0, 0.1)),
    source("бургер", "burger", 1, FOOD_A_START, (0.95, 0.45, 0.15)),
    source("энергетик", "monster", 2, FOOD_B_START, (0.45, 0.95, 0.20)),
    source("сигарета", "cigarette", 2, CIGARETTE_START, (0.90, 0.90, 0.85)),
    # Ещё два зрительных триггера: светящиеся экраны идут через ту же
    # зрительную цепочку, что и лампочка, но память у каждого своя.
    source("айфон", "iphone", 0, IPHONE_START, (0.40, 0.60, 1.00)),
    source("компьютер", "computer", 0, COMPUTER_START, (0.60, 0.85, 0.95)),
]

# Свежая муха к никотину равнодушна: вес ровно посередине, тяги нет.
sources[3]["memory"] = 0.5
sources[3]["lit"] = False  # загорается, когда Муза берёт её в лапки

# Лампочка по умолчанию тянет вдвое слабее остального.
sources[0]["strength"] = LIGHT_STRENGTH

# Айфон и ноутбук стоят экраном к середине комнаты и сами не вертятся.
for spot in sources:
    spot["facing"] = -spot["position"][:2] / max(np.linalg.norm(spot["position"][:2]), 1e-9)
    if spot["latin"] == "computer":
        spot["facing"] = np.array([0.0, -1.0])  # спиной к стене, экраном в комнату
    if spot["latin"] == "iphone":
        spot["facing"] = IPHONE_FACING / np.linalg.norm(IPHONE_FACING)
    spot["home_facing"] = spot["facing"].copy()
    # На какой высоте вещь стоит над опорой — на столе и на полу одинаково.
    spot["floor_z"] = spot["start"][2] - room.surface(spot["start"])


# Расстановка запоминается, как в Sims: куда ты поставила вещь, там она и
# будет при следующем запуске. HOME в меню вещей возвращает заводское место.
LAYOUT = Path(__file__).resolve().parent / "layout.json"


def save_layout():
    import json

    layout = {
        spot["latin"]: {
            "home": [round(float(v), 4) for v in spot["home"]],
            "facing": [round(float(v), 4) for v in spot["home_facing"]],
        }
        for spot in sources
    }
    LAYOUT.write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")


def load_layout():
    import json

    if not LAYOUT.exists():
        return
    try:
        layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return
    for spot in sources:
        saved = layout.get(spot["latin"])
        if not saved or spot["latin"] == "cigarette":  # сигарета — всегда в пепельнице
            continue
        spot["home"] = np.array(saved["home"], dtype=float)
        spot["position"][:] = spot["home"]
        spot["home_facing"] = np.array(saved["facing"], dtype=float)
        spot["facing"] = spot["home_facing"].copy()


load_layout()

light = sources[0]["position"]
odors = sources[1:]

journal = Journal(Path(__file__).resolve().parent.parent / "outputs")


def active():
    return feelings[state["active"]]


def show_active():
    """Панель и карта показывают ту цепочку, которая сейчас выбрана."""

    feeling = active()

    state["chain_name"] = feeling["name"]
    state["tab_label"] = feeling["latin"]
    state["mode"] = "memory" if feeling["name"] == "memory" else "chain"
    state["sources"] = sources
    state["weights"] = feeling["weights"]
    state["gains"] = feeling["gains"]
    state["activity"] = feeling["activity"]
    state["targets"] = feeling["targets"]
    state["functions"] = feeling["functions"]
    state["selected"] = min(state["selected"], len(feeling["targets"]) - 1)


show_active()


# ============================================================
# ДЕЙСТВИЯ
# ============================================================

def toggle_walk():
    state["walking"] = not state["walking"]
    journal.event("ходьба", "включила" if state["walking"] else "выключила")


def reset_weights():
    # Все цепочки сразу, а не только открытая сейчас.
    for feeling in feelings:
        feeling["gains"]["L"][:] = 1.0
        feeling["gains"]["R"][:] = 1.0
    journal.event("веса", "вернула как в мозге", "все цепочки")


def scrolling(seconds):
    """Удовольствие. Лента: пока листает — растёт, а энергия тает быстрее,
    чем от ходьбы; чем меньше сил, тем меньше радости; силы кончились —
    откладывает айфон (и пойдёт на матрас). Сигарета: радует, но энергии
    отнимает больше, чем даёт радости."""

    phone = held_item()
    holding = phone is not None and state["reach"] > 0.9
    using = holding and phone["latin"] == "iphone"
    smoking = holding and phone is cigarette() and phone["lit"]
    typing = holding and phone["latin"] == "computer"

    if typing:
        state["joy"] = min(1.0, state["joy"] + WORK_JOY * seconds)
        state["energy"] = max(0.0, state["energy"] - WORK_DRAIN * seconds)
        state["worked"] += seconds

        if state["worked"] >= WORK_SECONDS or state["energy"] <= 0.0:
            drop_item()
            state["grab_block"] = PHONE_BREAK
            state["work"] = 0.0
            state["worked"] = 0.0
            journal.event("компьютер", "поработала", f"энергия {state['energy']:.0%}")
        return

    state["work"] = min(1.0, state["work"] + seconds / WORK_EVERY)

    if smoking:
        state["joy"] = min(1.0, state["joy"] + SMOKE_JOY * seconds)
        state["energy"] = max(0.0, state["energy"] - SMOKE_DRAIN * seconds)
        state["puffs"] = state.get("puffs", 0.0) + seconds / 4.0  # одна тяга — 4 с

        if state["energy"] <= 0.0 and state["puffs"] >= SMOKE_PUFFS:
            drop_item()
            state["grab_block"] = PHONE_BREAK
            journal.event("сигарета", "сил нет, бросила", f"удовольствие {state['joy']:.0%}")
        return

    if using and state.pop("phone_session", False):
        state["phone_until"] = data.time + PHONE_SESSION
    if using and data.time > state.get("phone_until", np.inf):
        state.pop("phone_until")
        drop_item()
        state["grab_block"] = PHONE_BREAK
        journal.event("айфон", "насмотрелась, отложила", f"удовольствие {state['joy']:.0%}")
        return

    if using:
        if state.get("was_scrolling") is not True:
            journal.event("айфон", "листает ленту, нравится")
        state["energy"] = max(0.0, state["energy"] - seconds / PHONE_TIRES_SECONDS)
        fade = max(0.0, JOY_FADES_BELOW - state["energy"]) / JOY_FADES_BELOW
        wanted = 1.0 - fade
        state["joy"] += (wanted - state["joy"]) * min(1.0, JOY_RATE * seconds * 3.0)
    else:
        state["joy"] = max(0.0, state["joy"] - JOY_FADE * seconds)
        if not holding:
            state.pop("phone_until", None)

    state["was_scrolling"] = using

    if using and state["energy"] <= 0.0:
        drop_item()
        state["grab_block"] = PHONE_BREAK
        journal.event("айфон", "устала, отложила", f"удовольствие {state['joy']:.0%}")


def watch_stuck(seconds):
    """Если идёт, а за 3 секунды почти не сдвинулась — застряла."""

    if state["chore"] is not None or not state["walking"] or held_item() is not None:
        state["stuck_mark"] = None
        return

    here = data.xpos[thorax_id][:2].copy()
    mark = state.get("stuck_mark")

    if mark is None or float(np.linalg.norm(here - mark[0])) > 0.5:
        state["stuck_mark"] = (here, data.time)
        return

    if data.time - mark[1] > 3.0:
        state["unstick"] = data.time + 2.0
        state["stuck_mark"] = (here, data.time + 2.0)
        journal.event("Муза", "застряла, разворачивается")


def life(seconds):
    """Усталость, туалет, отдых, переворот и ток — всё, что идёт само."""

    # Сигарета не в лапках — гаснет.
    smoke = cigarette()
    if smoke["lit"] and held_item() is not smoke:
        smoke["lit"] = False
        journal.event("сигарета", "погасла: не в лапках")

    state["shock"] = max(0.0, state["shock"] - seconds)
    watch_stuck(seconds)
    scrolling(seconds)

    if state["chore"] is None:
        state["energy"] = max(
            0.0, state["energy"] - seconds / ENERGY_SECONDS * (1.0 if state["walking"] else 0.3)
        )
        state["poop"] += seconds / POOP_SECONDS

    state["full"] = max(0.0, state["full"] - seconds / FULL_SECONDS)
    state["water"] = max(0.0, state["water"] - seconds / WATER_SECONDS)

    run_lift(seconds)
    run_gate(seconds)
    run_cafe(seconds)

    # Дверь подъезда закрывается сама: Муза её только открывает. Закрывается,
    # когда Муза отошла от двери и проходить в неё сейчас не собирается.
    door_middle = np.array([room.HOUSE[0], STREET_DOOR_Y])
    away = float(np.linalg.norm(data.xpos[thorax_id][:2] - door_middle)) > 3.0
    passing = any(kind == "glide" for kind, _ in state["route"][:2])
    if room.STREET["open"] > 0.0 and room.WHERE["floor"] == 1 and away and state["chore"] is None and not passing:
        room.STREET["open"] = max(0.0, room.STREET["open"] - seconds / DOOR_SECONDS)

    # Входная дверь с доводчиком: когда Муза дома, закрывается сама.
    inside = room.WHERE["floor"] == 5 and data.xpos[thorax_id][1] < room.HALF - 1.6
    if room.FRONT["open"] > 0.0 and inside and state["chore"] is None and state["errand"] not in ("out", "cafe"):
        room.FRONT["open"] = max(0.0, room.FRONT["open"] - seconds / DOOR_SECONDS)

    # В кафе захотела в туалет — идёт в туалет кафе, потом дальше по делам.
    here = data.xpos[thorax_id]
    heading = state["route"][0][0] if state["route"] else None
    if (
        state["errand"] == "cafe" and state["poop"] >= 1.0 and state["chore"] is None
        and room.in_cafe(here) and heading in ("walk", "glide") and room.CAFE["cup"] != "held"
        and "after_toilet" not in state
    ):
        room.use_bathroom("cafe")
        state["after_toilet"] = state["route"]
        state["route"] = [
            ("walk", room.DOOR_OPEN_SPOT),
            ("door", "open"),
            ("walk", room.TOILET_FRONT),
            ("poop", None),
        ]
        journal.event("Муза", "идёт в туалет кафе")

    # Захотела — идёт сама, даже если ходьба выключена.
    if state["errand"] is None and state["chore"] is None and held_item() is None:

        if state["poop"] >= 1.0:
            start_errand("poop")
        elif room.DOOR["open"] > 0.2:
            start_errand("close")
        elif state["energy"] <= 0.0:
            start_errand("rest")
        elif state["full"] <= HUNGRY:
            start_errand("eat")
        elif state["water"] <= THIRSTY:
            start_errand("drink")
        elif state["work"] >= 1.0:
            start_errand("work")
        elif not state["walking"] and room.WHERE["floor"] == 5:
            free_time()
        else:
            state["idle_since"] = data.time

    # Идёт по маршруту: дошла до точки — следующий шаг.
    while state["errand"] is not None and state["chore"] is None and state["route"]:

        kind, where = state["route"][0]
        thorax = data.xpos[thorax_id]

        if kind == "walk":
            # К унитазу хватит подойти поближе: садиться всё равно плавно
            # подводим сами, а упор у унитаза бывает не даёт встать точно.
            close_enough = 1.5 if state["route"][1:2] == [("poop", None)] else 0.7
            if any(np.array_equal(where, spot) for spot in AROUND + [GATE_OUTSIDE]):
                close_enough = AROUND_REACH
            gap = float(np.linalg.norm(thorax[:2] - where))
            if gap > close_enough and not circling(where, gap):
                break
            state.pop("walk_best", None)
            state["route"].pop(0)

        elif kind == "mattress":
            if not room.on_mattress(thorax, margin=2.0):
                break
            state["route"].pop(0)

        elif kind == "item":
            # Дошла и взяла — дальше ест или печатает сама.
            if held_item() is None or held_item()["latin"] != where:
                break
            state["route"].pop(0)

        elif kind == "gate":
            # Ждёт, пока ворота отъедут.
            if room.GATE["open"] < 0.95:
                state["walking"] = False
                break
            state["walking"] = True
            state["route"].pop(0)

        elif kind == "look":
            # Смотрит в окно: стоит, радость понемногу прибывает.
            if "look_until" not in state:
                state["look_until"] = data.time + where
            if data.time < state["look_until"]:
                state["walking"] = False
                break
            state.pop("look_until")
            state["joy"] = min(1.0, state["joy"] + WINDOW_JOY)
            journal.event("Муза", "посмотрела в окно", "радость")
            state["route"].pop(0)

        elif kind == "joy":
            state["joy"] = min(1.0, state["joy"] + where)
            state["route"].pop(0)

        elif kind == "homebath":
            # Вышла из туалета кафе: «тот самый» туалет снова домашний.
            room.use_bathroom("home")
            state["route"].pop(0)

        elif kind == "zebra":
            # У края тротуара: ждёт, пока машина проедет зебру.
            if not room.zebra_clear():
                state["walking"] = False
                break
            state["walking"] = True
            state["route"].pop(0)

        elif kind == "cafe":
            # Ждёт, пока стеклянная дверь кафе разъедется.
            if room.CAFE["open"] < 0.95:
                state["walking"] = False
                break
            state["walking"] = True
            state["route"].pop(0)

        elif kind == "order":
            # У стойки: бариста поворачивается к кофемашине, наливает кофе в
            # стаканчик, ставит на стойку — Муза берёт его лапкой.
            cafe = room.CAFE
            if "order_start" not in state:
                state["order_start"] = data.time
                cafe.update(cup=None, fill=0.0, seat=None, pull=0.0, sip=0.0)
                journal.event("бариста", "варит кофе")
                state["joy"] = min(1.0, state["joy"] + BARISTA_JOY)
                journal.event("Муза", "поболтала с баристой", "радость")
            moment = data.time - state["order_start"]
            cafe["order"] = moment
            pose = barista_pose(moment)
            cafe["cup"], cafe["fill"] = pose["cup"], pose["fill"]
            if moment < BARISTA_TOTAL:
                state["walking"] = False
                break
            state.pop("order_start")
            cafe.update(order=None, cup="held")
            journal.event("Муза", "взяла стаканчик кофе лапкой")
            state["walking"] = True
            state["route"].pop(0)


        elif kind == "lift":
            # Ждёт у дверей, пока лифт приедет и откроется.
            if not (room.lift_here() and room.LIFT["doors"] >= 0.95):
                state["walking"] = False
                break
            state["walking"] = True
            state["route"].pop(0)

        else:
            state["route"].pop(0)
            start_chore(kind, where)

    if state["errand"] is not None and state["chore"] is None and not state["route"]:
        if state["errand"] in ("out", "cafe"):
            # Погуляла — немного радости, и пора домой.
            state["joy"] = min(1.0, state["joy"] + WALK_JOY)
            state["errand"] = "home"
            state["route"] = home_route()
        else:
            finish_errand()

    if state["chore"] is None:
        return

    state["chore_time"] += seconds
    moment = state["chore_time"]

    if state["chore"] == "poop":

        # Села, посидела — какашка падает в воду.
        if moment > SIT_SECONDS + POOP_PAUSE / 2 and not state["pooped"]:
            water = room.TOILET + np.array([0.35, 0.0])
            state["poops"].append(np.array([water[0], water[1], room.SEAT_HEIGHT + 0.05]))
            state["pooped"] = True
            state["poop_count"] += 1
            journal.event("Муза", "покакала в туалет", f"всего {state['poop_count']}")

        # Встала — смыла и уходит из ванной.
        if moment > 2 * SIT_SECONDS + POOP_PAUSE:
            state["poops"].clear()
            state["poop"] = 0.0
            data.qpos[0:2] = room.TOILET_FRONT
            data.qpos[3:7] = state["sit_from"]
            data.qvel[:6] = 0.0
            journal.event("Муза", "смыла")
            state["chore"] = None
            state["walking"] = True
            state["route"] = [
                ("walk", room.DOOR_OUTSIDE),
                ("walk", room.DOOR_KICK_SPOT),
                ("door", "close"),
                ("walk", room.DOOR_AWAY),
            ]
            if "after_toilet" in state:
                # Туалет в кафе: закрыла дверь — и дальше по своим делам.
                state["route"] += [("homebath", None)] + state.pop("after_toilet")
            else:
                state["errand"] = "leave"

    elif state["chore"] == "sit":

        # Отодвигает стул, садится, ставит стаканчик, отпивает, встаёт.
        cafe = room.CAFE
        sat, drunk, stood = cafe_times()
        cafe["pull"] = min(1.0, moment / PULL_SECONDS)
        if moment > sat and cafe["cup"] == "held":
            cafe["cup"] = "sip"
            journal.event("Муза", "села за столик, пьёт кофе")
        if cafe["cup"] == "sip":
            # Пьёт как энергетик: держит обеими передними лапками у рта и
            # на такте запрокидывает стаканчик.
            cafe["sip"] = float(np.clip(min(moment - sat, drunk - moment) / LIFT_CUP, 0.0, 1.0))
            cafe["beat"] = float(max(0.0, np.sin(np.pi * ((moment - sat) % 2.2) / 2.2)))
            # Кофе бодрит и поит: пока пьёт, прибывают энергия и вода.
            part = seconds / COFFEE_SECONDS
            state["energy"] = min(1.0, state["energy"] + COFFEE_ENERGY * part)
            state["water"] = min(1.0, state["water"] + COFFEE_WATER * part)
            state["full"] = min(1.0, state["full"] + COFFEE_FULL * part)
        if moment > drunk and cafe["cup"] == "sip":
            cafe.update(cup="table", sip=0.0)
            journal.event("Муза", "допила, оставила стаканчик на столике")
        if moment > stood:
            table = state["seat_table"]
            side = room.aisle_side(table)
            data.qpos[0:2] = room.beside_chair(table)
            data.qpos[3:7] = yaw_quat(np.arctan2(side, 0.0))
            data.qvel[:6] = 0.0
            state["chore"] = None
            state["walking"] = True

    elif state["chore"] == "door":

        if moment > DOOR_SECONDS:
            # Дверь доходит до конца: с TIME xN время дела прыгает кусками,
            # и дверь застревала приоткрытой — Муза билась о неё и бегала кругами.
            door, _, _ = door_parts()
            door["open"] = state["door_to"]
            if state["door_which"] == "button":
                room.LIFT["called"] = floor_z()
                journal.event("лифт", "вызван", f"{room.WHERE['floor']} этаж")
            if state["door_which"] == "gate":
                room.GATE["called"] = True
                journal.event("ворота", "открываются")
            state["chore"] = None
            state["walking"] = True

    elif state["chore"] in ("board", "glide"):

        if moment > state["board_seconds"]:
            state["chore"] = None
            state["walking"] = True

    elif state["chore"] == "ride":

        lift = room.LIFT
        shut = LIFT_DOOR_SECONDS
        travel = abs(state["ride_to"] - state["ride_from"]) / room.LIFT_SPEED

        # Двери съезжаются, кабина едет, двери разъезжаются.
        if moment < shut:
            lift["doors"] = max(0.0, 1.0 - moment / shut)
        elif moment < shut + travel:
            lift["doors"] = 0.0
            lift["z"] = state["ride_from"] + (state["ride_to"] - state["ride_from"]) * (moment - shut) / travel
        else:
            lift["z"] = state["ride_to"]
            lift["doors"] = min(1.0, (moment - shut - travel) / shut)

        if moment > 2 * shut + travel:
            lift["z"] = state["ride_to"]
            lift["doors"] = 1.0
            lift["idle"] = 0.0
            room.WHERE["floor"] = state["ride_storey"]
            state["ride_done"] = data.time
            state["chore"] = None
            state["walking"] = True
            journal.event("Муза", "приехала на лифте", f"{room.WHERE['floor']} этаж")

    elif state["chore"] == "rest":

        # Спит — энергия быстро возвращается: от нуля до полной за время сна.
        if LIE_SECONDS < moment < REST_SECONDS + LIE_SECONDS:
            state["energy"] = min(1.0, state["energy"] + seconds / REST_SECONDS)

        if moment > REST_SECONDS + 2 * LIE_SECONDS:
            state["energy"] = 1.0
            journal.event("Муза", "отдохнула и встала")
            state["chore"] = None
            finish_errand()

    elif state["chore"] == "flipped":

        if moment > FLIP_SECONDS + RIGHTING_SECONDS + 0.5:
            journal.event("Муза", "перевернулась обратно")
            state["chore"] = None
            finish_errand()


def circling(where, gap):
    """Кружит у точки маршрута: уже рядом, но 4 с не может подойти ближе —
    радиус поворота больше, чем нужно. Тогда точку засчитываем, иначе она
    ходит по кругу без конца."""

    key = tuple(np.round(np.asarray(where, dtype=float), 3))
    best = state.get("walk_best")
    if best is None or best[0] != key or gap < best[1] - 0.2:
        state["walk_best"] = (key, gap, data.time)
        return False
    return gap < 3.0 and data.time - best[2] > 4.0


def pastime_pulls():
    """Тяга к каждому занятию: что оно даст просевшим нуждам × вес синапса
    памяти × привыкание. Возвращает {занятие: тяга}."""

    memory = {spot["latin"]: spot["memory"] for spot in sources}
    bored = 1.0 - state["joy"]
    hungry = 1.0 - state["full"]
    thirsty = 1.0 - state["water"]
    tired = 1.0 - state["energy"]
    pulls = {
        "phone": memory["iphone"] * (0.9 * bored - 0.4 * tired),
        "window": memory["light"] * 0.4 * bored,
        "stroll": 0.35 * bored - 0.5 * tired,
        "rest": 0.9 * tired - 0.3,
        "cafe": (memory["burger"] + memory["monster"]) / 2
        * (0.5 * bored + 0.5 * hungry + 0.5 * thirsty + 0.25) - 0.9 * tired - 0.35,
        "out": 0.45 * bored - 0.7 * tired - 0.1,
    }
    for name, when in state.get("habits", {}).items():
        fade = max(0.0, 1.0 - (data.time - when) / HABIT_SECONDS)
        pulls[name] -= HABIT * fade
    return pulls


def free_time():
    """Нужд нет: постояв IDLE_SECONDS, выбирает занятие с самой сильной тягой."""

    since = state.setdefault("idle_since", data.time)
    if data.time - since < IDLE_SECONDS:
        return
    state["idle_since"] = data.time
    pulls = pastime_pulls()
    choice, pull = max(pulls.items(), key=lambda item: item[1])
    if pull < IDLE_THRESHOLD:
        return
    state.setdefault("habits", {})[choice] = data.time
    state["pastime"] = choice
    detail = ", ".join(f"{name} {value:+.2f}" for name, value in sorted(pulls.items(), key=lambda item: -item[1]))
    journal.event("Муза", f"выбрала занятие: {choice}", detail)
    if choice in ("out", "cafe"):
        (go_out if choice == "out" else go_cafe)()
    else:
        start_errand(choice)


def start_errand(what):
    """Составляет маршрут и идёт."""

    thorax = data.xpos[thorax_id]

    if what == "poop":
        route = []
        if not room.in_bathroom(thorax):
            route += [
                ("walk", room.DOOR_OPEN_SPOT),
                ("door", "open"),
            ]
        route += [("walk", room.TOILET_FRONT), ("poop", None)]
        journal.event("Муза", "идёт в туалет")
    elif what == "close":
        # Дверь открытой не оставляет: подходит и закрывает задней лапкой.
        route = leave_bathroom() or [
            ("walk", room.DOOR_KICK_SPOT),
            ("door", "close"),
        ]
        journal.event("Муза", "идёт закрыть дверь")
    elif what == "work":
        route = leave_bathroom() + [("item", "computer")]
        state["grab_block"] = 0.0
        journal.event("Муза", "идёт работать за компьютер")
    elif what == "eat":
        route = leave_bathroom() + [("item", "burger")]
        state["grab_block"] = 0.0
        journal.event("Муза", "проголодалась, идёт к бургеру")
    elif what == "drink":
        route = leave_bathroom() + [("item", "monster")]
        state["grab_block"] = 0.0
        journal.event("Муза", "хочет пить, идёт к энергетику")
    elif what == "out":
        route = leave_bathroom() + out_route()
        journal.event("Муза", "идёт из дома: к лифту и на первый этаж")
    elif what == "cafe":
        route = leave_bathroom() + out_route(cafe=True)
        journal.event("Муза", "идёт в кафе через дорогу")
    elif what == "phone":
        route = leave_bathroom() + [("item", "iphone")]
        state["grab_block"] = 0.0
        state["phone_session"] = True
        journal.event("Муза", "идёт полистать айфон")
    elif what == "window":
        route = leave_bathroom() + [("walk", WINDOW_SPOT), ("look", LOOK_SECONDS)]
        journal.event("Муза", "идёт посмотреть в окно")
    elif what == "stroll":
        route = leave_bathroom() + [("walk", spot) for spot in STROLL] + [("joy", WALK_JOY)]
        journal.event("Муза", "гуляет по квартире")
    elif what == "home":
        route = home_route()
        journal.event("Муза", "идёт домой")
    else:
        route = leave_bathroom() + [("mattress", None), ("rest", None)]
        journal.event("Муза", "идёт к матрасу")

    state["walk_before"] = state["walking"]
    state["walking"] = True
    state["errand"] = what
    state["route"] = route


def out_route(cafe=False):
    """Из квартиры на первый этаж: открыть входную дверь, выйти, закрыть её
    задней лапкой, нажать кнопку лифта, дождаться, зайти, спуститься и
    погулять по холлу."""

    return [
        ("walk", FRONT_INSIDE),
        ("front", "open"),
        ("walk", FRONT_OUTSIDE),
        ("walk", FRONT_KICK),
        ("front", "close"),
        ("walk", LIFT_CALL),
        ("button", 5),
        ("lift", None),
        ("walk", LIFT_DOORWAY),
        ("board", None),
        ("ride", 1),
        ("walk", LIFT_OUT),
    ] + [("walk", spot) for spot in LOBBY] + street_trip(cafe)


def street_trip(cafe=False):
    """Во двор и обратно: толкнуть дверь и выйти (закроется сама), по
    дорожке к воротам, кнопка, за ворота и назад, дверь на себя, зайти."""

    return [
        ("walk", STREET_INSIDE),
        ("street", "open"),
        ("glide", [STREET_DOORWAY_IN, STREET_THROUGH_OUT]),
        ("walk", SPINE_TOP),
        ("walk", SPINE_CORNER),
        ("walk", GATE_INSIDE),
        ("gatebutton", None),
        ("gate", None),
        ("walk", GATE_OUTSIDE),
    ] + (cafe_visit() if cafe else [("walk", spot) for spot in AROUND]) + [
        ("walk", GATE_OUTSIDE),
        ("walk", GATE_INSIDE),
        ("walk", SPINE_CORNER),
        ("walk", SPINE_TOP),
        ("walk", STREET_PULL),
        ("street", "pull"),
        ("glide", [STREET_DOORWAY_OUT, STREET_THROUGH_IN]),
    ]


def cafe_visit():
    """Через дорогу в кафе: по зебре, дверь сама разъезжается, к стойке —
    бариста варит кофе в стаканчик, Муза берёт его лапкой и идёт за
    случайный столик, пьёт — и обратно на свой тротуар."""

    table = room.CAFE_TABLES[np.random.randint(len(room.CAFE_TABLES))]
    beside = room.beside_chair(table)
    return [
        ("walk", CROSS_NEAR),
        ("zebra", None),
        ("walk", CROSS_FAR),
        ("walk", CAFE_OUTSIDE),
        ("cafe", None),
        ("glide", [CAFE_INSIDE]),
        # Внутри ходит по заданному пути: в тесноте между столиками
        # обычная ходьба проскакивала точку и шла кругом вдоль стен.
        ("glide", [CAFE_COUNTER]),
        ("order", None),
        ("glide", [np.array([beside[0], CAFE_AISLE_Y]), beside]),
        ("sit", table),
        ("glide", [np.array([beside[0], CAFE_AISLE_Y]), CAFE_INSIDE]),
        ("cafe", None),
        ("glide", [CAFE_OUTSIDE]),
        ("walk", CROSS_FAR),
        ("zebra", None),
        ("walk", CROSS_NEAR),
    ]


def yard_home():
    """Со двора домой: по дорожке к двери, дверь на себя, зайти."""
    here = data.xpos[thorax_id][:2]
    route = []
    if room.in_street(here) and here[0] < room.SPINE[0]:
        route += [("walk", GATE_INSIDE)] if here[0] < room.FENCE_X + 1.0 else []
        route += [("walk", SPINE_CORNER)]
    route += [("walk", SPINE_TOP)]
    if room.STREET["open"] < 0.8:
        route += [("walk", STREET_PULL), ("street", "pull")]
    return route + [("glide", [STREET_DOORWAY_OUT, STREET_THROUGH_IN])]


def home_route():
    """С первого этажа домой: лифт наверх, дверь на себя, зайти — дверь
    закроется доводчиком."""

    route = []
    if room.in_street(data.xpos[thorax_id]):
        route += yard_home()
    if room.WHERE["floor"] == 1:
        route += [
            ("walk", LIFT_CALL),
            ("button", 1),
            ("lift", None),
            ("walk", LIFT_DOORWAY),
            ("board", None),
            ("ride", 5),
            ("walk", LIFT_OUT),
        ]
    if room.FRONT["open"] < 0.8:
        route += [("walk", FRONT_PULL), ("front", "pull")]
    return route + [("walk", FRONT_INSIDE), ("walk", HOME_SPOT)]


def run_lift(seconds):
    """Лифт сам: едет на вызов, открывает двери, закрывает их, когда никто
    не заходит. Во время поездки им управляет дело «ride»."""

    lift = room.LIFT
    if state["chore"] == "ride":
        return

    step = seconds / LIFT_DOOR_SECONDS

    if lift["called"] is not None:
        goal = lift["called"]
        if abs(lift["z"] - goal) > 1e-6:
            if lift["doors"] > 0.0:
                lift["doors"] = max(0.0, lift["doors"] - step)
            else:
                move = room.LIFT_SPEED * seconds
                lift["z"] = goal if abs(goal - lift["z"]) <= move else lift["z"] + np.sign(goal - lift["z"]) * move
        else:
            lift["doors"] = min(1.0, lift["doors"] + step)
            if lift["doors"] >= 1.0:
                lift["called"] = None
                lift["idle"] = 0.0
        return

    # Ждёт, пока Муза зайдёт; вышла и отошла — двери закрываются.
    coming = any(kind == "ride" for kind, _ in state["route"][:4])
    thorax = data.xpos[thorax_id]
    clear = not room.in_cab(thorax) and thorax[0] < room.LIFT_X - 0.8
    if lift["doors"] > 0.0 and not coming and clear:
        lift["idle"] += seconds
        if lift["idle"] > 1.5:
            lift["doors"] = max(0.0, lift["doors"] - step)
    else:
        lift["idle"] = 0.0


def run_gate(seconds):
    """Ворота отъезжают по кнопке и закрываются сами, когда Муза вернулась
    во двор и за ворота больше не собирается."""

    gate = room.GATE
    here = data.xpos[thorax_id][:2]
    going_out = any(kind == "walk" and where[0] < room.FENCE_X for kind, where in state["route"])
    if gate["called"] and gate["open"] >= 1.0 and here[0] > room.FENCE_X + 2.0 and not going_out:
        gate["called"] = False
    target = 1.0 if gate["called"] else 0.0
    step = seconds / GATE_SECONDS
    gate["open"] = min(target, gate["open"] + step) if target > gate["open"] else max(target, gate["open"] - step)


def run_cafe(seconds):
    """Дверь кафе разъезжается, когда Муза рядом; на зебре машина ждёт;
    внутри — дом над кафе прозрачный."""

    here = data.xpos[thorax_id][:2]
    near = room.WHERE["floor"] == 1 and float(np.linalg.norm(here - np.array([room.CAFE_FACE, room.CAFE_Y]))) < 4.5
    step = seconds / 0.8
    room.CAFE["open"] = min(1.0, room.CAFE["open"] + step) if near else max(0.0, room.CAFE["open"] - step)
    room.CAFE["inside"] = room.in_cafe(here)
    x0 = room.LOT[0]
    # Машина встаёт, когда Муза сходит с тротуара на зебру, и ждёт, пока
    # она не перейдёт.
    room.CROSSING["on"] = bool(
        room.WHERE["floor"] == 1 and room.CAFE_FACE + 2.5 < here[0] < x0 - 3.5 and abs(here[1] - room.CAFE_Y) < room.CROSS_HALF + 1.0
    )


OUTINGS = ("out", "cafe", "home")  # походы из дома


def go_out():
    """Кнопка OUT: выйти из квартиры, спуститься на лифте на первый этаж,
    погулять по холлу и вернуться."""

    if state["errand"] in OUTINGS or state["chore"] is not None or room.WHERE["floor"] != 5:
        return

    drop_item()
    state["reach"] = 0.0
    start_errand("out")


def go_cafe():
    """Кнопка CAFE: вниз на лифте, через двор и ворота, по зебре в кафе за
    кофе — и домой."""

    if state["errand"] in OUTINGS or state["chore"] is not None or room.WHERE["floor"] != 5:
        return

    drop_item()
    state["reach"] = 0.0
    start_errand("cafe")


def leave_bathroom():
    """Если Муза в ванной — сначала выйти: открыть дверь, выйти, закрыть."""

    if not room.in_bathroom(data.xpos[thorax_id]):
        return []

    steps = []
    if room.DOOR["open"] < 0.8:
        steps.append(("door", "open"))
    return steps + [
        ("walk", room.DOOR_OUTSIDE),
        ("walk", room.DOOR_KICK_SPOT),
        ("door", "close"),
    ]


def start_chore(kind, where):
    """Остановилась и что-то делает: села, легла, открывает дверь, жмёт
    кнопку лифта, едет в лифте."""

    if kind in ("door", "front", "button", "street", "gatebutton"):
        state["door_which"] = {"door": "bath", "front": "front", "button": "button", "street": "street", "gatebutton": "gate"}[kind]
        # Дверь из дома толкает, снаружи тянет на себя, закрывает ногой.
        state["door_side"] = 1.0 if where == "pull" else -1.0
        state["door_mode"] = "handle" if where != "close" else "kick"
        kind = "door"
        if state["door_which"] in ("button", "gate"):
            where = "open"
        elif where == "pull":
            where = "open"

    state["chore"] = kind
    state["chore_time"] = 0.0
    state["walking"] = False
    state["pose_base"] = data.ctrl.copy()
    state["pooped"] = False
    state["sit_start"] = data.qpos[0:3].copy()
    state["sit_quat"] = data.qpos[3:7].copy()
    state["sit_from"] = np.array([1.0, 0.0, 0.0, 0.0])  # встаёт лицом в комнату

    if kind == "door":
        door, _, _ = door_parts()
        state["door_to"] = 1.0 if where == "open" else 0.0
        state["door_from"] = 0.0 if state["door_which"] in ("button", "gate") else door["open"]
        place_at_door(state["door_mode"] == "handle")
        journal.event(
            "Муза",
            "жмёт кнопку лифта" if state["door_which"] == "button"
            else "жмёт кнопку ворот" if state["door_which"] == "gate"
            else "открывает дверь лапкой" if where == "open" else "закрывает дверь лапкой",
            state["door_which"],
        )

    if kind in ("board", "glide"):
        # Идёт по заданному пути: в кабину — через середину проёма к её
        # середине; сквозь дверь подъезда — по прямой через проём.
        start = data.qpos[0:2].copy()
        doorway = np.array([room.LIFT_X - 0.6, room.LANE])
        points = [doorway, CAB_SPOT] if kind == "board" else where
        if kind == "glide":
            # Пока открывала дверь, могла уже шагнуть в проём: точки, что
            # остались позади, пропускаем — иначе она пятится и выходит дважды.
            end = np.asarray(points[-1], dtype=float)
            ahead = float(np.linalg.norm(end - start))
            points = [p for p in points[:-1] if np.linalg.norm(end - np.asarray(p, dtype=float)) < ahead] + [points[-1]]
        path = [start] + [np.asarray(point, dtype=float).copy() for point in points]
        lengths = [float(np.linalg.norm(b - a)) for a, b in zip(path, path[1:])]
        state["board_path"] = path
        state["board_lengths"] = lengths
        speed = CAFE_SPEED if kind == "glide" and room.in_cafe(path[-1]) and room.in_cafe(path[0]) else BOARD_SPEED
        state["board_seconds"] = max(sum(lengths) / speed, 0.5)
        state["board_quat"] = data.qpos[3:7].copy()
        journal.event("Муза", "заходит в лифт" if kind == "board" else "проходит в дверь")

    if kind == "sit":
        state["seat_table"] = tuple(where)
        room.CAFE.update(seat=tuple(where), pull=0.0, sip=0.0)
        journal.event("Муза", "отодвигает стул")

    if kind == "ride":
        state["ride_from"] = room.LIFT["z"]
        state["ride_to"] = room.level(where)
        state["ride_storey"] = where
        state["ride_body"] = data.qpos[0:3].copy()
        state["ride_quat"] = data.qpos[3:7].copy()
        state["ride_from_storey"] = room.WHERE["floor"]
        journal.event("Муза", "едет в лифте", f"на {where} этаж")


BUTTON = {"open": 0.0}  # у кнопки лифта «двери» нет: лапка к кнопке, как к ручке


def door_parts():
    """Дверь текущего дела: её состояние, полотно и ручка. Ванная, входная
    дверь (ручка со стороны Музы) или кнопка лифта."""

    which = state.get("door_which", "bath")

    if which == "front":
        side = state.get("door_side", -1.0)
        return room.FRONT, room.front_panel, lambda part=None: room.front_handle(part, side)

    if which == "button":
        spot = room.lift_button(room.WHERE["floor"])
        return BUTTON, room.door_panel, lambda part=None: spot

    if which == "gate":
        return BUTTON, room.door_panel, lambda part=None: room.GATE_BUTTON

    if which == "street":
        side = state.get("door_side", -1.0)
        return room.STREET, room.street_panel, lambda part=None: room.street_handle(part, side)

    return room.DOOR, room.door_panel, room.door_handle


def place_at_door(opening):
    """Куда встать, чтобы лапка правда касалась двери: передняя — ручки,
    задняя — края открытой двери. Лапку берёт ту, что ближе к двери."""

    thorax = data.qpos[0:2].copy()
    heading = data.xmat[thorax_id].reshape(3, 3)[:, 0][:2]
    _, panel_of, handle_of = door_parts()
    hinge, free = panel_of()

    if opening:
        target = handle_of()[:2]
        tip, pose, end = DOOR_TIP, DOOR_LEG, "f_"

        # Дверь слева — левой лапкой, справа — правой.
        towards = target - thorax
        left = heading[0] * towards[1] - heading[1] * towards[0] > 0.0
        reach = np.array([tip[0], tip[1] if left else -tip[1]])

        # Поворот, при котором кончик лапки смотрит на дверь.
        yaw = np.arctan2(towards[1], towards[0]) - np.arctan2(reach[1], reach[0])
    else:
        # Закрывает: стоит к востоку от края открытой двери мордой на север,
        # левая (ближняя к двери) задняя лапка бьёт вбок в торец двери. Дверь
        # закрывается на северо-запад — от неё, ни брюшко, ни лапки не задевает.
        target = free + np.array([0.05, 0.0])
        pose, end, left = KICK_LEG, "h_", True
        reach = KICK_TIP.copy()
        yaw = np.pi / 2
    turn = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    # Куда пойдёт после двери: к концу к этой точке и повернётся, чтобы не
    # закладывать круг по комнате (ноги поворачивают только на ходу).
    ahead = next((where for kind, where in state["route"] if kind == "walk"), None)
    if ahead is None and state["errand"] == "poop":
        ahead = room.TOILET_FRONT
    after = yaw
    if ahead is not None:
        place_xy = target - np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ reach
        way = np.asarray(ahead) - place_xy
        after = float(np.arctan2(way[1], way[0]))
    state["door_after_yaw"] = after

    state["door_leg"] = ("l" if left else "r") + end
    state["door_pose"] = pose
    state["door_place"] = target - turn @ reach
    state["door_yaw"] = yaw
    state["door_start"] = data.qpos[0:3].copy()
    state["door_start_quat"] = data.qpos[3:7].copy()
    state["door_handle"] = target.copy()


def finish_errand():
    # Привыкание — от конца занятия: после долгого похода кафе не тянет сразу снова.
    if "pastime" in state:
        state.setdefault("habits", {})[state.pop("pastime")] = data.time
    state["errand"] = None
    state["route"] = []
    state["walking"] = state["walk_before"]


def errand_target():
    """Куда идёт сейчас: следующая точка маршрута, матрас или компьютер.
    Если по прямой мешает ванная — сначала к её внешнему углу."""

    here = data.xpos[thorax_id][:2].copy()
    goal = here

    for kind, where in state["route"]:
        if kind == "walk":
            goal = np.asarray(where, dtype=float)
        elif kind == "mattress":
            goal = room.mattress_center()
        elif kind == "item":
            goal = next(x for x in sources if x["latin"] == where)["position"][:2].copy()
        break

    # В кабину лифта — через середину проёма: наискосок упиралась в косяк.
    if goal[0] > room.LIFT_X and here[0] < room.LIFT_X + 0.2 and room.in_corridor(here):
        return np.array([room.LIFT_X + 0.4, room.LANE])

    outside = not room.in_bathroom(here)
    doorway = np.array([room.BATH_X, room.TOILET[1]])
    through_door = here[0] > room.BATH_X and (
        abs(here[1] - room.DOOR_OUTSIDE[1]) < 1.2
        or (room.DOOR["open"] > 0.8 and float(np.linalg.norm(here - doorway)) < 3.2)
    )

    if outside and not through_door and not room.in_bathroom(goal) and room.blocked_by_bathroom(here, goal):
        return room.BATH_CORNER

    if outside and room.in_bathroom(goal) and not through_door:
        return room.DOOR_OUTSIDE

    return goal


def sitting_pose():
    """Сидит на унитазе как человек: спиной к бачку, тело почти стоймя,
    брюшко над водой."""

    turn = np.zeros(4)
    # Лицом в комнату (+x), затем выпрямиться: нос вверх.
    mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 1.0, 0.0]), -SIT_PITCH)

    place = np.array(
        [room.TOILET[0] + 0.55, room.TOILET[1], room.SEAT_HEIGHT + 1.4]
    )

    return place, turn


def cafe_times():
    """Когда села, когда допила, когда встала (секунды от начала дела)."""
    sat = PULL_SECONDS + SIT_SECONDS
    drunk = sat + COFFEE_SECONDS
    return sat, drunk, drunk + SIT_SECONDS


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def seated_quat(yaw):
    """Сидит как человек: повёрнута на yaw и выпрямлена, нос вверх."""
    pitch = np.zeros(4)
    mujoco.mju_axisAngle2Quat(pitch, np.array([0.0, 1.0, 0.0]), -SIT_PITCH)
    turned = np.zeros(4)
    mujoco.mju_mulQuat(turned, yaw_quat(yaw), pitch)
    return turned


def upright_quat():
    """Та же сторона, куда смотрит, но стоя на лапках."""

    heading = data.xmat[thorax_id].reshape(3, 3)[:, 0]
    yaw = np.arctan2(heading[1], heading[0])

    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def slerp(first, second, part):
    dot = float(first @ second)
    if dot < 0.0:
        second, dot = -second, -dot
    if dot > 0.9995:
        out = first + (second - first) * part
    else:
        angle = np.arccos(dot)
        out = (np.sin((1 - part) * angle) * first + np.sin(part * angle) * second) / np.sin(angle)
    return out / np.linalg.norm(out)


def chore_pose():
    """Каждый шаг физики: сидит на унитазе, лежит, дрыгается на спине,
    вздрагивает от тока."""

    if state["chore"] == "door":

        moment = state["chore_time"]
        opening = state["door_to"] > state["door_from"]
        door, _, handle_of = door_parts()

        # Доли одной секунды: встать → лапка к двери → толчок → лапку назад.
        seconds = DOOR_SECONDS
        settle = 0.25 * seconds

        step = min(1.0, moment / settle)
        start = state["door_start"]
        place = start[:2] + (state["door_place"] - start[:2]) * step
        yaw = state["door_yaw"]
        facing = slerp(
            state["door_start_quat"],
            np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
            step,
        )

        if opening or state.get("door_mode") == "handle":
            # Лапка на ручке: тянет (или толкает) дверь, сама идёт вместе с
            # ручкой; потом отпускает, и дверь доезжает сама.
            pull = float(np.clip((moment - settle) / (0.4 * seconds), 0.0, 1.0))
            coast = float(np.clip((moment - 0.65 * seconds) / (0.25 * seconds), 0.0, 1.0))
            coast = 0.5 - 0.5 * np.cos(np.pi * coast)
            progress = DOOR_PULL * pull + (1.0 - DOOR_PULL) * coast
            span = state["door_to"] - state["door_from"]
            door["open"] = state["door_from"] + span * progress
            if moment > settle:
                held = state["door_from"] + span * min(progress, DOOR_PULL)
                place = place + (handle_of(held)[:2] - state["door_handle"])
            leg_part = min(step, float(np.clip((seconds - moment) / (0.15 * seconds), 0.0, 1.0)))
        else:
            # Задняя лапка выстреливает назад в край двери — и дверь пошла.
            leg_part = min(
                float(np.clip((moment - settle) / (0.12 * seconds), 0.0, 1.0)),
                float(np.clip((seconds - moment) / (0.15 * seconds), 0.0, 1.0)),
            )
            swing = float(np.clip((moment - 0.37 * seconds) / (0.5 * seconds), 0.0, 1.0))
            swing = 0.5 - 0.5 * np.cos(np.pi * swing)
            door["open"] = state["door_from"] * (1.0 - swing)

        # Лапка отпущена — поворачивается туда, куда пойдёт дальше.
        turn_part = float(np.clip((moment - 0.8 * seconds) / (0.2 * seconds), 0.0, 1.0))
        if turn_part > 0.0:
            after = state["door_after_yaw"]
            facing = slerp(facing, np.array([np.cos(after / 2), 0.0, 0.0, np.sin(after / 2)]), turn_part)

        data.qpos[0:2] = place
        data.qpos[3:7] = facing
        data.qvel[:6] = 0.0

        for index, name in enumerate(dof_names):
            data.ctrl[index] = stand_ctrl[index]
            if name.startswith(state["door_leg"]):
                data.ctrl[index] += state["door_pose"].get(name.split("_", 1)[1], 0.0) * leg_part

        # Присоски отпускает: после ходьбы лапка приклеена к полу и не поднимется.
        data.ctrl[len(dof_names):] = 0.0

    if state["chore"] in ("board", "glide"):

        # Идёт по заданному пути: тело ведём, мордой по ходу.
        path, lengths = state["board_path"], state["board_lengths"]
        along = float(np.clip(state["chore_time"] / state["board_seconds"], 0.0, 1.0)) * sum(lengths)
        index = 0
        while index < len(lengths) - 1 and along > lengths[index]:
            along -= lengths[index]
            index += 1
        piece = path[index + 1] - path[index]
        part = along / lengths[index] if lengths[index] > 1e-6 else 1.0
        data.qpos[0:2] = path[index] + piece * min(part, 1.0)
        yaw = float(np.arctan2(piece[1], piece[0])) if lengths[index] > 1e-6 else 0.0
        turn = float(np.clip(state["chore_time"] / 0.4, 0.0, 1.0))
        data.qpos[3:7] = slerp(state["board_quat"], np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), turn)
        data.qvel[0:2] = 0.0
        data.qvel[3:6] = 0.0

    if state["chore"] == "ride":

        # Стоит в кабине: тело едет вместе с кабиной и полом, пока едет —
        # поворачивается мордой к дверям.
        lift = room.LIFT
        travel = abs(state["ride_to"] - state["ride_from"]) / room.LIFT_SPEED
        turn_part = float(np.clip(state["chore_time"] / (LIFT_DOOR_SECONDS + travel), 0.0, 1.0))
        body = state["ride_body"].copy()

        # Пока двери съезжаются, отходит в середину кабины: у входа брюшко
        # торчало в проём и в створки.
        middle = np.array([(room.CAB[0] + room.CAB[1]) / 2, room.LANE])
        settle = float(np.clip(state["chore_time"] / LIFT_DOOR_SECONDS, 0.0, 1.0))
        body[:2] = body[:2] + (middle - body[:2]) * settle

        data.qpos[0:3] = body
        data.qpos[3:7] = slerp(state["ride_quat"], np.array([0.0, 0.0, 0.0, 1.0]), turn_part)
        data.qvel[:6] = 0.0
        data.ctrl[: len(dof_names)] = stand_ctrl[: len(dof_names)]
        data.ctrl[len(dof_names):] = 0.0

    if state["chore"] == "poop":

        moment = state["chore_time"]
        seat, seated = sitting_pose()

        # Садится, сидит, встаёт — тело ведём сами, лапки свисают.
        down = min(1.0, moment / SIT_SECONDS)
        up = min(1.0, max(0.0, (2 * SIT_SECONDS + POOP_PAUSE - moment) / SIT_SECONDS))
        part = min(down, up)

        start = state["sit_start"] if moment < SIT_SECONDS + POOP_PAUSE else np.append(room.TOILET_FRONT, 1.05)
        facing = state["sit_quat"] if moment < SIT_SECONDS + POOP_PAUSE else state["sit_from"]

        data.qpos[0:3] = start + (seat - start) * part
        data.qpos[3:7] = slerp(facing, seated, part)
        data.qvel[:6] = 0.0

        for index, name in enumerate(dof_names):
            data.ctrl[index] = stand_ctrl[index] + SIT_LEGS.get(name.split("_", 1)[1], 0.0) * part
        data.ctrl[len(dof_names):] = 0.0

    if state["chore"] == "sit":

        moment = state["chore_time"]
        sat, drunk, stood = cafe_times()
        table = state["seat_table"]
        side = room.aisle_side(table)
        chair = np.array([table[0], table[1] + side * (room.CHAIR_TUCK + room.CHAIR_PULL)])
        place = np.array([chair[0], chair[1] - side * 0.6, room.CHAIR_SEAT + 1.4])  # ближе к столу: спинка не сквозь крылья
        seated = seated_quat(np.arctan2(-side, 0.0))  # лицом к столу

        down = float(np.clip((moment - PULL_SECONDS) / SIT_SECONDS, 0.0, 1.0))
        up = float(np.clip((stood - moment) / SIT_SECONDS, 0.0, 1.0))
        part = min(down, up)
        if moment < drunk:
            start, facing = state["sit_start"], state["sit_quat"]
        else:
            start = np.append(room.beside_chair(table), 1.05)  # слезает вбок, как с унитаза
            facing = yaw_quat(np.arctan2(side, 0.0))

        data.qpos[0:3] = start + (place - start) * part
        data.qpos[3:7] = slerp(facing, seated, part)
        data.qvel[:6] = 0.0

        # Правая передняя лапка тянет спинку стула; сидя пьёт как энергетик:
        # обе передние лапки держат стаканчик у рта и кивают на такте.
        pulling = float(np.sin(np.pi * min(1.0, moment / PULL_SECONDS))) if moment < PULL_SECONDS else 0.0
        hold, beat = room.CAFE["sip"], room.CAFE.get("beat", 0.0)
        coxa_nod, tibia_nod = NOD["monster"]
        for index, name in enumerate(dof_names):
            joint = name.split("_", 1)[1]
            data.ctrl[index] = stand_ctrl[index] + SIT_LEGS.get(joint, 0.0) * part
            if name.startswith("rf"):
                data.ctrl[index] += DOOR_LEG.get(joint, 0.0) * pulling
            if name.startswith(("lf", "rf")):
                nod = coxa_nod * beat if joint == "coxa:PITCH" else tibia_nod * beat if joint == "tibia:PITCH" else 0.0
                data.ctrl[index] += (SEAT_HOLD.get(joint, 0.0) + nod) * hold
        data.ctrl[len(dof_names):] = 0.0

    if state["chore"] == "rest":

        moment = state["chore_time"]
        down = min(1.0, moment / LIE_SECONDS)
        up = min(1.0, max(0.0, (REST_SECONDS + 2 * LIE_SECONDS - moment) / LIE_SECONDS))
        part = min(down, up)

        # От позы, в которой застыла, к позе лёжа: стойка плюс сложенные лапки.
        for index, name in enumerate(dof_names):
            lying = stand_ctrl[index] + LIE_POSE.get(name.split("_", 1)[1], 0.0)
            start = state["pose_base"][index]
            data.ctrl[index] = start + (lying - start) * part

        # Присоски на лапках отпускает, иначе лапки не разъедутся и она не ляжет.
        data.ctrl[len(dof_names):] = 0.0

    elif state["chore"] == "flipped":

        moment = state["chore_time"]

        if moment < FLIP_SECONDS:
            # На спине: лапки дрыгаются вразнобой.
            for index in range(len(dof_names)):
                data.ctrl[index] = stand_ctrl[index] + 0.6 * np.sin(
                    data.time * 25.0 + index * 1.7
                )
            data.ctrl[len(dof_names):] = 0.0
        else:
            # Переворачивается: тело плавно встаёт на лапки.
            if "flip_start" not in state:
                state["flip_quat"] = data.qpos[3:7].copy()
                state["flip_height"] = float(data.qpos[2])
                state["flip_xy"] = data.qpos[0:2].copy()
                state["flip_start"] = True

            # На месте: иначе лапки, упираясь в пол, отталкивают тело вбок.
            data.qpos[0:2] = state["flip_xy"]
            part = min(1.0, (moment - FLIP_SECONDS) / RIGHTING_SECONDS)
            if part >= 1.0:
                # Встала: полсекунды придерживаем, пока лапки нащупают пол.
                data.qpos[3:7] = state["flip_upright"]
                data.qvel[:6] = 0.0
                data.ctrl[: len(dof_names)] = stand_ctrl[: len(dof_names)]
                return
            data.qpos[3:7] = slerp(state["flip_quat"], state["flip_upright"], part)
            data.qpos[2] = state["flip_height"] + (1.2 - state["flip_height"]) * part
            data.qvel[:6] = 0.0
            # Лапки сразу в обычную стойку, иначе встанет криво.
            data.ctrl[: len(dof_names)] = stand_ctrl[: len(dof_names)]

    walking = state["walking"] and state["held"] is None and state["chore"] is None

    if state["shock"] > 0.0:
        # Слабый удар: лапки коротко вздрагивают от своей позы и возвращаются.
        if "shock_base" not in state:
            state["shock_base"] = data.ctrl.copy()
        base = data.ctrl if walking else state["shock_base"]
        for index in range(len(dof_names)):
            data.ctrl[index] = base[index] + SHOCK_TWITCH * np.sin(data.time * 60.0 + index * 2.3)

    elif "shock_base" in state:
        if not walking and state["chore"] is None and state["held"] is None:
            data.ctrl[:] = state["shock_base"]
        state.pop("shock_base")


def flip_musa():
    """Кнопка FLIP: перевернуть Музу на спину. Сама встанет через пару секунд."""

    if state["chore"] == "flipped":
        return

    drop_item()
    state["reach"] = 0.0

    if state["chore"] is None:
        state["walk_before"] = state["walking"]

    state["errand"] = None
    state["route"] = []
    state["chore"] = "flipped"
    state["chore_time"] = 0.0
    state["walking"] = False
    state["pose_base"] = data.ctrl.copy()

    # Поворот на 180 градусов вокруг своей продольной оси.
    upright = upright_quat()
    turned = np.zeros(4)
    mujoco.mju_mulQuat(turned, upright, np.array([0.0, 1.0, 0.0, 0.0]))

    data.qpos[3:7] = turned
    data.qpos[2] = 1.3
    data.qvel[:6] = 0.0

    # Когда начнёт вставать — откуда и куда.
    state["flip_quat"] = turned.copy()
    state["flip_upright"] = upright
    state["flip_height"] = 1.3
    state.pop("flip_start", None)

    journal.event("Муза", "перевёрнута на спину")


def shock():
    """Кнопка SHOCK: удар током. Учит так же, как наказание, и дёргает."""

    if state["shock"] > 0.0:
        return  # один удар за раз

    teach(-MEMORY_STEP, "ударена током")
    state["shock"] = SHOCK_SECONDS
    state["shock_seed"] = int(data.time * 1000)


def speed_up():
    """Кнопка TIME: жизнь идёт быстрее — голод, энергия, туалет, работа,
    сон и сидение на унитазе, — и физике достаётся больше времени на кадр,
    так что Муза и ходит быстрее (картинка обновляется реже)."""

    state["warp"] = {1: 2, 2: 4, 4: 8}.get(state["warp"], 1)
    journal.event("время", f"x{state['warp']}")


def reset_musa():
    # Сначала выпускает предмет: иначе поза хвата тут же вернула бы тело
    # на старое место, и сброс был бы не виден.
    drop_item()
    state["reach"] = 0.0
    state.pop("frozen", None)
    state.pop("base_ctrl", None)
    state["chore"] = None
    state["errand"] = None
    state["route"] = []
    state["shock"] = 0.0

    # Снова дома на 5 этаже: лифт наверху, двери закрыты. Была внизу —
    # камера тоже домой.
    if room.WHERE["floor"] != 5 or window.camera.lookat[2] < -1.0:
        camera_to(5)
    room.WHERE["floor"] = 5
    room.LIFT.update({"z": 0.0, "doors": 0.0, "called": None, "idle": 0.0})
    room.FRONT["open"] = 0.0
    room.STREET["open"] = 0.0
    room.GATE.update({"open": 0.0, "called": False})
    room.use_bathroom("home")
    state.pop("after_toilet", None)

    controller.reset(seed=0)
    sim.reset()
    state["walking"] = False
    journal.event("Муза", "поднята заново")


def switch_chain():
    """Переключает панель между светом и едой."""

    state["active"] = (state["active"] + 1) % len(feelings)
    show_active()
    panel.use(active()["map"])
    journal.event("панель", "переключила на", active()["stimulus"])


def change_gain(step):

    feeling = active()
    gains = feeling["gains"]
    chain = feeling["chain"]
    chosen = state["selected"]
    body_sides = ("L", "R") if state["side"] is None else (state["side"],)

    for body_side in body_sides:
        gains[body_side][chosen] = np.clip(
            gains[body_side][chosen] + step, 0.0, GAIN_MAX
        )

    journal.event(
        "синапс",
        f"{chain.stage_from[chosen]}>{chain.stage_to[chosen]} "
        f"{'обе стороны' if state['side'] is None else state['side']}",
        f"x{gains['L'][chosen]:.2f} / x{gains['R'][chosen]:.2f}",
    )


def toggle_nicotine():
    """Кнопка зажигает и гасит сигарету."""

    smoke = cigarette()
    smoke["lit"] = not smoke["lit"]

    journal.event("сигарета", "зажгла" if smoke["lit"] else "погасила")


def forget():
    for spot in sources:
        spot["memory"] = 1.0
        spot["taught"] = 0
    journal.event("память", "стёрта", "все источники как в коннектоме")


def choose_source(index):
    """Кнопка источника: им и двигают стрелки, его и учат."""

    def press():

        state["chosen_source"] = index

        # У сигареты своей цепочки нет, закладку на ней не меняем.
        if sources[index] is not cigarette():
            state["active"] = sources[index]["chain"]
            show_active()
            panel.use(active()["map"])

        journal.event("выбор", "стрелки двигают", sources[index]["name"])

    return press


def open_memory():
    """Кнопка MEMORY: сразу в закладку памяти, без перелистывания."""

    state["active"] = len(feelings) - 1
    show_active()
    panel.use(active()["map"])


actions = {
    "walk": toggle_walk,
    "memory": open_memory,
    "weights": reset_weights,
    "musa": reset_musa,
    "punish": shock,
    "flip": flip_musa,
    "warp": speed_up,
    "out": go_out,
    "cafe": go_cafe,
    "reward": lambda: teach(+MEMORY_STEP, "награждён"),
    "forget": forget,
    "nicotine": toggle_nicotine,
    **{f"source{i}": choose_source(i) for i in range(len(sources))},
}


# ============================================================
# KEYBOARD
# ============================================================

def key_callback(key):

    # SPACE — ходьба
    if key == 32:
        toggle_walk()

    # C — переключить цепочку в панели
    elif key == 67:
        switch_chain()

    # 1..N — выбрать переход
    elif 49 <= key < 49 + len(active()["targets"]):
        state["selected"] = key - 49

    # = и - — сила выбранного перехода
    elif key in (61, 45):
        change_gain(GAIN_STEP if key == 61 else -GAIN_STEP)

    # Q / E — перекос перехода влево / вправо
    elif key in (81, 69):
        step = GAIN_STEP if key == 81 else -GAIN_STEP
        gains = active()["gains"]
        chosen = state["selected"]
        gains["L"][chosen] = np.clip(gains["L"][chosen] + step, 0.0, GAIN_MAX)
        gains["R"][chosen] = np.clip(gains["R"][chosen] - step, 0.0, GAIN_MAX)

    # 0 — вернуть веса как в мозге
    elif key == 48:
        reset_weights()

    # P — наказать запах, O — наградить
    elif key == 80:
        shock()
    elif key == 79:
        teach(+MEMORY_STEP, "награждён")

    # M — забыть всё, что выучила
    elif key == 77:
        forget()

    # T — зажечь или погасить сигарету
    elif key == 84:
        toggle_nicotine()

    # U — выпустить предмет из лапок
    elif key == 85:
        drop_item()

    # N — переключить, чем двигают стрелки
    elif key == 78:
        state["chosen_source"] = (state["chosen_source"] + 1) % len(sources)

    # F — камера следит за Музой
    elif key == 70:
        state["follow"] = not state["follow"]

    # R — выбранный источник перед Музой, G — оба запаха
    elif key in (82, 71):

        thorax = data.xpos[thorax_id]
        heading = data.xmat[thorax_id].reshape(3, 3)[:, 0]
        aside = data.xmat[thorax_id].reshape(3, 3)[:, 1]

        if key == 71:
            for spot, shift in zip(odors, (-4.0, 4.0)):
                spot["position"][:] = thorax + heading * 2.5 + aside * shift
                room.keep_in_room(spot["position"])
                spot["position"][2] = FOOD_A_START[2]
                new_home(spot)
        else:
            target = moved_source()
            target[:] = thorax + heading * 2.5
            room.keep_in_room(target)
            new_home(sources[state["chosen_source"]])

    # X — поднять Музу заново
    elif key == 88:
        reset_musa()

    # B — выйти из дома на лифте и вернуться
    elif key == 66:
        go_out()


def move_source(target, keys, seconds):
    """Источник едет, пока клавиша зажата. Границ нет.

    Возвращает True, если его правда двигали.
    """

    before = target.copy()
    step = LIGHT_SPEED * seconds
    ahead, back, left, right = keys

    moved = False

    if window.key_held(ahead):
        target[0] += step
        moved = True
    if window.key_held(back):
        target[0] -= step
        moved = True
    if window.key_held(left):
        target[1] += step
        moved = True
    if window.key_held(right):
        target[1] -= step
        moved = True

    if moved:
        room.keep_in_room(target, before=before)

    return moved


def moved_source():
    """Стрелки двигают выбранный источник: свет или любой из запахов."""

    return sources[state["chosen_source"]]["position"]


def move_sources(seconds):
    """Стрелки и WASD двигают выбранный источник."""

    chosen = moved_source()

    moved = move_source(chosen, (glfw.KEY_W, glfw.KEY_S, glfw.KEY_A, glfw.KEY_D), seconds)
    moved |= move_source(
        chosen, (glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT), seconds
    )

    if moved:
        new_home(sources[state["chosen_source"]])



def new_home(spot):
    """Ты переставила вещь — это её новое место. Если была у Музы в лапках —
    забираешь."""

    spot_now = spot["position"].copy()

    if spot is held_item():
        drop_item()

    # Место сигареты — пепельница, всегда. Её не переставить.
    if spot is cigarette():
        spot["position"][:] = spot["start"]
        spot["home"] = spot["start"].copy()
        return

    spot["position"][:2] = spot_now[:2]
    spot["position"][2] = spot["floor_z"] + room.surface(spot_now)
    spot["home"] = spot["position"].copy()
    save_layout()


# ============================================================
# 3D СЦЕНА
# ============================================================

def beacon(scene, ngeom, position, color, radius):
    """Шарик и столб до пола, чтобы источник было видно издалека."""

    identity = np.eye(3).flatten()

    mujoco.mjv_initGeom(
        scene.geoms[ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0]),
        pos=position,
        mat=identity,
        rgba=np.array([*color, 1.0]),
    )

    ngeom += 1

    line = scene.geoms[ngeom]

    mujoco.mjv_initGeom(
        line,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.array([3.0, 0.0, 0.0]),
        position,
        identity,
        np.array([*color, 0.7]),
    )
    mujoco.mjv_connector(
        line,
        mujoco.mjtGeom.mjGEOM_LINE,
        3.0,
        np.array([position[0], position[1], 0.0]),
        position,
    )

    return ngeom + 1


def shock_bolts(scene, ngeom):
    """Молнии сверху в Музу, пока бьёт током."""

    if state["shock"] <= 0.0:
        return ngeom

    target = data.xpos[thorax_id]
    identity = np.eye(3).flatten()

    # Форма молнии одна на весь удар — это один разряд, а не мельтешение.
    rng = np.random.default_rng(state.get("shock_seed", 0))

    for bolt in range(1):

        point = target + np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), 4.0])

        for step in range(6):

            part = (step + 1) / 6
            aim = point + (target - point) / (6 - step)
            aim = aim + np.append(rng.uniform(-0.35, 0.35, 2), 0.0) * (1 - part)

            geom = scene.geoms[ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                np.zeros(3), identity, np.array([0.75, 0.85, 1.0, 1.0]),
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.04, point, aim)
            geom.emission = 1.0
            ngeom += 1
            point = aim

    # Пара искр вокруг.
    for spark in range(4):
        mujoco.mjv_initGeom(
            scene.geoms[ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.07, 0, 0]),
            target + rng.uniform(-0.9, 0.9, 3), identity, np.array([0.9, 0.95, 1.0, 1.0]),
        )
        scene.geoms[ngeom].emission = 1.0
        ngeom += 1

    return ngeom


def exhaled_smoke(scene, ngeom):
    """Облачка выдоха: растут, поднимаются и тают."""

    identity = np.eye(3).flatten()
    now = time.perf_counter()
    state["smoke"] = [p for p in state["smoke"] if 0.0 <= now - p[2] < EXHALE_SECONDS]

    for start, drift, born in state["smoke"]:

        age = now - born
        part = age / EXHALE_SECONDS
        where = start + drift * age + np.array([0.0, 0.0, 0.15 * age * age])

        mujoco.mjv_initGeom(
            scene.geoms[ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.12 + 0.45 * part, 0.0, 0.0]), where, identity,
            np.array([0.86, 0.86, 0.89, 0.40 * (1.0 - part)]),
        )
        ngeom += 1

    return ngeom


BARISTA = {"parts": None, "data": None}  # как выглядят части тела Музы; её второй «скелет»


def barista_pose(moment):
    """Где бариста и что делает через moment секунд после заказа (None —
    ждёт у стойки): место, куда смотрит, насколько встала на задние лапки,
    идёт ли, тянет ли передние лапки, где стаканчик."""

    pose = {"xy": BARISTA_SERVE, "yaw": 0.0, "rear": 0.0, "gait": False, "reach": 0.0, "cup": None, "fill": 0.0, "serve": 0.0}
    if moment is None:
        return pose
    right, left = -np.pi / 2, np.pi / 2  # к кофемашине (на юг) и обратно
    plan = {
        0: {"xy": BARISTA_SERVE, "yaw": (0.0, right)},
        1: {"xy": (BARISTA_SERVE, BARISTA_MACHINE), "yaw": right, "gait": True},
        2: {"xy": BARISTA_MACHINE, "yaw": (right, 0.0)},
        3: {"xy": BARISTA_MACHINE, "rear": (0.0, 1.0), "reach": (0.0, ON_MACHINE)},
        4: {"xy": BARISTA_MACHINE, "rear": 1.0, "reach": ON_MACHINE, "cup": "brew", "fill": (0.0, 1.0)},
        5: {"xy": BARISTA_MACHINE, "rear": (1.0, 0.0), "reach": (ON_MACHINE, 0.3), "cup": "barista"},
        6: {"xy": BARISTA_MACHINE, "yaw": (0.0, left), "reach": 0.3, "cup": "barista"},
        7: {"xy": (BARISTA_MACHINE, BARISTA_SERVE), "yaw": left, "gait": True, "reach": 0.3, "cup": "barista"},
        8: {"xy": BARISTA_SERVE, "yaw": (left, 0.0), "reach": 0.3, "cup": "barista"},
        9: {"xy": BARISTA_SERVE, "reach": (0.3, 1.0), "cup": "barista", "serve": (0.0, 1.0)},
    }
    start = 0.0
    for number, (_, seconds) in enumerate(BARISTA_STEPS):
        if moment < start + seconds:
            part = (moment - start) / seconds
            for key, value in plan[number].items():
                if isinstance(value, tuple) and len(value) == 2 and not isinstance(value[0], str):
                    first, last = (np.asarray(v, dtype=float) for v in value)
                    pose[key] = first + (last - first) * part
                else:
                    pose[key] = value
            return pose
        start += seconds
    pose["cup"] = "counter"
    return pose


def barista(scene, ngeom):
    """Бариста за стойкой кафе — вторая Муза со своим скелетом: те же суставы
    модели, свои углы; позу считает кинематика MuJoCo во втором MjData.
    Когда наша Муза подходит к стойке, бариста идёт к кофемашине, встаёт на
    задние лапки, передние — на машину, потом несёт стаканчик к стойке."""

    root = model.body_rootid[thorax_id]
    if BARISTA["parts"] is None:
        # Как выглядят части тела (сетки, материалы, цвета) — из сцены
        # первого кадра; где они — будем брать из её скелета.
        parts = []
        for index in range(ngeom):
            geom = scene.geoms[index]
            if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or model.geom_bodyid[geom.objid] == 0:
                continue
            if model.body_rootid[model.geom_bodyid[geom.objid]] != root:
                continue
            parts.append((geom.objid, geom.type, geom.dataid, geom.matid, geom.category, np.array(geom.size),
                          np.array(geom.rgba), geom.emission, geom.specular, geom.shininess))
        BARISTA["parts"] = parts
        BARISTA["data"] = mujoco.MjData(model)
        BARISTA["root"] = model.jnt_qposadr[model.body_jntadr[root]]
        BARISTA["joints"] = [model.jnt_qposadr[model.actuator_trnid[index, 0]] for index in range(len(dof_names))]
        return ngeom

    pose = barista_pose(room.CAFE["order"])
    skeleton = BARISTA["data"]
    body = BARISTA["root"]

    # Тело: место, поворот, на задних лапках — выпрямлена и выше.
    pitch = np.zeros(4)
    mujoco.mju_axisAngle2Quat(pitch, np.array([0.0, 1.0, 0.0]), -BARISTA_REAR * float(pose["rear"]))
    turned = np.zeros(4)
    mujoco.mju_mulQuat(turned, yaw_quat(float(pose["yaw"])), pitch)
    x, y = pose["xy"]
    skeleton.qpos[body:body + 3] = [x, y, room.GROUND + room.BARISTA_STEP + 1.05 + 0.7 * float(pose["rear"])]
    skeleton.qpos[body + 3:body + 7] = turned

    # Лапки: стойка; идёт — перебирает треногами; передние тянутся вперёд
    # (на кофемашину, со стаканчиком к стойке).
    reach = float(pose["reach"])
    for index, name in enumerate(dof_names):
        angle = stand_ctrl[index]
        joint = name.split("_", 1)[1]
        if pose["gait"] and joint == "coxa:PITCH":
            group = 0.0 if name[:2] in ("lf", "rm", "lh") else np.pi
            angle += 0.35 * np.sin(data.time * 40.0 + group)
        if name.startswith(("lf", "rf")):
            angle += 1.6 * DOOR_LEG.get(joint, 0.0) * reach
        skeleton.qpos[BARISTA["joints"][index]] = angle
    mujoco.mj_kinematics(model, skeleton)

    for objid, kind, dataid, matid, category, size, rgba, emission, specular, shininess in BARISTA["parts"]:
        if ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[ngeom]
        mujoco.mjv_initGeom(geom, kind, size, skeleton.geom_xpos[objid], skeleton.geom_xmat[objid], rgba)
        geom.dataid = dataid
        geom.matid = matid
        geom.category = category
        geom.emission = emission
        geom.specular = specular
        geom.shininess = shininess
        ngeom += 1

    def put(kind, size, pos, color, rotation=np.eye(3), shiny=0.3):
        nonlocal ngeom
        geom = scene.geoms[ngeom]
        mujoco.mjv_initGeom(geom, kind, np.array(size, dtype=float), np.array(pos, dtype=float),
                            np.asarray(rotation, dtype=float).flatten(), np.array(color, dtype=float))
        geom.specular = shiny
        ngeom += 1

    # Стаканчик в лапках баристы, пока несёт, — всегда над столешницей;
    # ставит его на стойку сверху, по дуге.
    if pose["cup"] == "barista" and ngeom + 4 < scene.maxgeom:
        paws = (skeleton.xpos[hand_ids[0]] + skeleton.xpos[hand_ids[1]]) / 2.0
        surface = room.GROUND + room.COUNTER_TOP + 0.1
        carried = np.array([paws[0], paws[1], max(paws[2] - 0.05, surface + 0.35)])
        placed = np.array([room.COUNTER_SERVE[0], room.COUNTER_SERVE[1], surface])
        part = float(pose["serve"])
        spot = carried + (placed - carried) * part + np.array([0.0, 0.0, 0.3 * np.sin(np.pi * part)])
        room.takeaway(put, tuple(spot), 1.0)

    # Стаканчик в лапках Музы: у кончиков передних лапок, на её высоте; за
    # столиком — стоит на столе и поднимается ко рту, когда она отпивает.
    if room.CAFE["cup"] in ("held", "sip") and ngeom + 4 < scene.maxgeom:
        if room.CAFE["cup"] == "held":
            paws = hands_spot()
            spot = np.array([paws[0], paws[1], paws[2] + shown_z() - 0.05])
            tilt = np.eye(3)
        else:
            # Со стола — в обе лапки у рта; на глотке запрокидывает к себе.
            table = np.array(room.cup_on_table(room.CAFE["seat"])) + np.array([0.0, 0.0, room.GROUND])
            paws = hands_spot() + np.array([0.0, 0.0, shown_z() - 0.2])
            spot = table + (paws - table) * room.CAFE["sip"]
            toward = np.array([0.0, room.aisle_side(room.CAFE["seat"]), 0.0])  # от стола к ней
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, np.cross(np.array([0.0, 0.0, 1.0]), toward), CAN_POUR * 0.8 * room.CAFE.get("beat", 0.0) * room.CAFE["sip"])
            tilt = np.zeros(9)
            mujoco.mju_quat2Mat(tilt, quat)
            tilt = tilt.reshape(3, 3)
        room.takeaway(put, tuple(spot), 1.0, rotation=tilt)
    return ngeom


def fly_geoms(scene, ngeom):

    identity = np.eye(3).flatten()

    # Тело Музы — на этаже, где она (в лифте — вместе с кабиной).
    dz = shown_z()
    if dz:
        lower_musa(scene, 0, ngeom, dz)

    # Виден только этаж, где Муза: на первом — ни квартиры, ни вещей.
    storey = view_floor()
    ngeom = room.draw(scene, ngeom, state["poops"], data.time, home=storey == 5)
    ngeom = building.draw(scene, ngeom, storey)
    if storey is None:
        # Кабина едет, камера снаружи: Муза в закрытой кабине не видна —
        # иначе она висит в холле за стеклянной дверью подъезда.
        for index in range(ngeom):
            geom = scene.geoms[index]
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and model.geom_bodyid[geom.objid] != 0:
                geom.rgba[3] = 0.0
    ngeom = barista(scene, ngeom)
    ngeom = exhaled_smoke(scene, ngeom)
    bolts = ngeom
    ngeom = shock_bolts(scene, ngeom)
    if dz:
        lower_musa(scene, bolts, ngeom, dz)

    for index, spot in enumerate(sources):

        if storey != 5:
            break

        # Выученный источник тускнеет: вес KC -> MBON у него упал.
        shade = 0.35 + 0.65 * float(np.clip(spot["memory"], 0.0, 1.0))
        radius = 0.30 if index == 0 else 0.35

        # Вокруг выбранного — светлое кольцо: им сейчас двигают стрелки.
        if index == state["chosen_source"]:

            mujoco.mjv_initGeom(
                scene.geoms[ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([radius * 1.7, 0.0, 0.0]),
                pos=spot["position"],
                mat=identity,
                rgba=np.array([1.0, 1.0, 1.0, 0.25]),
            )

            ngeom += 1

        if spot is cigarette():

            heading = data.xmat[thorax_id].reshape(3, 3)[:, 0]

            # В лапках — поменьше и горизонтально, даже когда она стоит
            # на задних лапках у окна.
            flat = np.array([heading[0], heading[1], 0.0])
            flat = flat / max(np.linalg.norm(flat), 1e-9)

            # У окна — как зажата в правой лапке: фильтром к морде.
            if spot is held_item() and "pose_to" in state and "hold_dir" in spot:
                flat = -spot["hold_dir"]

            ngeom = props.cigarette(
                scene,
                ngeom,
                spot["position"],
                spot["lit"],
                data.time,
                direction=-flat if spot is held_item() else None,
                scale=HELD_CIGARETTE if spot is held_item() else 1.0,
                smoke=spot is not held_item(),
            )

        elif index == 0:
            # Свет — это лампочка. Без луча до пола: он торчал как палка.
            ngeom = props.bulb(scene, ngeom, spot["position"], True, shade)

        elif index == 1:
            # Каждый укус делает бургер чуть меньше.
            eaten = 1.45 - 0.6 * spot["eaten"] / PORTION["burger"]
            ngeom = props.burger(scene, ngeom, spot["position"], shade, eaten)

        elif index == 2:
            # Когда пьёт — банка наклоняется ко рту.
            up = spot.get("up") if spot is held_item() else None
            ngeom = props.monster(scene, ngeom, spot["position"], shade, up=up)

        elif spot["latin"] in ("iphone", "computer"):
            # Стоят как поставили и за Музой не поворачиваются. Чуть над
            # полом: иначе тонкий корпус ноутбука тонет в ковре.
            facing = spot["facing"]
            lifted = spot["position"] + np.array([0.0, 0.0, 0.04])

            if spot["latin"] == "computer":
                # Когда печатает — на экране терминал.
                typed = spot.get("typed", 0.0) if spot is held_item() else None
                ngeom = props.laptop(scene, ngeom, lifted, facing, data.time, shade, typed=typed)
            elif spot is held_item():
                # В лапках — открыт инстаграм, лента прокручивается.
                heading = data.xmat[thorax_id].reshape(3, 3)[:, 0]
                ngeom = props.iphone(
                    scene, ngeom, lifted, -heading, data.time, shade,
                    scale=PHONE_SCALE, feed=spot.get("scroll", 0.0),
                )
            else:
                ngeom = props.iphone(scene, ngeom, lifted, facing, data.time, shade)

        else:
            ngeom = beacon(scene, ngeom, spot["position"], spot["color"] * shade, radius)

    return ngeom


# ============================================================
# START
# ============================================================

print()
print("🪰 МУЗА — комната, потребности и мозг из коннектома")
print("=" * 62)
print("SPACE        ходьба вкл/выкл")
print("стрелки      переставляют выбранную вещь (это её новое место)")
print("W A S D      то же самое")
print("C            переключить закладку: свет / запах / память")
print("1..N         выбрать переход")
print("= и -        сила выбранного перехода")
print("Q / E        перекос перехода влево / вправо")
print("0            вернуть веса как в мозге")
print("N            перебрать, что двигают стрелки (или кнопки вещей в панели)")
print("T            погасить сигарету в лапках")
print("U            отпустить вещь из лапок — вернётся на своё место")
print("P / O        ток / награда: тому, что в лапках, к чему подходит, иначе выбранному")
print("M            забыть всё выученное")
print("R            выбранную вещь перед Музой, G — бургер и энергетик")
print("F            камера следит за Музой")
print("X            поднять Музу заново (или кнопка RESET MUSA)")
print("ESC          выйти")
print("мышь         слева камера Музы, по мозгу справа — крутить и выбирать")
print("=" * 62)

floor_names = {"sense": "ЧУВСТВО", "link": "СВЯЗЬ", "motor": "ДВИЖЕНИЕ"}

for feeling in feelings:

    chain = feeling["chain"]
    print(f"\n########## {feeling['stimulus'].upper()}")

    shown = None

    for i, (first, second) in enumerate(zip(chain.stage_from, chain.stage_to)):

        if chain.floors[i] != shown:
            shown = chain.floors[i]
            print(f"\n--- {floor_names.get(shown, shown)}")

        print(
            f"  {i + 1}. {first:6} -> {second:6}"
            f"  L {chain.weights['L'][i]:7.0f}  R {chain.weights['R'][i]:7.0f}"
        )
        medium = chain.transmitters[i]
        mark = "возбуждает" if chain.signs[i] > 0 else "тормозит"
        print(f"     медиатор {medium} — {mark}")
        print(f"     {chain.details[i]}")

print()


window = Window(model, data, "Musa - light and food", key_callback)
music = Music()
music.start()
# Камера свободная: при запуске видна вся комната, дальше мышью как хочешь.
camera_to(5)

# Стены и матрас квартиры — геометрия модели: на другом этаже прячем их в
# выключенную группу.
HIDDEN = 5
window.option.geomgroup[HIDDEN] = 0
APARTMENT = [model.geom(f"room_wall_{name}").id for name in room.WALLS] + [model.geom("room_mattress").id]

walls = room.wall_ids(model)
panel = Panel(active()["map"], state, actions, model, data, gain_max=GAIN_MAX)

window.panel = panel

wall_mark = time.perf_counter()
sim_mark = data.time
speed = 0.0
frame_mark = time.perf_counter()

while window.is_running():

    # TIME xN: физике достаётся в N раз больше времени на кадр — Муза и
    # правда движется быстрее, а картинка обновляется реже.
    frame_deadline = time.perf_counter() + state["warp"] / FPS

    # Мозг решает, куда поворачивать, раз в кадр: раньше он считался на
    # каждом шаге физики, 10 000 раз в секунду, и съедал половину времени.
    signal, steering = brain_step()
    circles = obstacles()
    walls_now = room.wall_table()
    tick = 0

    while time.perf_counter() < frame_deadline:

        walking = state["walking"] and state["held"] is None and state["chore"] is None
        boarding = state["chore"] in ("board", "glide")

        if (walking or boarding) and tick % CONTROL_EVERY == 0:
            obs = HybridControllerObservation.from_sim(sim, fly.name)
            # В кабину идёт прямо: путь ведём сами, ноги просто шагают.
            step_signal = np.ones(2) if boarding else signal
            apply_locomotion_action(sim, fly.name, controller.step(step_signal, obs))

        hold_pose()
        chore_pose()

        # У двери, на унитазе, у окна и за компьютером телом управляем
        # сами — упор не нужен.
        if state["chore"] not in ("door", "poop", "sit", "ride", "board", "glide") and "pose_to" not in state:
            room.hold_inside(data, circles, BODY_PARTS, walls_now)

        sim.step()
        tick += 1

    now = time.perf_counter()

    move_sources(now - frame_mark)
    take_item(now - frame_mark)
    crave(now - frame_mark)
    life(max(0.0, data.time - state.get("life_mark", data.time)) * state["warp"])
    state["life_mark"] = data.time
    frame_mark = now

    state["teach_name"] = teach_target()["latin"]
    state["doing"] = {
        "rest": "resting",
        "poop": "on the toilet",
        None: "scrolling the feed" if state.get("was_scrolling") else (
            "working at the computer" if held_item() is not None and held_item()["latin"] == "computer" else None
        ) or (
            f"drawn to {state['focus']['latin']}"
            if state["walking"] and held_item() is None and state.get("focus") is not None else None
        ),
        "flipped": "on her back",
        "door": "calling the lift" if state.get("door_which") == "button" else "opening the gate" if state.get("door_which") == "gate" else "opening the door" if state.get("door_to") else "closing the door",
        "ride": "in the lift",
        "board": "stepping into the lift",
        "glide": "going through the door",
    }.get(state["chore"]) or (
        {"poop": "going to the toilet", "rest": "going to bed", "leave": "leaving the bathroom", "work": "going to work", "eat": "hungry, going to eat", "drink": "thirsty, going to drink", "close": "closing the door", "out": "going out", "cafe": "going to the cafe", "home": "going home", "phone": "going to scroll", "window": "looking out of the window", "stroll": "strolling around"}[state["errand"]]
        if state["errand"] else ""
    )
    lift_camera()
    room.cut_away(model, walls, window.camera)
    model.geom_group[APARTMENT] = 0 if view_floor() == 5 else HIDDEN

    if now - wall_mark > 0.5:
        speed = (data.time - sim_mark) / (now - wall_mark)
        wall_mark, sim_mark = now, data.time

    fly_rect, panel_rect = window.panes()

    if state["follow"]:
        window.look_at(data.xpos[thorax_id] + np.array([0.0, 0.0, shown_z()]))

    window.draw(fly_rect, window.camera, window.option, fly_geoms)

    panel.draw(window, panel_rect)
    music.set_volume(state.get("volume", music.volume))
    music.choose(state.get("track"))

    window.finish()

    journal.sample(
        data.xpos[thorax_id],
        {spot["name"]: spot["position"] for spot in sources},
        {spot["name"]: spot["turn"] for spot in sources},
        signal,
        state["walking"],
    )

    print(
        f"\rWALK={'ON ' if state['walking'] else 'OFF'} "
        + " ".join(
            f"| {spot['name']} {spot['turn']:+.2f} x{spot['memory']:.2f}"
            for spot in sources
        )
        + f" | поворот {steering:+.2f} "
        f"| ноги {signal[0]:.2f}/{signal[1]:.2f} "
        f"| панель {state['chain_name']} "
        f"| {speed:.2f}x",
        end="",
        flush=True,
    )

window.close()
music.stop()

print()
print()
print("=" * 62)
print("ЧТО БЫЛО В ЭТОМ ЗАПУСКЕ")
print("=" * 62)

for line in journal.summary(feelings):
    print(line)

print()
print("ЧТО МУЗА ЗАПОМНИЛА")
print(f"{'источник':10} {'вес KC>MBON':>12} {'тяга':>7} {'уроков':>7}")

for spot in sources:
    pull = 2.0 * spot["memory"] - 1.0
    print(f"{spot['name']:10} {spot['memory']:>12.2f} {pull:>+7.2f} {spot['taught']:>7}")

print()
print("ГДЕ СТОЯТ ВЕЩИ (их места)")
for spot in sources:
    print(
        f"{spot['name']:10} x{spot['home'][0]:+6.2f} y{spot['home'][1]:+6.2f}"
        f"  экран/поворот ({spot['home_facing'][0]:+.2f}, {spot['home_facing'][1]:+.2f})"
    )
    journal.event("места", spot["name"], f"{spot['home'][0]:.3f} {spot['home'][1]:.3f} "
                  f"{spot['home_facing'][0]:.3f} {spot['home_facing'][1]:.3f}")

for path in journal.save():
    print("таблица:", path)

print()
