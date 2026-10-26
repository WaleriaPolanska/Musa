"""Фоновая музыка: неспешная баллада в духе 80-х, загадочная, как из сна,
с настроением отпуска и тропиков (мелодия своя). Звуки ре минора, светлые
мажорные аккорды, 80 ударов в минуту, мягкий ровный грув: бархатный голос мальчика («оу») с
хорусом и эхом, тихое пианино, медленно тающие аккорды, электропиано, бас,
стил-пэн, пульс низкого бархатного саксофона, мягкий ритм, длинная реверберация. Сверху — старая
кассета: звук чуть плавает, мягко сжат, тихо шипит. Генерируется при
запуске; играет player.py, пока Муза стоит."""


import numpy as np

RATE = 22050
TEMPO = 80
BEAT = 60.0 / TEMPO
BARS = 16

# Аккорды по такту (полутоны от ре): Fmaj9, B♭maj7♯11 (загадочный ми
# сверху), Gm9, C9sus, Fmaj9, Dm9, B♭maj7♯11, C9sus — дважды. Звуки те же, что в ре миноре, но круг
# начинается с мажора — светло, как в отпуске; до ведёт обратно к фа.
CHORDS = [
    (3, 7, 10, 14, 17), (-4, 0, 3, 7, 14), (5, 8, 12, 15, 19), (-2, 3, 5, 8, 12),
    (3, 7, 10, 14, 17), (0, 3, 7, 10, 14), (-4, 0, 3, 7, 14), (-2, 3, 5, 8, 12),
]
D = 146.83  # ре малой октавы
SCALE = [0, 2, 3, 5, 7, 8, 10]  # ре минор: те же звуки, что в аккордах


def hz(semitone):
    return D * 2.0 ** (semitone / 12.0)


def times(seconds):
    return np.arange(int(RATE * seconds)) / RATE


def envelope(t, attack, release, total):
    rise = np.minimum(1.0, t / attack)
    fall = np.clip((total - t) / release, 0.0, 1.0)
    return rise * fall


# Гласная «оу» детского голоса: форманты (Гц) и их ширина.
VOWEL = ((480.0, 120.0, 1.0), (950.0, 150.0, 0.30), (2900.0, 250.0, 0.03))


def voice(freq, seconds, rng):
    """Бархатный голос мальчика: гласная «оу» из гармоник, у которых
    громкость задают форманты; мягкая атака, высота ровная, немного
    дыхания, второй голос чуть в стороне — теплее."""
    t = times(seconds)  # затихает в своей длине: ноты не наслаиваются
    out = np.zeros(t.size)
    for detune, gain in ((1.0, 1.0), (1.004, 0.5)):
        phase = 2 * np.pi * freq * detune * t  # высота ровная, без подъездов и вибрато
        for k in range(1, int(1800 / freq) + 1):  # без верхних обертонов — не режет
            f = k * freq
            shape = sum(g / np.sqrt(1.0 + ((f - F) / (B / 2)) ** 2) for F, B, g in VOWEL)
            out += gain * np.sin(k * phase) * shape / k ** 1.4
    breath = rng.normal(0, 1, t.size)
    breath = np.convolve(breath, np.ones(12) / 12, mode="same") * 0.05  # мягкий шёпот
    return (out / 1.5 + breath * 0.5) * envelope(t, 0.18, 0.25, t[-1])


def pad(freqs, seconds):
    """Тёплые аккорды, как во сне: медленно нарастают и тают, каждый звук —
    два слегка расстроенных голоса, сверху тихое мерцание на октаву выше."""
    t = times(seconds + 1.0)
    out = np.zeros(t.size)
    for f in freqs:
        for detune in (0.997, 1.003):
            phase = 2 * np.pi * f * detune * t
            out += sum(np.sin(k * phase) / k ** 1.5 for k in range(1, 5))
        shimmer = 1.0 + 0.5 * np.sin(2 * np.pi * 0.3 * t + f)  # медленно дышит
        out += 0.25 * np.sin(4 * np.pi * f * t) * shimmer
    return out * envelope(t, 1.2, 1.0, t[-1]) / len(freqs)


def piano(freq, seconds):
    """Электропиано: чистый тон с колокольчиком в атаке."""
    t = times(seconds)
    bell = np.sin(2 * np.pi * freq * 7.0 * t) * np.exp(-t * 25) * 0.05
    return (np.sin(2 * np.pi * freq * t) + bell) * np.exp(-t * 2.5) * envelope(t, 0.02, 0.1, t[-1])


def low_sax(freq, seconds, rng):
    """Низкий бархатный саксофон: только нижние обертоны с плавным спадом,
    мягкая атака, немного дыхания — без высоких нот и без визга."""
    t = times(seconds)
    phase = 2 * np.pi * freq * t
    tone = sum(np.sin(k * phase) * 0.8 ** k / k ** 0.5 for k in range(1, int(1400 / freq) + 1))
    breath = np.convolve(rng.normal(0, 1, t.size), np.ones(10) / 10, mode="same") * 0.08
    return (tone + breath) * envelope(t, 0.035, 0.08, t[-1])


def soft_piano(freq, seconds):
    """Мягкое пианино: тёплый тон без звонкой атаки, долго тает."""
    t = times(seconds)
    tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t) * np.exp(-t * 4)
    return tone * np.exp(-t * 1.6) * envelope(t, 0.015, 0.3, t[-1])


def bass(freq, seconds):
    t = times(seconds)
    return (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t)) * envelope(t, 0.02, 0.15, t[-1])


def steel(freq, seconds):
    """Стил-пэн: звонкий, но мягкий удар, обертоны чуть не в лад, долго тает."""
    t = times(seconds)
    tone = (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2.0 * freq * t) * np.exp(-t * 3)
            + 0.25 * np.sin(2 * np.pi * 3.01 * freq * t) * np.exp(-t * 6))
    return tone * np.exp(-t * 2.2) * envelope(t, 0.02, 0.2, t[-1])


def bongo(freq):
    t = times(0.25)
    return np.sin(2 * np.pi * freq * (1.0 + 0.3 * np.exp(-t * 40)) * t) * np.exp(-t * 18)


def kick():
    t = times(0.35)
    return np.sin(2 * np.pi * (45 + 60 * np.exp(-t * 30)) * t) * np.exp(-t * 9)


def snare(rng):
    t = times(0.3)
    noise = rng.normal(0, 1, t.size)
    noise = noise - np.concatenate(([0.0], noise[:-1])) * 0.6  # светлее
    noise = np.convolve(noise, np.ones(8) / 8, mode="same") * 2.0  # щётка, не хлопок
    return (noise * 0.5 + 0.3 * np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 10) * envelope(t, 0.012, 0.05, t[-1])


def hat(rng, open_part=0.0):
    t = times(0.08 + 0.2 * open_part)
    noise = rng.normal(0, 1, t.size)
    noise = noise - np.concatenate(([0.0], noise[:-1]))
    return noise * np.exp(-t * (70 - 50 * open_part))


# Ритм фразы на два такта (начало и длина в долях): только доли и
# полудоли, никаких быстрых пробежек.
RHYTHMS = (
    ((0.0, 1.5), (1.5, 0.5), (2.0, 2.0), (4.0, 1.0), (5.0, 1.5)),
    ((0.5, 1.0), (1.5, 1.0), (2.5, 1.5), (4.0, 2.5)),
    ((0.0, 2.0), (2.0, 1.0), (3.0, 1.0), (4.0, 2.5)),
)
LOW, HIGH = 7, 12  # ступени мелодии: в пределах одной октавы


def melody(rng):
    """Своя мелодия: фразы по два такта, голос идёт плавно — на соседнюю
    ступень, изредка через одну; на сильных долях — звук аккорда.
    Последняя нота тянется до самого конца круга, первая фраза вступает
    через полдоли — стык без паузы и без наслоений."""
    notes = []  # (начало в долях, длина в долях, полутон)
    degree = 9
    for phrase in range(BARS // 2):
        start = phrase * 8.0
        if phrase in (3, BARS // 2 - 1):
            # Долгая нота аккорда: в середине — вдох, в конце — до стыка.
            degree = chord_degree(degree, start)
            span = 7.0 if phrase == BARS // 2 - 1 else 5.0
            notes.append((start + 1.0, span, degree_to_semitone(degree)))
            continue
        for offset, span in RHYTHMS[(phrase + 1) % len(RHYTHMS)]:
            degree = int(np.clip(degree + rng.choice([-1, -1, 0, 1, 1, 2, -2]), LOW, HIGH))
            if offset in (0.0, 4.0):
                degree = chord_degree(degree, start + offset)
            notes.append((start + offset, span, degree_to_semitone(degree)))
    return notes


def chord_degree(degree, beat):
    """Ближайшая к degree ступень, которая звучит в аккорде этого такта."""
    chord = CHORDS[int(beat // 4) % len(CHORDS)]
    tones = {c % 12 for c in chord}
    options = [d for d in range(LOW, HIGH + 1) if degree_to_semitone(d) % 12 in tones]
    return min(options, key=lambda d: abs(d - degree)) if options else degree


def degree_to_semitone(degree):
    return 12 * (degree // 7) + SCALE[degree % 7]


def wobble(sound, parts):
    """Задержка, которая плавает: parts — (частота Гц, размах в отсчётах).
    Частоты подгоняются так, чтобы за круг было целое число колебаний —
    стык без шва. Звук читается по кругу."""
    n = np.arange(sound.size)
    shift = np.zeros(sound.size)
    for freq, depth in parts:
        cycles = max(1, round(freq * sound.size / RATE))
        shift += depth * np.sin(2 * np.pi * cycles * n / sound.size)
    return np.interp((n - shift) % sound.size, n, sound, period=sound.size)


def compose(seed=11):
    rng = np.random.default_rng(seed)
    length = int(RATE * BEAT * 4 * BARS)
    song = np.zeros(length + RATE * 4)
    airy = np.zeros(song.size)  # голос и стил-пэн: к ним ещё эхо

    def add(sound, beat, gain, into=None):
        into = song if into is None else into
        i = int(beat * BEAT * RATE)
        end = min(into.size, i + sound.size)
        into[i:end] += gain * sound[: end - i]

    for bar in range(BARS):
        chord = CHORDS[bar % len(CHORDS)]
        first = bar * 4.0
        add(pad([hz(c) for c in chord], 4 * BEAT), first, 0.16)
        # Электропиано: короткие аккорды — на 1 и на слабых долях.
        for place, span, gain in ((0.0, 1.0, 0.06), (1.5, 0.5, 0.04), (2.5, 0.75, 0.05)):
            for c in chord[1:4]:
                add(piano(hz(c + 12), span * BEAT), first + place, gain)
        # Бас пружинит: корень с подхватом, квинта, корень, октава и ход к
        # следующему аккорду.
        root = hz(chord[0] - 12)
        for place, span, ratio, gain in (
            (0.0, 0.7, 1.0, 0.30), (0.75, 0.25, 1.0, 0.18), (1.5, 0.5, 1.5, 0.22),
            (2.5, 0.7, 1.0, 0.26), (3.25, 0.25, 2.0, 0.16), (3.5, 0.5, 2.0 ** (2 / 12), 0.18),
        ):
            add(bass(root * ratio, span * BEAT), first + place, gain)
        # Ритм мягкий, но ровный: бочка на 1, «и» второй и «и» третьей,
        # щётка на 2 и 4, тихий шейкер шестнадцатыми с упором на восьмые.
        add(kick(), first, 0.22)
        add(kick(), first + 1.5, 0.10)
        add(kick(), first + 2.5, 0.14)
        add(snare(rng), first + 1, 0.09)
        add(snare(rng), first + 3, 0.09)
        # Пульс низкого саксофона: короткие ноты аккорда на слабых долях,
        # одинаково в каждом такте — держит грув.
        low = ((chord[0] + 5) % 12) - 5  # корень аккорда около ре малой октавы
        for place, span, step in ((0.5, 0.4, 0), (1.5, 0.4, 7), (2.5, 0.9, 12), (3.5, 0.4, 7)):
            add(low_sax(hz(low + step), span * BEAT, rng), first + place, 0.12)
        for step in range(16):
            add(hat(rng), first + step / 4, 0.022 if step % 2 == 0 else 0.012)
        # Тропики: два тихих удара бонго и стил-пэн переливами по звукам
        # аккорда — с эхом.
        for place, pitch in ((1.5, 250.0), (3.5, 330.0)):
            add(bongo(pitch), first + place, 0.05)
        tones = sorted(chord[1:4])
        for place, pick in ((0.5, 0), (2.0, 2), (3.5, 1)):
            add(steel(hz(tones[pick] + 24), 1.2 * BEAT), first + place, 0.045, into=airy)

    for beat, span, semitone in melody(rng):
        freq = hz(semitone)
        add(voice(freq, span * BEAT, rng), beat, 0.20, into=airy)
        add(soft_piano(freq, span * BEAT + 0.3), beat, 0.10)  # тихо вторит голосу

    # Хвосты последних тактов — в начало: круг без шва.
    for part in (song, airy):
        part[: part.size - length] += part[length:]
    song, airy = song[:length], airy[:length]

    # Голос как в 80-х: хорус — копия с плавающей задержкой около 15 мс —
    # и ещё одна чуть позже, будто поют вдвоём.
    base = int(0.015 * RATE)
    airy = 0.7 * airy + 0.35 * np.roll(wobble(airy, ((0.8, 60.0),)), base) + 0.25 * np.roll(wobble(airy, ((0.53, 45.0),)), 2 * base)

    # Эхо голосу и стил-пэну: через три восьмых, тише и тише (по кругу).
    step = int(0.75 * BEAT * RATE)
    song = song + airy + sum(g * np.roll(airy, n * step) for n, g in ((1, 0.35), (2, 0.18), (3, 0.08)))

    # Реверберация по кругу (свёртка через FFT сама замыкает петлю):
    # длинная и тёмная, как в большом пустом зале.
    t = times(4.5)
    impulse = np.zeros(length)
    tail = np.convolve(rng.normal(0, 1, t.size), np.ones(6) / 6, mode="same") * np.exp(-t * 1.4) * 0.026
    impulse[: t.size] = tail
    impulse[0] = 1.0
    # И мягкий фильтр: верха выше 2.5 кГц тише — звук тёплый, не режет.
    freqs = np.fft.rfftfreq(length, 1.0 / RATE)
    soften = 1.0 / (1.0 + (freqs / 2500.0) ** 2)
    song = np.fft.irfft(np.fft.rfft(song) * np.fft.rfft(impulse) * soften, n=length)

    # Старая кассета: звук чуть плавает (медленно и мелкой дрожью),
    # плёнка мягко сжимает громкие места, тихо шипит.
    song = wobble(song, ((0.5, 10.0), (6.0, 0.35)))
    song /= max(1e-9, np.abs(song).max())
    song = np.tanh(1.4 * song) / np.tanh(1.4)
    hiss = np.convolve(rng.normal(0, 1, length), np.ones(3) / 3, mode="same")
    song += hiss * 0.004

    song /= max(1e-9, np.abs(song).max()) / 0.6
    return song
