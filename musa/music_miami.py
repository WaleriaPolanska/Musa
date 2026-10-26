"""Третий трек — «Майами, ночь из офиса»: саксофонная баллада 80-х, с
надрывом, как Careless Whisper по настроению (мелодия своя). Миллионер
смотрит сквозь стекло своего офиса на ночные огни Майами.

Ре минор, 76 ударов в минуту. Бархатный тенор-саксофон ведёт мелодию:
долгие ноты с вибрато, подхваты шестнадцатыми, в конце фраз — стон вниз.
Под ним — FM-электропиано как у DX7, аналоговые аккорды с хорусом,
безладовый бас, драм-машина с «гейтовым» малым барабаном 80-х, хай-хэт
шестнадцатыми, далёкий гул ночного города. Сверху — старая кассета.
"""

import numpy as np

from music import RATE, envelope, hz, kick, times, wobble

TEMPO = 76
BEAT = 60.0 / TEMPO
BARS = 16

# Аккорды по такту (полутоны от ре): Dm9, B♭maj7, Gm9, A7sus → A7 — и
# снова; во второй половине F вместо Dm, чтобы свет мелькнул.
CHORDS = [
    (0, 3, 7, 10, 14), (-4, 0, 3, 7, 10), (5, 8, 12, 15, 19), (-5, 0, 2, 5, 7),
    (0, 3, 7, 10, 14), (-4, 0, 3, 7, 10), (5, 8, 12, 15, 19), (-5, -1, 2, 5, 7),
    (3, 7, 10, 14, 17), (-4, 0, 3, 7, 10), (5, 8, 12, 15, 19), (-5, 0, 2, 5, 7),
    (0, 3, 7, 10, 14), (-4, 0, 3, 7, 10), (-7, -4, 0, 3, 7), (-5, -1, 2, 5, 7),
]
MINOR = [0, 2, 3, 5, 7, 8, 10]
HARMONIC = [0, 2, 3, 5, 7, 8, 11]  # над ля-септаккордом — до-диез, надрыв


def degree_to_semitone(degree, scale):
    return 12 * (degree // 7) + scale[degree % 7]


def sax(freq, seconds, rng, scoop=False, fall=False):
    """Бархатный тенор-саксофон: тёплые нижние обертоны с «рупорным»
    подъёмом около 1.2 кГц, дыхание в атаке; на долгих нотах вибрато
    раскачивается к концу; scoop — подъезд снизу в начале фразы, fall —
    стон вниз в конце."""
    t = times(seconds)
    depth = 0.006 * np.clip((t - 0.35) / 0.6, 0.0, 1.0) if seconds > 0.6 else 0.0
    bend = 1.0 + depth * np.sin(2 * np.pi * 5.3 * t)
    if scoop:
        bend = bend * (1.0 - 0.06 * np.exp(-t / 0.06))
    if fall:
        bend = bend * (1.0 - 0.10 * np.clip((t - (seconds - 0.25)) / 0.25, 0.0, 1.0) ** 2)
    phase = 2 * np.pi * np.cumsum(freq * bend) / RATE
    tone = np.zeros(t.size)
    for k in range(1, int(2600 / freq) + 1):
        f = k * freq
        horn = 1.0 + 0.8 / (1.0 + ((f - 1200.0) / 500.0) ** 2)
        tone += np.sin(k * phase) * horn * 0.78 ** k
    breath = np.convolve(rng.normal(0, 1, t.size), np.ones(8) / 8, mode="same")
    breath *= 0.25 * np.exp(-t * 12) + 0.04
    swell = 0.8 + 0.2 * np.clip(t / max(seconds, 1e-6), 0.0, 1.0)  # долгая нота крепнет
    return (tone / 3.0 + breath) * swell * envelope(t, 0.04, 0.08, t[-1])


def fm_piano(freq, seconds):
    """FM-электропиано как у DX7: колокольчик в атаке тает, остаётся тёплый тон."""
    t = times(seconds)
    index = 1.6 * np.exp(-t * 5.0) + 0.3
    tone = np.sin(2 * np.pi * freq * t + index * np.sin(2 * np.pi * freq * t))
    tine = 0.15 * np.sin(2 * np.pi * freq * 14.0 * t) * np.exp(-t * 30)
    return (tone + tine) * np.exp(-t * 1.2) * envelope(t, 0.004, 0.2, t[-1])


def juno(freqs, seconds):
    """Аналоговые аккорды с хорусом: мягкая «пила» из гармоник, два голоса
    слегка плывут друг относительно друга."""
    t = times(seconds + 0.8)
    out = np.zeros(t.size)
    for f in freqs:
        for side in (-1.0, 1.0):
            drift = 1.0 + side * (0.002 + 0.0015 * np.sin(2 * np.pi * 0.6 * t))
            phase = 2 * np.pi * np.cumsum(f * drift) / RATE
            out += sum(np.sin(k * phase) / k ** 1.3 for k in range(1, 7))
    return out * envelope(t, 0.4, 0.8, t[-1]) / len(freqs)


def fretless(freq, seconds):
    """Безладовый бас: круглый тон, мягко въезжает в ноту."""
    t = times(seconds)
    glide = 1.0 - 0.02 * np.exp(-t / 0.04)
    phase = 2 * np.pi * np.cumsum(freq * glide) / RATE
    return (np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)) * envelope(t, 0.02, 0.12, t[-1])


def gated_snare(rng):
    """Малый барабан 80-х: удар и густой хвост реверберации, резко обрезанный."""
    t = times(0.34)
    noise = np.convolve(rng.normal(0, 1, t.size), np.ones(3) / 3, mode="same")
    body = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 25)
    gate = np.where(t < 0.28, 1.0, np.exp(-(t - 0.28) * 120))
    return (0.7 * noise * np.exp(-t * 4) + body) * gate * envelope(t, 0.002, 0.02, t[-1])


def hat16(rng, open_part=0.0):
    t = times(0.05 + 0.18 * open_part)
    noise = rng.normal(0, 1, t.size)
    noise = noise - np.concatenate(([0.0], noise[:-1]))
    return np.convolve(noise, np.ones(2) / 2, mode="same") * np.exp(-t * (80 - 60 * open_part))


# Фразы саксофона по два такта: (начало в долях, длина, шаг по ладу).
# Подхват шестнадцатыми — в долгую ноту — ответ — стон вниз.
PHRASES = (
    ((0.0, 0.25, 0), (0.25, 0.25, 1), (0.5, 0.25, 1), (0.75, 2.25, 1), (3.0, 0.5, -1),
     (3.5, 0.5, -1), (4.0, 1.5, -1), (5.5, 0.5, 2), (6.0, 1.5, -2)),
    ((0.5, 0.5, 0), (1.0, 0.5, 2), (1.5, 2.0, 1), (3.5, 0.25, -1), (3.75, 0.25, -1),
     (4.0, 1.0, -1), (5.0, 0.5, 1), (5.5, 2.0, -1)),
    ((0.0, 1.5, 0), (1.5, 0.25, 1), (1.75, 0.25, 1), (2.0, 2.0, 1), (4.0, 0.5, -2),
     (4.5, 0.5, 1), (5.0, 2.5, -1)),
    ((0.0, 0.5, 0), (0.5, 0.5, -1), (1.0, 3.0, 2), (4.5, 0.5, -1), (5.0, 0.5, -1), (5.5, 2.0, -1)),
)


def melody():
    notes = []  # (начало в долях, длина в долях, полутон, подъезд, стон)
    degree = 15  # ми первой октавы: середина тенора, есть куда рваться вверх
    for phrase in range(BARS // 2):
        start = phrase * 8.0
        pattern = PHRASES[phrase % len(PHRASES)]
        for number, (offset, span, step) in enumerate(pattern):
            degree = int(np.clip(degree + step, 9, 18))
            beat = start + offset
            chord = CHORDS[int(beat // 4) % len(CHORDS)]
            scale = HARMONIC if (-1 % 12) in {c % 12 for c in chord} else MINOR
            if offset in (0.0, 4.0) or span >= 2.0:
                tones = {c % 12 for c in chord}
                near = [d for d in range(9, 19) if degree_to_semitone(d, scale) % 12 in tones]
                degree = min(near, key=lambda d: abs(d - degree))
            last = number == len(pattern) - 1
            notes.append((beat, span, degree_to_semitone(degree, scale), number == 0, last))
    return notes


def compose(seed=21):
    rng = np.random.default_rng(seed)
    length = int(RATE * BEAT * 4 * BARS)
    song = np.zeros(length + RATE * 4)
    lead = np.zeros(song.size)

    def add(sound, beat, gain, into=None):
        into = song if into is None else into
        i = int(beat * BEAT * RATE)
        end = min(into.size, i + sound.size)
        into[i:end] += gain * sound[: end - i]

    for bar in range(BARS):
        chord = CHORDS[bar]
        first = bar * 4.0
        add(juno([hz(c) for c in chord], 4 * BEAT), first, 0.10)
        # Электропиано: аккорд на 1, отклик на «и» второй, на 4.
        for place, span, gain in ((0.0, 1.5, 0.07), (1.5, 1.0, 0.05), (3.0, 1.0, 0.05)):
            for c in chord[1:4]:
                add(fm_piano(hz(c + 12), span * BEAT), first + place, gain)
        # Бас: корень, подхват, квинта, ход к следующему.
        root = hz(chord[0] - 12)
        for place, span, ratio, gain in ((0.0, 1.4, 1.0, 0.32), (1.5, 0.5, 1.0, 0.18),
                                          (2.0, 1.4, 1.5, 0.26), (3.5, 0.5, 2.0, 0.18)):
            add(fretless(root * ratio, span * BEAT), first + place, gain)
        # Драм-машина: бочка на 1 и «и» второй, гейтовый малый на 2 и 4,
        # хай-хэт шестнадцатыми, открытый на последней.
        add(kick(), first, 0.34)
        add(kick(), first + 1.5, 0.18)
        add(kick(), first + 2.75, 0.14)
        add(gated_snare(rng), first + 1, 0.20)
        add(gated_snare(rng), first + 3, 0.20)
        for step in range(16):
            add(hat16(rng, 1.0 if step == 14 else 0.0), first + step / 4, 0.035 if step % 2 == 0 else 0.018)

    for beat, span, semitone, scoop, fall in melody():
        add(sax(hz(semitone - 12), span * BEAT, rng, scoop=scoop, fall=fall), beat, 0.30, into=lead)

    for part in (song, lead):
        part[: part.size - length] += part[length:]
    song, lead = song[:length], lead[:length]

    # Саксофон — в большой зал: длинная тёплая реверберация и эхо.
    step = int(0.75 * BEAT * RATE)
    lead = lead + 0.22 * np.roll(lead, step) + 0.10 * np.roll(lead, 2 * step)

    def hall(sound, seconds, decay, amount):
        t = times(seconds)
        impulse = np.zeros(length)
        impulse[: t.size] = np.convolve(rng.normal(0, 1, t.size), np.ones(5) / 5, mode="same") * np.exp(-t * decay) * amount
        impulse[0] = 1.0
        return np.fft.irfft(np.fft.rfft(sound) * np.fft.rfft(impulse), n=length)

    song = hall(song, 1.8, 2.5, 0.015) + hall(lead, 3.5, 1.6, 0.022)

    # Далёкий город за стеклом: низкий гул, медленно дышит.
    hum = np.fft.irfft(np.fft.rfft(rng.normal(0, 1, length)) / (1.0 + (np.fft.rfftfreq(length, 1 / RATE) / 180.0) ** 2), n=length)
    cycles = max(1, round(0.07 * length / RATE))
    hum *= 0.7 + 0.3 * np.sin(2 * np.pi * cycles * np.arange(length) / length)
    song += hum / max(1e-9, np.abs(hum).max()) * 0.04 * np.abs(song).max()

    # Мягко срезать верха и — старая кассета.
    freqs = np.fft.rfftfreq(length, 1.0 / RATE)
    song = np.fft.irfft(np.fft.rfft(song) / (1.0 + (freqs / 4500.0) ** 2), n=length)
    song = wobble(song, ((0.5, 8.0), (6.0, 0.3)))
    song /= max(1e-9, np.abs(song).max())
    song = np.tanh(1.4 * song) / np.tanh(1.4)
    song += np.convolve(rng.normal(0, 1, length), np.ones(3) / 3, mode="same") * 0.003
    song /= max(1e-9, np.abs(song).max()) / 0.6
    return song
