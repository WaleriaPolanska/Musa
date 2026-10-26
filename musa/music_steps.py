"""Второй трек — «шаги Музы»: на основе первого (music.py, его не меняем),
но живее и светлее. Темп 96: восьмые совпадают с шагами Музы — она делает
около 3.2 шага в секунду при обычной скорости симуляции (0.19x).

Отличия от первого трека: бочка чаще, малый барабан на 2 и 4 чётче,
хай-хэт шестнадцатыми с открытым в конце такта, бас восьмыми, «шажки» —
тихие деревянные щелчки на каждую восьмую, левый-правый по очереди,
саксофон и стил-пэн чаще. Фильтр мягче (верха до 6 кГц), реверберация
короче, кассета едва заметна. Голос, мелодия и аккорды — те же.
"""

import numpy as np

import music
from music import (CHORDS, RATE, bass, bongo, envelope, hat, hz, kick, low_sax,
                   melody, pad, piano, snare, soft_piano, steel, times, voice, wobble)

TEMPO = 96
BEAT = 60.0 / TEMPO
BARS = music.BARS


def tick(freq):
    """Шажок: короткий деревянный щелчок."""
    t = times(0.06)
    return np.sin(2 * np.pi * freq * t) * np.exp(-t * 90) * envelope(t, 0.001, 0.01, t[-1])


def compose(seed=11):
    rng = np.random.default_rng(seed)
    length = int(RATE * BEAT * 4 * BARS)
    song = np.zeros(length + RATE * 4)
    airy = np.zeros(song.size)

    def add(sound, beat, gain, into=None):
        into = song if into is None else into
        i = int(beat * BEAT * RATE)
        end = min(into.size, i + sound.size)
        into[i:end] += gain * sound[: end - i]

    for bar in range(BARS):
        chord = CHORDS[bar % len(CHORDS)]
        first = bar * 4.0
        add(pad([hz(c) for c in chord], 4 * BEAT), first, 0.12)
        for place, span, gain in ((0.0, 0.5, 0.06), (1.5, 0.5, 0.05), (2.5, 0.5, 0.05), (3.5, 0.5, 0.04)):
            for c in chord[1:4]:
                add(piano(hz(c + 12), span * BEAT), first + place, gain)
        # Бас восьмыми: корень, корень, квинта, корень, октава — идёт вперёд.
        root = hz(chord[0] - 12)
        for place, ratio in ((0, 1), (0.5, 1), (1, 1.5), (1.5, 1), (2, 1), (2.5, 1), (3, 2), (3.5, 1.5)):
            add(bass(root * ratio, 0.45 * BEAT), first + place, 0.24)
        # Ритм: бочка на 1, «и» второй, 3; малый на 2 и 4; хай-хэт 16-ми,
        # открытый на последней восьмой.
        add(kick(), first, 0.30)
        add(kick(), first + 1.5, 0.16)
        add(kick(), first + 2, 0.24)
        add(snare(rng), first + 1, 0.16)
        add(snare(rng), first + 3, 0.16)
        for step in range(16):
            add(hat(rng, open_part=1.0 if step == 14 else 0.0), first + step / 4, 0.035 if step % 2 == 0 else 0.02)
        # Шажки на каждую восьмую: левый и правый тон по очереди.
        for step in range(8):
            add(tick(900.0 if step % 2 == 0 else 760.0), first + step / 2, 0.05)
        # Пульс низкого саксофона и бонго.
        low = ((chord[0] + 5) % 12) - 5
        for place, span, step in ((0.5, 0.35, 0), (1.0, 0.35, 7), (1.5, 0.35, 0), (2.5, 0.8, 12), (3.5, 0.35, 7)):
            add(low_sax(hz(low + step), span * BEAT, rng), first + place, 0.12)
        for place, pitch in ((0.75, 330.0), (1.5, 250.0), (2.75, 330.0), (3.5, 250.0)):
            add(bongo(pitch), first + place, 0.07)
        tones = sorted(chord[1:4])
        for place, pick in ((0.5, 0), (1.0, 1), (2.0, 2), (2.5, 1), (3.5, 2)):
            add(steel(hz(tones[pick] + 24), 1.0 * BEAT), first + place, 0.045, into=airy)

    for beat, span, semitone in melody(rng):
        freq = hz(semitone)
        add(voice(freq, span * BEAT, rng), beat, 0.20, into=airy)
        add(soft_piano(freq, span * BEAT + 0.3), beat, 0.10)

    for part in (song, airy):
        part[: part.size - length] += part[length:]
    song, airy = song[:length], airy[:length]

    base = int(0.015 * RATE)
    airy = 0.75 * airy + 0.3 * np.roll(wobble(airy, ((0.8, 60.0),)), base)
    step = int(0.75 * BEAT * RATE)
    song = song + airy + sum(g * np.roll(airy, n * step) for n, g in ((1, 0.25), (2, 0.1)))

    # Реверберация короче, фильтр мягче: верха до 6 кГц.
    t = times(2.0)
    impulse = np.zeros(length)
    impulse[: t.size] = np.convolve(rng.normal(0, 1, t.size), np.ones(4) / 4, mode="same") * np.exp(-t * 2.5) * 0.02
    impulse[0] = 1.0
    freqs = np.fft.rfftfreq(length, 1.0 / RATE)
    soften = 1.0 / (1.0 + (freqs / 6000.0) ** 2)
    song = np.fft.irfft(np.fft.rfft(song) * np.fft.rfft(impulse) * soften, n=length)

    # Кассета едва заметна.
    song = wobble(song, ((0.5, 4.0),))
    song /= max(1e-9, np.abs(song).max())
    song = np.tanh(1.2 * song) / np.tanh(1.2)
    song /= max(1e-9, np.abs(song).max()) / 0.6
    return song
