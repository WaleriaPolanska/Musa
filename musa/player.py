"""Проигрыватель фоновой музыки: встроенный в macOS afplay, по кругу.

Три трека — «dream» (music.py), «steps» (music_steps.py) и «miami»
(music_miami.py); игрок выбирает в списке MUSIC на панели. Ничего не
выбрано (ALL) — играют по очереди с первого, каждый LOOPS кругов; выбран
трек — он по кругу. Переход — за FADE секунд: один трек тает, другой
нарастает.

afplay не умеет начинать с середины и менять громкость на ходу, поэтому
каждый раз пишем файл, который начинается с нужного места, и запускаем
новый afplay чуть раньше, чем гасим старый, — без щелчка тишины.
MUSA_MUSIC=0 — без музыки (тестовые прогоны)."""

import atexit
import os
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

import music
import music_miami
import music_steps

RATE = music.RATE
REPEATS = 12  # столько кругов подряд в файле: хватает минут на десять
FADE = 2.0  # секунд на переход между треками
ORDER = ("dream", "steps", "miami")
LOOPS = 4  # столько кругов играет трек в режиме ALL (около трёх минут)
HOLD = 0.2  # щёлкаешь по кнопкам подряд — меняется только последний выбор
AHEAD = 0.4  # запас на запись файла: новый afplay начинает с этого места


class Music:
    def __init__(self):
        self.volume = 0.15  # то, что на ползунке
        self.playing = None  # громкость, с которой запущен afplay
        self.changed = 0.0
        self.songs = {}
        self.track = "dream"
        self.wanted = "dream"
        self.wanted_since = 0.0
        self.choice = None  # выбранный трек; None — все по очереди
        self.track_since = time.time()  # когда начал играть текущий трек
        self.offset = 0.0  # с какого места трека идёт afplay (с)
        self.started = 0.0
        self.process = None
        self.stopped = False
        self.turn = 0
        self.folder = Path(tempfile.gettempdir())

    def start(self):
        if os.environ.get("MUSA_MUSIC", "1") == "0":
            return
        threading.Thread(target=self._loop, daemon=True).start()
        atexit.register(self.stop)

    def set_volume(self, volume):
        volume = float(np.clip(volume, 0.0, 1.0))
        if abs(volume - self.volume) > 1e-3:
            self.volume = volume
            self.changed = time.time()

    def set_track(self, name):
        if name != self.wanted:
            self.wanted = name
            self.wanted_since = time.time()

    def choose(self, name):
        """Выбор в списке: трек — он по кругу; None — все по очереди."""
        if name == self.choice:
            return
        self.choice = name
        if name is None:
            self.track_since = time.time()  # очередь — с того, что играет
            self.set_track(self.track)
        else:
            self.set_track(name)

    def _next_in_order(self, now):
        """В режиме ALL: трек отыграл свои круги — следующий по очереди."""
        if self.choice is not None or not self.songs:
            return
        rounds = self.songs[self.track].size / RATE * LOOPS
        if self.wanted == self.track and now - self.track_since > rounds:
            self.wanted = ORDER[(ORDER.index(self.track) + 1) % len(ORDER)]
            self.wanted_since = now - HOLD

    def _position(self, moment):
        seconds = self.songs[self.track].size / RATE
        if self.process is None:
            return self.offset
        return (self.offset + moment - self.started) % seconds

    def _launch(self, samples, offset, moment):
        """Пишет файл и запускает его ровно в moment; старый afplay гасит
        чуть позже, чтобы между ними не было тишины."""
        self.turn += 1
        path = self.folder / f"musa_music_{self.turn % 2}.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            out.writeframes((samples * 32767).astype(np.int16).tobytes())
        time.sleep(max(0.0, moment - time.time()))
        old = self.process
        self.process = subprocess.Popen(["afplay", "-v", f"{self.volume:.2f}", str(path)])
        self.offset = offset
        self.started = time.time()
        self.playing = self.volume
        if old is not None and old.poll() is None:
            time.sleep(0.1)
            old.terminate()

    def _play(self, offset, moment):
        song = self.songs[self.track]
        start = int(offset * RATE) % song.size
        self._launch(np.tile(np.roll(song, -start), REPEATS), offset, moment)

    def _cross(self, moment):
        """Старый трек тает, новый нарастает с начала."""
        old = self.songs[self.track]
        start = int(self._position(moment) * RATE) % old.size
        self.track = self.wanted
        self.track_since = moment
        new = np.tile(self.songs[self.track], REPEATS)
        fade = int(FADE * RATE)
        ramp = np.linspace(0.0, np.pi / 2, fade)
        new[:fade] = new[:fade] * np.sin(ramp) + np.roll(old, -start)[:fade] * np.cos(ramp)
        self._launch(new, 0.0, moment)

    def _loop(self):
        self.songs = {"dream": music.compose(), "steps": music_steps.compose(), "miami": music_miami.compose()}
        self.track_since = time.time()
        while not self.stopped:
            now = time.time()
            self._next_in_order(now)
            running = self.process is not None and self.process.poll() is None
            settled = now - self.changed > 0.25  # ползунок отпустили
            switch = self.wanted != self.track and now - self.wanted_since > HOLD
            if self.volume <= 0.0:
                if running:
                    self.offset = self._position(now)
                    self.process.terminate()
                    self.process = None
            elif running and switch:
                self._cross(now + AHEAD)
            elif running and self.playing != self.volume and settled:
                self._play(self._position(now + AHEAD), now + AHEAD)
            elif not running and settled:
                if switch:
                    self.track, self.offset = self.wanted, 0.0
                    self.track_since = now
                    self._play(self.offset, now)
                else:
                    self._play(self._position(now), now)
            time.sleep(0.05)

    def stop(self):
        self.stopped = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
