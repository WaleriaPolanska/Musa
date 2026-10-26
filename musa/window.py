"""Своё окно MuJoCo: работают только наши клавиши.

Встроенный вьювер MuJoCo (mujoco.viewer) сам перехватывает буквы и цифры —
включает каркас, тени, прозрачность и прочее. Здесь окно наше, поэтому
никаких чужих горячих клавиш нет: мышь крутит камеру, клавиши идут в наш код.

Окно — два экрана: слева Муза, справа мозг (musa/panel.py).
"""

import glfw
import mujoco
import numpy as np

MAX_GEOM = 40000

PANEL_PART = 0.36  # какую долю окна занимает правый экран с мозгом


class Window:
    """Окно на два экрана: 3D Муза и панель мозга."""

    def __init__(self, model, data, title, key_handler, width=1700, height=980):

        if not glfw.init():
            raise SystemExit("не удалось запустить glfw")

        self.model = model
        self.data = data
        self.key_handler = key_handler
        self.panel = None

        self.window = glfw.create_window(width, height, title, None, None)

        if not self.window:
            glfw.terminate()
            raise SystemExit("не удалось открыть окно")

        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, MAX_GEOM)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        mujoco.mjv_defaultFreeCamera(model, self.camera)

        self.camera.distance = 6.0
        self.camera.azimuth = 130.0
        self.camera.elevation = -25.0

        self._last_mouse = (0.0, 0.0)
        self._buttons = {"left": False, "right": False, "middle": False}
        self._in_panel = False

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_move)
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    # --------------------------------------------------------
    # ДВА ЭКРАНА
    # --------------------------------------------------------

    def panes(self):
        """Прямоугольники в пикселях: слева Муза, справа мозг."""

        width, height = glfw.get_framebuffer_size(self.window)

        panel_width = int(width * PANEL_PART)

        fly = mujoco.MjrRect(0, 0, width - panel_width, height)
        panel = mujoco.MjrRect(width - panel_width, 0, panel_width, height)

        return fly, panel

    def cursor(self):
        """Курсор в пикселях кадра, начало отсчёта снизу — как у MuJoCo."""

        x, y = glfw.get_cursor_pos(self.window)

        win_width, win_height = glfw.get_window_size(self.window)
        buf_width, buf_height = glfw.get_framebuffer_size(self.window)

        return (
            x * buf_width / max(win_width, 1),
            buf_height - y * buf_height / max(win_height, 1),
        )

    def pixel_scale(self):
        """Во сколько раз пиксели кадра крупнее точек окна (retina)."""

        win_width, _ = glfw.get_window_size(self.window)
        buf_width, _ = glfw.get_framebuffer_size(self.window)

        return buf_width / max(win_width, 1)

    # --------------------------------------------------------
    # ВВОД
    # --------------------------------------------------------

    def key_held(self, key):
        """Зажата ли клавиша прямо сейчас — для плавного движения света."""
        return glfw.get_key(self.window, key) == glfw.PRESS

    def _on_key(self, window, key, scancode, action, mods):

        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
            return

        self.key_handler(key)

    def _on_button(self, window, button, action, mods):

        self._buttons["left"] = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self._buttons["right"] = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )
        self._buttons["middle"] = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )

        self._last_mouse = glfw.get_cursor_pos(window)

        if self.panel is None or button != glfw.MOUSE_BUTTON_LEFT:
            return

        x, y = self.cursor()
        _, panel_rect = self.panes()

        if action == glfw.PRESS:
            self._in_panel = x >= panel_rect.left
            if self._in_panel:
                self.panel.on_press(x, y, panel_rect)
        else:
            self.panel.on_release()
            self._in_panel = False

    def _on_move(self, window, xpos, ypos):

        dx = xpos - self._last_mouse[0]
        dy = ypos - self._last_mouse[1]
        self._last_mouse = (xpos, ypos)

        if self._in_panel:
            x, y = self.cursor()
            self.panel.on_drag(x, y, self.panes()[1])
            return

        if not any(self._buttons.values()):
            return

        width, height = glfw.get_window_size(window)

        if self._buttons["left"]:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self._buttons["right"]:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(
            self.model, action, dx / height, dy / height, self.scene, self.camera
        )

    def _on_scroll(self, window, xoffset, yoffset):

        x, _ = self.cursor()
        _, panel_rect = self.panes()

        if self.panel is not None and x >= panel_rect.left:
            _, y = self.cursor()
            self.panel.on_scroll(yoffset, x, y, panel_rect)
            return

        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.scene,
            self.camera,
        )

    # --------------------------------------------------------
    # КАДР
    # --------------------------------------------------------

    def is_running(self):
        return not glfw.window_should_close(self.window)

    def look_at(self, position):
        """Камера следит за точкой — например за грудью Музы."""
        self.camera.lookat[:] = position

    def draw(self, rect, camera, option, fill=None):
        """Собирает сцену и рисует её в указанный прямоугольник окна."""

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

        ngeom = self.scene.ngeom

        if fill is not None:
            ngeom = fill(self.scene, ngeom)

        self.scene.ngeom = ngeom

        mujoco.mjr_render(rect, self.scene, self.context)

    def overlay(self, rect, left, right, corner=mujoco.mjtGridPos.mjGRID_TOPRIGHT):
        """Подпись в углу экрана. Шрифт MuJoCo знает только латиницу."""

        if not left and not right:
            return

        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            corner,
            rect,
            left,
            right,
            self.context,
        )

    def finish(self):
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def screenshot(self):
        """Картинка текущего кадра, для проверки без глаз."""

        width, height = glfw.get_framebuffer_size(self.window)
        pixels = np.zeros((height, width, 3), dtype=np.uint8)

        mujoco.mjr_readPixels(
            pixels, None, mujoco.MjrRect(0, 0, width, height), self.context
        )

        return np.flipud(pixels)

    def close(self):
        glfw.terminate()
