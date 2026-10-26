"""Pygame control panel for the v2 ROBOCON course.

This is the modular replacement for the old ``ui.py``: obstacle presets on the
left, a top-down route map on the right, mouse waypoint navigation, and
keyboard telemetry.  It talks only to ``EpisodeRunner``; it does not know how
the policy or the MJCF were built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from .navigation import NavigationController


@dataclass(frozen=True)
class Scenario:
    name: str
    spawn: tuple[float, float, float, float]
    note: str
    target: tuple[float, float] | None = None


# Coordinates are for robocon2027.v2.xml (obstacles around x=-6.7..6.5,
# y=-2.5..2.5).  Base z is the Go2W stand height; the plum platform preset is
# raised to its 0.20 m top.
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("全地图起点", (0.0, 0.0, 0.45, 0.0), "平地起步 / 手动驾驶"),
    Scenario("高墙", (0.80, 2.0, 0.45, 0.0), "high_wall_00，面向 +x", (1.25, 2.0)),
    Scenario("梅花桩", (3.225, 2.0, 0.65, 0.0), "入口平台 + 6 块石，面向 +x", (3.90, 2.0)),
    Scenario("单木桥", (-3.0, 2.0, 0.45, 0.0), "3 m x 0.4 m 木桥，面向 +x", (-2.55, 2.0)),
    Scenario("轮胎阵", (-3.8, -2.0, 0.45, 0.0), "2 x 4 轮胎，面向 +x", (-3.15, -2.0)),
    Scenario("台阶 / 平台", (-3.0, -2.0, 0.45, math.pi), "两级台阶 + 0.36 m 平台，面向 -x", (-3.85, -2.0)),
    Scenario("限高框", (0.0, -2.0, 0.45, 0.0), "0.29 m 净空框，面向 +x", (0.25, -2.0)),
    Scenario("L 形绕桩", (6.0, -3.0, 0.45, math.pi / 2), "L_pile，面向 +y", (6.0, -2.55)),
    Scenario("砂砾 / 坑区", (-3.9, 2.0, 0.45, math.pi), "gravel + wood + pit，面向 -x", (-4.30, 2.0)),
    Scenario("坡道 / 下平台", (-6.2, -2.8, 0.45, math.pi / 2), "ramp_10，面向 +y", (-6.2, -2.0)),
)


class Button:
    def __init__(self, rect: pygame.Rect, label: str, callback, enabled=lambda: True) -> None:
        self.rect = rect
        self.label = label
        self.callback = callback
        self.enabled = enabled

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, selected: bool = False) -> None:
        active = bool(self.enabled())
        if not active:
            fill, border, text = (42, 46, 55), (65, 70, 82), (105, 110, 120)
        elif selected:
            fill, border, text = (23, 104, 118), (76, 220, 205), (240, 255, 252)
        else:
            fill, border, text = (45, 55, 68), (85, 104, 124), (226, 234, 242)
        pygame.draw.rect(surface, fill, self.rect, border_radius=5)
        pygame.draw.rect(surface, border, self.rect, width=1, border_radius=5)
        rendered = font.render(self.label, True, text)
        surface.blit(rendered, rendered.get_rect(center=self.rect.center))

    def click(self, pos: tuple[int, int]) -> bool:
        if self.enabled() and self.rect.collidepoint(pos):
            self.callback()
            return True
        return False


class ControlPanel:
    def __init__(
        self,
        runner,
        scene,
        navigation: NavigationController | None = None,
        sim_speed: float = 1.0,
        model_catalog=None,
        model_loader=None,
        current_model: str | Path | None = None,
        residual_scale: float = 0.40,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Go2W Sim2Sim Control - ROBOCON v2")
        self.surface = pygame.display.set_mode((1120, 770))
        self.clock = pygame.time.Clock()
        self.runner = runner
        self.scene = scene
        self.navigation = navigation or NavigationController()
        self.sim_speed = float(sim_speed)
        self.paused = False
        self.closed = False
        self.draw_mode = False
        self.drawing = False
        self.drawing_points: list[tuple[float, float]] = []
        self.selected_scenario = 0
        self.command = np.zeros(3, dtype=np.float32)
        self.command_limit = np.asarray(runner.cfg.max_command, dtype=np.float32)
        self.map_rect = pygame.Rect(585, 82, 510, 650)
        self.map_geometries, self.map_bounds = self._build_map(scene)
        xmin, xmax, ymin, ymax = self.map_bounds
        world_w = max(xmax - xmin, 1.0)
        world_h = max(ymax - ymin, 1.0)
        self.map_scale = min(self.map_rect.width / world_w, self.map_rect.height / world_h)
        self.map_cx = (xmin + xmax) / 2.0
        self.map_cy = (ymin + ymax) / 2.0
        self.font = self._font(18)
        self.small = self._font(14)
        self.title = self._font(24)
        self.buttons: list[Button] = []
        self._build_buttons()
        self.model_catalog = model_catalog
        self.model_loader = model_loader
        self.current_model = None if current_model is None else Path(current_model)
        self.residual_scale = float(residual_scale)
        self.model_page = False
        self._paused_before_model = False
        self.model_entries: list = []
        self.model_selected = 0
        self.model_scroll = 0
        self.model_status = "点击“选择模型”扫描 .pt checkpoint"
        self.model_buttons: list[Button] = []
        self._model_list_top = 102
        self._model_row_h = 40
        self._model_visible_rows = 13
        self._model_list_bottom = self._model_list_top + self._model_visible_rows * self._model_row_h

    # ------------------------------------------------------------------
    # Font / map helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _font(size: int) -> pygame.font.Font:
        for path in (
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
        ):
            if path.exists():
                return pygame.font.Font(str(path), size)
        return pygame.font.Font(pygame.font.get_default_font(), size)

    @staticmethod
    def _color_for(kind: str) -> tuple[int, int, int]:
        return {
            "wall": (220, 96, 72),
            "stone": (96, 176, 84),
            "platform": (72, 140, 210),
            "step": (214, 148, 56),
            "frame": (230, 190, 52),
            "pile": (232, 150, 58),
            "tire": (70, 70, 78),
            "pit": (150, 120, 80),
            "course": (110, 124, 140),
        }.get(kind, (120, 130, 145))

    def _build_map(self, scene):
        candidates = {"wall", "stone", "platform", "step", "frame", "pile", "tire", "pit"}
        geoms: list[tuple[str, str, float, float, float, float, tuple[int, int, int]]] = []
        xs: list[float] = []
        ys: list[float] = []
        gravel_box: list[float] | None = None
        for feature in scene.build.features:
            name = feature.name
            if name.startswith("gravel") or name.startswith("wood"):
                x, y = feature.center[0], feature.center[1]
                if gravel_box is None:
                    gravel_box = [x, y, x, y]
                else:
                    gravel_box[0] = min(gravel_box[0], x)
                    gravel_box[1] = min(gravel_box[1], y)
                    gravel_box[2] = max(gravel_box[2], x)
                    gravel_box[3] = max(gravel_box[3], y)
                continue
            if feature.kind not in candidates:
                continue
            sx = max(0.02, float(feature.size[0]))
            sy = max(0.02, float(feature.size[1]))
            if sx < 0.015 and sy < 0.015:
                continue
            geoms.append(
                (
                    name,
                    feature.kind,
                    float(feature.center[0]),
                    float(feature.center[1]),
                    sx,
                    sy,
                    self._color_for(feature.kind),
                )
            )
            xs.extend((feature.center[0] - sx / 2.0, feature.center[0] + sx / 2.0))
            ys.extend((feature.center[1] - sy / 2.0, feature.center[1] + sy / 2.0))
        if gravel_box is not None:
            geoms.append(
                (
                    "gravel_pit",
                    "pit",
                    (gravel_box[0] + gravel_box[2]) / 2.0,
                    (gravel_box[1] + gravel_box[3]) / 2.0,
                    max(0.2, gravel_box[2] - gravel_box[0]),
                    max(0.2, gravel_box[3] - gravel_box[1]),
                    (118, 96, 70),
                )
            )
            xs.extend((gravel_box[0], gravel_box[2]))
            ys.extend((gravel_box[1], gravel_box[3]))
        if xs and ys:
            xmin, xmax = min(xs) - 1.0, max(xs) + 1.0
            ymin, ymax = min(ys) - 1.0, max(ys) + 1.0
        else:
            xmin, xmax, ymin, ymax = -7.5, 7.5, -4.0, 4.0
        # Keep the aspect ratio readable: at least 4 m of lateral extent.
        if ymax - ymin < 4.0:
            cy = (ymax + ymin) / 2.0
            ymin, ymax = cy - 2.0, cy + 2.0
        return geoms, (xmin, xmax, ymin, ymax)

    def _world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        # Equal scale in x/y so boxes keep their true aspect ratio.
        sx = self.map_rect.centerx + (x - self.map_cx) * self.map_scale
        sy = self.map_rect.centery - (y - self.map_cy) * self.map_scale
        return int(sx), int(sy)

    def _screen_to_world(self, x: int, y: int) -> tuple[float, float]:
        wx = self.map_cx + (x - self.map_rect.centerx) / self.map_scale
        wy = self.map_cy - (y - self.map_rect.centery) / self.map_scale
        return wx, wy

    # ------------------------------------------------------------------
    # Buttons / commands
    # ------------------------------------------------------------------
    def _button(self, x, y, w, h, label, callback, enabled=lambda: True) -> None:
        self.buttons.append(Button(pygame.Rect(x, y, w, h), label, callback, enabled))

    def _build_buttons(self) -> None:
        self._button(430, 20, 118, 30, "选择模型", self.open_model_page)
        y = 88
        for index, scenario in enumerate(SCENARIOS):
            row, col = divmod(index, 2)
            self._button(
                18 + col * 274,
                y + row * 38,
                258,
                32,
                scenario.name,
                lambda i=index: self.select_scenario(i),
            )
        speed_y = y + math.ceil(len(SCENARIOS) / 2) * 38 + 14

        def add_speed(label, axis, delta, x):
            self._button(x, speed_y, 76, 32, label, lambda: self.change_speed(axis, delta))

        add_speed("VX -", 0, -0.05, 18)
        add_speed("VX +", 0, 0.05, 100)
        add_speed("VY -", 1, -0.05, 188)
        add_speed("VY +", 1, 0.05, 270)
        add_speed("Yaw -", 2, -0.10, 358)
        add_speed("Yaw +", 2, 0.10, 440)

        action_y = speed_y + 46
        self._button(18, action_y, 98, 36, "开始 / 继续", self.resume)
        self._button(122, action_y, 72, 36, "暂停", self.pause)
        self._button(200, action_y, 72, 36, "停止", self.stop)
        self._button(278, action_y, 94, 36, "复位", self.reset)
        self._button(378, action_y, 170, 36, "导航到选中障碍", self.start_scenario_navigation)

        rate_y = action_y + 50
        for index, rate in enumerate((0.5, 1.0, 2.0)):
            self._button(18 + index * 100, rate_y, 88, 32, f"仿真 {rate:g}x", lambda r=rate: self.set_sim_speed(r))
        self._button(322, rate_y, 104, 32, "鼠标画线", self.toggle_draw_mode)
        self._button(434, rate_y, 114, 32, "取消导航", self.cancel_navigation)
        self.status_y = rate_y + 56

    def change_speed(self, axis: int, delta: float) -> None:
        limit = float(self.command_limit[axis])
        self.command[axis] = float(np.clip(self.command[axis] + delta, -limit, limit))

    def select_scenario(self, index: int) -> None:
        self.selected_scenario = int(index)
        self.cancel_navigation()
        self.runner.set_spawn(SCENARIOS[index].spawn)
        self.runner.reset()
        self.command[:] = 0.0
        self.paused = False

    def reset(self) -> None:
        self.cancel_navigation()
        self.runner.set_spawn(SCENARIOS[self.selected_scenario].spawn)
        self.runner.reset()
        self.command[:] = 0.0
        self.paused = False

    def start_scenario_navigation(self) -> None:
        scenario = SCENARIOS[self.selected_scenario]
        if scenario.target is None:
            return
        self.navigation.start_target(scenario.target[0], scenario.target[1], 0.25)
        self.command[:] = 0.0
        self.paused = False

    def resume(self) -> None:
        self.paused = False

    def pause(self) -> None:
        self.paused = True

    def stop(self) -> None:
        self.cancel_navigation()
        self.command[:] = 0.0
        self.paused = True

    def set_sim_speed(self, value: float) -> None:
        self.sim_speed = float(value)

    def toggle_draw_mode(self) -> None:
        self.draw_mode = not self.draw_mode
        if not self.draw_mode:
            self.drawing = False
            self.drawing_points.clear()

    def cancel_navigation(self) -> None:
        self.navigation.cancel()
        self.drawing = False
        self.drawing_points.clear()
        self.command[:] = 0.0

    # ------------------------------------------------------------------
    # Model picker page
    # ------------------------------------------------------------------
    def _build_model_buttons(self) -> None:
        y = 640
        self.model_buttons = [
            Button(pygame.Rect(18, y, 90, 34), "返回", self.close_model_page),
            Button(pygame.Rect(116, y, 100, 34), "重新扫描", self.rescan_models),
            Button(pygame.Rect(224, y, 120, 34), "加载选中", self.load_selected_model),
            Button(pygame.Rect(352, y, 80, 34), "residual 0.35", lambda: self.set_residual_scale(0.35)),
            Button(pygame.Rect(440, y, 80, 34), "residual 0.40", lambda: self.set_residual_scale(0.40)),
        ]

    def open_model_page(self) -> None:
        self._paused_before_model = self.paused
        self.model_page = True
        self.paused = True
        self._build_model_buttons()
        self.rescan_models()

    def close_model_page(self) -> None:
        self.model_page = False
        # Returning from the picker restores the previous run/pause state, so
        # it cannot leave the viewer silently paused forever.
        self.paused = self._paused_before_model

    def rescan_models(self) -> None:
        if self.model_catalog is None:
            self.model_entries = []
            self.model_status = "未配置模型目录（SIM2SIM_MODEL_DIRS）"
            return
        try:
            self.model_entries = list(self.model_catalog.scan())
        except Exception as exc:  # noqa: BLE001
            self.model_entries = []
            self.model_status = f"扫描失败: {exc}"
            return
        self.model_selected = min(self.model_selected, max(0, len(self.model_entries) - 1))
        self.model_scroll = 0
        self.model_status = f"发现 {len(self.model_entries)} 个 checkpoint；点击一行后按“加载选中”"

    def set_residual_scale(self, value: float) -> None:
        self.residual_scale = float(value)
        self.model_status = f"residual_scale={self.residual_scale:g}（PlumBlossom=0.35, Mixed=0.40）"

    def _set_model_selection(self, index: int) -> None:
        if not self.model_entries:
            return
        self.model_selected = int(np.clip(index, 0, len(self.model_entries) - 1))
        if self.model_selected < self.model_scroll:
            self.model_scroll = self.model_selected
        if self.model_selected >= self.model_scroll + self._model_visible_rows:
            self.model_scroll = self.model_selected - self._model_visible_rows + 1

    def load_selected_model(self) -> None:
        if not self.model_entries:
            self.model_status = "列表为空；把 .pt 放到 models/ 或 D:\mujoco_test 根目录"
            return
        entry = self.model_entries[self.model_selected]
        if self.model_loader is None:
            self.model_status = "当前启动方式没有提供模型加载回调"
            return
        self.model_status = f"正在加载 {entry.name} ..."
        try:
            ok, message = self.model_loader(entry, self.residual_scale)
        except Exception as exc:  # noqa: BLE001
            ok, message = False, f"加载异常: {exc}"
        self.model_status = message
        if ok:
            self.current_model = entry.path
            self.model_page = False
            # Resume immediately so switching a model does not look like the
            # simulator is stuck in pause.
            self.paused = False

    def _model_page_event(self, event) -> None:
        if event.type == pygame.QUIT:
            self.closed = True
            return
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                self.close_model_page()
            elif event.key in (pygame.K_UP, pygame.K_w):
                self._set_model_selection(self.model_selected - 1)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self._set_model_selection(self.model_selected + 1)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.load_selected_model()
            elif event.key == pygame.K_r:
                self.rescan_models()
            return
        if event.type == pygame.MOUSEWHEEL:
            self.model_scroll = int(
                np.clip(self.model_scroll - event.y, 0, max(0, len(self.model_entries) - self._model_visible_rows))
            )
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
            self.model_scroll = int(
                np.clip(
                    self.model_scroll + (-1 if event.button == 4 else 1),
                    0,
                    max(0, len(self.model_entries) - self._model_visible_rows),
                )
            )
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.model_buttons:
                if button.click(event.pos):
                    return
            if self._model_list_top <= event.pos[1] < self._model_list_bottom:
                row = (event.pos[1] - self._model_list_top) // self._model_row_h
                index = self.model_scroll + int(row)
                if 0 <= index < len(self.model_entries):
                    self._set_model_selection(index)

    def key(self, key: int) -> None:
        """Handle a key from either pygame or the MuJoCo viewer callback."""

        if self.model_page:
            try:
                code = int(key)
            except (TypeError, ValueError):
                return
            if code in (27, 256):
                self.close_model_page()
            return

        if isinstance(key, str):
            char = key.lower()
        else:
            try:
                char = chr(int(key)).lower()
            except (TypeError, ValueError):
                return
        if char in {"w", "s", "a", "d", "q", "e", "x"}:
            # Keyboard input takes manual control back from the waypoint
            # follower instead of being silently overwritten by it.
            self.navigation.cancel()
            self.command[:] = 0.0
        if char == "w":
            self.command[0] = 0.25
        elif char == "s":
            self.command[0] = -0.20
        elif char == "a":
            self.command[1] = 0.18
        elif char == "d":
            self.command[1] = -0.18
        elif char == "q":
            self.command[2] = 0.45
        elif char == "e":
            self.command[2] = -0.45
        elif char == "r":
            self.reset()
        elif char == " ":
            self.paused = not self.paused

    def command_for_policy(self) -> np.ndarray:
        if self.navigation.active:
            data = self.runner.scene.data
            q = data.qpos[3:7]
            yaw = math.atan2(
                2.0 * (q[0] * q[3] + q[1] * q[2]),
                1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]),
            )
            return self.navigation.update(float(data.qpos[0]), float(data.qpos[1]), yaw)
        return self.command.copy()

    # ------------------------------------------------------------------
    # Mouse / keyboard / draw
    # ------------------------------------------------------------------
    def _start_draw(self, pos: tuple[int, int]) -> None:
        self.drawing = True
        self.drawing_points = [self._screen_to_world(*pos)]
        self.navigation.cancel()

    def _finish_draw(self) -> None:
        if len(self.drawing_points) < 2:
            self.drawing = False
            self.drawing_points.clear()
            return
        waypoints = []
        for point in self.drawing_points:
            if not waypoints or math.dist(point, waypoints[-1][:2]) >= 0.25:
                waypoints.append((point[0], point[1], 0.25, "rough"))
        if waypoints:
            self.navigation.start_mission("Mouse path", waypoints)
            self.command[:] = 0.0
        self.drawing = False
        self.drawing_points.clear()

    def poll(self) -> None:
        for event in pygame.event.get():
            if self.model_page:
                self._model_page_event(event)
                continue
            if event.type == pygame.QUIT:
                self.closed = True
            elif event.type == pygame.KEYDOWN:
                self.key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.map_rect.collidepoint(event.pos):
                    if self.draw_mode:
                        self._start_draw(event.pos)
                    else:
                        wx, wy = self._screen_to_world(*event.pos)
                        self.navigation.start_target(wx, wy)
                        self.command[:] = 0.0
                else:
                    for button in self.buttons:
                        if button.click(event.pos):
                            break
            elif event.type == pygame.MOUSEMOTION and self.drawing and self.map_rect.collidepoint(event.pos):
                point = self._screen_to_world(*event.pos)
                if not self.drawing_points or math.dist(point, self.drawing_points[-1]) >= 0.12:
                    self.drawing_points.append(point)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.drawing:
                self._finish_draw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_map(self) -> None:
        pygame.draw.rect(self.surface, (13, 20, 32), self.map_rect)
        pygame.draw.rect(self.surface, (70, 92, 115), self.map_rect, width=1)
        xmin, xmax, ymin, ymax = self.map_bounds
        for x in range(math.ceil(xmin), math.floor(xmax) + 1):
            sx, _ = self._world_to_screen(x, ymin)
            pygame.draw.line(self.surface, (28, 42, 61), (sx, self.map_rect.top), (sx, self.map_rect.bottom), 1)
        for y in range(math.ceil(ymin), math.floor(ymax) + 1):
            _, sy = self._world_to_screen(xmin, y)
            pygame.draw.line(self.surface, (28, 42, 61), (self.map_rect.left, sy), (self.map_rect.right, sy), 1)

        for _name, kind, x, y, sx, sy, color in self.map_geometries:
            px, py = self._world_to_screen(x, y)
            half_x = max(2, int(sx * self.map_scale / 2.0))
            half_y = max(2, int(sy * self.map_scale / 2.0))
            if kind == "tire":
                pygame.draw.circle(self.surface, color, (px, py), max(2, half_x))
            else:
                pygame.draw.rect(
                    self.surface,
                    color,
                    pygame.Rect(px - half_x, py - half_y, half_x * 2, half_y * 2),
                )

        if len(self.navigation.drawn_path) > 1:
            points = [self._world_to_screen(x, y) for x, y in self.navigation.drawn_path]
            pygame.draw.lines(self.surface, (16, 185, 129), False, points, 3)
            for point in points:
                pygame.draw.circle(self.surface, (6, 182, 212), point, 4)
        if len(self.drawing_points) > 1:
            points = [self._world_to_screen(x, y) for x, y in self.drawing_points]
            pygame.draw.lines(self.surface, (245, 158, 11), False, points, 3)

        data = self.runner.scene.data
        rx, ry = self._world_to_screen(float(data.qpos[0]), float(data.qpos[1]))
        pygame.draw.circle(self.surface, (6, 182, 212), (rx, ry), 7)
        q = data.qpos[3:7]
        yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))
        nose = (rx + int(math.cos(yaw) * 16), ry - int(math.sin(yaw) * 16))
        pygame.draw.line(self.surface, (16, 185, 129), (rx, ry), nose, 3)

        if self.navigation.active and self.navigation.target_x is not None and self.navigation.target_y is not None:
            target = self._world_to_screen(self.navigation.target_x, self.navigation.target_y)
            pygame.draw.circle(self.surface, (234, 179, 8), target, 9, width=2)
            pygame.draw.line(self.surface, (234, 179, 8), (target[0] - 12, target[1]), (target[0] + 12, target[1]), 1)
            pygame.draw.line(self.surface, (234, 179, 8), (target[0], target[1] - 12), (target[0], target[1] + 12), 1)

    def _draw_model_page(self) -> None:
        self.surface.blit(self.title.render("选择模型 checkpoint", True, (236, 244, 250)), (18, 24))
        self.surface.blit(
            self.small.render(
                "优先扫描 sim2sim\\model | 点击一行选中，Enter/“加载选中”切换，ESC 返回",
                True,
                (153, 170, 184),
            ),
            (18, 58),
        )
        current = "-" if self.current_model is None else self.current_model.name
        self.surface.blit(
            self.small.render(f"当前模型: {current}    residual_scale={self.residual_scale:g}", True, (16, 185, 129)),
            (18, 78),
        )
        list_rect = pygame.Rect(18, self._model_list_top, 530, self._model_visible_rows * self._model_row_h)
        pygame.draw.rect(self.surface, (13, 20, 32), list_rect)
        pygame.draw.rect(self.surface, (70, 92, 115), list_rect, width=1)
        if not self.model_entries:
            self.surface.blit(
                self.small.render("没有发现 .pt；把模型放进 sim2sim\\model 后点“重新扫描”", True, (230, 170, 80)),
                (28, self._model_list_top + 18),
            )
        visible = self.model_entries[self.model_scroll : self.model_scroll + self._model_visible_rows]
        for row, entry in enumerate(visible):
            y = self._model_list_top + row * self._model_row_h
            rect = pygame.Rect(20, y + 2, 526, self._model_row_h - 4)
            selected = (self.model_scroll + row) == self.model_selected
            is_current = self.current_model is not None and entry.path == self.current_model
            if selected:
                pygame.draw.rect(self.surface, (23, 104, 118), rect, border_radius=4)
            elif is_current:
                pygame.draw.rect(self.surface, (22, 52, 48), rect, border_radius=4)
            pygame.draw.rect(
                self.surface,
                (76, 220, 205) if selected else ((60, 120, 100) if is_current else (48, 60, 74)),
                rect,
                width=1,
                border_radius=4,
            )
            name = entry.name
            if len(name) > 34:
                name = name[:31] + "..."
            self.surface.blit(
                self.small.render(f"{name}", True, (236, 244, 250)),
                (28, y + 5),
            )
            self.surface.blit(
                self.small.render(
                    f"iter={entry.iteration_text}   {entry.size_mb:.1f} MB   {entry.group}",
                    True,
                    (164, 180, 193),
                ),
                (28, y + 21),
            )
            self.surface.blit(
                self.small.render(entry.date_text, True, (125, 145, 160)),
                (430, y + 13),
            )
        if len(self.model_entries) > self._model_visible_rows:
            track = pygame.Rect(552, self._model_list_top, 4, self._model_visible_rows * self._model_row_h)
            pygame.draw.rect(self.surface, (40, 52, 66), track)
            total = len(self.model_entries)
            height = max(20, int(track.height * self._model_visible_rows / total))
            top = track.top + int((track.height - height) * self.model_scroll / max(1, total - self._model_visible_rows))
            pygame.draw.rect(self.surface, (76, 220, 205), pygame.Rect(track.left, top, 4, height))
        for button in self.model_buttons:
            button.draw(self.surface, self.small)
        self.surface.blit(self.small.render(self.model_status, True, (230, 190, 80)), (18, 688))
        self.surface.blit(
            self.small.render("模型目录: sim2sim\\model（只扫描此目录及其子目录）", True, (111, 135, 150)),
            (18, 724),
        )
        self.surface.blit(
            self.small.render("选中不兼容的 checkpoint 会显示错误，不会替换当前模型", True, (111, 135, 150)),
            (18, 746),
        )

    def draw(self) -> None:
        bg, panel = (20, 25, 32), (29, 36, 47)
        self.surface.fill(bg)
        pygame.draw.rect(self.surface, panel, (10, 10, 566, 750), border_radius=8)
        pygame.draw.rect(self.surface, panel, (576, 10, 530, 750), border_radius=8)
        if self.model_page:
            self._draw_model_page()
            self.surface.blit(self.title.render("路线 / 障碍地图", True, (236, 244, 250)), (594, 24))
            self.surface.blit(
                self.small.render("模型页面打开时仿真暂停；加载后按“开始/继续”或键盘驾驶", True, (153, 170, 184)),
                (594, 58),
            )
            self._draw_map()
            pygame.display.flip()
            self.clock.tick(30)
            return
        self.surface.blit(self.title.render("Go2W Sim2Sim 控制面板", True, (236, 244, 250)), (18, 24))
        self.surface.blit(
            self.small.render("ROBOCON2027 v2 地图  |  选择障碍预设，或直接键盘驾驶", True, (153, 170, 184)),
            (18, 58),
        )
        self.surface.blit(self.title.render("路线 / 障碍地图", True, (236, 244, 250)), (594, 24))
        self.surface.blit(
            self.small.render("点击地图设目标；开启鼠标画线后拖动生成航点；点击机器人附近可取消", True, (153, 170, 184)),
            (594, 58),
        )
        self._draw_map()
        for index, button in enumerate(self.buttons):
            selected = index == self.selected_scenario
            button.draw(self.surface, self.small, selected=selected)

        scenario = SCENARIOS[self.selected_scenario]
        y = self.status_y
        pygame.draw.line(self.surface, (67, 80, 94), (18, y - 12), (548, y - 12), 1)
        data = self.runner.scene.data
        lines = (
            f"当前障碍: {scenario.name}  ({scenario.note})",
            f"指令 vx={self.command[0]:+.2f}  vy={self.command[1]:+.2f}  yaw={self.command[2]:+.2f}",
            f"仿真倍率: {self.sim_speed:g}x    状态: {'暂停' if self.paused else '运行'}",
            f"位置 x={data.qpos[0]:+.2f}  y={data.qpos[1]:+.2f}  z={data.qpos[2]:.3f}",
            f"碰撞={self.runner.collision_count}    摔倒={'是' if self.runner.fallen else '否'}    "
            f"时间={self.runner.elapsed:.1f}s",
        )
        for index, line in enumerate(lines):
            color = (226, 234, 242) if index == 0 else (164, 180, 193)
            self.surface.blit(self.small.render(line, True, color), (18, y + index * 25))
        nav_status = "AUTO" if self.navigation.active else "MANUAL"
        target = "-" if self.navigation.target_x is None else f"({self.navigation.target_x:.2f}, {self.navigation.target_y:.2f})"
        self.surface.blit(
            self.small.render(f"导航={nav_status}  目标={target}", True, (164, 180, 193)),
            (18, y + 132),
        )
        self.surface.blit(
            self.small.render("键盘 W/S 前后  A/D 横移  Q/E 转向  X 停  Space 暂停  R 复位", True, (111, 135, 150)),
            (18, 724),
        )
        self.surface.blit(
            self.small.render("MuJoCo 窗口与面板同时可用；策略约 2 s 摔倒时窗口保持，按 R 重试", True, (111, 135, 150)),
            (18, 746),
        )
        pygame.display.flip()
        self.clock.tick(30)

    def close(self) -> None:
        pygame.quit()
