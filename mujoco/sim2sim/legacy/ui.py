from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pygame


MAP_RECT = pygame.Rect(570, 88, 500, 610)
MAP_BOUNDS = (-7.2, 7.2, -4.4, 4.4)


@dataclass(frozen=True)
class Scenario:
    name: str
    spawn: tuple[float, float, float, float]
    note: str


SCENARIOS = (
    Scenario("全地图起点", (0.0, 0.0, 0.262, 0.0), "平地起步"),
    Scenario("高墙 / 踏石", (1.0, 2.0, 0.262, 0.0), "ROBOCON 障碍区"),
    Scenario("ROBOCON 梅花桩", (3.5, 2.1, 0.262, 0.0), "两侧平台 + 交错方块"),
    Scenario("单木桥", (-3.2, 2.0, 0.262, 0.0), "3 m x 0.4 m 木桥"),
    Scenario("轮胎阵", (-3.5, -2.0, 0.262, 0.0), "ROBOCON 轮胎障碍"),
    Scenario("台阶 / 平台", (-6.8, -2.0, 0.262, 0.0), "ROBOCON 台阶与平台"),
    Scenario("坡道", (-6.8, -1.1, 0.262, 0.0), "ROBOCON 坡道"),
    Scenario("限高框", (0.0, -2.0, 0.262, 0.0), "内高约 290 mm"),
    Scenario("碎石 / 坑区", (-7.0, 0.6, 0.262, 0.0), "ROBOCON 碎石与坑"),
    Scenario("L 形绕桩", (6.0, -3.0, 0.262, math.pi / 2), "ROBOCON L 形桩"),
    Scenario("rough + 梅花桩", (-2.9, 3.35, 0.45, 0.0), "Go2W 独立训练车道"),
    Scenario("平地 + 高墙", (-2.9, 0.85, 0.45, 0.0), "独立三道高墙"),
    Scenario("低杆车道", (-2.9, -0.85, 0.45, 0.0), "Go2W 低杆"),
    Scenario("Gap 车道", (-2.9, -3.35, 0.45, 0.0), "0.42 m 宽、0.22 m 深沟壑"),
)


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    callback: Callable[[], None]
    enabled: Callable[[], bool] = lambda: True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, selected: bool = False) -> None:
        active = self.enabled()
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

    def click(self, position: tuple[int, int]) -> bool:
        if self.enabled() and self.rect.collidepoint(position):
            self.callback()
            return True
        return False


class ControlPanel:
    """Small pygame HUD modeled after the reference sim2sim control panel."""

    def __init__(self, sim, command, initial_spawn, sim_speed: float = 1.0, navigation=None, scene_path=None):
        pygame.init()
        pygame.display.set_caption("Go2W Sim2Sim Control")
        self.surface = pygame.display.set_mode((1100, 760))
        self.clock = pygame.time.Clock()
        self.sim = sim
        self.command = command
        self.navigation = navigation
        self.map_geometries = self._load_map_geometries(scene_path)
        self.initial_spawn = tuple(initial_spawn)
        self.selected_scenario = 0
        self.sim_speed = sim_speed
        self.paused = False
        self.closed = False
        self.draw_mode = False
        self.drawing = False
        self.drawing_points: list[tuple[float, float]] = []
        self.font = self._font(20)
        self.small = self._font(16)
        self.title = self._font(25)
        self.buttons: list[Button] = []
        self._build_buttons()

    @staticmethod
    def _parse_floats(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            return default
        try:
            return tuple(float(item) for item in value.split())
        except ValueError:
            return default

    def _load_map_geometries(self, scene_path) -> list[tuple[str, str, float, float, float, float, tuple[int, int, int]]]:
        """Read course-only geometry for the 2D route map."""
        if scene_path is None or not Path(scene_path).exists():
            return []
        try:
            root = ET.parse(scene_path).getroot()
        except (OSError, ET.ParseError):
            return []

        geometries = []

        def visit_body(body: ET.Element, parent_x: float, parent_y: float) -> None:
            body_pos = self._parse_floats(body.get("pos"), (0.0, 0.0, 0.0))
            body_x = parent_x + body_pos[0]
            body_y = parent_y + body_pos[1]
            for geom in body.findall("geom"):
                if geom.get("class") != "course":
                    continue
                name = geom.get("name", "course_geom")
                if name in {"arena_floor", "floor"} or name.startswith("boundary"):
                    continue
                pos = self._parse_floats(geom.get("pos"), (0.0, 0.0, 0.0))
                size = self._parse_floats(geom.get("size"), (0.1, 0.1, 0.1))
                rgba = self._parse_floats(geom.get("rgba"), (0.45, 0.45, 0.45, 1.0))
                color = tuple(max(30, min(255, int(channel * 255))) for channel in rgba[:3])
                if geom.get("type") == "box" and len(size) >= 2:
                    sx, sy = size[0], size[1]
                elif geom.get("type") == "cylinder" and size:
                    sx = sy = size[0]
                else:
                    sx = sy = max(size[0] if size else 0.1, 0.1)
                if sx < 0.015 and sy < 0.015:
                    continue
                geometries.append((name, geom.get("type", "box"), body_x + pos[0], body_y + pos[1], sx, sy, color))
            for child in body.findall("body"):
                visit_body(child, body_x, body_y)

        worldbody = root.find("worldbody")
        if worldbody is not None:
            for body in worldbody.findall("body"):
                visit_body(body, 0.0, 0.0)
            for geom in worldbody.findall("geom"):
                if geom.get("class") != "course":
                    continue
                name = geom.get("name", "course_geom")
                if name in {"arena_floor", "floor"} or name.startswith("boundary"):
                    continue
                pos = self._parse_floats(geom.get("pos"), (0.0, 0.0, 0.0))
                size = self._parse_floats(geom.get("size"), (0.1, 0.1, 0.1))
                rgba = self._parse_floats(geom.get("rgba"), (0.45, 0.45, 0.45, 1.0))
                color = tuple(max(30, min(255, int(channel * 255))) for channel in rgba[:3])
                sx, sy = (size[0], size[1]) if len(size) >= 2 else (0.1, 0.1)
                geometries.append((name, geom.get("type", "box"), pos[0], pos[1], sx, sy, color))
        return geometries

    @staticmethod
    def _world_to_screen(x: float, y: float) -> tuple[int, int]:
        xmin, xmax, ymin, ymax = MAP_BOUNDS
        sx = MAP_RECT.left + (x - xmin) / (xmax - xmin) * MAP_RECT.width
        sy = MAP_RECT.bottom - (y - ymin) / (ymax - ymin) * MAP_RECT.height
        return int(sx), int(sy)

    @staticmethod
    def _screen_to_world(x: int, y: int) -> tuple[float, float]:
        xmin, xmax, ymin, ymax = MAP_BOUNDS
        wx = xmin + (x - MAP_RECT.left) / MAP_RECT.width * (xmax - xmin)
        wy = ymin + (MAP_RECT.bottom - y) / MAP_RECT.height * (ymax - ymin)
        return wx, wy

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

    def _button(self, x, y, w, h, label, callback, enabled=lambda: True):
        self.buttons.append(Button(pygame.Rect(x, y, w, h), label, callback, enabled))

    def _build_buttons(self) -> None:
        # Scenario selection is a spawn preset; the complete 27rc course stays
        # loaded so the viewer always shows the same physical map.
        y = 105
        for index, scenario in enumerate(SCENARIOS):
            row, col = divmod(index, 2)
            self._button(20 + col * 260, y + row * 42, 248, 34, scenario.name,
                         lambda i=index: self.select_scenario(i))

        speed_y = y + math.ceil(len(SCENARIOS) / 2) * 42 + 20
        self._button(20, speed_y, 78, 34, "VX -", lambda: self.change_speed(0, -0.05))
        self._button(105, speed_y, 78, 34, "VX +", lambda: self.change_speed(0, 0.05))
        self._button(205, speed_y, 78, 34, "VY -", lambda: self.change_speed(1, -0.05))
        self._button(290, speed_y, 78, 34, "VY +", lambda: self.change_speed(1, 0.05))
        self._button(390, speed_y, 68, 34, "Yaw -", lambda: self.change_speed(2, -0.10))
        self._button(465, speed_y, 78, 34, "Yaw +", lambda: self.change_speed(2, 0.10))

        action_y = speed_y + 52
        self._button(20, action_y, 120, 38, "开始 / 继续", self.resume)
        self._button(150, action_y, 100, 38, "暂停", self.pause)
        self._button(260, action_y, 100, 38, "停止", self.stop)
        self._button(370, action_y, 173, 38, "复位当前障碍", self.reset)

        rate_y = action_y + 54
        for index, rate in enumerate((0.5, 1.0, 2.0)):
            self._button(20 + index * 105, rate_y, 92, 34, f"仿真 {rate:g}x",
                         lambda r=rate: self.set_sim_speed(r))
        self._button(335, rate_y, 100, 34, "鼠标画线", self.toggle_draw_mode)
        self._button(440, rate_y, 103, 34, "取消导航", self.cancel_navigation)
        self.status_y = rate_y + 62

    def _scenario_spawn(self, index: int) -> tuple[float, float, float, float]:
        x, y, _, yaw = SCENARIOS[index].spawn
        return (x, y, self.initial_spawn[2], yaw)

    def select_scenario(self, index: int) -> None:
        self.selected_scenario = index
        self.cancel_navigation()
        self.sim.reset(self._scenario_spawn(index))
        self.command.value[:] = (0.0, 0.0, 0.0)
        self.paused = False

    def change_speed(self, index: int, delta: float) -> None:
        limit = float(self.command.limits[index])
        self.command.value[index] = float(
            np_clip(self.command.value[index] + delta, -limit, limit)
        )

    def resume(self) -> None:
        self.paused = False

    def pause(self) -> None:
        self.paused = True

    def stop(self) -> None:
        self.cancel_navigation()
        self.command.value[:] = 0.0
        self.paused = True

    def reset(self) -> None:
        self.cancel_navigation()
        self.sim.reset(self._scenario_spawn(self.selected_scenario))
        self.command.value[:] = 0.0
        self.paused = False

    def set_sim_speed(self, value: float) -> None:
        self.sim_speed = value

    def toggle_draw_mode(self) -> None:
        self.draw_mode = not self.draw_mode
        if not self.draw_mode:
            self.drawing = False
            self.drawing_points.clear()

    def cancel_navigation(self) -> None:
        if self.navigation is not None:
            self.navigation.cancel()
        self.drawing = False
        self.drawing_points.clear()
        self.command.value[:] = 0.0

    def navigation_command(self):
        if self.navigation is None or not self.navigation.active:
            return self.command.value.copy()
        qpos = self.sim.d.qpos
        yaw = math.atan2(
            2.0 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
            1.0 - 2.0 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
        )
        return self.navigation.update(float(qpos[0]), float(qpos[1]), yaw)

    def _start_draw(self, position: tuple[int, int]) -> None:
        if self.navigation is None:
            return
        self.drawing = True
        self.drawing_points = [self._screen_to_world(*position)]
        self.navigation.cancel()

    def _finish_draw(self) -> None:
        if self.navigation is None or len(self.drawing_points) < 2:
            self.drawing = False
            self.drawing_points.clear()
            return
        waypoints = []
        for point in self.drawing_points:
            if not waypoints or math.dist(point, waypoints[-1][:2]) >= 0.25:
                waypoints.append((point[0], point[1], 0.25, "rough"))
        if waypoints:
            self.navigation.start_mission("Mouse path", waypoints)
            self.command.value[:] = 0.0
        self.drawing = False
        self.drawing_points.clear()

    def poll(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if MAP_RECT.collidepoint(event.pos):
                    if self.draw_mode:
                        self._start_draw(event.pos)
                    elif self.navigation is not None:
                        wx, wy = self._screen_to_world(*event.pos)
                        self.navigation.start_target(wx, wy)
                        self.command.value[:] = 0.0
                else:
                    for button in self.buttons:
                        if button.click(event.pos):
                            break
            elif event.type == pygame.MOUSEMOTION and self.drawing and MAP_RECT.collidepoint(event.pos):
                point = self._screen_to_world(*event.pos)
                if not self.drawing_points or math.dist(point, self.drawing_points[-1]) >= 0.12:
                    self.drawing_points.append(point)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.drawing:
                self._finish_draw()

    def _draw_map(self) -> None:
        pygame.draw.rect(self.surface, (13, 20, 32), MAP_RECT)
        pygame.draw.rect(self.surface, (70, 92, 115), MAP_RECT, width=1)

        xmin, xmax, ymin, ymax = MAP_BOUNDS
        for x in range(math.ceil(xmin), math.floor(xmax) + 1):
            sx, _ = self._world_to_screen(x, ymin)
            pygame.draw.line(self.surface, (28, 42, 61), (sx, MAP_RECT.top), (sx, MAP_RECT.bottom), 1)
        for y in range(math.ceil(ymin), math.floor(ymax) + 1):
            _, sy = self._world_to_screen(xmin, y)
            pygame.draw.line(self.surface, (28, 42, 61), (MAP_RECT.left, sy), (MAP_RECT.right, sy), 1)

        for name, geom_type, x, y, sx, sy, color in self.map_geometries:
            px, py = self._world_to_screen(x, y)
            half_x = max(2, int(sx / (xmax - xmin) * MAP_RECT.width))
            half_y = max(2, int(sy / (ymax - ymin) * MAP_RECT.height))
            if geom_type == "cylinder":
                pygame.draw.circle(self.surface, color, (px, py), max(2, half_x))
            else:
                pygame.draw.rect(
                    self.surface,
                    color,
                    pygame.Rect(px - half_x, py - half_y, half_x * 2, half_y * 2),
                )

        if self.navigation is not None and len(self.navigation.drawn_path) > 1:
            points = [self._world_to_screen(x, y) for x, y in self.navigation.drawn_path]
            pygame.draw.lines(self.surface, (16, 185, 129), False, points, 3)
            for point in points:
                pygame.draw.circle(self.surface, (6, 182, 212), point, 4)

        if len(self.drawing_points) > 1:
            points = [self._world_to_screen(x, y) for x, y in self.drawing_points]
            pygame.draw.lines(self.surface, (245, 158, 11), False, points, 3)

        qpos = self.sim.d.qpos
        robot_x, robot_y = self._world_to_screen(float(qpos[0]), float(qpos[1]))
        pygame.draw.circle(self.surface, (6, 182, 212), (robot_x, robot_y), 7)
        yaw = math.atan2(
            2.0 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
            1.0 - 2.0 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
        )
        nose_x = robot_x + int(math.cos(yaw) * 16)
        nose_y = robot_y - int(math.sin(yaw) * 16)
        pygame.draw.line(self.surface, (16, 185, 129), (robot_x, robot_y), (nose_x, nose_y), 3)

        if self.navigation is not None and self.navigation.active:
            target = self._world_to_screen(self.navigation.target_x, self.navigation.target_y)
            pygame.draw.circle(self.surface, (234, 179, 8), target, 9, width=2)
            pygame.draw.line(self.surface, (234, 179, 8), (target[0] - 12, target[1]), (target[0] + 12, target[1]), 1)
            pygame.draw.line(self.surface, (234, 179, 8), (target[0], target[1] - 12), (target[0], target[1] + 12), 1)

    def draw(self) -> None:
        bg = (20, 25, 32)
        panel = (29, 36, 47)
        self.surface.fill(bg)
        pygame.draw.rect(self.surface, panel, (12, 12, 536, 736), border_radius=8)
        pygame.draw.rect(self.surface, panel, (558, 12, 530, 736), border_radius=8)
        self.surface.blit(self.title.render("Go2W Sim2Sim 控制面板", True, (236, 244, 250)), (20, 25))
        self.surface.blit(self.small.render("选择障碍车道（地图仍为完整 27rc 障碍赛）", True, (153, 170, 184)), (20, 68))
        self.surface.blit(self.title.render("路线规划地图", True, (236, 244, 250)), (580, 25))
        self.surface.blit(self.small.render("点击地图设置目标；开启鼠标画线后拖动生成航点", True, (153, 170, 184)), (580, 60))
        self._draw_map()

        for index, button in enumerate(self.buttons):
            selected = index == self.selected_scenario
            button.draw(self.surface, self.small, selected=selected)

        scenario = SCENARIOS[self.selected_scenario]
        status_y = self.status_y
        pygame.draw.line(self.surface, (67, 80, 94), (20, status_y - 14), (540, status_y - 14), 1)
        lines = (
            f"当前障碍: {scenario.name}",
            f"指令 vx={self.command.value[0]:+.2f}  vy={self.command.value[1]:+.2f}  yaw={self.command.value[2]:+.2f}",
            f"仿真倍率: {self.sim_speed:g}x    状态: {'暂停' if self.paused else '运行'}",
            f"位置 x={self.sim.d.qpos[0]:+.2f}  y={self.sim.d.qpos[1]:+.2f}  z={self.sim.d.qpos[2]:.3f}",
            f"碰撞={self.sim.collision_count}    摔倒={'是' if self.sim.fallen else '否'}",
        )
        for index, line in enumerate(lines):
            color = (226, 234, 242) if index == 0 else (164, 180, 193)
            self.surface.blit(self.small.render(line, True, color), (20, status_y + index * 28))
        if self.navigation is not None:
            nav_status = "AUTO" if self.navigation.active else "MANUAL"
            target = "-" if self.navigation.target_x is None else f"({self.navigation.target_x:.2f}, {self.navigation.target_y:.2f})"
            self.surface.blit(self.small.render(f"导航={nav_status}  目标={target}", True, (164, 180, 193)), (20, status_y + 145))
        self.surface.blit(self.small.render("MuJoCo 窗口仍显示物理仿真；面板按钮立即生效", True, (111, 135, 150)), (20, 716))
        self.surface.blit(self.small.render("也可用键盘 W/S/A/D/Q/E/X 控制速度", True, (111, 135, 150)), (20, 740))
        pygame.display.flip()
        self.clock.tick(30)

    def close(self) -> None:
        pygame.quit()


def np_clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
