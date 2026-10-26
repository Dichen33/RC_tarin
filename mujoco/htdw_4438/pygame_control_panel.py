import time
from dataclasses import dataclass

import mujoco
import numpy as np
import pygame
from mujoco import viewer

from ik_control import (
    DOWN_BASE_HEIGHT,
    MotionCommand,
    MODEL_PATH,
    STAND_BASE_HEIGHT,
    STRAIGHT_LEG_BASE_HEIGHT,
    VMCPositionIKController,
)


WIDTH, HEIGHT = 560, 470
BG = (24, 28, 34)
PANEL = (34, 40, 48)
TEXT = (230, 235, 242)
MUTED = (145, 154, 166)
ACCENT = (55, 135, 255)
ACCENT_DARK = (34, 82, 150)
TRACK = (74, 84, 98)
BUTTON = (52, 60, 72)
BUTTON_ACTIVE = (38, 112, 78)
WARNING = (188, 92, 62)


@dataclass
class Button:
    label: str
    mode: str
    rect: pygame.Rect

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, active_mode: str) -> None:
        color = BUTTON_ACTIVE if self.mode == active_mode else BUTTON
        if self.mode == "zero_velocity":
            color = ACCENT_DARK
        pygame.draw.rect(screen, color, self.rect, border_radius=6)
        pygame.draw.rect(screen, (86, 96, 110), self.rect, 1, border_radius=6)
        text = font.render(self.label, True, TEXT)
        screen.blit(text, text.get_rect(center=self.rect.center))

    def contains(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


@dataclass
class Slider:
    label: str
    min_value: float
    max_value: float
    value: float
    rect: pygame.Rect
    unit: str = ""
    dragging: bool = False

    def set_from_x(self, x: int) -> None:
        ratio = (x - self.rect.left) / max(1, self.rect.width)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        self.value = self.min_value + ratio * (self.max_value - self.min_value)

    def knob_x(self) -> int:
        ratio = (self.value - self.min_value) / (self.max_value - self.min_value)
        return int(self.rect.left + np.clip(ratio, 0.0, 1.0) * self.rect.width)

    def hit_knob_or_track(self, pos: tuple[int, int]) -> bool:
        knob = pygame.Rect(0, 0, 22, 22)
        knob.center = (self.knob_x(), self.rect.centery)
        track = self.rect.inflate(0, 18)
        return knob.collidepoint(pos) or track.collidepoint(pos)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, small: pygame.font.Font) -> None:
        title = font.render(self.label, True, TEXT)
        value = small.render(f"{self.value:+.3f} {self.unit}".rstrip(), True, MUTED)
        screen.blit(title, (self.rect.left, self.rect.top - 28))
        screen.blit(value, value.get_rect(topright=(self.rect.right, self.rect.top - 25)))

        pygame.draw.rect(screen, TRACK, self.rect, border_radius=4)
        fill = pygame.Rect(self.rect.left, self.rect.top, self.knob_x() - self.rect.left, self.rect.height)
        pygame.draw.rect(screen, ACCENT_DARK, fill, border_radius=4)
        pygame.draw.circle(screen, ACCENT, (self.knob_x(), self.rect.centery), 11)
        pygame.draw.circle(screen, (210, 225, 255), (self.knob_x(), self.rect.centery), 11, 1)


def make_widgets() -> tuple[list[Button], dict[str, Slider]]:
    buttons = [
        Button("Stand", "stand", pygame.Rect(32, 48, 144, 48)),
        Button("Trot", "trot", pygame.Rect(208, 48, 144, 48)),
        Button("Down", "down", pygame.Rect(384, 48, 144, 48)),
        Button("Zero", "zero_velocity", pygame.Rect(384, 104, 144, 36)),
    ]

    sliders = {
        "vx": Slider("X velocity", -0.18, 0.18, 0.0, pygame.Rect(48, 175, 464, 8), "m/s"),
        "vy": Slider("Y velocity", -0.09, 0.09, 0.0, pygame.Rect(48, 240, 464, 8), "m/s"),
        "yaw": Slider("Yaw rate", -1.2, 1.2, 0.0, pygame.Rect(48, 305, 464, 8), "rad/s"),
        "h": Slider("Standing height h", DOWN_BASE_HEIGHT, STRAIGHT_LEG_BASE_HEIGHT, STAND_BASE_HEIGHT, pygame.Rect(48, 370, 464, 8), "m"),
    }
    return buttons, sliders


def draw_panel(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small: pygame.font.Font,
    buttons: list[Button],
    sliders: dict[str, Slider],
    mode: str,
    contacts: int,
    base_z: float,
) -> None:
    screen.fill(BG)
    pygame.draw.rect(screen, PANEL, pygame.Rect(16, 20, WIDTH - 32, HEIGHT - 40), border_radius=8)

    for button in buttons:
        button.draw(screen, font, mode)

    for slider in sliders.values():
        slider.draw(screen, font, small)

    status_color = WARNING if base_z < 0.12 else MUTED
    status = small.render(f"mode={mode}  contacts={contacts}  base_z={base_z:.3f} m", True, status_color)
    screen.blit(status, (32, HEIGHT - 40))
    hint = small.render("Drag sliders. Buttons switch IK target mode.", True, MUTED)
    screen.blit(hint, hint.get_rect(right=WIDTH - 32, bottom=HEIGHT - 20))


def main() -> None:
    pygame.init()
    pygame.display.set_caption("HTDW-4438 IK Control")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Segoe UI", 20)
    small = pygame.font.SysFont("Segoe UI", 15)

    buttons, sliders = make_widgets()
    mode = "stand"

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    controller = VMCPositionIKController(model, ik_iterations=10)
    controller.reset(data)

    running = True
    sim_start = time.perf_counter()
    with viewer.launch_passive(model, data) as handle:
        while running and handle.is_running():
            t = time.perf_counter() - sim_start
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for button in buttons:
                        if button.contains(event.pos):
                            if button.mode == "zero_velocity":
                                sliders["vx"].value = 0.0
                                sliders["vy"].value = 0.0
                                sliders["yaw"].value = 0.0
                                mode = "stand"
                                controller.zero_motion(data, t, mode)
                                continue
                            mode = button.mode
                            if mode == "stand":
                                sliders["vx"].value = 0.0
                                sliders["vy"].value = 0.0
                                sliders["yaw"].value = 0.0
                                sliders["h"].value = STAND_BASE_HEIGHT
                            elif mode == "down":
                                sliders["vx"].value = 0.0
                                sliders["vy"].value = 0.0
                                sliders["yaw"].value = 0.0
                                sliders["h"].value = DOWN_BASE_HEIGHT
                    for slider in sliders.values():
                        if slider.hit_knob_or_track(event.pos):
                            slider.dragging = True
                            slider.set_from_x(event.pos[0])
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    for slider in sliders.values():
                        slider.dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    for slider in sliders.values():
                        if slider.dragging:
                            slider.set_from_x(event.pos[0])

            command = MotionCommand(
                mode,
                sliders["vx"].value,
                sliders["vy"].value,
                sliders["yaw"].value,
                sliders["h"].value,
            )
            controller.step(data, command, t)
            mujoco.mj_step(model, data)
            handle.sync()

            draw_panel(screen, font, small, buttons, sliders, mode, data.ncon, float(data.qpos[2]))
            pygame.display.flip()
            clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
