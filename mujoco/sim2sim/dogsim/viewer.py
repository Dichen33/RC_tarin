"""Interactive MuJoCo viewer plus the pygame v2-course control panel."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import RunConfig
from .evaluators import ObstacleResult
from .model_catalog import ModelCatalog
from .navigation import NavigationController
from .policies import PolicyBackend, load_policy
from .scene import compose_scene
from .simulation import EpisodeRunner


def _make_panel(
    runner: EpisodeRunner,
    scene,
    use_panel: bool,
    model_catalog: ModelCatalog | None = None,
    model_loader=None,
):
    if not use_panel:
        return None
    try:
        from .panel import ControlPanel

        checkpoint = runner.policy.contract.checkpoint
        return ControlPanel(
            runner,
            scene,
            navigation=NavigationController(),
            model_catalog=model_catalog,
            model_loader=model_loader,
            current_model=checkpoint,
            residual_scale=runner.cfg.policy.residual_scale,
        )
    except Exception as exc:  # noqa: BLE001 - UI optional, physics still runs
        print(f"[UI] pygame panel unavailable: {exc}")
        print("[UI] Continue with MuJoCo viewer only; install pygame for the panel.")
        return None


def load_model_into_runner(cfg: RunConfig, runner: EpisodeRunner, entry, residual_scale: float):
    """Load a catalog checkpoint, validate the contract and swap it in."""

    old_checkpoint = cfg.policy.checkpoint
    old_scale = cfg.policy.residual_scale
    cfg.policy.checkpoint = Path(entry.path)
    cfg.policy.residual_scale = float(residual_scale)
    try:
        new_policy = load_policy(cfg)
        old_contract = runner.policy.contract
        new_contract = new_policy.contract
        if (
            new_contract.action_dim != old_contract.action_dim
            or new_contract.proprio_dim != old_contract.proprio_dim
            or new_contract.history_length != old_contract.history_length
            or new_contract.scan_dim != old_contract.scan_dim
            or new_contract.scan_size != old_contract.scan_size
        ):
            raise ValueError(
                f"observation/action contract mismatch: old={old_contract}, new={new_contract}"
            )
        runner.set_policy(new_policy)
        runner.reset()
        return True, f"已加载 {entry.name} (iter={entry.iteration_text}, scale={residual_scale:g})"
    except Exception as exc:  # noqa: BLE001 - report in the panel
        cfg.policy.checkpoint = old_checkpoint
        cfg.policy.residual_scale = old_scale
        return False, f"加载失败: {exc}"


def run_viewer(
    cfg: RunConfig,
    policy: PolicyBackend,
    scene_xml=None,
    use_panel: bool = True,
) -> ObstacleResult:
    scene = compose_scene(cfg, scene_xml)
    runner = EpisodeRunner(scene, cfg, policy, seed=cfg.simulation.randomization.seed)
    runner.reset()
    model_catalog = ModelCatalog() if use_panel else None

    def load_model(entry, residual_scale: float):
        return load_model_into_runner(cfg, runner, entry, residual_scale)

    panel = _make_panel(runner, scene, use_panel, model_catalog=model_catalog, model_loader=load_model)
    paused = False

    def key_callback(key: int) -> None:
        nonlocal paused
        if panel is not None:
            panel.key(key)
            paused = panel.paused
            return
        try:
            char = chr(key).lower()
        except (TypeError, ValueError):
            return
        command = runner.command.copy()
        if char in {"w", "s", "a", "d", "q", "e", "x"}:
            command[:] = 0.0
        if char == "w":
            command[0] = 0.25
        elif char == "s":
            command[0] = -0.20
        elif char == "a":
            command[1] = 0.18
        elif char == "d":
            command[1] = -0.18
        elif char == "q":
            command[2] = 0.45
        elif char == "e":
            command[2] = -0.45
        elif char == "r":
            runner.reset()
            paused = False
            return
        elif char == " ":
            paused = not paused
            return
        if char in {"w", "s", "a", "d", "q", "e", "x"}:
            runner.set_command(command)

    print("[Viewer] W/S forward, A/D lateral, Q/E yaw, X stop, Space pause, R reset")
    if panel is not None:
        print("[UI] obstacle presets, map click navigation, mouse-drawn waypoints enabled")
    from mujoco import viewer as mujoco_viewer

    duration = float(cfg.simulation.duration)
    try:
        with mujoco_viewer.launch_passive(scene.model, scene.data, key_callback=key_callback) as viewer:
            viewer.cam.distance = 4.5
            viewer.cam.elevation = -24.0
            viewer.cam.azimuth = 90.0
            viewer.cam.lookat[:] = scene.data.qpos[:3]
            while viewer.is_running():
                # ``duration`` is simulation time, matching headless eval.
                if duration > 0.0 and runner.time >= duration:
                    break
                if panel is not None:
                    panel.poll()
                    if panel.closed:
                        viewer.close()
                        break
                    paused = panel.paused
                    runner.set_command(panel.command_for_policy())
                started = time.perf_counter()
                finished = False
                if not paused:
                    finished = runner.step()
                viewer.cam.lookat[:] = scene.data.qpos[:3]
                viewer.sync()
                if panel is not None:
                    panel.draw()
                if finished:
                    state = runner.last_state
                    print(
                        f"[Viewer] episode finished t={runner.time:.2f}s "
                        f"along={0.0 if state is None else state.along:.2f} "
                        f"fallen={False if state is None else state.fallen}. "
                        "Press R to reset, or close the window to exit."
                    )
                    if panel is not None:
                        panel.paused = True
                    else:
                        break
                rate = panel.sim_speed if panel is not None else 1.0
                remaining = cfg.simulation.control_dt / max(rate, 0.1) - (time.perf_counter() - started)
                if remaining > 0.0 and not paused:
                    time.sleep(remaining)
    finally:
        if panel is not None:
            panel.close()
    return runner.evaluator.result(runner.time, runner.step_index)
