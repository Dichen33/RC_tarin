from __future__ import annotations

from legged_gym.envs.htdw_4438.htdw_4438 import Htdw4438Env
from legged_gym.envs.htdw_4438.htdw_4438_config import (
    Htdw4438Cfg,
    Htdw4438CfgPPO,
    RobotCrawlCfg,
    RobotCrawlCfgPPO,
    RobotFlatCfg,
    RobotFlatCfgPPO,
    RobotRoughCfg,
    RobotRoughCfgPPO,
)
from legged_gym.utils.task_registry import task_registry


task_registry.register("htdw_4438", Htdw4438Env, Htdw4438Cfg, Htdw4438CfgPPO)
task_registry.register("Robot-Flat-v0", Htdw4438Env, RobotFlatCfg, RobotFlatCfgPPO)
task_registry.register("Robot-Rough-v0", Htdw4438Env, RobotRoughCfg, RobotRoughCfgPPO)
task_registry.register("Robot-Crawl-v0", Htdw4438Env, RobotCrawlCfg, RobotCrawlCfgPPO)
