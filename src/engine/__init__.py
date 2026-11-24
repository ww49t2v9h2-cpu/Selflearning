"""智训引擎·自进化引擎层（B 实现）。

模块与 docs/03 §4 冻结契约一一对应：
    monitor.DriftMonitor / MonitorReport      §4.1 漂移监测
    diagnosis.Diagnoser                       §4.2 归因诊断
    playbooks.PlaybookLibrary/PlaybookContext §4.3 动作库
    planner.Planner                           §4.4 预算规划 + 策略层 bandit
    gates.Gate                                §4.5 上线门禁
    registry.ModelRegistry                    §4.6 模型仓库
    safety.SafetyGuard / AuditLog             §4.7 安全护栏与哈希链审计
    watchdog.Watchdog                         §4.8 上线看门狗
"""

from .monitor import DriftMonitor, MonitorReport
from .diagnosis import Diagnoser
from .playbooks import (ExperimentResult, InfeasiblePlaybook, PlaybookContext,
                        PlaybookLibrary)
from .planner import Planner
from .gates import Gate
from .registry import ModelRegistry
from .safety import AuditLog, SafetyGuard
from .watchdog import Watchdog
from .defaults import resolve_config, ENGINE_DEFAULTS

__all__ = [
    "DriftMonitor", "MonitorReport", "Diagnoser", "PlaybookLibrary",
    "PlaybookContext", "ExperimentResult", "InfeasiblePlaybook", "Planner",
    "Gate", "ModelRegistry", "SafetyGuard", "AuditLog", "Watchdog",
    "resolve_config", "ENGINE_DEFAULTS",
]
