"""Exception hierarchy. Messages must never contain private (truth) data."""
from __future__ import annotations


class TraceForgeError(Exception):
    pass


class ConfigError(TraceForgeError):
    pass


class GenerationError(TraceForgeError):
    pass


class EpisodeValidationError(TraceForgeError):
    pass


class EnvError(TraceForgeError):
    pass


class NotReset(EnvError):
    pass


class EpisodeFinished(EnvError):
    pass


class CalculatorError(TraceForgeError):
    pass


class LeakError(TraceForgeError):
    pass


class AgentProtocolError(TraceForgeError):
    pass
