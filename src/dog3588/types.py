from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class JointState:
    """Motor-space radians, rad/s and per-joint monotonic acquisition times."""

    position: np.ndarray
    velocity: np.ndarray
    stamps: np.ndarray


@dataclass(frozen=True)
class ImuState:
    """Body angular velocity (rad/s), unit downward gravity, monotonic sample time."""

    omega: np.ndarray
    gravity: np.ndarray
    stamp: float


class JointIO(Protocol):
    def start(self) -> None: ...
    def read(self) -> JointState: ...
    def command(self, targets: np.ndarray, now: float) -> None: ...
    def close(self) -> None: ...


class ImuIO(Protocol):
    def start(self) -> None: ...
    def read(self) -> ImuState: ...
    def close(self) -> None: ...

