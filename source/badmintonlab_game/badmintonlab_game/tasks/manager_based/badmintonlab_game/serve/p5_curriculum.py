"""P5 curriculum (P5_RECEIVE_DESIGN 5). Pure Python, no Isaac import.

* **Feeder level** L0..L3 by ``d_need``. EMA (``ema_iters``) of the feeder-rally receive rates
  (``f_touch``, ``f_crossed``); a level passes its gate after ``min_iters``; at ``max_iters`` without a pass the
  level stays and an alert event is raised once (no stop). L3 has no gate. ``max_level`` caps the level
  (the pilot runs L0-L1 only).
* **Reward stage** R1 for ``r1_iters`` iterations of the run, then a linear blend to R2 over ``blend_iters``.
* **Natural share** of the rallies, set from outside (the monitor's play evaluation, 5.2) and clamped to
  ``[natural_min, natural_max]``; ``natural_min`` may be 0 for the feeder-only pilot.

``update(iteration, rates)`` takes ``ReceiveRewardComputer.rates`` (keys ``f_rallies``, ``f_touch`` ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .receive import RECV_STAGE_WEIGHTS, RECV_TERMS


@dataclass
class LevelSpec:
    min_iters: int = 300
    max_iters: int = 3000
    gate: dict | None = None  # metric -> (">=" | "<=", threshold); None: no promotion


DEFAULT_LEVELS = (
    LevelSpec(gate={"f_touch": (">=", 0.6), "f_crossed": (">=", 0.3)}),
    LevelSpec(gate={"f_touch": (">=", 0.6), "f_crossed": (">=", 0.3)}),
    LevelSpec(gate={"f_touch": (">=", 0.5), "f_crossed": (">=", 0.25)}),
    LevelSpec(gate=None),
)


@dataclass
class P5Curriculum:
    levels: tuple = DEFAULT_LEVELS
    ema_iters: int = 200
    min_rallies: float = 32.0  # feeder rallies an iteration needs to move the EMA
    max_level: int = 3
    r1_iters: int = 1500
    blend_iters: int = 300
    natural_min: float = 0.3
    natural_max: float = 0.7
    # State.
    level: int = 0
    iters_in_level: int = 0
    iters: int = 0
    natural_share: float = 0.5
    ema: dict = field(default_factory=dict)
    ema_count: int = 0
    history: list = field(default_factory=list)
    alerted: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def __post_init__(self):
        self.natural_share = self.clamp_share(self.natural_share)
        self.level = min(self.level, self.max_level)

    @property
    def alpha(self) -> float:
        return 2.0 / (self.ema_iters + 1.0)

    def clamp_share(self, x: float) -> float:
        return float(min(max(x, self.natural_min), self.natural_max))

    def set_natural_share(self, x: float) -> bool:
        new = self.clamp_share(x)
        changed = abs(new - self.natural_share) > 1e-9
        self.natural_share = new
        return changed

    def stage_mix(self) -> float:
        """0 = R1, 1 = R2."""
        if self.iters < self.r1_iters:
            return 0.0
        if self.blend_iters <= 0:
            return 1.0
        return min((self.iters - self.r1_iters) / float(self.blend_iters), 1.0)

    def stage(self) -> str:
        f = self.stage_mix()
        return "R1" if f <= 0.0 else "R2" if f >= 1.0 else "R1>R2"

    def weights(self) -> dict:
        f = self.stage_mix()
        a, b = RECV_STAGE_WEIGHTS["R1"], RECV_STAGE_WEIGHTS["R2"]
        return {k: (1.0 - f) * a[k] + f * b[k] for k in RECV_TERMS}

    def gate_ok(self) -> bool:
        gate = self.levels[self.level].gate
        if gate is None:
            return False
        for key, (op, thr) in gate.items():
            v = self.ema.get(key)
            if v is None or (op == ">=" and v < thr) or (op == "<=" and v > thr):
                return False
        return True

    def update(self, iteration: int, rates: dict) -> dict:
        """Returns the receive weights for the next iteration; ``events`` lists what happened."""
        self.events = []
        self.iters += 1
        self.iters_in_level += 1
        if rates.get("f_rallies", 0.0) >= self.min_rallies:
            a = self.alpha if self.ema_count > 0 else 1.0
            for k, v in rates.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    self.ema[k] = a * v + (1.0 - a) * self.ema.get(k, v)
            self.ema_count += 1
        spec = self.levels[self.level]
        enough = self.iters_in_level >= spec.min_iters and self.ema_count >= min(self.ema_iters, spec.min_iters)
        if self.level < self.max_level and enough and self.gate_ok():
            self.history.append(dict(level=self.level, iteration=iteration, iters=self.iters_in_level,
                                     ema={k: round(v, 4) for k, v in self.ema.items()}))
            self.events.append(f"passed_L{self.level}")
            self.level += 1
            self.iters_in_level = 0
        elif self.iters_in_level >= spec.max_iters and spec.gate is not None and self.level < self.max_level:
            key = f"max_L{self.level}"
            if key not in self.alerted:
                self.alerted.append(key)
                self.events.append(key)
        if self.r1_iters <= self.iters < self.r1_iters + 1:
            self.events.append("R2_blend_start")
        return self.weights()

    def log_values(self) -> dict:
        out = {f"P5Curr/ema_{k}": v for k, v in self.ema.items() if k in ("f_touch", "f_crossed", "f_in", "f_fell",
                                                                          "n_touch", "n_crossed", "touch", "crossed")}
        out.update({"P5Curr/level": float(self.level), "P5Curr/iters_in_level": float(self.iters_in_level),
                    "P5Curr/stage_mix": self.stage_mix(), "P5Curr/natural_share": self.natural_share})
        return out

    def state_dict(self) -> dict:
        return dict(level=self.level, iters_in_level=self.iters_in_level, iters=self.iters,
                    natural_share=self.natural_share, ema=dict(self.ema), ema_count=self.ema_count,
                    history=list(self.history), alerted=list(self.alerted))

    def load_state_dict(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)
        self.level = min(self.level, self.max_level)
        self.natural_share = self.clamp_share(self.natural_share)
