"""P5 V2 receive curriculum (P5_RECEIVE_DESIGN_V2 4.1, 4.3). Pure Python, no Isaac import.

* **Stages** R0 (footwork and turn) -> R1 (touch) -> R2 (return), gated on EMAs (``ema_iters``) of the training
  receive rates (``ReceiveRewardComputer.pop_iteration``): R0 needs ``st_err_p50 <= 0.35`` m,
  ``tf_err_p50 <= 35`` deg and ``fell <= 0.03``; R1 needs ``touch >= 0.5``; R2 runs to the end. A stage passes
  its gate after ``min_iters``; at ``max_iters`` it is promoted anyway (event ``max_<stage>``, the monitor alerts).
* **Weights** of the receive terms switch linearly from the previous stage over ``blend_iters`` after a promotion.
* **Group weights** ``w_g`` of the multi-critic advantage (V2 2.5, order ``receive.REWARD_GROUPS`` = serve, recv,
  fall, body): R0 1.0 / 1.0 / 0.8 / 0.2, R1 and R2 1.0 / 1.5 / 0.5 / 0.2; blended with the receive weights.
* **Face tolerance** (flat part of the recv_face heading factor) anneals per stage from ``face_tol[s][0]`` to
  ``face_tol[s][1]`` over ``face_anneal_iters`` iterations of the stage (R1 90 -> 45, R2 45 -> 25); its own counter,
  independent of the blend.
* **Feeder level**: L0-L2 (``early_level``) in R0 and R1, L3 from R2 (``r2_level``); the environment draws a level
  uniformly in ``0..level`` per rally.
* **Natural share** fixed at 0.5, clamped to ``[natural_min, natural_max]`` = [0.4, 0.6]; can be set from outside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .receive import NUM_GROUPS, RECV_STAGE_WEIGHTS, RECV_STAGES, RECV_TERMS, REWARD_GROUPS


@dataclass
class StageSpec:
    min_iters: int = 300
    max_iters: int | None = None
    gate: dict | None = None  # metric -> (">=" | "<=", threshold); None: no promotion


DEFAULT_STAGES = (
    StageSpec(300, 1500, {"st_err_p50": ("<=", 0.35), "tf_err_p50": ("<=", 35.0), "fell": ("<=", 0.03)}),
    StageSpec(300, 2000, {"touch": (">=", 0.5)}),
    StageSpec(300, None, None),
)
DEFAULT_FACE_TOL = ((90.0, 90.0), (90.0, 45.0), (45.0, 25.0))
# V2 2.5: per stage, (w_serve, w_recv, w_fall, w_body).
DEFAULT_GROUP_WEIGHTS = {"R0": (1.0, 1.0, 0.8, 0.2), "R1": (1.0, 1.5, 0.5, 0.2), "R2": (1.0, 1.5, 0.5, 0.2)}
LOGGED_EMA = ("st_err_p50", "tf_err_p50", "tf_lit_p50", "fell", "touch", "crossed", "in", "f_touch", "n_touch",
              "f_crossed", "n_crossed", "hd_abs_med", "vc_med")


@dataclass
class P5Curriculum:
    stages: tuple = DEFAULT_STAGES
    ema_iters: int = 200
    min_rallies: float = 32.0  # receive rallies an iteration needs to move the EMA
    blend_iters: int = 300
    face_tol: tuple = DEFAULT_FACE_TOL
    face_anneal_iters: int = 1000
    group_weight_table: dict = field(default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_GROUP_WEIGHTS.items()})
    early_level: int = 2
    r2_level: int = 3
    natural_min: float = 0.4
    natural_max: float = 0.6
    # State.
    stage: int = 0
    iters_in_stage: int = 0
    iters: int = 0
    blend_from: int | None = None
    natural_share: float = 0.5
    ema: dict = field(default_factory=dict)
    ema_count: int = 0
    history: list = field(default_factory=list)
    alerted: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def __post_init__(self):
        self.natural_share = self.clamp_share(self.natural_share)
        for k, v in self.group_weight_table.items():
            if k not in RECV_STAGES or len(v) != NUM_GROUPS:
                raise ValueError(f"group weights {k}: {v} (need {NUM_GROUPS} for stages {RECV_STAGES})")

    @property
    def alpha(self) -> float:
        return 2.0 / (self.ema_iters + 1.0)

    @property
    def stage_name(self) -> str:
        return RECV_STAGES[self.stage]

    @property
    def level(self) -> int:
        return self.r2_level if self.stage >= 2 else self.early_level

    def clamp_share(self, x: float) -> float:
        return float(min(max(x, self.natural_min), self.natural_max))

    def set_natural_share(self, x: float) -> bool:
        new = self.clamp_share(x)
        changed = abs(new - self.natural_share) > 1e-9
        self.natural_share = new
        return changed

    def blend(self) -> float:
        """1 = the current stage's weights; < 1 during the switch from ``blend_from``."""
        if self.blend_from is None or self.blend_iters <= 0:
            return 1.0
        return min(self.iters_in_stage / float(self.blend_iters), 1.0)

    def weights(self) -> dict:
        cur = RECV_STAGE_WEIGHTS[self.stage_name]
        f = self.blend()
        if f >= 1.0:
            return {k: float(cur[k]) for k in RECV_TERMS}
        prev = RECV_STAGE_WEIGHTS[RECV_STAGES[self.blend_from]]
        return {k: (1.0 - f) * prev[k] + f * cur[k] for k in RECV_TERMS}

    def group_weights(self) -> tuple:
        """``w_g`` (``REWARD_GROUPS`` order), blended like the receive weights."""
        cur = self.group_weight_table[self.stage_name]
        f = self.blend()
        if f >= 1.0:
            return tuple(float(x) for x in cur)
        prev = self.group_weight_table[RECV_STAGES[self.blend_from]]
        return tuple((1.0 - f) * a + f * b for a, b in zip(prev, cur))

    def face_tolerance(self) -> float:
        a, b = self.face_tol[self.stage]
        f = 1.0 if self.face_anneal_iters <= 0 else min(self.iters_in_stage / float(self.face_anneal_iters), 1.0)
        return a + (b - a) * f

    def gate_ok(self) -> bool:
        gate = self.stages[self.stage].gate
        if gate is None:
            return False
        for key, (op, thr) in gate.items():
            v = self.ema.get(key)
            if v is None or not math.isfinite(v) or (op == ">=" and v < thr) or (op == "<=" and v > thr):
                return False
        return True

    def _promote(self, iteration: int, why: str) -> None:
        self.history.append(dict(stage=self.stage_name, iteration=iteration, iters=self.iters_in_stage, why=why,
                                 ema={k: round(v, 4) for k, v in self.ema.items()}))
        self.events.append(f"{why}_{self.stage_name}")
        self.blend_from = self.stage
        self.stage += 1
        self.iters_in_stage = 0

    def update(self, iteration: int, rates: dict) -> dict:
        """Returns the receive weights for the next iteration; ``events`` lists what happened."""
        self.events = []
        self.iters += 1
        self.iters_in_stage += 1
        if rates.get("rallies", 0.0) >= self.min_rallies:
            a = self.alpha if self.ema_count > 0 else 1.0
            for k, v in rates.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    self.ema[k] = a * v + (1.0 - a) * self.ema.get(k, v)
            self.ema_count += 1
        spec = self.stages[self.stage]
        last = self.stage >= len(self.stages) - 1
        enough = self.iters_in_stage >= spec.min_iters and self.ema_count >= min(self.ema_iters, spec.min_iters)
        if not last and enough and self.gate_ok():
            self._promote(iteration, "passed")
        elif not last and spec.max_iters is not None and self.iters_in_stage >= spec.max_iters:
            self.alerted.append(f"max_{self.stage_name}")
            self._promote(iteration, "max")
        return self.weights()

    def log_values(self) -> dict:
        out = {f"P5Curr/ema_{k}": v for k, v in self.ema.items() if k in LOGGED_EMA}
        out.update({"P5Curr/stage": float(self.stage), "P5Curr/iters_in_stage": float(self.iters_in_stage),
                    "P5Curr/blend": self.blend(), "P5Curr/face_tol": self.face_tolerance(),
                    "P5Curr/level": float(self.level), "P5Curr/natural_share": self.natural_share})
        out.update({f"P5Curr/w_{g}": w for g, w in zip(REWARD_GROUPS, self.group_weights())})
        return out

    def state_dict(self) -> dict:
        return dict(stage=self.stage, iters_in_stage=self.iters_in_stage, iters=self.iters, blend_from=self.blend_from,
                    natural_share=self.natural_share, ema=dict(self.ema), ema_count=self.ema_count,
                    history=list(self.history), alerted=list(self.alerted))

    def load_state_dict(self, d: dict) -> None:
        for k, v in d.items():
            if k in ("level", "iters_in_level"):  # V1 checkpoints
                continue
            setattr(self, k, v)
        self.natural_share = self.clamp_share(self.natural_share)
