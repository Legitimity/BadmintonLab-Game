"""Stage curriculum for the serve task (design doc V3 section 7.3). Pure Python, no Isaac import.

Per iteration the runner calls :meth:`ServeCurriculum.update` with the iteration's rates
from :meth:`ServeRewardComputer.pop_iteration`. The curriculum keeps an EMA per metric,
promotes when the stage's gate holds for the EMA after the minimum iterations, blends
weights linearly over ``blend_iters`` iterations, and requests a stop when a stage reaches
its maximum without promotion (no automatic promotion, V3 7.3). Passing the C gate marks
the curriculum ``finished`` (M1 reached in training metrics); training continues in C.
``state_dict`` goes into
the checkpoint ``infos`` so a resumed run continues in the same stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .serve_reward import STAGE_WEIGHTS, TERMS

ORDER = ("A", "B", "C")


@dataclass
class StageSpec:
    min_iters: int
    max_iters: int
    # metric -> (">=" or "<=", threshold)
    gate: dict


# Gates revised after P0-P2 (REPORT_P0_P2_RESULTS): the scripted swing of the default pose (candidate 2)
# reaches 47% in with 34% pass-through, and the contact model caps a single-contact serve at about
# 85-90% in under realistic noise. "fall" is the server's fall rate (root < 0.45 m or trunk > 45 deg).
DEFAULT_STAGES = {
    "A": StageSpec(200, 1500, {"legal": (">=", 0.70), "fall": ("<=", 0.10)}),
    "B": StageSpec(300, 4000, {"crossed": (">=", 0.70), "in": (">=", 0.60), "fall": ("<=", 0.10)}),
    "C": StageSpec(300, 4000, {"in": (">=", 0.85), "fault": ("<=", 0.02), "fall": ("<=", 0.05)}),
}


@dataclass
class ServeCurriculum:
    stages: dict = field(default_factory=lambda: dict(DEFAULT_STAGES))
    ema_iters: int = 200
    blend_iters: int = 300
    min_rallies: float = 64.0  # an iteration with fewer resolved rallies does not move the EMA
    stage: str = "A"
    iters_in_stage: int = 0
    blend_from: dict | None = None
    blend_left: int = 0
    ema: dict = field(default_factory=dict)
    ema_count: int = 0
    finished: bool = False
    stop_reason: str | None = None
    history: list = field(default_factory=list)
    events: list = field(default_factory=list)

    @property
    def alpha(self) -> float:
        return 2.0 / (self.ema_iters + 1.0)

    def weights(self) -> dict:
        target = STAGE_WEIGHTS[self.stage]
        if not self.blend_from or self.blend_left <= 0:
            return dict(target)
        f = self.blend_left / float(self.blend_iters)
        # A term missing in blend_from (checkpoint from before the term existed) blends from its target.
        return {k: f * self.blend_from.get(k, target[k]) + (1.0 - f) * target[k] for k in TERMS}

    def update(self, iteration: int, rates: dict) -> dict:
        """Returns the weights for the next iteration. Sets ``stop_reason`` to stop the run.

        ``self.events`` lists what happened in this call (``passed_A``, ..., ``finished``) so the
        runner can write a named checkpoint.
        """
        self.events = []
        n_hist = len(self.history)
        was_finished = self.finished
        if self.blend_left > 0:
            self.blend_left -= 1
        self.iters_in_stage += 1
        if rates.get("rallies", 0.0) >= self.min_rallies:
            a = self.alpha if self.ema_count > 0 else 1.0
            for k, v in rates.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    self.ema[k] = a * v + (1.0 - a) * self.ema.get(k, v)
            self.ema_count += 1
        spec = self.stages[self.stage]
        if self.finished:
            return self.weights()
        if self.iters_in_stage >= spec.min_iters and self.ema_count >= min(self.ema_iters, spec.min_iters) and self.gate_ok():
            self._promote(iteration)
        elif self.iters_in_stage >= spec.max_iters:
            self.stop_reason = f"stage_{self.stage}_max_iters"
        if len(self.history) > n_hist:
            self.events.append(f"passed_{self.history[-1]['stage']}")
        if self.finished and not was_finished:
            self.events.append("finished")
        return self.weights()

    def gate_ok(self) -> bool:
        for key, (op, thr) in self.stages[self.stage].gate.items():
            v = self.ema.get(key)
            if v is None:
                return False
            if (op == ">=" and v < thr) or (op == "<=" and v > thr):
                return False
        return True

    def _promote(self, iteration: int) -> None:
        self.history.append(dict(stage=self.stage, iteration=iteration, iters=self.iters_in_stage,
                                 ema={k: round(v, 4) for k, v in self.ema.items()}))
        i = ORDER.index(self.stage)
        if i + 1 >= len(ORDER):
            self.finished = True
            return
        self.blend_from = self.weights()
        self.blend_left = self.blend_iters
        self.stage = ORDER[i + 1]
        self.iters_in_stage = 0

    def log_values(self) -> dict:
        out = {f"Curriculum/ema_{k}": v for k, v in self.ema.items()}
        out["Curriculum/stage"] = float(ORDER.index(self.stage))
        out["Curriculum/iters_in_stage"] = float(self.iters_in_stage)
        out["Curriculum/blend_left"] = float(self.blend_left)
        return out

    def state_dict(self) -> dict:
        return dict(stage=self.stage, iters_in_stage=self.iters_in_stage, blend_from=self.blend_from,
                    blend_left=self.blend_left, ema=dict(self.ema), ema_count=self.ema_count,
                    finished=self.finished, history=list(self.history))

    def load_state_dict(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)
        self.stop_reason = None
