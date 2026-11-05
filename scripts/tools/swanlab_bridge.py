"""Mirror an RSL-RL TensorBoard log directory to SwanLab from an isolated interpreter.

Why a sidecar: the SwanLab project requires SDK >= 0.9, which needs
``wrapt>=1.17``, while ``isaacsim-kernel 5.1.0`` pins ``wrapt==1.16.0``.  The
training interpreter therefore keeps TensorBoard only, and this script runs in
``/root/autodl-tmp/env_swanlab`` (swanlab 0.10.x + tensorboard), tailing the
event files and forwarding every scalar with its iteration as step.

Credentials come from ``swanlab login`` (``~/.swanlab``); nothing is read from
or written to the repository.

Stops when ``<logdir>/.swanlab_done`` appears, the parent process exits, or on
SIGTERM/SIGINT -- always after a final drain of the event files.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import time

import swanlab
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util

SKIP_SUFFIXES = ("/time",)  # wall-clock x-axis duplicates of Train/* scalars


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logdir", required=True)
    p.add_argument("--project", default="BadmintonLab-Game")
    p.add_argument("--workspace", default=None)
    p.add_argument("--name", required=True)
    p.add_argument("--group", default=None)
    p.add_argument("--config", default=None, help="JSON file with the run configuration.")
    p.add_argument("--mode", default="cloud", choices=("cloud", "local", "offline"))
    p.add_argument("--parent_pid", type=int, default=None)
    p.add_argument("--interval", type=float, default=15.0)
    return p.parse_args()


def _alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _value(v) -> float | None:
    if v.HasField("simple_value"):
        return float(v.simple_value)
    if v.HasField("tensor"):
        arr = tensor_util.make_ndarray(v.tensor)
        if arr.size == 1:
            return float(arr.reshape(()))
    return None


class Tailer:
    def __init__(self, logdir: str):
        self.logdir = logdir
        self.loaders: dict[str, EventFileLoader] = {}
        self.sent = 0

    def drain(self) -> int:
        for path in sorted(glob.glob(os.path.join(self.logdir, "events.out.tfevents.*"))):
            if path not in self.loaders:
                self.loaders[path] = EventFileLoader(path)
        batch: dict[int, dict[str, float]] = {}
        for loader in self.loaders.values():
            for event in loader.Load():
                if not event.HasField("summary"):
                    continue
                for v in event.summary.value:
                    if v.tag.endswith(SKIP_SUFFIXES):
                        continue
                    val = _value(v)
                    if val is not None:
                        batch.setdefault(int(event.step), {})[v.tag] = val
        for step in sorted(batch):
            swanlab.log(batch[step], step=step)
        self.sent += len(batch)
        return len(batch)


def main() -> None:
    args = _parse()
    config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            config = json.load(f)
    swanlab.init(
        project=args.project,
        workspace=args.workspace,
        experiment_name=args.name,
        group=args.group,
        mode=args.mode,
        config=config,
        logdir=os.path.join(args.logdir, "swanlog"),
    )
    stop = {"flag": False}

    def _on_signal(signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    tailer = Tailer(args.logdir)
    sentinel = os.path.join(args.logdir, ".swanlab_done")
    try:
        while not stop["flag"]:
            tailer.drain()
            if os.path.exists(sentinel) or not _alive(args.parent_pid):
                break
            for _ in range(int(args.interval * 10)):
                if stop["flag"] or os.path.exists(sentinel):
                    break
                time.sleep(0.1)
        time.sleep(1.0)
        tailer.drain()
    finally:
        print(f"[swanlab_bridge] forwarded {tailer.sent} steps from {args.logdir}", flush=True)
        swanlab.finish()


if __name__ == "__main__":
    main()
