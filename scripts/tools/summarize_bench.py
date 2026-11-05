"""Summarise the P0 throughput benchmark written by ``p0_bench.sh``.

Reads each run's TensorBoard scalars (``Perf/collection time``,
``Perf/learning_time``) for iterations ``[--first, --last)`` and the sampled
GPU / cgroup memory, and prints a Markdown table.
"""

import argparse
import csv
import glob
import os
import re
import statistics

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

STEPS_PER_ENV = 24
PHYSX_PATTERNS = re.compile(r"overflow|exceeds|PxGpu|buffer.*(too small|capacity)", re.IGNORECASE)


def run_dir(log_root: str, run_name: str) -> str | None:
    dirs = sorted(glob.glob(os.path.join(log_root, f"*_{run_name}")))
    return dirs[-1] if dirs else None


def perf(path: str, first: int, last: int) -> dict:
    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()
    def window(tag):
        return [e.value for e in acc.Scalars(tag) if first <= e.step < last]
    coll, learn = window("Perf/collection time"), window("Perf/learning_time")
    return dict(coll=statistics.mean(coll), learn=statistics.mean(learn), n=len(coll))


def mem(path: str) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    gpu = [float(r["gpu_mem_mib"]) for r in rows if r["gpu_mem_mib"]]
    util = [float(r["gpu_util"]) for r in rows if r["gpu_util"]]
    cg = [float(r["cgroup_mem_bytes"]) for r in rows if r["cgroup_mem_bytes"]]
    busy = [u for u in util if u > 0] or [0.0]
    return dict(gpu_peak_gib=max(gpu) / 1024, util_mean=statistics.mean(busy), cg_peak_gib=max(cg) / 2**30)


def physx_warnings(log: str) -> int:
    with open(log, errors="ignore") as f:
        return sum(1 for line in f if PHYSX_PATTERNS.search(line))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", nargs="?", default="/root/autodl-tmp/logs/p0/bench")
    p.add_argument("--log_root", default="logs/rsl_rl/badmintonlab_game_serve")
    p.add_argument("--first", type=int, default=20)
    p.add_argument("--last", type=int, default=60)
    a = p.parse_args()
    groups = {"2048": [("bench_2048", 2048)], "4096": [("bench_4096", 4096)], "8192": [("bench_8192", 8192)],
              "dual2048": [("bench_dual2048_a", 2048), ("bench_dual2048_b", 2048)]}
    print("| config | iter s (coll+learn) | collection s | learning s | player samples/s | GPU peak GiB | GPU util % | cgroup peak GiB | PhysX warnings |")
    print("|---|---|---|---|---|---|---|---|---|")
    for cfg, runs in groups.items():
        mem_csv = os.path.join(a.out_dir, f"{cfg}.mem.csv")
        if not os.path.isfile(mem_csv):
            continue
        total_sps, iters, colls, learns, warns = 0.0, [], [], [], 0
        for name, n in runs:
            d = run_dir(a.log_root, name)
            if d is None:
                continue
            r = perf(d, a.first, a.last)
            it = r["coll"] + r["learn"]
            iters.append(it); colls.append(r["coll"]); learns.append(r["learn"])
            total_sps += 2 * n * STEPS_PER_ENV / it
            warns += physx_warnings(os.path.join(a.out_dir, f"{name}.log"))
        if not iters:
            continue
        m = mem(mem_csv)
        print(f"| {cfg} | {statistics.mean(iters):.2f} | {statistics.mean(colls):.2f} | {statistics.mean(learns):.2f} | "
              f"{total_sps:,.0f} | {m['gpu_peak_gib']:.1f} | {m['util_mean']:.0f} | {m['cg_peak_gib']:.1f} | {warns} |")


if __name__ == "__main__":
    main()
