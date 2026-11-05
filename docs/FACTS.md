# FACTS：实测与源码核实的事实

> 依据：`TRAINING_DESIGN_V3.md` 第 0 节第 2 条。这里记录实测结果，以及经源码核实、与 V3 不一致或 V3 未写到的事实。以实测为准。
> 标记：**【实测】** 在本机 Isaac 中运行得到；**【代码】** 读源码确认。
> 环境：Isaac Sim 5.1.0、Isaac Lab 2.3.2、PyTorch 2.7.0+cu128、RSL-RL 3.1.2，RTX 4080 SUPER 32 GB（AutoDL）。

## 1. 与 V3 不一致的条目

| # | V3 原文 | 事实 | 依据 | 影响 |
|---|---|---|---|---|
| F1 | §2.1：手臂力矩上限 38.3 N·m | 显式执行器实际按 **18 N·m** 裁剪（URDF `effort="18"`）。38.3 只写入 PhysX 的 `effort_limit_sim`，并用于计算 `T1_ACTION_SCALE`。膝关节为 60 N·m。 | 【代码】`BoosterDelayedPDActuatorCfg` 未设置 `effort_limit`；Isaac Lab `actuator_base.py:215` 规定 `cfg.effort_limit is None` 时取 USD 值，`_clip_effort` 用的就是这个值 | P1 测挥拍速度时以 18 N·m 为准；可能达不到 §2.4 估算的 6–7 m/s |
| F2 | §4.3-4：无人击球 → 开球方负 | 判负方是**落点所在半场**的一方（`landing_side = x >= net_x`）。开球点固定在开球方半场，所以结果通常等于“开球方负”；球若被身体弹过网，则对方负。 | 【代码】play `_update_rally_result`；【实测】场景 4a/4b | 只影响边界情况；场景测试按“落点半场”判定 |
| F3 | §0-3、§11.6：SwanLab 固定 0.6.1，在进程内 `swanlab.init()` + `sync_tensorboard_torch()` | 项目 `@Legitimity/BadmintonLab-Game` 拒绝 0.6.1：`422 新版本项目请使用新版 SDK（版本号 >= 0.9.0）`。SwanLab 0.9.0 起依赖 `wrapt>=1.17`，而 `isaacsim-kernel 5.1.0` 固定 `wrapt==1.16.0`，两者无法装进同一环境。 | 【实测】API 返回值；`pip --dry-run` 报依赖冲突 | 改用旁路进程：独立 venv `/root/autodl-tmp/env_swanlab`（swanlab 0.10.4 + tensorboard）读取 TensorBoard 事件后上传，见 `scripts/tools/swanlab_bridge.py`。Isaac 环境内的包未作任何改动 |
| F4 | §4.4：内存上限“用户称 62 GB，brief 称约 503 GiB” | cgroup `memory.max = 66,571,993,088 B = 62.0 GiB`；`free -g` 显示的 503 GiB 是宿主机内存 | 【实测】`/sys/fs/cgroup/memory.max` | 规划时按 62 GiB 计算 |
| F5 | §2.2：main 的网碰撞过滤“可能没有生效” | **确认没有生效**：默认任务中，球以 4 m/s 飞向网，停在 `x=-0.07`，`vx=0`，10 个球场全部如此。Serve 任务和 play 中球直接穿过（`x_end=0.5`，`vx=4.0`）。 | 【实测】场景 8；main 的 FilteredPairsAPI 写在 `NetCollision`（Xform）上；Serve/play 写在 `NetCollision/geometry/mesh` 上，且每个球场的 target 都指向本场的 `Shuttle`（已检查 env_0 和 env_9） | 默认任务里球撞网会物理反弹并落回本方，`net_fault` 永远不会触发 |

## 2. 规则对齐（P0 门槛）【实测】

`scripts/tools/p0_rule_scenarios.py` 每个球场构造一个场景，10 个球场并行。机器人每个控制步都被重新写回固定位姿，动作为 0；场景 5–8 直接写入球的状态和击球方状态。

| 场景 | Serve 任务 | play | 默认任务（仅作对照） |
|---|---|---|---|
| 1 拍在 1.30 m 接球 | 超高，玩家 1 胜 ✔ | 同左 ✔ | 不判超高；1.306 m 的接触算合法击球，球落地后才结束 |
| 2 拍在 0.95 m 接球 | 合法击球，触球高度 0.956 m ✔ | 同左 ✔ | 合法击球（无 fault 字段） |
| 3 同一方二次触球 | 二次击球，玩家 1 胜 ✔ | 同左 ✔ | 错误接触被忽略，不判负 |
| 4a/4b 无人击球（开球方 0/1） | 落点方负 ✔ | 同左 ✔ | 一致 |
| 5a/5b 下网（P0/P1 击球） | 击球方负，第 9 步判定 ✔ | 同左 ✔ | 胜负相同，但原因不同：球被网弹回，第 30 步落地才判定 |
| 6a 空中越过边线 | 击球方负，第 4 步 ✔ | 同左 ✔ | 一致 |
| 6b 空中越过底线 | 击球方负，第 17 步 ✔ | 同左 ✔ | 一致 |
| 7 合法击球落入对方半场 | 击球方胜 ✔ | 同左 ✔ | 一致 |
| 8 球穿过网面不反弹 | 10/10 个球场穿过 ✔ | 10/10 ✔ | 0/10，全部被网挡住 |

Serve 任务与 play 的胜方、fault 标志、判定所在控制步和触球高度**逐项相同**。原始结果：`/root/autodl-tmp/logs/p0/scenarios_{serve,play,default}.json`。

play 环境测试方式：`PYTHONPATH=<play>/source/badmintonlab_game` 优先于 editable 安装生效，脚本用 `--expect_source` 断言 `badmintonlab_game.__file__` 来自 play 源码树。

## 3. 其他已核实事实

| # | 事实 | 依据 |
|---|---|---|
| F6 | play 的 `[RESULT]` 会打印两次：`MatchScore.update` 达到回合数时打印一次，`summary()` 结束时再打印一次 | 【代码】`match_score.py:29-40` |
| F7 | play 没有超时，只在 `rally_resolved` 时 reset。球若停在机器人身上不落地，回合永远不会结束 | 【代码】play `dual_agent_env.py:65` |
| F8 | play 的 reset 事件顺序是 player1 → player0 → shuttle（`DualEventsCfg(Player0EventsCfg, Player1EventsCfg)` 的 MRO）。`next_hitter` 由最后执行的 shuttle 事件写入，所以 player 事件中读到的 `next_hitter` 是上一回合的旧值。**`player_cfg.py` 的 reset 事件不能依赖本回合的发球方** | 【代码】 |
| F9 | main 的 `PPORunnerCfg` 未设置 `obs_groups`，RSL-RL 3.1 默认 actor 和 critic 都使用 `policy` 组 | 【代码】 |
| F10 | `reset_joints_by_scale` 是乘法，默认角为 0 的关节无法被扰动 | 【代码】 |
| F11 | 所有执行器组的延迟都按回合、按球场在 2–8 个物理步内随机，train 和 play 一致 | 【代码】`booster.py` |
| F12 | CCD 关闭；`sim.dt=0.005`，`decimation=4` | 【代码】 |
| F13 | 球拍碰撞盒为 0.01×0.14×0.2 m，中心在 `racket_link` 局部坐标 z=0.52 处，与 `racket_sweet_spot` 重合；拍面法线为该坐标系的 x 轴 | 【代码】URDF `racket_link` / `racket_to_sweet_spot` |
| F14 | 默认关节姿态下（机器人朝 +x），甜区约在根部前方 0.55 m、右侧 0.30 m，拍面法线约为 `(0.05, -0.96, 0.26)`，即拍面接近竖直、朝向机器人右侧，而不是朝向对方 | 【实测】场景 1 的 throw 记录 |
| F15 | 球以 2.5 m/s 沿法线撞向静止的球拍时，接触在 1 个控制步内被检测到，并且只算一次（上升沿） | 【实测】场景 2/3 |
| F16 | Serve 任务在 4 个球场上随机动作训练 3 轮正常；`Metrics/Rule/*` 和 `Metrics/Serve/*` 已记录。默认任务 2 轮 smoke 正常，没有回归 | 【实测】 |

## 4. 吞吐基准【实测】

测试对象为 `BadmintonLab-Game-Serve` 任务，每档跑 60 轮，取第 20–60 轮的均值。`num_steps_per_env=24`，每轮样本数 = 2 × 球场数 × 24。脚本为 `scripts/tools/p0_bench.sh`，汇总脚本为 `summarize_bench.py`，原始日志在 `/root/autodl-tmp/logs/p0/bench/`。

| 配置 | 每轮 s | 采集 s | 学习 s | 玩家样本/s（合计） | GPU 显存峰值 GiB | GPU 利用率 % | cgroup 内存峰值 GiB | PhysX 溢出 |
|---|---|---|---|---|---|---|---|---|
| 2048 | 5.09 | 4.26 | 0.83 | 19,304 | 4.7 | 27 | 8.4 | 0 |
| 4096 | 5.83 | 5.01 | 0.82 | 33,731 | 6.7 | 33 | 10.8 | **2** |
| 8192 | 7.44 | 6.60 | 0.84 | 52,833 | 9.9 | 42 | 15.5 | **2** |
| 2×2048 双进程 | 7.91（每个进程） | 7.05 | 0.85 | 24,871 | 9.4 | 88 | 13.8 | 0 |

说明：

- GPU 利用率是 `nvidia-smi` 每 2 秒采样、去掉 0 值后的均值。双进程时两个进程的利用率会叠加计入。
- 内存峰值包含 Isaac Sim 启动阶段，取的是 cgroup 的 `memory.current`。
- 2048 档比默认任务（4.82 s/轮）慢约 6%，额外开销来自每个物理子步都做一次规则判定。
- 4096 和 8192 档在启动后约 40–65 秒报 `PhysX error: Patch buffer overflow detected, please increase its size to at least 176128 / 352256`。当前值是 Isaac Lab 默认的 `gpu_max_rigid_patch_count = 5·2^15 = 163,840`。溢出后接触 patch 会被丢弃，**击球检测和碰撞结果都不可信**。按 V3 §12.3，这两档不能直接使用。
- 双进程的合计吞吐只有单进程的 **1.29 倍**，低于 V3 规定的 1.5 倍门槛。原因是两个进程的采集都变慢了：GPU 利用率已到 88%，相互抢占。

**规模决策**：

- 默认使用 `num_envs=2048` 单进程，约 5.1 s/轮，即约 700 轮/小时。
- 不默认使用双进程。只有在需要同时跑两个种子、而且能接受每个 run 慢约 55% 时才用。
- 若以后要用 8192（吞吐 2.7 倍），需要把 `sim.physx.gpu_max_rigid_patch_count` 至少调到 2^19，并重新测试，确认没有溢出。这只是缓冲区容量，不改变物理参数，但 V3 未授权，需要用户确认后再做。
