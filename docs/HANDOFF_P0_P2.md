# P0–P2 代码交接（远程 agent 必读）

> 作者：本地设计侧。日期：2026-11-05。
> 分支：`serve-v3-local`，基于 `main` c5528bf。只新增文件和一行任务注册，默认任务 `BadmintonLab-Game` 的行为不变。
> **以下所有内容都没有在 Isaac/GPU 上运行过。** 只做了 CPU 检查（21 项测试，见第 4 节）。

## 1. 与 V3 和你上次反馈的对应关系

你的 7 条修改我都核对过，结论如下：

| 你的反馈 | 结论 | 代码处理 |
|---|---|---|
| 手臂力矩上限 18 N·m | **属实**。显式执行器按 URDF 的 18 N·m 截断，38.3 只用于 action scale | `swing_search.py` 用 18 N·m 和速度-力矩曲线 |
| 无人击球按落点判负 | 属实 | `rules.py` 照搬；`test_no_hit_loss_follows_landing_half` |
| 球停在身上会卡死评测 | 属实 | P1 `plate` 模式统计 `rest`；`eval_serve.py` 带 60 s 看门狗 |
| 静止拍面有二次击球风险 | 属实，**这推翻了 V3 第 2.5 节** | 改为“球拍在第一次接触时已有足够速度”，见第 2 节 |
| reset 时读不到本回合发球方 | 属实 | 初始姿态不区分角色 |
| `main` 的网过滤可能无效 | 属实 | `serve_env_cfg.py` 用 `play` 的写法 |
| 日期偏乐观 | 同意 | 不影响代码 |

## 2. 发球方案的变化（重要）

V3 的“球先落在静止拍面，再加速挥拍”会造成二次击球，已放弃。

新方案：**准备姿势把球拍放在球路下方约 0.6 m，球下落到约 0.66 m 时球拍已在挥动**。只有一次接触，接触时法向速度约 5–6 m/s。

CPU 动力学搜索（`scripts/tools/swing_search.py`）找到了这样的开环挥拍。模型包括 URDF 惯量、电机转子惯量、18 N·m 截断、速度-力矩曲线、无重力补偿的 PD 和课程阻力模型。候选 0：

- 躯干位置 `(-2.961, 0.916)`，偏航 0.177 rad；
- 准备关节：`Waist 0.308, R_Shoulder_Pitch 1.029, R_Shoulder_Roll 0.167, R_Elbow_Pitch -0.22, R_Elbow_Yaw -0.035`；
- 第 27 个控制步切换到挥拍目标 `qc`（见 `docs/ready_pose.json`）；
- 接触高度约 0.66 m，拍面仰角约 53°，法向速度约 5.6 m/s；
- 手臂和腰各自的延迟在 2–8 个物理步中组合（14 种），全部发球入界，落点 x=1.6–2.8 m；
- 碰撞判定改为“球从哪一面来就用哪一面”时，结果相同。

这些结论都**只是上限**：模型忽略脚的反作用力和平衡，碰撞用的是简化的球-盒模型。`racket_impact_test.py --mode robot` 才是真正的检验。

`player_cfg.py` 已写入候选 0 的准备姿势。还有 7 个备选在 `docs/swing_search.json`，可用 `scripts/tools/ready_pose.py --candidate N --write` 切换。

## 3. 文件说明

| 文件 | 阶段 | 状态 |
|---|---|---|
| `.../serve/rules.py` | P0 | `play` 规则的纯 torch 移植，与 `play` 源码差分测试一致 |
| `.../serve/serve_env.py` | P0 | 新环境：子步判定、锁存、reset 后 `sim.forward()`、`permanent_wrench_composer`、发球指标 |
| `.../serve/serve_env_cfg.py` | P0/P2 | 网碰撞过滤修正；actor 和 critic 观测组 |
| `.../serve/player_cfg.py` | P2 | **提交文件**。95 维观测、无对手信息、固定准备姿势、自定义 reset |
| `.../serve/obs_critic.py` | P2 | critic 特权输入，包括执行器延迟 |
| `.../serve/agents.py` | P2 | `obs_groups` 和 `init_noise_std=0.3` |
| `.../serve/wrapper.py` | P2 | 输出 critic 组；动作偏置初始化 |
| `.../serve/runner.py` | P2/工程 | 原子 checkpoint、`latest.pt`、时间预算、信号、恢复课程状态和 RNG |
| `scripts/rsl_rl/train_serve.py` | 工程 | 训练入口，支持 `--auto_resume`、`--time_budget_hours`、`--swanlab` |
| `scripts/export_torchscript.py` | P2 | 导出，结构与 `model_test.pt` 完全相同 |
| `scripts/eval_serve.py` + `_eval_serve_inner.py` | P2 | 在 `play` 中评测；临时替换 `play` 的 player 配置，结束后还原 |
| `scripts/tools/p0_scenarios.py` | P0 | 场景测试，train 和 play 各跑一次后比较 |
| `scripts/tools/bench.sh` | P0 | 吞吐基准 |
| `scripts/tools/racket_impact_test.py` | P1 | `plate` 和 `robot` 两种模式 |
| `scripts/tools/swing_search.py`, `ready_pose.py` | P1/P2 | CPU 挥拍搜索和准备姿势 |
| `tests/` | — | 21 项 CPU 测试 |

**奖励还没改**：新任务暂时沿用默认奖励。P3 的发球奖励（V3 第 7.2 节）尚未实现，这是下一步。

## 4. 已做的 CPU 检查

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests -q   # 21 passed
```

- `test_rules_vs_play.py`：用 stub 加载**未修改的 `play` 源码**，与 `rules.py` 在 5 个种子×3000 步的随机事件流中逐步比较全部状态，结果完全一致；另检查事件流覆盖所有结局。
- `test_export_and_init.py`：导出结果与 `act_inference` 误差小于 1e-5；用 `model_test.pt` 自身权重重建后误差小于 1e-6，TorchScript 代码与课程模型一致；动作偏置初始化有效。
- `test_player_cfg.py`：用真正的 Isaac `configclass` 检查观测为 95 维且不含对手；按 `play` 方式合并两侧事件时名称不冲突；Player 1 是旋转而非镜像；reset 写入正确并截断到软限位。

**未检查**：Isaac 中的实际运行、观测函数的张量计算、PhysX 碰撞、`racket_impact_test.py` 和 `p0_scenarios.py` 中依赖物理的部分。这些脚本很可能需要小修。

## 5. 建议的 GPU 使用顺序

尽量不让 GPU 空转：

1. **冒烟**（几分钟）：`python scripts/rsl_rl/train_serve.py --headless --num_envs 4 --max_iterations 2`。若报错，多半是 configclass 字段、critic 组或 `body_names` 解析问题。
2. **P0 场景**：train 和 play 各跑一次 `p0_scenarios.py`，再用 `--compare` 比较。
3. **P1 robot 模式**：`racket_impact_test.py --mode robot --candidate 0`。**这是最关键的一步**：它决定新发球方案是否成立。不成立就换候选，或回到 V3 第 7.4 节的回退方案。
4. **P1 plate 模式**：测穿模、恢复系数、二次接触和球停留。
5. **吞吐基准**：`bash scripts/tools/bench.sh`。
6. **P2 闭环**：用短训的 checkpoint 导出，再用 `eval_serve.py --opponent zero` 和 `--opponent model_test` 各跑一次，两侧都测。

步骤 2–4 互不依赖。若某一步卡住，可以先跑别的。

`racket_impact_test.py` 的 `plate` 模式是“把整台机器人沿球拍法线瞬移”。我不确定 PhysX 对每步瞬移的刚体能否给出正确的接触速度。若结果不可信，就以 `robot` 模式为准，并告诉我。

## 6. 合并方式

我没有推送到 GitHub。建议用户把本地分支推上去，或者把以下目录打包传到远程：

```text
source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/serve/
source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/__init__.py  (末尾一行注册)
scripts/rsl_rl/train_serve.py
scripts/export_torchscript.py
scripts/eval_serve.py
scripts/_eval_serve_inner.py
scripts/tools/
tests/
docs/
```

若你已在 `feat/serve-v3` 写了 P0 代码，请比较两者：`rules.py` 已有差分测试保障，建议以它为准。冲突之处以 Isaac 实测结果为准。

## 7. P3 发球奖励与课程（2026-11-05 晚补充，仅 CPU 验证）

新增 `serve/serve_reward.py`、`serve/curriculum.py`、`tests/test_serve_reward.py`，并改动 `serve_env.py`、`serve_env_cfg.py`、`runner.py`、`train_serve.py`。

- **奖励**：按 V3 第 7.2 节的 9 项实现，阶段权重表 `STAGE_WEIGHTS` 与 V3 一致。父类的 `interval_approach`、`outgoing_quality`、`rally_result` 已关闭，其余默认成本保留。所有服务项在 `ServeRewardComputer` 中统一算好，奖励项权重固定为 1.0，便于课程改权重和做每回合正奖励上限（8）。
- **触球特征**：在击球所在的物理子步读取 `racket_sweet_spot` 的姿态和速度；拍面法线取 sweet spot 的 x 轴，按运动方向选择击球面。**这个轴向是根据 URDF 推断的，冒烟时请打印 `Serve/el_mean` 并与 `racket_impact_test.py` 的结果对照。**
- **相对 V3 的改动**：
  1. `outcome_pred` 增加长球项 `x_long_safe=6.4`，否则预测出底线的发球仍能拿到约一半分数。
  2. 发球方击球后不再计 `hold_ready`。
  3. M1 中接球方触到合法发球即结束回合，并记为发球成功（`end_on_return=True`）；P5 要改成 False。
  4. 超时回合计入统计（`Serve/timeout`），用来发现球停在机器人身上的情况。
- **课程**：200 轮 EMA、晋级后 300 轮线性过渡；某阶段到最多轮数仍未晋级就停止（`stop_reason=stage_X_max_iters`），不会自动晋级。通过 C 阶段门槛后标记 `finished` 并继续训练。每次晋级都会保存 `model_passed_X_<it>.pt`。课程状态写在 checkpoint 中，`--auto_resume` 后会从原阶段继续。
- **P1 实测后必须修改的参数**（hydra 覆盖）：`env.serve_reward.v_target`（默认 6.0）、`elevation_target_deg`（55）、`x_target`（3.0）。阻力模型下，0.7 m 处以 45° 出球，10 m/s 落点约网后 2.0 m，8 m/s 约 1.2 m，7 m/s 下网。
- **新命令行参数**：`--stage A|B|C`（起始阶段）、`--no_curriculum`（固定权重，用于调试）。
- **日志**：每轮写 `Serve/*`（legal、crossed、in、fault、no_hit、fall、timeout、vn_mean、el_mean、outcome_mean 等）、`Curriculum/*` 和 `Serve/weight_*`，终端每轮打印一行 `[Serve]`。

**冒烟检查**：用 4 个球场跑 2 轮，确认 Isaac 能接受 `None` 奖励项和新的终止项，且 `[Serve]` 行里的数值有限。然后用 2048 个球场在 A 阶段跑约 100 轮，看 `legal` 是否随准备姿势上升、`timeout` 是否接近 0。

### 7.1 按 P1 实测调整的默认值（2026-11-06）

- `v_target` 从 6.0 改为 **4.5 m/s**：plate 测试显示相对法向速度 10 m/s 时约 54% 穿过拍面，恢复系数约 0.65。4.5 m/s 的球拍速度在 55° 时可打出约 9–10 m/s，足够过网，再快主要是增加穿模。`swing_speed` 在达到目标后封顶，不鼓励更快。
- `x_target` 从 3.0 改为 **2.0 m**：约 10 m/s 出球落点在网后约 2 m。
- 若 P1 的穿模—速度曲线在 4–6 m/s 区间另有拐点，按实测再调 `env.serve_reward.v_target`。
