# P3 发球训练指令 V2（2026-11-06，依据 REPORT_P0_P2_RESULTS.md）

本文档取代 `HANDOFF_P0_P2.md` 第 7 节和 V3 设计文档第 7 节中与此冲突的内容。代码只经过 CPU 验证（50 项 pytest 通过），未在 Isaac 中运行。

## 1. P0–P2 结果对 P3 的含义

| 实测结果 | 对 P3 的含义 |
|---|---|
| 500 轮默认奖励：play 入界约 80%，但每次成功发球后都倒地 | 策略已能触球，**主要问题是站稳**。倒地发球即使入界，也会让我方接不到回球，且不符合 E 档“保持平衡” |
| 拍面接触：恢复系数 0.65；击中率 ≈ min(1, 6.2 / 闭合速度) | 穿过拍面由**球与拍沿法线的闭合速度**决定（含球自身约 4.3 m/s 的下落速度），不是单纯的挥拍速度 |
| 候选 2 开环挥拍：入界 47%，穿过 34%，下网 7%，落点约网后 1.05 m | 准备姿势可用；穿过拍面是挥空的主因，下网说明出球偏低 |
| CPU 挥拍搜索把出球速度低估约一半 | 不再依赖 `swing_search.py` 的数值；目标值改为根据 plate 表标定 |
| 8192 球场吞吐为 2048 的 3.03 倍，无溢出 | 正式训练改用 8192，但先做短对比 |

### 接触模型标定（本地计算，依据远程 plate 表）

把远程 plate 表（仰角 45°，拍速 2–10 m/s）拟合成：恢复系数 0.65、摩擦 0.11、击中概率 = min(1, 6.2 / v_close)。5 个速度点的击中率与出球速度误差都小于 0.02 和 0.1 m/s。

在这个模型下，**单次接触发球**的最佳窗口是：

| 量 | 值 |
|---|---|
| 接触高度 | 1.0–1.1 m（越高越好，但 1.15 m 以上判违例） |
| 拍面仰角 | 50–55° |
| 球拍沿法线速度 | 3–4 m/s |
| 闭合速度 v_close | 6.5–7.5 m/s |
| 出球 | 约 8–9 m/s，仰角约 41–48° |
| 落点 | 网后 1.1–1.7 m，过网余量 0.1–0.8 m |
| 预期入界率 | 无扰动约 0.97；加 ±5° 拍面、±0.5 m/s 扰动后约 0.85 |

所以**发球入界率的现实上限约 85–90%**。V3 的 M1 门槛“入界 ≥85%”保留，C 阶段晋级门槛也降到 85%。这是基于标定模型的估计，不是实测。

## 2. 奖励改动（相对上一版 P3 包）

| 项 | 改动 | 原因 |
|---|---|---|
| `fall` | 权重 −5 → **−10**；躯干倾斜超过 45°（投影重力 z > −0.70）也算摔倒 | 原来入界 +10、摔倒 −5，扑倒式发球仍然净赚 |
| 发球方摔倒即结束回合 | 新终止项 `robot_fell`；球还在空中时，该次发球记为**失败**（−10） | 让“扑倒后球才落地入界”拿不到入界奖励 |
| `serve_result` | 只有回合结束时发球方仍站立，才算成功 | 同上；关闭 `terminate_on_fall` 时由这一条兜底 |
| `post_stand`（新） | 发球方击球后保持站立且直立，每秒 +1（C 阶段 +0.5） | 稠密信号，帮助学会挥拍后的恢复 |
| `swing_speed` | 改用闭合速度 v_close：线性升到 6.5 m/s 得满分，6.5–8 m/s 平台，8 m/s 以上线性扣分，10 m/s 时为 0，最低 −1 | 按标定，v_close 超过 8 m/s 后击中率低于 0.78 |
| `tunnel`（新） | 击球前，球在拍面 12 cm 范围内从一侧穿到另一侧且没有接触：每回合一次，−1（A）/ −2（B、C） | 穿过拍面是最主要的失败方式，需要直接信号 |
| `face_quality` | 目标仰角 55° → **52°**，宽度 20° → **15°** | 按标定窗口 |
| `x_target` | 2.0 → **1.6 m**，`x_sigma` 2.0 → 1.5 | 9 m/s 左右出球的自然落点；M1 不追求发深 |
| A 阶段 | `swing_speed`、`face_quality` 从 0 改为 0.5，`outcome_pred` 1 → 2 | 默认奖励 500 轮已能触球，A 阶段不必只学接触 |
| `v_target` | 删除，由 `v_close_target`、`v_close_max` 取代 | |

每回合正向过程奖励仍封顶 8；各阶段正向事件项之和都小于 `serve_result`（有测试保证）。

### 晋级门槛

| 阶段 | 门槛（200 轮 EMA） | 最少 / 最多轮数 |
|---|---|---|
| A | 合法发球 ≥70%，摔倒 ≤10% | 200 / 1500 |
| B | 过网 ≥70%，入界 ≥60%，摔倒 ≤10% | 300 / 4000 |
| C | 入界 ≥85%，违例 ≤2%，摔倒 ≤5% | 300 / 4000 |

“摔倒”指发球方在回合内根部低于 0.45 m 或躯干倾斜超过 45°。

### 新增指标

每轮记录在 `Serve/*`，终端 `[Serve]` 行也会打印：

- `tunnel`：穿过拍面的回合比例；
- `fall_after_hit`：击球后摔倒的回合比例；
- `vc_mean`：触球时的平均闭合速度；
- 原有的 `legal`、`crossed`、`in`、`fault`、`no_hit`、`fall`、`el_mean`。

## 3. 需要先确认的两件事

1. **拍面法线轴**：代码取 `racket_sweet_spot` 的 x 轴作法线。URDF 中拍面碰撞盒是 `0.01 × 0.14 × 0.2`，薄边在 x 方向，所以 x 轴应该就是法线。远程的 plate 测试也把 x 当作法线，结果自洽。冒烟时打印 `Serve/el_mean`，候选 2 准备姿势下应在 40–60° 之间；如果接近 0° 或 90°，先停下修正。
2. **穿过检测**：`tunnel` 按物理步比较球相对拍面平面的符号变化。在 Isaac 中跑 100 轮后，`tunnel + legal + no_hit` 的关系应与 robot 模式实测（穿过约 34%）同量级；若 `tunnel` 远低于 `no_hit`，检查 `tunnel_face_radius`（默认 12 cm）。

## 4. 执行顺序

所有命令在仓库根目录、用 Isaac 的 python 运行。

### 4.1 合并与检查（约 10 分钟）

```bash
git switch serve-v3-local          # 远程 HEAD 0f4009f（含候选 2 姿势 81861a5）
git am --3way <包>/patches/*.patch
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests -q   # 应全部通过
```

冲突时保留远程的修复：候选 2 姿势、`PLAY_ROOT`/`ISAACLAB_SRC` 路径、`gpu_max_rigid_patch_count`、swanlab 旁路、`os._exit(1)`、extras 诊断键、`obs_critic` 的属性兜底。

### 4.2 评测统计确认（约 5 分钟）

按 `EVAL_STATS_AUDIT.md` 用 500 轮短训模型重跑一次 play 评测，确认 `fell_after_hit` ≈ `served_in`。结果写进报告即可，不阻塞后续步骤。

### 4.3 冒烟（约 10 分钟）

```bash
python scripts/rsl_rl/train_serve.py --headless --num_envs 4 --max_iterations 2 --stage B
```

确认：Isaac 接受被设为 `None` 的奖励项和两个新终止项（`serve_returned`、`robot_fell`）；`[Serve]` 行数值有限；`el_mean` 合理（见第 3 节）。

### 4.4 规模短对比（约 1.5 小时 GPU）

两个配置各跑 **45 分钟**，同一种子，B 阶段起步，`--no_curriculum`（固定 B 权重，便于比较）：

```bash
python scripts/rsl_rl/train_serve.py --headless --num_envs 2048 --seed 1 --stage B --no_curriculum \
    --run_name p3cmp_2048 --time_budget_hours 0.75 --swanlab
python scripts/rsl_rl/train_serve.py --headless --num_envs 8192 --seed 1 --stage B --no_curriculum \
    --mini_batches 64 --run_name p3cmp_8192 --time_budget_hours 0.75 --swanlab
```

8192 时 `--mini_batches 64`，使每个 mini-batch 的样本数与 2048/16 相同。学习率保持自适应（`desired_kl=0.01`），不改。

**按墙钟时间**比较 `Serve/in`、`Serve/fall`、`Serve/tunnel`。8192 在相同时间内不更差（`in` 差距 ≤3 个百分点，`fall` 不更高），就用 8192；否则用 2048。

### 4.5 正式训练（单种子，预算 8–12 小时）

```bash
python scripts/rsl_rl/train_serve.py --headless --num_envs <选定> [--mini_batches 64] --seed 1 --stage B \
    --run_name p3_s1 --time_budget_hours 12 --auto_resume --swanlab
```

- 从 B 阶段起步：默认奖励已能触球，A 阶段只会浪费轮数。若前 300 轮 `legal` 低于 50%，改为 `--stage A` 重启。
- 课程每次晋级保存 `model_passed_X_<it>.pt`。
- 第二个种子不并行：双进程吞吐只有 1.32 倍（2048）。等第一个种子进入 C 阶段，或训满 4 小时后再决定。

### 4.6 监控与处理

在前 300 轮内重点看：

| 现象 | 处理 |
|---|---|
| `fall` 下降、`in` 也随之下降到 50% 以下 | 正常的过渡期。若 600 轮后 `in` 仍低于 50%，把 `fall` 降到 −7，`--auto_resume` 继续 |
| `tunnel` 持续高于 0.25，且 `vc_mean` 高于 8 m/s | 把 `tunnel` 权重调到 −3，或把 `env.serve_reward.v_close_max` 降到 7.5 |
| `tunnel` 高，但 `vc_mean` 低于 7 m/s | 穿过不是速度导致的。可能是拍面边缘接触，或 `tunnel_face_radius` 偏大；报告给本地 |
| `no_hit` 高、`tunnel` 低 | 挥空。看 `el_mean` 是否正常，并检查准备姿势是否被 `hold_ready` 锁得太死 |
| `timeout` 大于 0.01 | 球可能停在机器人身上，录 2–3 个回合视频报告 |
| `fall_after_hit` 下降，但 `in` 卡在 60–70% | 正常，课程会在 B 阶段继续推进 |

### 4.7 每 1000 轮评测一次

导出最新的 checkpoint，用修正后的评测脚本在 play 中测两侧、两个对手（zero、`model_test`），每次 200 回合以上。记录：发球入界、违例、未击中、`fell_after_hit`、`timing_s`。

M1 门槛（play 中、两侧、全部对手都满足）：

| 指标 | 门槛 |
|---|---|
| 合法发球 | ≥90% |
| 入界 | ≥85% |
| 违例 | ≤2% |
| 发球后摔倒 | ≤10% |

达到 M1 后先导出并做练习赛打包（P4），再继续训练冲 B 档其余指标。

## 5. 必须停下来报告的情况

- 冒烟中 `el_mean` 不在 30–70° 之间：拍面法线轴可能取错。
- B 阶段 1500 轮后 `in` 低于 40%。
- 训练中 `in` 与 play 评测入界率相差超过 10 个百分点：训练和评测的规则或摔倒判定不一致。
- 训练中 `fall` 已低于 10%，但 play 评测的 `fell_after_hit` 高于 30%：训练的摔倒判定（倾斜或高度）与评测不一致。
- 任何阶段达到最多轮数仍未晋级（训练会自动停止，`stop_reason=stage_X_max_iters`）。

## 6. 未改动的部分

- 候选 2 准备姿势保持不变；候选 7 只作备选，不在训练中途切换。
- PPO 其余超参数、网络结构和 95 维观测不变。
- `end_on_return=True`：M1 中接球方触球即结束并记为发球成功。P5 接发训练时改为 False。
- 评测侧（play）的规则代码完全不动；摔倒终止只存在于训练环境。
