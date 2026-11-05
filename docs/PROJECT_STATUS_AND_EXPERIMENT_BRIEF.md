# Booster T1 羽毛球 RL：项目现状与训练/实验设计说明

> 日期：2026-11-05
>
> 用途：交给不直接操作代码仓库的高思考模型，设计训练脚本、配置和实验方案。
>
> 重要边界：当前只完成环境验证与短基准，**尚未启动完整 baseline**。本地电脑尚未实测。本文不包含任何 API Key、PAT 或账号密码。

## 1. 希望设计模型解决的问题

用户计划：

- 本地电脑平时闲置，用于运行默认/轻量 baseline；
- 远程 AutoDL 用于高负载、大并行数或后续改进实验；
- 希望单次主要实验尽量控制在约 20 小时，但本地速度尚未测量；
- 需要保留可比较、可恢复、可导出、可评测、可写入课程报告的实验记录；
- 后续可使用 SwanLab 统一查看训练曲线，不使用 W&B；
- 默认 baseline 与后续算法改进必须明确区分，避免同时改变多个变量。

设计输出应包含：

1. 本地与远程的配置差异及理由；
2. 明确命令或配置项，包括球场数、迭代数、mini-batch、种子和日志目录；
3. 根据实测迭代时间计算时间预算的方法；
4. checkpoint、恢复、评测和 TorchScript 导出方案；
5. 对照实验应固定哪些变量，比较相同样本量还是相同 wall-clock；
6. 预期效果、主要风险、停止条件和最小验证；
7. 不应为了速度改变物理频率、回合规则或评测接口。

## 2. 软件与代码基线

课程固定版本：

- Python 3.11
- Isaac Sim 5.1.0
- Isaac Lab 2.3.2，Git 标签 `v2.3.2`
- PyTorch `2.7.0+cu128`
- torchvision `0.22.0+cu128`
- RSL-RL 3.1.2，满足课程 `>=3.0.1`
- 课程包 `badmintonlab_game 0.1.0`

课程官方基线提交：

`c5528bf3c976a58a8d4c7c6ce89977e373dc028e`

个人 GitHub fork：

`https://github.com/Legitimity/BadmintonLab-Game`

远程开发工作树：

`/root/autodl-tmp/BadmintonLab-Game-Dev`

Git 分支约定：

- `main` 跟踪 `origin/main`，保留官方基线；
- `dev` 跟踪 `origin/dev`，用于个人开发；
- `upstream` 指向 `https://github.com/ailab-pku/BadmintonLab-Game`；
- 算法实验宜从 `dev` 新建独立 feature 分支；
- 提交署名使用 `Legitimity <legitimity233@gmail.com>`，不修改全局 Git 配置。

当前 `env_isaaclab` 的 editable 包已指向 `BadmintonLab-Game-Dev`。

其他目录：

- 原始训练克隆：`/root/autodl-tmp/BadmintonLab-Game`
- play 评测：`/root/autodl-tmp/BadmintonLab-Game-Play`
- 个人工具：`/root/autodl-tmp/projects/badminton_rl`
- 官方参考动作：`/root/autodl-tmp/BadmintonLab-Motion`
- 验证日志：`/root/autodl-tmp/logs/verification`

注意：个人工具目录目前独立于课程 fork，不应误以为其中的代码已经被课程仓库 Git 跟踪。需要纳入训练流程时，应将所需模块有目的地移入 fork 并提交，而不是保留两份长期分叉的实现。

## 3. 当前硬件与访问边界

### 远程 AutoDL

- RTX 4080 SUPER，32760 MiB 显存；
- 16 CPU 核，约 503 GiB 内存；
- 数据盘剩余约 30 GiB；
- 当前没有训练或评测进程；
- CUDA 对 PyTorch 可用，Isaac Sim Vulkan 已验证。

远程访问能力：当前编码助手只能直接操作这台远程实例。

### 用户本地电脑

- RTX 5070 Laptop，8 GiB 显存；
- 32 GiB 系统内存；
- 平时闲置，用户希望在此运行 baseline；
- 当前编码助手没有本地文件或 shell 访问权限；
- 尚未验证本地操作系统、驱动、Isaac Sim、Vulkan、吞吐、温度或长时稳定性。

本地若希望助手直接操作，需要在本地运行编码助手，或提供受控 SSH 环境。否则只能提供文档、脚本和命令由用户或本地智能体执行。

## 4. 已完成验证

### 环境

- PyTorch `2.7.0+cu128` 导入和 CUDA 检查通过；
- 课程包无卡导入通过；
- AutoDL 曾因缺少 `libXt.so.6`、`libGLU.so.1` 和 Vulkan loader 失败；
- 已安装系统库，并按 AutoDL 官方说明将 headless Vulkan ICD 配置为 `libEGL_nvidia.so.0`；
- 修复后 Isaac Sim 正常识别 RTX 4080 SUPER。

### 训练与评测

- 4 球场、2 次 PPO 迭代成功；
- play 分支自带 `model_test.pt` 完成 1 球场、2 回合评测；
- 评测正常输出 `[SCORE]` 和最终 `[RESULT]`；
- 256 球场、20 次迭代短基准成功；
- 默认 2048 球场、10 次迭代短基准成功；
- 没有启动 50,000 次或其他完整训练。

### 参考动作个人框架

位于 `/root/autodl-tmp/projects/badminton_rl`，已通过 9 项纯 CPU 测试：

- 安全加载 NPZ；
- 52 维 AMP 单帧状态和 208 维四帧转移；
- 采样、归一化、顺序播放；
- Game URDF 关节映射；
- 双机器人四帧历史与排列；
- AMP 判别器和风格奖励基础组件。

该框架尚未接入 Game PPO，也没有启动 AMP 训练。

## 5. 默认任务与 PPO 配置

### 仿真与并行

- 默认 2048 个双机器人球场；
- 共享策略包装后为 4096 个玩家样本；
- 物理 `sim.dt=0.005`，即 200 Hz；
- `decimation=4`，控制频率 50 Hz；
- 默认训练回合上限 10 秒，即最多约 500 个控制步；
- play 评测不设置同样的 10 秒 rally 超时，存在训练/评测时长分布差异。

### 动作与观测

- 每侧动作 23 维关节位置目标；
- 默认动作 scale 为 0.5，再叠加到机器人默认关节角；
- 默认每侧观测 120 维：
  - 自身场地坐标朝向 2、位置 3；
  - 球在自身机体系的位置 3、速度 3；
  - 对手朝向 2、位置 3、重力投影 3、关节角 23；
  - 自身线速度 3、角速度 3、重力投影 3；
  - 自身关节角 23、关节速度 23、上一动作 23。

观测包含较多对手状态。面对与训练对手不同分布的策略时，可能出现 OOD。

### PPO

- `num_steps_per_env=24`
- `max_iterations=50000`
- `save_interval=500`
- actor/critic 隐藏层均为 `[512, 256, 128]`
- 每侧输入 120 维、输出 23 维
- actor/critic 均启用观测归一化
- `num_learning_epochs=5`
- `num_mini_batches=16`
- 初始学习率 `1e-3`，adaptive KL 调度
- `gamma=0.995`，`lam=0.95`
- `entropy_coef=0.001`
- `init_noise_std=1.0`

默认每次迭代玩家样本数：

\[
2\times2048\times24=98,304
\]

默认 50,000 次约 49.15 亿个玩家样本。

### 奖励

双方有镜像奖励管理器，主要包括：

- 机身低姿态和 root 姿态成本；
- 球拍撑地、关节限位、动作变化率、脚部滑移成本；
- 来球接近事件成本；
- 击球后出球速度和方向质量成本；
- 回合胜负奖励；
- 过程奖励只覆盖每侧每回合前四次击球，降低刷奖励风险。

`rally_result` 权重为 10，事件型奖励通过 `_event_scale` 抵消 RewardManager 的 dt 缩放。获胜方需要至少一次有效击球，避免完全依靠对方失误被动获得胜方奖励。

最新 main 已记录：

- 双方平均击球数；
- Player 0 开球方比例；
- 双方最近八次接球机会的回球成功率；
- 各奖励分项和回合终止原因。

仍建议增加更直接的指标：开球成功/过网率、每回合击球次数、下网/出界原因、摔倒率、长回合稳定性和真正独立评测胜率。

## 6. 实测性能与时间约束

远程 RTX 4080 SUPER 实测：

| 配置 | 稳态表现 | 资源 |
|---|---|---|
| 256 球场，20 次 | 约 3,000–3,400 steps/s，约 3.6–4 秒/轮 | 约 3.3 GiB 显存 |
| 2048 球场，10 次 | 平均约 20,500 steps/s、4.82 秒/轮 | 峰值约 4.8 GiB，GPU 利用率采样峰值约 44% |

默认 2048 球场的主要时间构成：

- collection 约 3.8 秒；
- learning 约 0.7–1.0 秒。

由此推算：

- 50,000 次约 67 小时纯训练；
- 12,000 次约 16.1 小时；
- 14,000 次约 18.8 小时；
- 15,000 次约 20.1 小时。

实测 checkpoint 约 5.49 MB。每 500 次保存时，完整 50,000 次约 100 个、合计约 0.55 GB，磁盘不是主要瓶颈。

这些数字只适用于当前远程实例。不能直接套用到 5070 Laptop，也不能仅按显存线性外推吞吐。

## 7. 本地与远程建议分工

### 本地：轻量默认 baseline

建议保持 observation、reward、PPO 超参和回合逻辑不变，先测试：

1. 4 球场、2 次迭代；
2. 512 或 1024 球场、20 次迭代；
3. 至少 30 分钟稳态测试，再估算长训。

本地 8 GiB 显存可能能运行 2048 球场，但桌面和其他程序会争用显存。优先从 512/1024 开始，保持 headless、无 video，并关闭其他 GPU 程序。

本地长训需要关注：

- 电源、高性能模式、禁止睡眠；
- 温度和持续降频；
- Windows/WSL/Linux 的 Isaac Sim 与 Vulkan 支持差异；
- checkpoint 定期备份；
- 不用前几轮速度直接承诺 20 小时完成。

本地 baseline 可固定为“默认算法的低并行版本”，但它与远程默认 2048 球场并非相同 batch，应在报告中明确。

### 远程：高负载和后续实验

优先短测 4096、8192 球场，再决定正式规模。远程还可承担：

- 默认 2048 的完整或较长基线；
- 大 batch/高吞吐实验；
- AMP 或其他增加计算量的实验；
- 多 checkpoint、多种子评测。

不要在没有短测时直接把 8192 或更高规模用于长训。

## 8. 训练脚本可修改方向

### A. 时间预算与优雅停止

当前已有 `--max_iterations`，可设置 12,000/14,000 等预算。更完善的脚本可增加：

- `--time_budget_hours`；
- 每轮检查累计训练时间，而非依赖外部 `timeout` 强杀；
- 到预算后正常执行最终保存和日志关闭；
- SIGINT/SIGTERM 时尽量保存最近 checkpoint；
- 记录 `elapsed_seconds`、实际完成迭代数和累计玩家样本数。

不应只靠外部杀进程控制 20 小时，否则可能丢失距上次 500 轮 checkpoint 的进展。

### B. 配置预设而不是反复改默认值

建议建立明确的 local/remote 预设，例如：

- local smoke：4 球场、2 次；
- local baseline：512/1024 球场，迭代数由实测时间决定；
- remote default：2048 球场；
- remote high-throughput：4096/8192 球场，先短测；
- evaluation：独立 play 配置，不占用训练配置。

预设应记录所有关键参数、Git 提交、随机种子和总样本量，避免只保存命令而丢失环境差异。

### C. batch 与吞吐实验

可研究：

- `num_envs`：2048、4096、8192；
- `num_mini_batches`：16 对比 8；
- 必要时再讨论 `num_steps_per_env`，但它会改变 PPO rollout 语义，不是纯性能参数。

增大球场数会同时增加每轮样本量。如果仍跑 50,000 轮，总样本和总时间都可能增加。公平比较时应明确选择：

- 相同玩家样本数；
- 相同 wall-clock；
- 或相同迭代数。

三者不能混为一谈。

### D. SwanLab 日志

RSL-RL 不原生支持 `agent.logger=swanlab`。当前兼容方案：

- 保持 TensorBoard logger；
- 在 runner 创建 `SummaryWriter` 前初始化 SwanLab；
- 调用 `swanlab.sync_tensorboard_torch()`；
- 训练结束正常 `swanlab.finish()`；
- 同时保留本地 TensorBoard 和 checkpoint。

当前安装 SwanLab 0.6.1，因为最新版本会升级 Isaac Sim 不兼容的 `wrapt`。不要为了升级 SwanLab 破坏 Isaac Sim 固定依赖。

### E. checkpoint、恢复与导出

训练脚本已支持：

- `--resume`
- `--load_run`
- `--checkpoint`

长训计划应明确恢复命令和定期备份，而不是中断后从头开始。

当前课程 main 没有官方 TorchScript 导出脚本。完整闭环需要补充导出工具：

- 加载 actor 和训练时的固定观测归一化；
- 将必要预处理嵌入 TorchScript；
- 输入 `(N, 120)`、输出 `(N, 23)`；
- 支持不同 batch；
- 检查所有输出有限；
- 与 RSL actor 在同一批观测上对齐输出；
- 在 play 中完成至少 2 个回合，再增加评测回合数。

普通 checkpoint 不能直接作为最终提交模型。

### F. 评测与训练分离

不建议在同一张卡、同一训练进程中频繁启动独立 Isaac Sim 评测。更稳妥的方式：

- 训练持续保存 checkpoint；
- 在关键迭代后暂停或另择时间导出；
- 用固定种子和固定对手做 play 评测；
- 保存比分、回合长度和失败类型；
- 训练曲线与 play 结果共同决定是否继续。

建议关注 500、2000、5000、10000、14000 等节点，而非只看最终模型。

## 9. 后续算法实验方向（不属于默认 baseline）

每次只改一类因素：

1. 直接过网/开球成功 shaping，改善早期稀疏学习；
2. 课程学习：平衡与开球，简单接球，稳定回球，再自对弈；
3. 历史 checkpoint 对手或 league，减少共享最新版自对弈的循环和过拟合；
4. AMP 参考动作风格奖励，改善拟人性和姿态，但必须保留任务奖励主导；
5. 训练/评测长回合分布对齐；
6. 对手观测消融，评估 OOD 与收益。

AMP 框架不是即插即用的行为克隆。官方路线是四帧状态转移、判别器、梯度惩罚和风格奖励，不是随机抽帧播放或直接给 PPO 做监督标签。

## 10. 参考动作数据事实

官方 T1 数据共 20 个 NPZ：

- 总计 2592 帧；
- 全部 50 FPS；
- 四帧窗口共 2532 个；
- `dof_pos [F,23]`、`base_pos [F,3]`、`root_rot [F,4]`；
- `root_rot` 是 SciPy xyzw，而 Isaac/Isaac Lab 通常使用 wxyz；
- 参考字段使用 float64，训练特征可转换为 float32；
- 数据不是击球动作监督标签。

默认后退步伐子集有 404 个窗口，前进步伐子集有 450 个窗口。

静态 URDF 检查发现 228 个帧级关节限位越界。用于 AMP 速度特征不等于可以直接用于参考重置或逐帧播放；后两种用途必须逐关节审查，不能静默 clip 后声称动作一致。

## 11. 已知风险与未完成事项

- 本地环境尚未安装和验证；
- 本地 baseline 尚未开始；
- 远程完整训练尚未开始；
- SwanLab 尚未接入训练脚本；
- TorchScript 导出工具尚未实现；
- 默认 baseline 的最终击球/比赛能力未知；
- 训练 10 秒与 play 无 rally 超时的分布差异需要评测；
- 共享策略自对弈不保证对其他同学模型泛化；
- `pip check` 仍有课程上游 FastAPI/Starlette 固定版本提示，但核心导入和 GPU 冒烟已通过；
- 不应通过升级 torch、Isaac Sim 或 Isaac Lab 来消除一般依赖提示。

## 12. 推荐的最低实验记录

每次实验至少保存：

- Git 分支和提交；
- 本地/远程硬件、GPU 和驱动；
- 完整配置及命令；
- seed、球场数、迭代数、总玩家样本数；
- 实测 FPS、collection/learning 时间、显存和 wall-clock；
- 训练曲线与关键 checkpoint；
- play 对手、种子、回合数、比分和失败类型；
- 导出模型与 `player_cfg.py` 的对应关系；
- 结论和下一步只准备改变的一个变量。

最终目标不是单纯提高 reward 或 GPU 利用率，而是在可接受训练时间内得到可复现、可导出、可评测并具有实际回球能力的策略。
