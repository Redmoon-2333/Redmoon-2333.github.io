---
topic: reinforcement-learning
layout: post
title: "强化学习 04 · REINFORCE、baseline 与 actor–critic"
date: 2026-09-26 11:00:00 +0800
categories:
  - 技术实践
series_order: 4
tags:
  - 强化学习
  - REINFORCE
  - A2C
  - GAE
excerpt: 从 log-prob 技巧到 baseline、GAE 与 actor–critic，附两方法同预算对照。
math: true
---
> 前几章的价值表方法（Q 表、DQN）学的是“每个状态值多少钱”，策略只是从价值里读出来的副产品。这一章换一条路：把策略本身写成带参数的概率分布，用采样到的那条轨迹去改参数。要回答三个问题：没有价值表时更新信号从哪来；这条信号的方差为什么大、从哪里来、怎么压；actor–critic 拿 V(s) 换掉全回合回报之后，为什么能做到边走边学。

## 一 · 核心问题：没有价值表，怎么更新策略

价值表方法的前提是能算出每个状态（或状态–动作对）的数值，再从中取 argmax 或 softmax（把一组分数按指数归一化成动作概率）得到策略。连续状态靠第 3 章的函数逼近学会了价值，但策略仍从价值里间接读出——价值估偏了，策略跟着偏。另一条路线是策略梯度：设策略为 $\pi_\theta(a\mid s)$，直接参数化，目标是最大化带折扣的期望回报

$$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\big[R(\tau)\big].$$

难点在于：$J$ 是对**所有**轨迹求期望，而一次交互只采到一条轨迹。策略梯度定理给出的答案是：$J$ 的梯度可以写成一个期望形式，用采到的轨迹做蒙特卡洛估计——不需要枚举动作空间，也不需要价值表，只需要两样东西：采到的动作 $a_t$，以及给这步动作打分的标量（回报或优势）。

打分是正还是负决定了这一步是加强还是抑制；真正的难点是打分标准怎么定、方差怎么控，这是 4.2 到 4.6 的主线。

## 二 · 策略梯度直觉与 log-prob 技巧

先把期望展开成对轨迹的求和，其中 $\pi_\theta(\tau)$ 是轨迹似然：

$$\nabla_\theta J(\theta)=\nabla_\theta\sum_\tau \pi_\theta(\tau)R(\tau)=\sum_\tau \pi_\theta(\tau)\,\nabla_\theta\log\pi_\theta(\tau)\,R(\tau)=\mathbb{E}_\tau\big[\nabla_\theta\log\pi_\theta(\tau)\,R(\tau)\big].$$

第二步用的就是 log-prob 技巧：对 $\nabla\log\pi=\nabla\pi/\pi$ 移项得 $\nabla\pi=\pi\,\nabla\log\pi$。它把“梯度没法采样”的困境换成了“采样一个对数导数”——$\nabla\log\pi_\theta(\tau)$ 与 $\theta$ 直接可算，可由 autograd 完成，于是梯度变成“采到什么就算什么”的普通样本均值。

完整轨迹的似然是 $p_\theta(\tau)=\rho_0(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$，其中 $\rho_0(s_0)$ 是初始状态分布（环境开局落在哪个状态的概率）。上文的 $\pi_\theta(\tau)$ 表示这个完整分布，动作概率的乘积是其中的一个因子。在初始分布与环境转移不依赖 $\theta$ 时，求对数再求梯度才得到 $\nabla_\theta\log p_\theta(\tau)=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)$。若 $R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t$、$G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k$，则单轨迹梯度估计为

$$\widehat{\nabla_\theta J}=\sum_{t=0}^{T-1}\gamma^t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t.$$

其中 $G_t$ 是从 $t$ 起的折扣回报，由 `discounted_returns` 的三行反向递推算出（$G_t=r_t+\gamma G_{t+1}$，初值 0），逐点把一条轨迹的总收益分配回各步。直觉上：$G_t$ 高于参照 → 上升 $\log\pi(a_t\mid s_t)$，该动作概率变大，其他动作的概率被归一化压低；$G_t$ 低于参照 → 反向。整个过程只碰采到的动作，“没采到的动作怎么办”由概率归一化代劳。

本项目的 REINFORCE 实现（`rlpractice/algos/single/pg.py`）在回合结束后把整条轨迹堆起来做一次更新：

```python
G = discounted_returns(rewards, gamma)          # 反向递推整条轨迹的 G_t
adv = th.as_tensor(G - b, dtype=th.float32)
loss = -(adv * th.stack(logps)).mean()          # 采样梯度：log-prob × 回报
```

这段代码按单条轨迹长度取均值，且没有外层 $\gamma^t$，与上面的有限折扣回报目标并非严格相同的梯度估计。随机回合长度下，逐回合除以 $T$ 会改变不同轨迹的权重——长短轨迹都被压成同样的“一次平均”，长轨迹里每一步分到的信号更稀——不能只当作固定学习率缩放；4.3 的零期望 baseline 证明也不能直接推广到依赖整条轨迹的 $1/T$ 权重。实现还在时间上限或预算耗尽时用零尾值计算已收集回报，与后文 A2C 在截断处 bootstrap 的处理不同。

## 三 · baseline：保持期望的条件与方差变化

直接用 $G_t$ 打分有个明显问题：CartPole 里一条好轨迹的 $G_t$ 全是正数且数值都很大，每一步的 $\log\pi$ 都被推高，区分不出“这一步比平均水平好”还是“这条轨迹碰巧活了 500 步”。减一个与动作无关的参照值 $b(s)$ 能解决：

$$\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}\big[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)\big]=b(s)\sum_a\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s)=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)=b(s)\nabla_\theta 1=0.$$

推导对给定状态 $s$ 下的动作求条件期望，用的是概率归一化。只要 baseline 在给定状态和采样前历史后不依赖当前动作，且 actor 更新把它视为常数（stop-gradient：求导时当它不存在，不把梯度传回 baseline），减去 baseline 就不改变原有策略梯度估计的期望；方差是升是降取决于 baseline 的选取，回报或优势估计本身的偏差也不受这一步影响。

固定常数、采样前已确定的历史回报均值和状态价值 $V(s)$ 都可以作为 baseline。降方差取决于 baseline 与回报、score 梯度的关系：若以条件梯度协方差的迹为准，最优标量为 $b^*(s)=\mathbb{E}[G\|g\|^2\mid s]/\mathbb{E}[\|g\|^2\mid s]$，其中 $g=\nabla_\theta\log\pi_\theta(a\mid s)$，分母非零；选得不合适反而会增大方差。直接用包含当前动作后果的同一回合均值给该回合打分，一般不满足上述条件独立要求。

## 四 · 本项目的指数滑动均值 baseline（β=0.9）

REINFORCE 版本用一个全局标量做指数滑动均值（`baseline="moving"`，`baseline_beta=0.9`）：

```python
b = baseline                                    # 先用旧 baseline 给本回合打分
baseline = 0.9 * baseline + 0.1 * float(np.mean(G))   # 再用本回合均值更新
adv = G - b
```

细节有两处：**先用后更新**，本回合的打分只来自历史回合，避免用自己算自己；$\beta=0.9$ 对应约 10 个回合的有效记忆窗口，比全程均值反应快，又不至于被单条运气轨迹带偏。这个旧 $b$ 满足 4.3 对 baseline 本身的条件独立要求，轨迹长度加权则按 4.2 的口径单独处理；本回合的 $\mathrm{mean}(G)$ 只进入后续回合使用的 baseline。`baseline="none"` 时 $b=0$，退回教科书上未加 baseline 的版本，可作为消融对照。

## 五 · 从全回合回报到 V(s)：advantage 与 actor–critic

REINFORCE 有两个固有约束：$G_t$ 要等**整条回合结束**才算得出来，更新节奏被回合长度绑死；且同一轨迹上各时刻的样本相关，$G_t$ 混入了后续状态的随机性。先用学出来的状态价值 $V(s_t)$ 作为 baseline，而非直接替代回报：

$$\hat A_t = G_t - V(s_t).$$

这就是 critic：$V$ 把“从这里往后平均能拿多少”这个公共项吸走，剩下的 $\hat A_t$ 衡量“这一步实际比平均好多少”。更进一步只看一步：

$$\delta_t = r_t+\gamma V(s_{t+1})-V(s_t)$$

是 advantage 的单步（TD）估计，也是 actor–critic 里最常用的即时信号。结构上，actor 是策略头 $\pi_\theta$，critic 是价值头 $V_w$，复现包 `ActorCritic` 用共享 trunk（共享的主体网络）、两个线性头（一个输出动作 logits、一个输出状态价值；logits 指网络末层的原始得分，过 softmax 后才是动作概率），一次前向同时得到两者。REINFORCE 的“无 critic + 回合级 baseline”与 A2C 的“有 critic + 步级 advantage”由此区分开：前者等回合、方差大但实现最简；后者边走边学、每步都有信号，但多了一个价值回归要共同训练。

## 六 · GAE(γ, λ)：δ 展开与回合结束的两种处理

单步 TD 依赖后继价值估计，完整回报不做这种自举，通常方差更大。若 $V=V^\pi$，完整回报减 $V(s_t)$ 才是 $A^\pi(s_t,a_t)$ 的无偏估计；近似 $V$ 作 baseline 时，可保持策略梯度期望的条件见第三节。GAE 用 $\lambda$ 在多步估计之间加权：

$$\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\,\delta_{t+l},$$

等价的反向递推（`gae()` 的实现形式）是

$$\hat A_t=\delta_t+\gamma\lambda\,\mathbb{1}[\text{t 非回合末}]\,\hat A_{t+1}.$$

$\lambda=0$ 时 $\hat A_t=\delta_t$。$\lambda=1$ 时中间的 $V$ 项一正一负相消（望远镜相消），得到 $\sum_{l=0}^{T-t-1}\gamma^l r_{t+l}+\gamma^{T-t}V(s_T)-V(s_t)$；只有 $T$ 是真实终止边界、后继价值置零时，才化为完整蒙特卡洛回报减 baseline。有限 rollout（固定长度采集的一段轨迹）或时间截断边界仍带 bootstrap。增大 $\lambda$ 通常减少自举带来的偏差、增加方差，但不是对任意任务都严格单调。本项目取 $\gamma=0.99$、$\lambda=0.95$。

回合结束的处理要分三种情形，这是实现里最容易出错的地方：

- **terminated（杆倾角或小车位置越界，真实终止）**：真实终止后不再有后续回报，后继价值按 0 处理，bootstrap 项置 0，即 `bootstrap = 0.0 if terminated_flags[t] else next_vals[t]`；
- **truncated（撑满 500 步上限被人为掐断）**：世界并没有结束，仍用下一步状态的价值外推，即照常 `next_vals[t]`，否则会人为低估截断点前的优势；
- **λ 链在两种结束处都切断**：`end_flags = term or trunc`，递推式里的指示函数在两种情形都为 0。截断处虽然保留一步价值外推，但 $\delta$ 的递归不跨过回合边界——下一时刻已经是新回合的状态，它的 δ 与当前回合无关。

逐记录 bootstrap 价值 `next_vals` 正是为了把这件事做对。rollout 是固定长度的分段，截断可能发生在段中间，回合截断后随即 `reset()` 开新回合；若按朴素写法用 `values[t+1]`，截断那一步会取到**新回合初始状态**的价值——一个数值上碰巧存在、语义上完全错误的外推。`next_vals[t]` 在每步执行后立即记录“该步之后状态”的价值，与缓冲区逐步对齐，规避了这个索引陷阱：

```python
vn = 0.0 if term else float(model(th.as_tensor(obs2[None], ...))[1])  # 采样时顺手记下
next_val_buf.append(vn)
...
adv, rets = gae(rew_buf, val_buf, term_buf, end_buf, last_value=0.0,
                gamma=gamma, lam=lam, next_vals=next_val_buf)
```

## 七 · A2C 实现要点

REINFORCE 与 A2C 共用同一环境包装和评估协议，差别只在**用什么信号、在什么时机更新**：

| 维度 | REINFORCE | A2C |
|---|---|---|
| 更新时机 | 回合结束才更新一次 | 每 `rollout=32` 步更新一次 |
| 优势信号 | $G_t-b$（EMW baseline） | GAE(0.99, 0.95) |
| 网络 | 仅策略 MLP | 共享 trunk + 策略/价值双头 |
| 归一化 | 无 | 优势归一化默认关闭 |

- **rollout 分段与更新时机**：内层循环固定收集 32 步就做一次 GAE + 反向传播，缓冲区可以跨回合边界（`end_buf`/`term_buf` 记录哪些步是结束步）。REINFORCE 要等整条回合才更新一次，A2C 把更新切成每 32 步一次，参数在整个预算内持续更新；代价是每次更新只看到 32 个样本，梯度噪声仍不小。跨 rollout 的 `ep_ret`/`ep_len` 只在回合真正结束时落盘，与分段无关。
- **与回合级更新的对照**：REINFORCE 在回合内完全冻结参数，长回合意味着一次更新要承担大量高度相关的样本，短回合则更新频繁但样本少；rollout 分段把这种时间相关性切碎，更新次数与回合长度解耦，评估点也因此能更均匀地插入训练全程。
- **on-policy 的数据代价**：采样在 `no_grad`（只前向、不记录梯度）下进行，更新时对缓冲区重新前向；一批数据用一次就丢弃，100,000 步预算共产生约 3,125 次更新。
- **熵正则**：`loss = pg_loss + vf_coef * vf_loss - ent_coef * ent`，其中 `ent_coef=0.001` 对策略熵取负号加进损失，鼓励保持探索（熵衡量策略分布的随机程度：越接近均匀分布越大；这一项阻止策略过早坍缩到单一动作）。
- **value 系数**：`vf_coef=0.5` 控制价值 MSE 的权重，太大网络容量会偏向拟合价值头，策略梯度被稀释。
- **梯度裁剪**：A2C 的 `grad_clip=5.0`，REINFORCE 为 10.0。裁剪限制梯度范数，不等同于限制 Adam 的参数步长。初版阈值为 0.5，v2 放宽了约束；是否频繁触发、改善来自哪一项配置，需要梯度日志与逐项消融，不能只由最终回报判断。

## 八 · 实测结果

实验沿用第 3 章的 CartPole 任务和成败判据（杆铰接在小车上，靠向左或向右推车维持平衡；撑满 500 步算成功，杆倾角超过 12° 或小车出界即失败终止），回答两个问题：两种策略梯度方法能否在 100,000 环境步内稳定撑满 500 步；把全回合回报换成 critic 的 GAE 信号、把整回合更新换成每 32 步更新之后，结果有什么不同。两者的预算、种子、隐藏层宽度、学习率和评估协议相同，差别是第七节表中的四项，外加梯度裁剪阈值（10.0 与 5.0）。REINFORCE 始终带着第四节的滑动均值 baseline，没有去掉 baseline 的对照组，所以第三节说的方差变化在这里没有单独测量。

环境 CartPole-v1，预算 100000 环境步，5 个随机种子，测试为每种子 100 局 greedy（无探索）汇总，验证评估每点 20 局，误差带为跨种子 SD。最终数字取**一次性复评**（种子 30000–30099，配置冻结后评估，过程见第 11 章）。

REINFORCE 的实测：5 种子总回合 1737，墙钟 137s ± 1s；复评回报 **499.286 ± 1.335**，成功率 98.6%，截断率 98.6%（撑满 500 步上限即视为成功），失败终止率 1.4%；验证成功率从首个验证点的 13% 升到末个验证点的 99.6%（5 种子均值；复评逐种子回报 500、497、500、500、500）。

| 算法 | 复评回报 mean±SD | 成功率 | 截断率 | 失败终止率 | 验证成功率（首个→末个验证点） |
|---|---|---|---|---|---|
| REINFORCE | 499.286 ± 1.335 | 98.6% | 98.6% | 1.4% | 13% → 99.6% |
| A2C（v2） | 500.000 ± 0.000 | 100.0% | 100.0% | 0.0% | 0% → 100% |

A2C 走过一轮配置修正：初版（rollout 5、lr $3\times10^{-4}$、`ent_coef` 0.01、`grad_clip` 0.5、批内优势归一化）在同预算只有 33.2 ± 42.5 步，成功率 0%。小批量归一化和较紧裁剪是可能的影响因素，但这组联合调参不能分离各项贡献。冻结为 v2（**rollout 32、lr $1\times10^{-3}$、`ent_coef` 0.001、`grad_clip` 5.0、归一化关闭**）后以 5 个种子重跑：5 个种子全部满分，复评 500.000 ± 0.000。旧运行原样归档。两行数字同预算、同协议，可以直接并读；A2C 的其余配置见复现包附带的数据摘要。

![CartPole 上四种方法的验证成功率曲线（每点 20 局贪心评估，5 种子；阴影为跨种子 SD）](/assets/img/rl-practice/04_cartpole_curves.png)

图注：横轴为环境步，纵轴为验证成功率（每点 20 局），5 种子均值 ± 跨种子 SD，数据来源为复现包内的 `runs/` 原始日志。图中末点只到最后一个验证点，上表引用的一次性复评不在该曲线内。

这些数字对应本地 CartPole 实现与预算，读法见下一节易错点。

## 九 · 易错点

1. **混用训练回报与独立评估**。训练时策略持续更新，单局回报不能直接替代冻结策略的性能。本项目在 held-out 种子上做 greedy 评估；随机采样评估也有效，它测量的是随机策略本身，须注明模式并在同一协议下比较。
2. **把 baseline 说成必然降方差或降偏差**。在 4.3 的条件独立与 stop-gradient 条件下，减 baseline 不改变相应策略梯度估计的期望；合适的 baseline 可降方差，不合适的也可增大方差。同批均值含当前样本的后果，不能直接套用无偏证明。
3. **忽略小批量归一化的统计波动**。`normalize_adv` 使用批内均值与标准差，小批量时这两个统计量可能不稳定。当前 A2C v2 每批 32 步，旧版才是 5 步；默认 `normalize_adv=False`。是否归一化应结合批量与消融结果判断，不能断言它对所有小批量都有害、对所有大批量都有益。
4. **熵系数不是越大越好**。`ent_coef` 过大时策略被熵项主导、长期停留在接近均匀的随机分布，探索变成不收敛的原因；它只是温和正则（本项目 0.001），需要与策略梯度项的量级相称，而非单调调大。

## 参考文献

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. https://incompleteideas.net/book/the-book-2nd.html
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3–4), 229–256. https://doi.org/10.1007/BF00992696
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR 2016 (arXiv:1506.02438). https://arxiv.org/abs/1506.02438
- Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. *ICML 2016* (arXiv:1602.01783). https://arxiv.org/abs/1602.01783 —— A3C 原文；本项目的 A2C 是其同步化变体。


## 附 · 资料下载

- [复现指南（环境、命令、数据口径）](/assets/attach/rl-practice/复现指南.md)
- [实验数据摘要（逐组配置与结果）](/assets/attach/rl-practice/数据摘要.md)
- [代码与原始日志打包](/assets/attach/rl-practice/rl-practice_code.zip)
