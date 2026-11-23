---
title: LLM 与强化学习 P2：从 token 概率到 PPO、GRPO 的训练闭环
date: 2026-11-23 00:02:00 +0800
categories:
- 论文精读
topic: llm-rl
series_order: 2
series_label: P2 · 策略优化
tags:
- PPO
- GRPO
- DAPO
- 优势
excerpt: 把奖励、优势、旧策略、参考策略与 token 掩码放在同一个训练流程里，用一组能手算的结果理解 GRPO。
math: true
slug: llm-rl-p2-policy-optimization
reading_prev:
  title: P1 · Qwen、Actor 和优化器，究竟谁在学习
  url: /2026/11/23/llm-rl-p1-learning-object/
reading_next:
  title: P3 · 团队成功以后，怎样评价消息、角色和每一步
  url: /2026/11/23/llm-rl-p3-credit-assignment/
---

货车重 5 吨，北桥限重 6 吨，南桥限重 3 吨。给 Qwen 同一份输入，独立采样四次，得到两次「北桥」和两次「南桥」。判题规则给能通行的回答 1 分。现在要把这四条分数变成一次参数更新。

## 1. 奖励与优势为什么要分开

| 回答 | 生成内容示意 | 规则奖励 |
|---|---|---:|
| 1 | 北桥可通行 | 1 |
| 2 | 北桥 | 1 |
| 3 | 南桥可通行 | 0 |
| 4 | 南桥 | 0 |

这四条纸上样本的奖励是 `[1, 1, 0, 0]`。直接用奖励加权时，答错的两条恰好是零权重。组内比较先减去均值 `0.5`，把两条答错的回答变成负权重。

GRPO（组相对策略优化）的一种常见结果优势写作

$$
A_i=\frac{r_i-\bar r}{s_r+\varepsilon}.
$$

$r_i$ 是第 $i$ 条回答的奖励，$\bar r$ 是同题组内均值，$s_r$ 是组内标准差，$\varepsilon$ 是防止除零的小数。使用总体标准差时，例子的均值为 0.5，标准差为 0.5，因此优势约为 `[1, 1, -1, -1]`。

组优势衡量一条回答在同组中的表现。同组全对或全错时，各条回答的奖励相同，减均值后优势都为零。KL 约束与熵奖励（鼓励保留多种选择）仍可提供梯度，其他题也会通过共享参数影响本题表现。

## 2. PPO 多保存了一份旧概率

一次采样很贵，因此常会对同一批回答做多轮参数更新。采样后，当前策略逐渐离开产生数据的旧策略。PPO 用重要性比率跟踪这种变化：

$$
\rho_{i,t}=
\frac{\pi_\theta(y_{i,t}\mid x_i,y_{i,1:t-1})}
{\pi_{\text{old}}(y_{i,t}\mid x_i,y_{i,1:t-1})}.
$$

分母是采样时保存的旧策略概率，更新中保持固定。分子来自当前模型对**同一条实际输出**重新计算的概率，保留梯度。两次计算都输入同一个题目和已经生成的前 `t−1` 个 token。

若旧概率为 0.2，当前变为 0.26，比率就是 1.3。它表达相对于采样时，模型把这个动作看重了多少。

PPO 最大化裁剪代理目标：

$$
J_{\text{clip}}
=\mathbb E\left[
\min\left(\rho A,\,
\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A\right)
\right].
$$

训练代码通常最小化它的负数。$\epsilon=0.2$ 表示裁剪区间为 `[0.8, 1.2]`。

设四条回答各自被标出的一个动作，其旧概率恰好都是 `0.20`。一次参数更新后，当前策略给它们的概率如下。目标逐行取两项中的较小值。

| 回答 | 优势 | 当前概率 | 比率 | 原项 | 裁剪项 | 取值 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | +1 | 0.26 | 1.30 | +1.30 | +1.20 | +1.20 |
| 2 | +1 | 0.14 | 0.70 | +0.70 | +0.80 | +0.70 |
| 3 | −1 | 0.26 | 1.30 | −1.30 | −1.20 | −1.30 |
| 4 | −1 | 0.14 | 0.70 | −0.70 | −0.80 | −0.80 |

第 1 条的正向激励在比率 `1.2` 后封顶。第 3 条是答错的回答，提高它的概率仍会压低目标值。第 4 条已经往奖励鼓励的方向降低概率，激励在比率 `0.8` 处封顶。表里的实际比率取到了 `1.3` 和 `0.7`，裁剪发生在计算目标时。

## 3. 旧策略与参考策略的区别

旧策略回答“这批数据由谁采样”，参考策略回答“希望模型与哪个长期起点保持联系”。

| 对象 | 更新频率与用途 |
|---|---|
| 当前策略 | 每个优化器步骤更新；提供可导概率 |
| 旧策略 / 行为策略 | 在这批更新期间固定；提供概率比的分母 |
| 参考策略 | 常固定为某个预训练或 SFT 起点；提供 KL 约束 |
| Critic | 预测回报；PPO 常用它估计优势 |

KL 衡量两个概率分布的差异。对完整词表的一种精确定义是：

$$
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
=\sum_v\pi_\theta(v)\log
\frac{\pi_\theta(v)}{\pi_{\mathrm{ref}}(v)}.
$$

实现中可采用抽样估计器，其取样分布与公式要配套。把单个已选 token 的 `logp_new - logp_ref` 直接当作处处非负的精确 KL，会误读日志。

## 4. GRPO 相对 PPO 改了哪一处

常见 PPO 实现用 Critic 预测回报，再用 TD 偏差和 GAE 得到优势，见下方算例。GRPO 则用同一个问题的多条回答作参照：答对得 1 分，高于这一组的平均 `0.5`，就有正优势。这样可以省去单独的价值网络。

完整的裁剪式更新会保存旧概率，并按所用实现加入正则项。基础策略梯度可以直接写成 `loss = -A * log_prob`。[实践 E8](/2026/11/23/llm-rl-e8-compaction-code/) 的小网络就用组相对优势配这个基础损失。

换成 GRPO 后，评分器继续负责给答案打分，组内比较负责产生优势。桥是否可通行仍由规则决定。测试集、奖励模型与语言裁判也可以充当评分器，具体如何验证这些分数见 P4。

Critic 的状态价值 $V(s)$ 预测从当前状态继续行动的预期回报。最简单的一步偏差为：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
$$

例如奖励 0、折扣 0.9、下一状态价值 0.8、当前价值 0.5，偏差为 `0+0.9×0.8−0.5=0.22`。这一步比原先预期更好。把下一状态的预测接入目标，叫 bootstrap。

GAE（广义优势估计）把连续多步偏差按 $(\gamma\lambda)^l$ 加权累积，$\lambda$ 控制较远偏差参与多少。Critic 用回报目标另行拟合。更新 Actor 时，把估计优势当作固定权重。到达真正终点时，未来价值取零。采样到长度上限但任务还可以继续时，用下一状态的预测价值。

## 5. token 怎样对齐，哪些位置算动作

因果语言模型用截至当前位置的内容预测下一个 token。因此模型输出的 logits 要去掉最后一个位置，目标 token 则去掉第一个位置，二者才能对齐。

以下假定按原始 softmax 随机采样，`temperature=1`，不使用 top-k/top-p 截断；旧概率按同一口径保存。使用其他生成分布时，需要与该实现的概率记录和修正方法配套。

```python
# input_ids: 提示词与实际回答拼在一起，形状 [批量, 总长度]
# logits: 模型对每个位置给出的词表分数，[批量, 总长度, 词表大小]
logits = qwen(input_ids=input_ids, attention_mask=attention_mask).logits
next_token_logps = logits[:, :-1].log_softmax(dim=-1)
actual_next_tokens = input_ids[:, 1:]
chosen_logps = next_token_logps.gather(
    dim=-1, index=actual_next_tokens.unsqueeze(-1)
).squeeze(-1)
```

这里的 `qwen` 是已经加载的模型对象。`gather` 从每个词表分布中取出本次实际生成 token 的对数概率。`attention_mask` 让提示词与有效回答参与前向计算，`response_mask` 决定哪些移位后的输出位置参与策略损失：回答位置为 1，提示词、补齐和无效位置为 0；是否纳入 EOS 需要与采样记录一致。

用上表第 2 条举例。假设一条相同长度的批量记录切成 `[题1, 题2, 北桥, EOS, PAD]`，其中 `EOS` 表示生成终止，`PAD` 用来补齐。此处约定训练包含生成出的 EOS。

| 去掉最后一项后的输入位置 | 它预测的目标 | 计入策略损失 |
|---|---|---:|
| `题1` | `题2` | 0 |
| `题2` | `北桥` | 1 |
| `北桥` | `EOS` | 1 |
| `EOS` | `PAD` | 0 |

`北桥` 的概率来自 `题2` 对应的输出，恰好说明为什么代码要移位。实际分词由模型的 tokenizer 决定，表中的符号是手算单位。四条回答长短不同时，各自按这种方法得到掩码，然后放进同一个批量。

## 6. 一个可以独立运行的裁剪计算

下面把四条样本缩到每条最多两个位置。第 2、4 条保留两个有效位置，第 1、3 条的第二个位置是补齐，掩码为 `0`。代码用独立的 log-prob 数值观察每个位置的目标梯度。

```python
import torch

rewards = torch.tensor([[1., 1., 0., 0.]])
advantages = (
    (rewards - rewards.mean(dim=1, keepdim=True))
    / (rewards.std(dim=1, unbiased=False, keepdim=True) + 1e-6)
).flatten()

# 4 条回答，每条预留 2 个位置
old_logps = torch.full((4, 2), -1.5)
new_logps = torch.nn.Parameter(old_logps.clone())
mask = torch.tensor([[1., 0.], [1., 1.], [1., 0.], [1., 1.]])

ratio = (new_logps - old_logps.detach()).exp()
raw = ratio * advantages[:, None].detach()
clipped = ratio.clamp(0.8, 1.2) * advantages[:, None].detach()
token_objective = torch.minimum(raw, clipped)
loss = -(token_objective * mask).sum() / mask.sum()

loss.backward()
print("优势：", advantages)
print("当前损失：", loss.item())
print("梯度：", new_logps.grad)

with torch.no_grad():
    first_position_ratios = torch.tensor([1.3, 0.7, 1.3, 0.7])
    first_position_scores = torch.minimum(
        first_position_ratios * advantages,
        first_position_ratios.clamp(0.8, 1.2) * advantages,
    )
    print("四个示范位置的裁剪取值：", first_position_scores)
```

反传时新旧概率起初相同，比率都为 1。三个正优势位置与三个负优势位置相消，总损失约为零。逐位置看，正优势项的梯度约为 `−1/6`，负优势项约为 `+1/6`，补齐位置为 `0`。最后一行单独代入表中的四个比率，得到 `[1.2, 0.7, -1.3, -0.8]`。

代码把 6 个有效位置一起平均，每个位置权重都是 `1/6`。若先按回答取平均，再把 4 条回答等权合并，单 token 回答的词权重是 `1/4`，双 token 回答的每个词则是 `1/8`。这就是两种常见写法训练时会产生差别的地方。

## 7. DAPO 为什么值得接着看

DAPO 把大规模语言 RL 中几个实现选择放在一起研究：不对称裁剪、动态采样、token 级损失聚合与超长回答的奖励处理。归档版本第 2–5 页介绍这些机制，实验使用 Qwen2.5-32B。

动态采样会滤掉全对或全错等没有组相对奖励信号的组，再继续采集。前面的算例也说明，改为按 token 平均会改变长回答的权重。要单独测试一种信用规则，就让各组沿用同样的采样、长度处理和损失聚合方式。

## 8. 一次更新的完整顺序

1. 固定行为策略，为每道训练题采样一组回答，记录实际 token 和旧概率。
2. 用约定评分器算奖励，形成结果优势或步骤优势。
3. 当前模型对这些输出重新前向，取得可导 token 概率。
4. 对齐掩码，计算概率比、裁剪目标及所选 KL / 熵项。
5. 反向传播并更新授权训练的参数。
6. 在独立题目上用原始任务指标评价，之后采集新一批数据。

单个 Qwen 的回答可以整体打分。两位智能体共同成功时，还要决定各自的动作怎么分到这份学习信号。接下来读 [P3：团队得分怎样成为个体信号](/2026/11/23/llm-rl-p3-credit-assignment/)。

## 原文入口

- [PPO](https://arxiv.org/abs/1707.06347)，第 3 节。
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)，第 4 节：GRPO。
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)，归档 v2 第 2–3 页：组相对优化；后续章节区分 R1-Zero 与 R1 的多阶段训练。
- [DAPO](https://arxiv.org/abs/2503.14476)，归档 v2 第 2–5 页。
