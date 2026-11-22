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

读完 [P1](/2026/11/23/llm-rl-p1-learning-object/)，已经知道奖励通过输出概率改变策略。接下来需要把这条路径变成稳定的训练循环：**采样时的策略、更新中的策略、长期参照的策略，分别承担什么作用？**

## 1. 奖励与优势为什么要分开

假设 Qwen 针对同一道题生成四个答案，奖励为 `[1, 1, 0, 0]`。用奖励直接更新，会给前两个答案正权重，后两个零权重；用组内比较，则可以表达后两个答案低于本组平均水平。

GRPO 的一种常见结果优势为：

$$
A_i=\frac{r_i-\bar r}{s_r+\varepsilon}.
$$

$r_i$ 是第 $i$ 条回答的奖励，$\bar r$ 是同题组内均值，$s_r$ 是组内标准差，$\varepsilon$ 是防止除零的小数。使用总体标准差时，例子的均值为 0.5，标准差为 0.5，因此优势约为 `[1, 1, -1, -1]`。

这是相对组内其他回答的评价，不是“真实贡献分”。同组全对或全错时，奖励项没有区分度；该组的结果优势为零。若总损失还包含 KL 或熵项，那些项仍可产生梯度；其他题目的更新也会通过共享参数改变本题表现。

## 2. PPO 多保存了一份旧概率

一次采样很贵，因此常会对同一批回答做多轮参数更新。采样后，当前策略逐渐离开产生数据的旧策略。PPO 用重要性比率跟踪这种变化：

$$
\rho_{i,t}=
\frac{\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})}
{\pi_{\text{old}}(y_{i,t}\mid x_i,y_{i,<t})}.
$$

分母是采样时保存的旧策略概率，更新中不接收梯度。分子来自当前模型对**同一条实际输出**重新计算的概率，保留梯度。

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

当 $A=1$、$\rho=1.3$，两项分别为 1.3 与 1.2，取 1.2，继续抬高这个样本概率不会继续增加该项收益。当 $A=-1$、$\rho=1.3$，取 −1.3，过度抬高差动作仍会受到惩罚。**裁剪限制的是目标里的激励，不是把真实策略概率锁死在区间内。**

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

基础 PPO 常由 Critic 估计状态价值，再用 TD/GAE 形成优势。GRPO 用同一问题的一组回答构造相对优势，省去这类单独价值网络。

它仍然需要策略模型、旧概率、明确的奖励，以及所选实现的裁剪与正则化。`loss = -A * log_prob` 能解释基本策略梯度，却没有完整实现 PPO 的旧概率比和裁剪。实践 E5 的压缩先导代码就属于这种组相对更新的简化形式。

“没有 Critic”也不等于“没有评分器”。答案可以由精确规则、测试集、奖励模型或语言裁判评分，评分器的可靠性在 P4 单独讨论。

Critic 的状态价值 $V(s)$ 预测从当前状态继续行动的预期回报。最简单的一步偏差为：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
$$

例如奖励 0、折扣 0.9、下一状态价值 0.8、当前价值 0.5，偏差为 `0+0.9×0.8−0.5=0.22`。这一步比原先预期更好。把下一状态的预测接入目标，叫 bootstrap。

GAE 把连续多步偏差按 $(\gamma\lambda)^l$ 加权累积，$\lambda$ 控制较远偏差参与多少。Critic 用回报目标另行拟合；Actor 更新时，把估计优势当作固定权重。环境真正终止时未来价值为零；仅因采样长度到限而截断时，应按任务的终止定义处理，不能一律抹去未来价值。

## 5. token 怎样对齐，哪些位置算动作

因果语言模型用当前位置之前的内容预测下一个 token。因此模型输出的 logits 通常要去掉最后一个位置，目标 token 则去掉第一个位置，二者才能对齐。

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

若总序列长 100，而真正回答只有 20 个 token，直接把 100 个位置平均，会把提示词和补齐也混进目标。长短回答之间的权重也会变化。

## 6. 一个可以独立运行的裁剪计算

下面只计算一批保存好的概率和优势。它演示 Actor 目标中的裁剪与掩码，没有调用 Qwen 或重新采样。

```python
import torch

rewards = torch.tensor([[1., 1., 0., 0.]])
advantages = (
    (rewards - rewards.mean(dim=1, keepdim=True))
    / (rewards.std(dim=1, unbiased=False, keepdim=True) + 1e-6)
).flatten()

# 4 条回答，每条在这里用 2 个有效 token 表示
old_logps = torch.full((4, 2), -1.5)
new_logps = torch.nn.Parameter(old_logps.clone())
mask = torch.ones_like(new_logps)

ratio = (new_logps - old_logps.detach()).exp()
raw = ratio * advantages[:, None].detach()
clipped = ratio.clamp(0.8, 1.2) * advantages[:, None].detach()
token_objective = torch.minimum(raw, clipped)
loss = -(token_objective * mask).sum() / mask.sum()

loss.backward()
print("优势：", advantages)
print("当前损失：", loss.item())
print("梯度：", new_logps.grad)
```

当前新旧概率相同，所以比率都为 1；正负项相消，总损失约为零，但梯度不为零。正优势两条的梯度为负，负优势两条为正。**损失值为零，并不表示这一步没有学习信号。**

这里采用有效 token 总平均。若先对每条回答平均、再对回答平均，会得到另一种长度权重；两者应在方法和代码中说清。

## 7. DAPO 为什么值得接着看

DAPO 把大规模语言 RL 中几个实现选择放在一起研究：不对称裁剪、动态采样、token 级损失聚合与超长回答的奖励处理。归档版本第 2–5 页介绍这些机制，实验使用 Qwen2.5-32B。

它提醒读者：算法名称相同，长度处理、全对全错组筛选和截断处理不同，也会改变训练行为。比较新方法时，先把这些基础项对齐，否则一次“信用分配改进”可能实际来自更换了采样和聚合规则。

## 8. 一次更新的完整顺序

1. 固定行为策略，为每道训练题采样一组回答，记录实际 token 和旧概率。
2. 用约定评分器算奖励，形成结果优势或步骤优势。
3. 当前模型对这些输出重新前向，取得可导 token 概率。
4. 对齐掩码，计算概率比、裁剪目标及所选 KL / 熵项。
5. 反向传播并更新授权训练的参数。
6. 在独立题目上用原始任务指标评价，之后采集新一批数据。

这条循环中，优化器决定怎样更新；奖励与信用机制决定更新朝什么方向。接下来读 [P3：团队得分怎样成为个体信号](/2026/11/23/llm-rl-p3-credit-assignment/)，再到实践 E1 认识本地实验的证据层级。

## 原文入口

- [PPO](https://arxiv.org/abs/1707.06347)，第 3 节。
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)，第 4 节：GRPO。
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)，归档 v2 第 2–3 页：组相对优化；后续章节区分 R1-Zero 与 R1 的多阶段训练。
- [DAPO](https://arxiv.org/abs/2503.14476)，归档 v2 第 2–5 页。
