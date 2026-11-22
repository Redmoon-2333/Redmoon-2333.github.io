---
title: LLM 与强化学习 E8：把压缩与执行变成一个能看懂的训练任务
date: 2026-11-23 00:18:00 +0800
categories:
- 技术实践
topic: llm-rl-practice
series_order: 8
series_label: E8 · 压缩训练代码
tags:
- 记忆压缩
- 策略梯度
- 代码解析
excerpt: 逐段解释十轮事实流、槽位覆盖与丢弃、查询注意力和四种信用规则，明确哪些部分实际接受了参数更新。
math: true
slug: llm-rl-e8-compaction-code
reading_prev:
  title: P7 · 把信用给关键步骤，为什么有时反而不好学
  url: /2026/11/23/llm-rl-p7-coverage/
reading_next:
  title: E9 · 160 次压缩训练之后，哪些现象值得继续追
  url: /2026/11/23/llm-rl-e9-compaction-results/
---

这组先导实验把长程记忆问题缩成一个可控任务：连续看十条事实，记忆只有四或六个槽位，最后回答某实体的最新值。压缩器决定保留什么，执行器决定怎样从剩余内容中作答。

这里实际训练的是 MLP 压缩器和查询注意力执行器，没有加载 Qwen。它用于研究机制与设计比较，相关原论文的方法见 [P6](/2026/11/23/llm-rl-p6-memory/)。

## 1. 一局任务怎样生成

共有六个实体，编号 0–3 可能被提问，4–5 是干扰；每个实体有四种离散值。四个相关实体的首次赋值放在前六轮的随机位置，其余轮次可能更新已经出现的相关实体，或生成干扰。

更新概率取 0.2 或 0.6。更新时新值与当前值不同；最后从四个相关实体中均匀选一个，目标是它的当前值。

例如下面是一条教学事实流：

```text
实体0=1，实体4=2，实体1=0，实体2=3，实体0=2，
实体3=1，实体5=0，实体1=3，实体4=1，实体2=0
```

若最后问实体 0，答案是 2。只记住最早的 1 就会答错。

## 2. 每条事实为什么是 12 维

特征包含六维实体 one-hot、四维数值 one-hot、一个归一化时间与一个有效位：

```python
feature_size = 6 + 4 + 1 + 1  # 12
```

one-hot 是只有一个位置为 1 的编码。时间帮助区分同一实体的旧值与新值；有效位标记记录存在。

容量为 $B$ 时，前 $B$ 条自动填槽，不是压缩器学出的动作。之后每轮有 $B+1$ 个动作：覆盖任意一个槽，或丢弃新事实。

## 3. 压缩器怎样产生动作

对每个槽，将“旧槽内容、新事实、是否同实体、时间差”拼起来，用 MLP 打分；另一个 MLP 给丢弃动作打分，再拼成 logits：

```python
# slots: [M, B, 12]；incoming: [M, 12]
# M 是此批所有任务的所有采样轨迹数
slot_logits = slot_scorer(pair_features).squeeze(-1)  # [M, B]
drop_logit = drop_scorer(drop_features)              # [M, 1]
logits = torch.cat([slot_logits, drop_logit], dim=-1)
dist = torch.distributions.Categorical(logits=logits)
action = dist.sample()
log_prob = dist.log_prob(action)
```

`slot_scorer` 与 `drop_scorer` 是已定义的可训练 MLP；片段摘出实际动作采样步骤。训练采样，评估取 `argmax`，因此二者的轨迹规则不同。

选择覆盖时用新事实替换对应槽；选择丢弃时全部槽保持。程序还记录每个终局槽最后由哪一步写入，用于来源加权。

## 4. 执行器怎样回答

执行器把查询实体编码成 query，槽位编码成 key/value，计算注意力，再将加权内容与 query 一起输入四分类头。

```python
# q_vec: [M, 1, H]；keys/vals: [M, B, H]
scores = (
    torch.bmm(q_vec, keys.transpose(1, 2)).squeeze(1)
    / (hidden_size ** 0.5)
)
attention = scores.softmax(dim=-1)       # [M, B]
context = torch.bmm(
    attention.unsqueeze(1), vals
).squeeze(1)                            # [M, H]
```

这段解释张量轴：`M` 条轨迹各自对 `B` 个槽分配权重，`H` 是隐藏宽度。分类头输出四种值的概率，答案等于目标时奖励 1。

源码的“最新事实留存率”实际检查终局槽中是否存在查询实体且值等于当前目标的记录，没有严格比对它是否就是最新时间戳那条记录。若值经历循环变回旧值，该指标与严格来源留存不同；本文按“当前值可见率”解释。

## 5. 同题采样八条，再形成组相对优势

每步生成 16 个新任务，每题采样 8 条轨迹，因此 $M=128$。奖励 reshape 成 `[16,8]`，按题标准化：

```python
reward = rollout["reward"].view(batch_size, group_size)
mean = reward.mean(dim=1, keepdim=True)
std = reward.std(dim=1, unbiased=False, keepdim=True)
active = (std > 1e-5).float()
advantage = ((reward - mean) / (std + 1e-6)) * active
advantage = advantage.reshape(-1)
```

全对、全错组的结果优势为零，仍可由熵项产生更新。后面的失败屏蔽只对负优势样本工作，不能说所有零奖励轨迹都经过责任拆分。

## 6. 四种规则怎样改变权重

| 名称 | 成功 / 正优势 | 负优势 |
|---|---|---|
| Joint | 两模块共享终局优势 | 两模块共享 |
| Pos-Route | 压缩步骤按终局槽位注意力来源加权 | 均匀广播 |
| Random-Split | 同 Joint | 随机屏蔽该样本在一个模块上的负优势策略梯度项 |
| PR-Split | 同 Joint | 当前执行器恢复全历史后，按答对与否屏蔽该样本在一个模块上的负优势策略梯度项 |

Pos-Route 的未归一化步骤权重为：

$$
w_t=0.25+0.75\,n_{\text{comp}}\,a_t,
$$

$a_t$ 是最后由该步骤写入的槽位获得的终局注意力之和。之后按轨迹均值归一化，所以未被引用动作也保留正底座。

PR-Split 在恢复全历史后答对时保留压缩器负权重，将该样本的执行器负优势项置零；仍错时反向处理。熵项与同批其他样本对该模块的更新保留。它把恢复当作诊断规则，诊断是否可靠需要另外验证。

Random-Split 用于检查收益是否来自简单少罚一个模块。少了它，恢复拆分的任何变化都可能被误解释成正确归责。

## 7. 源码实际使用的优化目标

```python
comp_pg = -(comp_weights.detach() * comp_logprobs).mean()
exec_pg = -(exec_weights.detach() * exec_logprob).mean()
entropy_bonus = 0.01 * (
    comp_entropies.mean() + exec_entropy.mean()
)
loss = comp_pg + exec_pg - entropy_bonus

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(parameters, 1.0)
optimizer.step()
```

这是在线组相对策略梯度加熵奖励。变量名虽然含 `GRPO`，实现没有旧策略概率比、PPO 裁剪或 Critic，不能当作完整 GRPO、CompactionRL 或 ECHO 复现。

权重 `detach` 后作为固定评分；采样动作的 `log_prob` 保留梯度，训练压缩器和执行器。离散覆盖与规则奖励本身无须可导，这与 P1 的梯度路径一致。

下一篇 [E9](/2026/11/23/llm-rl-e9-compaction-results/) 读取这套程序留下的 160 次运行，解释容量、随机种子和末端门槛。
