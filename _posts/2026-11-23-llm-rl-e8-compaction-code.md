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

先看到「实体 0 的值是 1」，后来收到更新「实体 0 的值改为 2」。读完十条事实才知道要问谁，笔记却只能留四条或六条。压缩器每轮决定往哪一格写，或把新事实丢掉；执行器在最后根据留下的事实回答。

实际训练对象是 MLP 压缩器和查询注意力执行器。MLP 是由多个线性层和激活函数组成的小网络，这里给「写入这个槽」和「丢弃这条」打分。执行器从保留下来的事实里选答案。原语言模型方法见 [P6](/2026/11/23/llm-rl-p6-memory/)。

## 1. 一局任务怎样生成

共有六个实体，编号 0–3 可能被提问，4–5 是干扰；每个实体有四种离散值。四个相关实体的首次赋值放在前六轮的随机位置，其余轮次可能更新已经出现的相关实体，或生成干扰。

更新概率取 0.2 或 0.6。更新时新值与当前值不同；最后从四个相关实体中均匀选一个，目标是它的当前值。

例如下面是一条教学事实流：

```text
实体0=1，实体4=2，实体1=0，实体2=3，实体0=2，
实体3=1，实体5=0，实体1=3，实体4=1，实体2=0
```

若最后问实体 0，答案是 2。只记住最早的 1 就会答错。

取四个槽位，可以手动走完一条答对的路径。`E0=2` 表示「实体 0 的值是 2」，槽位编号从 `0` 开始。

| 轮次 | 读到的事实 | 动作 | 四个槽里的内容 |
|---|---|---|---|
| 1–4 | 前四条 | 按到达顺序自动填入 | `E0=1` `E4=2` `E1=0` `E2=3` |
| 5 | `E0=2` | 覆盖槽 0 | `E0=2` `E4=2` `E1=0` `E2=3` |
| 6 | `E3=1` | 覆盖槽 1 | `E0=2` `E3=1` `E1=0` `E2=3` |
| 7 | `E5=0` | `drop` | 四个槽维持原内容 |
| 8 | `E1=3` | 覆盖槽 2 | `E0=2` `E3=1` `E1=3` `E2=3` |
| 9 | `E4=1` | `drop` | 四个槽维持原内容 |
| 10 | `E2=0` | 覆盖槽 3 | `E0=2` `E3=1` `E1=3` `E2=0` |

评估时可以沿事实流检查最后要问的实体还在不在。第 7、9 轮的 `drop` 让干扰事实留在槽外，保护了先前写入的内容。

## 2. 每条事实为什么是 12 维

特征包含六维实体 one-hot、四维数值 one-hot、一个归一化时间与一个有效位：

```python
feature_size = 6 + 4 + 1 + 1  # 12
```

one-hot 是只有一个位置为 1 的编码。`E0=2` 的实体部分是 `[1,0,0,0,0,0]`，值部分是 `[0,0,1,0]`。拼上归一化时间和有效位 `1` 就得到 12 维。时间帮助区分同一实体的旧值与新值，有效位标记记录存在。

容量为 $B$ 时，程序自动把前 $B$ 条填槽。之后每轮有 $B+1$ 个候选动作，其中 $B$ 个对应覆盖各槽，另一个对应丢弃新事实。

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

`slot_scorer` 给每个槽打一个分，`drop_scorer` 给「丢弃」打一个分。`torch.cat` 将它们拼成一次选择。四个槽时，每条轨迹得到 5 个分数。训练按概率抽动作，评估则用 `argmax` 取分数最大的那个。

选择覆盖时用新事实替换对应槽；选择丢弃时全部槽保持。程序还记录每个终局槽最后由哪一步写入，用于来源加权。

四个槽时，`logits` 最后一个分数对应 `drop`，动作编号是 `4`。第 5 轮选 `0`，程序就把 `E0=2` 写入槽 0，将该槽的来源时间记为第 5 轮。第 7 轮选 `4`，槽内容和来源记录都照旧。`log_prob` 仍然记录选中 `drop` 的概率，更新时可以给这个动作加权。

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

`M` 条轨迹各自对 `B` 个槽分配权重，`H` 是每条内部向量的长度。假设查询 `E0` 后四个槽的注意力为 `[0.7, 0.1, 0.1, 0.1]`。`context` 就是给槽 0 的内容向量乘 `0.7`，给三个其他内容向量各乘 `0.1`，再相加。分类头用这份加权内容和查询一起输出四种值的分布，最后采到 `2` 得 1 分。

「当前值可见率」检查终局槽里是否有匹配查询实体和目标值的事实。设 `E0` 的值先为 1、后为 2、最后改回 1，保留最初那条 `E0=1` 也会命中这个指标。要测是否留住了最新来源，就同时比对时间戳。

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

例如八条轨迹的奖励为 `[1,1,1,1,0,0,0,0]`。均值与总体标准差都为 `0.5`，前四条得到约 `+1`，后四条得到约 `−1`。如果同题八条都答错，组内均值和优势都是 `0`，负优势屏蔽不会触发。熵项仍能维持策略的探索。

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

$a_t$ 是最后由该步骤写入的槽位获得的终局注意力之和。手算轨迹里，第 5 至 10 轮共 `6` 个压缩动作。槽 0 最后由第 5 轮写入，注意力为 `0.7`，该动作的原始权重是 `0.25+0.75×6×0.7=3.4`。第 7 轮丢弃，没给终局槽写入内容，权重为 `0.25`。最后各步都除以同一个轨迹均值，所以写入动作的正向权重仍是这次丢弃的 `3.4/0.25=13.6` 倍。

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

`comp_pg` 用每次写入或丢弃的对数概率训练压缩器，`exec_pg` 用选答案的对数概率训练执行器。两项熵奖励鼓励探索。此处从 GRPO 借用的是同题组优势，代码直接用它乘当前 `log_prob`。用旧概率比率和裁剪的版本见 [P2](/2026/11/23/llm-rl-p2-policy-optimization/)。

权重 `detach` 后作为固定评分；采样动作的 `log_prob` 保留梯度，训练压缩器和执行器。离散覆盖与规则奖励本身无须可导，这与 P1 的梯度路径一致。

下一篇 [E9](/2026/11/23/llm-rl-e9-compaction-results/) 读取这套程序留下的 160 次运行，解释容量、随机种子和末端门槛。
