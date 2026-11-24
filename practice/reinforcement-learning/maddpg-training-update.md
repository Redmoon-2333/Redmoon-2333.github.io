---
layout: post
title: MADDPG 实作：从一条联合经历到一次网络更新
date: 2026-10-12 01:40:00 +0800
categories:
- 技术实践
topic: reinforcement-learning
permalink: /2026/10/12/rl-day8-maddpg-training/training-update/
math: true
reading_prev:
  title: Day8 · MADDPG：连续动作与集中式评分
  url: /2026/10/12/rl-day8-maddpg-training/
---

两人各写一个 `[-1,1]` 内的数。每轮奖励是两数差的平方取负：相差 `0.6` 得 `-0.36`，写得相同就得 `0`，这是最高分。两轮之后结束。同一步的两人观测、动作、奖励和下一观测一起保存，训练时可以从里面抽一批经历。

[返回主文](/2026/10/12/rl-day8-maddpg-training/)

Actor 根据自己看到的信息选数。Critic 同时读两个人的观测和动作，预测从这一步起能拿到的折扣回报。一次更新里有三组动作：真实执行过的动作让 Critic 学会评分，下一状态的目标动作补上未来分数，本人当前的动作让 Actor 知道下一次应该往哪里改。

## 4. 网络分工：两个局部 Actor、两份集中式 Q、四个目标副本

Actor 输入 `[自己的数值, 公共阶段]`，输出一个连续动作。末尾 `Tanh` 将输出限制在 `(-1,1)`。

```python
import random

import torch
from torch import nn

device = torch.device("cpu")
seed = 7
torch.manual_seed(seed)

def local_observations(state):
    x0, x1, phase = state
    return torch.tensor([[x0, phase], [x1, phase]], dtype=torch.float32, device=device)

class LocalActor(nn.Module):
    def __init__(self, initial_bias):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1), nn.Tanh(),
        )
        with torch.no_grad():
            self.net[-2].bias.fill_(initial_bias)

    def forward(self, own_observation):
        return self.net(own_observation)
```

`state` 里是双方上一步写的数和已完成的轮数。第 0 行只给 Actor 0，第 1 行只给 Actor 1，两人都看得见轮数。`Linear(2,32)` 把两个观测数变成 32 个隐藏特征，`Linear(32,1)` 产生一个输出。`+0.6` 和 `-0.6` 是最后一层的初始偏置，用来让两人的起始策略不同；实际动作还要经过其他权重和最终的 `Tanh`。

Critic 把双方观测和动作拼接起来，预测一条经历的未来折扣回报：

```python
class CentralCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, obs, actions):
        inputs = torch.cat(
            [obs.flatten(start_dim=1), actions.flatten(start_dim=1)],
            dim=1,
        )
        return self.net(inputs)
```

设批量大小为 `B`，观测是 `[B,2,2]`，动作是 `[B,2,1]`。从第 1 维开始展平，分别得到 `[B,4]` 和 `[B,2]`，再拼成 `[B,6]`：

```text
[x0, phase, x1, phase, a0, a1] → 一个 Q 预测
```

公共阶段出现两次，来自直接拼接局部观测。对这个任务，两行观测的并集已经包含全局状态信息。Critic 输出 `[B,1]`，每条转移一个数；输出层不加 `Tanh`，因为 Q 的尺度由回报决定。

```python
import copy

actor_lr, critic_lr = 0.0005, 0.002
actors = [LocalActor(0.6).to(device), LocalActor(-0.6).to(device)]
critics = [CentralCritic().to(device) for _ in range(2)]
target_actors = copy.deepcopy(actors)
target_critics = copy.deepcopy(critics)
for target in target_actors + target_critics:
    target.requires_grad_(False)

actor_optimizers = [
    torch.optim.Adam(a.parameters(), lr=actor_lr) for a in actors
]
critic_optimizers = [
    torch.optim.Adam(c.parameters(), lr=critic_lr) for c in critics
]
```

每个 Actor 有对应的集中式 Q Critic。[1] 在这个共享奖励的环境里，两份 Critic 估计的是同一个团队回报。若对同一组观测和动作给出 `-0.2` 和 `-0.3`，相差 `0.1` 说明两份估计不同。没有真实期望回报作参照，单凭这个差值判断不了哪份更准，也不能把差值当成队员的贡献。

目标副本用于构造比较缓慢变化的 TD 目标：目标 Actor 提供下一步动作，目标 Critic 评价这些动作。`deepcopy` 得到独立参数；它们不会因在线优化器 `step()` 自动变化，而是由后面的软同步更新。

这份练习的 Actor 输出一个固定数值，训练采样在外面加高斯噪声，让它试到附近的动作。原作者的实现见参考 [2]。

## 5. 联合回放：同一局、同一步的双方数据必须配对

回放中一行记录一次共同交互。`obs` 与 `actions` 同时包含两人，`next_obs` 与 `terminated` 来自这次交互的真实结果。

```python
from collections import deque
from typing import NamedTuple

class Transition(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor

replay_capacity, batch_size = 5000, 128
replay = deque(maxlen=replay_capacity)
replay_rng = random.Random(seed + 1)
```

`NamedTuple` 给五项数据起名字，方便通过 `batch.obs` 等访问。`deque(maxlen=5000)` 最多保留 5000 条转移，超过容量时淘汰最旧条目。回放抽样使用独立随机数生成器。

```python
def save_transition(state, actions, reward, next_state, terminated):
    replay.append(Transition(
        local_observations(state),
        torch.tensor(actions, dtype=torch.float32, device=device).view(2, 1),
        torch.full((2, 1), reward, dtype=torch.float32, device=device),
        local_observations(next_state),
        torch.tensor([float(terminated)], dtype=torch.float32, device=device),
    ))

def sample_batch():
    rows = replay_rng.sample(list(replay), batch_size)
    return Transition(*(torch.stack(field) for field in zip(*rows)))
```

保存的是加噪声、裁剪后真正执行的动作，因为真实奖励来自这些动作。保存干净动作却配上带噪声动作的奖励，会让 Critic 学到错误配对。

共同奖励在 `rewards` 中复制两份，便于分别训练两个 Critic；每个 Critic 取自己对应的一份，整局环境奖励仍然只计一次。复制数据不意味着把团队回报乘以 2。

`sample_batch()` 抽取完整行，`zip(*rows)` 按字段分组，`torch.stack` 增加批量维度：

| 字段 | 单条形状 | 批量形状 | 含义 |
|---|---|---|---|
| `obs` | `[2,2]` | `[B,2,2]` | 双方本步局部观测 |
| `actions` | `[2,1]` | `[B,2,1]` | 双方实际执行动作 |
| `rewards` | `[2,1]` | `[B,2,1]` | 分给两份 Critic 的共同奖励 |
| `next_obs` | `[2,2]` | `[B,2,2]` | 同一步产生的下一观测 |
| `terminated` | `[1]` | `[B,1]` | 第二轮真正终止为 1，否则 0 |

不能给两名智能体各自随机抽样，再把不同局、不同时间的动作拼起来。联合 Critic 要评价的是同一次真实发生的配合。

回放记录是当时已经发生的事，下次抽到它时，状态、动作和奖励仍是原来的数字。TD 目标要经过网络重新算；目标网络改变后，同一条旧经历也可能得到新目标。

## 6. TD 目标：下一步先生成联合动作，再评价

连续动作不需要枚举所有数字做 `argmax`。当前实现直接让两个目标 Actor 在下一观测上生成动作，再交给目标 Critic：

$$
y_i=r_i+\gamma(1-d)\,
Q_i^-\!\left(o_0',o_1',\mu_0^-(o_0'),\mu_1^-(o_1')\right).
$$

上标 `-` 表示目标网络。`d` 表示真正终止，`gamma=0.9` 表示未来一轮奖励只按九成计。第一轮后目标还含下一状态的估值；第二轮真正结束，`d=1`，目标就是当前奖励。

```python
gamma, tau = 0.9, 0.01

@torch.no_grad()
def td_targets(batch):
    next_actions = torch.stack(
        [target_actors[i](batch.next_obs[:, i]) for i in range(2)],
        dim=1,
    )
    targets = []
    for i in range(2):
        next_q = target_critics[i](batch.next_obs, next_actions)
        y = batch.rewards[:, i] + gamma * (1 - batch.terminated) * next_q
        targets.append(y)
    return targets
```

`batch.next_obs[:,i]` 取批次中智能体 `i` 的下一观测，形状 `[B,2]`。每个目标 Actor 输出 `[B,1]`，`stack(...,dim=1)` 得到 `[B,2,1]` 的下一联合动作。

这里既不用本步旧动作，也不加训练探索噪声。它回答的是“在这条经历的下一状态，目标策略会怎样做，目标 Critic 怎样评价”。

`rewards[:,i]`、终止标记和 `next_q` 都是 `[B,1]`，结果也必须是 `[B,1]`。若误把一个量压成 `[B]`，它与 `[B,1]` 运算可能广播成 `[B,B]`，把不同样本混在一起而不立即报错。

下面是两条转移的教学核算：

```python
reward = torch.tensor([[-0.36], [-0.04]])
terminated = torch.tensor([[0.0], [1.0]])
next_q = torch.tensor([[-0.10], [-0.90]])
y = reward + 0.9 * (1 - terminated) * next_q
print(y)  # [[-0.45], [-0.04]]
```

第二条的下一 Q 虽然算出了 -0.90，终止标记仍把未来项清零。终止步的奖励不能删除；需要删除的是它之后的未来价值。

`no_grad()` 把 TD 目标作为这次优化的固定数字。本环境在第二轮完成任务；如果只是被时间限制提前叫停，是否要保留最后观测的估值，见[终止和截断的两种掩码](/2026/10/06/rl-day4-reinforce-gae/gae/)，处理方式不同。

## 7. 一次联合更新：先训练 Critic，再替换本人的动作

Critic 先用真实联合动作拟合 TD 目标。更新 Actor 时，再从回放里换掉本人的动作，换进本人当前网络的输出，队友动作保持原样。这对应作者实现中只替换正在更新的智能体动作的写法。[2]

```python
from torch.nn import functional as F

def update_from_replay():
    batch = sample_batch()
    targets = td_targets(batch)
    critic_losses, actor_losses, grad_norms = [], [], []

    for i in range(2):
        q = critics[i](batch.obs, batch.actions)
        loss = F.mse_loss(q, targets[i])
        critic_optimizers[i].zero_grad(set_to_none=True)
        loss.backward()
        critic_optimizers[i].step()
        critic_losses.append(loss.item())

    for i in range(2):
        critic = critics[i]
        critic_optimizers[i].zero_grad(set_to_none=True)
        critic.requires_grad_(False)

        action_parts = [batch.actions[:, j] for j in range(2)]
        action_parts[i] = actors[i](batch.obs[:, i])
        actions_for_actor = torch.stack(action_parts, dim=1)
        loss = -critic(batch.obs, actions_for_actor).mean()
        actor_optimizers[i].zero_grad(set_to_none=True)
        loss.backward()
        norm_squared = sum(p.grad.square().sum() for p in actors[i].parameters())
        grad_norms.append(norm_squared.sqrt().item())
        actor_optimizers[i].step()
        critic.requires_grad_(True)
        actor_losses.append(loss.item())

    for online, target in zip(actors + critics, target_actors + target_critics):
        soft_update(online, target)
    return critic_losses, actor_losses, grad_norms
```

`zero_grad()` 清除先前的梯度，`backward()` 计算当前梯度，`step()` 才改参数。返回的梯度范数表示 Actor 收到的更新信号有多大；合作得分要等实际执行完两轮再看。

假设一条回放是 `(0.2,0.8)`，当前两个 Actor 输出为 0.9、0.7：

| 更新阶段 | 交给 Critic 的动作 | 目的 |
|---|---|---|
| Critic 拟合 | `(0.2,0.8)` | 学真实经历的评分 |
| Actor 0 更新 | `(0.9,0.8)` | 固定旧队友动作，调整本人 |
| Actor 1 更新 | `(0.2,0.7)` | 固定旧队友动作，调整本人 |

表里是便于手算的一条示例。每次循环重新从 `batch.actions` 构造列表，所以轮到 Actor 1 时，拿到的队友动作仍是回放中的 `0.2`。

`loss=-Q.mean()` 让优化器通过减小损失来提高平均评分。训练会对批里的队友动作求平均；Actor 真正选数时，输入仍只有自己的数和公共阶段。

`critic.requires_grad_(False)` 冻结的是 Critic 的参数，求导仍能经过 Critic 的输入回到本人 Actor。对本人动作做 `detach()`，或者把评分整个包进 `no_grad()`，才会切断这条路径。[3]

```python
actor = nn.Linear(1, 1, bias=False)
critic = nn.Linear(1, 1, bias=False)
with torch.no_grad():
    actor.weight.fill_(0.2)
    critic.weight.fill_(2.0)
critic.requires_grad_(False)

action = actor(torch.ones(1, 1))
loss = -critic(action).mean()
loss.backward()
print(actor.weight.grad)   # tensor([[-2.]])
print(critic.weight.grad)  # None
```

这个独立小例子里，Critic 权重不收梯度，Actor 仍能收到动作方向。若 `action.detach()` 再评分，本例的损失就不需要梯度，无法沿这条路径更新 Actor。

最后软同步：

```python
@torch.no_grad()
def soft_update(online, target):
    for online_p, target_p in zip(online.parameters(), target.parameters()):
        target_p.mul_(1 - tau).add_(online_p, alpha=tau)
```

每个目标参数保留 99% 的旧值，再加入 1% 的在线值。比如旧目标 0.2、在线 0.8，一次同步后为 `0.99×0.2+0.01×0.8=0.206`。它不是“每 100 次硬复制一次”，而是每轮小幅移动。

两个目标 Actor 和两个目标 Critic 都同步。只同步目标 Critic，会继续用没有跟上的目标 Actor 生成下一步动作；只同步目标 Actor，则评分器没有跟上。一次联合更新包含两次 Critic `step()`、两次 Actor `step()`，最后四个目标副本完成软同步。

## 8. 主训练循环：随机预热、带噪声动作、每步回放更新

本次实际配置如下：

| 配置 | 值 | 作用 |
|---|---:|---|
| 随机种子 | 7 | 固定本次初始化与随机过程 |
| 共同训练回合 | 2500 | 每局两轮 |
| 随机预热环境步 | 256 | 先积累较丰富的联合动作 |
| 批量大小 | 128 | 每次抽取 128 条共同转移 |
| 回放容量 | 5000 | 保留转移数量 |
| Actor / Critic 学习率 | 0.0005 / 0.002 | 两类参数更新幅度 |
| 折扣 / 软同步系数 | 0.9 / 0.01 | 未来价值与目标移动幅度 |
| 探索标准差 | 0.3 → 0.08 | 逐回合减少高斯噪声 |

这一段用到[主文第 3 节的 `TwoRoundAgreementEnv`](/2026/10/12/rl-day8-maddpg-training/#3-两轮实验动作直接设置数值共同奖励来自差距)。环境的 `reset()` 返回 `(x0, x1, phase)`；`step(actions)` 接收两人实际写的数，返回下一状态、共同奖励以及任务是否结束。将该环境类与上面的网络、回放和更新函数放在一起，就能接上完整训练循环。

环境交互只需要干净局部动作：

```python
@torch.no_grad()
def local_actions(actor_models, state):
    obs = local_observations(state)
    return tuple(
        actor_models[i](obs[i:i+1]).item() for i in range(2)
    )
```

`obs[i:i+1]` 保留批次维度，得到 `[1,2]`；`.item()` 把单元素输出变成环境接受的 Python 数值。这段用于真实采样与评估，不能替换第七节需要梯度的 Actor 调用。

```python
num_episodes, start_steps = 2500, 256
noise_start, noise_end = 0.3, 0.08
exploration_rng = random.Random(seed + 2)

train_env = TwoRoundAgreementEnv(seed + 3)
environment_steps = joint_updates = target_syncs = 0
episode_rewards = []

for episode in range(num_episodes):
    state, finished, total_reward = train_env.reset(), False, 0.0
    noise_std = (
        noise_start
        + (noise_end - noise_start) * episode / (num_episodes - 1)
    )
    while not finished:
        if environment_steps < start_steps:
            actions = tuple(
                exploration_rng.uniform(-1, 1) for _ in range(2)
            )
        else:
            clean_actions = local_actions(actors, state)
            actions = tuple(
                max(-1.0, min(1.0, a + exploration_rng.gauss(0, noise_std)))
                for a in clean_actions
            )

        next_state, reward, finished = train_env.step(actions)
        save_transition(state, actions, reward, next_state, finished)
        environment_steps += 1
        total_reward += reward
        if environment_steps >= start_steps:
            c_losses, a_losses, grad_norms = update_from_replay()
            joint_updates += 1
            target_syncs += 1
        state = next_state
    episode_rewards.append(total_reward)
```

每步的顺序是：选动作 → 环境真实执行 → 保存共同转移 → 更新网络。预热结束后，`clean_actions` 加独立高斯噪声，再裁剪到动作范围；回放必须保存这里的 `actions`。

第 256 个动作仍在随机预热中。它执行并保存之后，计数达到 256，立即发生第一次更新。因此 5000 个环境步对应：

```text
联合更新 = 5000 - 256 + 1 = 4745
优化器 step = 4745 × 4 = 18980
完整软同步 = 4745
```

这些计数分别描述环境交互、网络优化和目标移动，不能混成一个“训练步”。冻结评估用独立环境，不写入回放，也不增加训练环境步数。

## 资料

1. [Lowe 等，MADDPG 论文](https://arxiv.org/abs/1706.02275)，第 4.1 节给出集中式 Critic 与确定性策略梯度。
2. [作者实现 `maddpg.py`](https://github.com/openai/maddpg/blob/master/maddpg/trainer/maddpg.py)，可看 `p_train` 如何用当前 Actor 输出替换本人的回放动作。
3. [PyTorch 自动求导说明](https://docs.pytorch.org/docs/stable/notes/autograd.html)，解释 `requires_grad` 和 `no_grad()` 分别控制什么。
