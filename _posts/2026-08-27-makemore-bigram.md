---
title: "Day3：从 bigram 统计到可训练的字符语言模型"
date: 2026-08-27 08:00:00 +0800
categories: [技术实践]
tags: [makemore, bigram, 字符语言模型, PyTorch, softmax, autograd]
excerpt: 追随Karpathy大佬的脚步
math: true
---
# 从 bigram 统计到可训练的字符语言模型（活字乱刷术）

![Day3：活字乱刷术](/assets/img/day3-cover.png)

## 前言

昨天的打脸来的好快，本来想CNN，RNN暂且放下不表，但是AI给我的下一个学习建议赫然又回到了基础巩固。不过也好，那就让我们找回从前的记忆吧。

## 0. 这个项目从什么讲到了什么

```text
名字数据
→ 添加起止边界
→ Python 字典统计 bigram
→ 27×27 计数矩阵 N
→ 按行归一化得到条件概率
→ 加一平滑
→ 按概率采样
→ log-likelihood / NLL 评价
→ one-hot × W
→ logits → softmax
→ autograd 反向传播
→ L2 正则化
→ 梯度下降
→ 训练后采样
```

前半段是“算出来的模型”：直接从数据里统计字符转移次数；后半段是“学出来的模型”：用一个可训练矩阵 `W` 表示同样的转移分布。最值得记住的转折是：

> 计数矩阵和神经网络不是两个完全无关的模型。one-hot 输入乘以 `W`，本质上仍然是在查询“当前字符应该转移到哪些字符”；区别只在于表中的数值是直接统计得到的，还是通过损失函数和梯度下降学出来的。

![直接计数模型与可训练神经网络的两条路径](/assets/img/day3-count-vs-neural.svg)

## 1. 数据集与起止标记

### 1.1 先看看数据长什么样

notebook 从 `names.txt` 中读取数据：每一行是一个英文名字。当前文件共有 **32033** 个名字，最短长度为 **2**，最长长度为 **15**；数据中出现的普通字符是 `a-z` 共 26 个。

```python
words = open('names.txt', 'r', encoding='utf-8').read().splitlines()

words[:10]
len(words)
min(len(w) for w in words)
max(len(w) for w in words)
```

前十个样本包括：

```text
emma, olivia, ava, isabella, sophia,
charlotte, mia, amelia, harper, evelyn
```

这里的四个检查分别回答了四个问题：文件有没有正常读进来、样本数量是多少、名字长度范围多大、后面每个名字最多可能产生多少个字符转移。

### 1.2 一个名字为什么要加边界

只看 `emma` 本身，模型只能看到：

```text
 e → m → m → a
```

它不知道 `e` 是名字的第一个字符，也不知道 `a` 后面应该停止。因此要显式加上开始和结束标记：

```python
chs = ['<S>'] + list(w) + ['<E>']
```

`emma` 变成：

```text
<S>  e  m  m  a  <E>
```

于是得到 5 个 bigram：

```text
(<S>, e), (e, m), (m, m), (m, a), (a, <E>)
```

一个长度为 `L` 的名字会贡献 `L + 1` 个转移。整个数据集一共贡献：

$$
\sum_{w\in D}(|w|+1)=228146
$$

这 `228146` 个转移就是后面神经网络训练时的监督样本数。

### 1.3 两套边界写法

notebook 前面的 Python 字典示例使用 `<S>` 和 `<E>` 两个不同标记；为了把矩阵类别数压缩到 27 个，后面的 Tensor 版本使用同一个 `.` 同时表示开始和结束。（这里是因为形如“结束”+“开始”这种形态根本不可能出现，会导致内存空间的浪费）

因此：

- 第 0 行 `N[0, :]` 主要表示从名字开始边界出发的首字符分布；
- 第 0 列 `N[:, 0]` 主要表示某个字符后面结束名字的次数；
- 但矩阵本身看不出“这个 0 是开始还是结束”，只是借助行列位置赋予不同语义。

这是为了教学和实现方便的简化，不是说开始和结束在语言建模中天然是同一个状态。

## 2. Bigram：先数相邻字符

### 2.1 用字典统计转移次数

最直接的写法是用 Python 字典：

```python
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
```

`zip(chs, chs[1:])` 是这里的关键：

```text
chs       = [<S>, e, m, m, a, <E>]
chs[1:]   = [ e,  m, m, a, <E>]
zip(...)  = [(<S>,e), (e,m), (m,m), (m,a), (a,<E>)]
```

`b.get(bigram, 0) + 1` 的意思是：如果这对字符第一次出现，就从 0 开始；否则就在原有计数上加 1。

数学上，字典中的每个值都是：

$$
C(a,b)=\#\{t\mid c_t=a,\ c_{t+1}=b\}
$$

它只描述相邻两个字符的共现次数，因此叫 **bigram**。这是一个一阶 Markov 假设：下一个字符只依赖当前字符，不看更早的历史。

### 2.2 高频转移不等于高频字符

按计数排序后，前几名是：

| bigram | 次数 |
| --- | ---: |
| `n → <E>` | 6763 |
| `a → <E>` | 6640 |
| `a → n` | 5438 |
| `<S> → a` | 4410 |
| `e → <E>` | 3983 |

准确的说法应该是“`n → <E>` 这条转移很常见”，而不是“`n` 是最常见字符”。因为这里统计的是有方向的相邻字符对，而且结束标记也参与了统计。

## 3. 计数矩阵：把字典变成一张表

### 3.1 字符和索引

为了进行矩阵运算，notebook 建立两个方向的映射：

```python
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
```

`stoi` 是 string-to-integer，`itos` 是 integer-to-string：

```text
. → 0
 a → 1
 b → 2
...
 z → 26
```

然后创建计数矩阵：

```python
N = torch.zeros((27, 27), dtype=torch.int32)
```

它的含义是：

- `N.shape == (27, 27)`；
- 行索引 `i` 表示当前字符；
- 列索引 `j` 表示下一个字符；
- `N[i, j]` 表示 `i → j` 出现了多少次；
- `dtype=torch.int32` 说明这里存的是计数，不是概率。

### 3.2 把 bigram 映射到矩阵坐标

```python
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
```

这段代码只是把前面的字典键 `(ch1, ch2)` 换成了 `(ix1, ix2)`。例如：

```text
. → a  对应 N[0, 1]
a → n  对应 N[1, 14]
```

notebook 中可以看到：

```python
N[0]
```

其前几项是：

```text
[0, 4410, 1306, 1542, 1690, 1531, ...]
```

所以 `.` 到 `a` 的原始计数是 **4410**，到 `k` 是 **2963**，到 `m` 是 **2538**。

### 3.3 热图应该怎么看

![字符 Bigram 计数矩阵](/assets/img/bigram-count-matrix.svg)

横轴是下一个字符，纵轴是当前字符。每一行回答一个问题：

> 当前字符固定为 `i` 时，数据中下一字符 `j` 的转移次数如何分布？

图中颜色使用 `log(1 + count)` 映射，而不是直接使用原始计数。因为高频转移和低频转移差距很大，直接映射会让大部分低频格子看起来几乎一样；对数变换只改变显示尺度，不改变计数矩阵里的事实。

有一个细节值得注意：`. → a` 的计数位于第 0 行，而 `a → .` 的计数位于第 1 行第 0 列。它们都使用索引 0，但在行、列中的语义不同。

## 4. 从计数到条件概率

### 4.1 未平滑的首字符分布

只看第 0 行：

```python
p = N[0].float()
p = p / p.sum()
p
```

得到的是从开始边界出发时的首字符分布：

$$
P(j\mid i)=\frac{N_{ij}}{\sum_kN_{ik}}
$$

特别地，首字符分布是：

$$
P(j\mid .)=\frac{N_{0j}}{\sum_kN_{0k}}
$$

notebook 中 `. → a` 的**未平滑**概率约为 `0.1377`。它来自：

```python
N[0, 1] / N[0].sum()
```

这个数字不要和后面的加一平滑概率混淆。

### 4.2 加一平滑：避免零概率

如果一个 bigram 在训练集里从未出现，那么直接归一化会得到：

$$
N_{ij}=0\quad\Longrightarrow\quad P(j\mid i)=0
$$

而 NLL 中需要取对数：

$$
\log 0=-\infty
$$

于是 notebook 使用加一平滑：

```python
P = (N + 1).float()
P = P / P.sum(1, keepdim=True)
```

数学上是：

$$
P(j\mid i)=
\frac{N_{ij}+1}{\sum_k(N_{ik}+1)}
$$

一行有 27 个类别，因此分母等于：

$$
\sum_kN_{ik}+27
$$

它的含义不是“所有字符变得同样可能”，而是给每个没有观察到的转移留下一个很小的概率，避免模型因为“这次数据没出现”就断言“永远不可能”。

## 5. `keepdim=True`：广播方向决定归一化方向（视频里强调）

### 5.1 三个形状

这一行代码看起来简单，但它同时考察了维度、求和和广播：

```python
P = P / P.sum(1, keepdim=True)
```

对一个 `P.shape == (27, 27)` 的矩阵来说：

| 表达式 | shape | 含义 |
| --- | --- | --- |
| `P` | `(27, 27)` | 27 个当前字符 × 27 个候选下一个字符 |
| `P.sum(1)` | `(27,)` | 每一行的总和，但压掉了被求和的维度 |
| `P.sum(1, keepdim=True)` | `(27, 1)` | 每一行的总和，并保留一个长度为 1 的列维度 |

正确写法的除法是：

```text
(27, 27) / (27, 1)
```

`(27, 1)` 会沿着列方向广播成 `(27, 27)`，于是：

```text
第 0 行 ÷ 第 0 行的总和
第 1 行 ÷ 第 1 行的总和
...
```

每一行都被自己的分母归一化。

### 5.2 为什么不建议省略 `keepdim`

若写成：

```python
P = P / P.sum(1)
```

分母形状是 `(27,)`。PyTorch 广播从最后一个维度对齐，因此它把这个向量视作“最后一维上的 27 个分母”，也就是按列对齐，而不是明确按行对齐。

在一个非方阵小例子中，它甚至会直接报形状错误：

```python
A = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.],
])
A.sum(1).shape                  # torch.Size([2])
A.sum(1, keepdim=True).shape    # torch.Size([2, 1])
```

`(2, 3) / (2,)` 的最后一维是 `3` 对 `2`，无法广播；而 `(2, 3) / (2, 1)` 可以明确按行工作。

当前 notebook 的矩阵恰好是方阵 `(27, 27)`，省略 `keepdim` 可能不报错，但这只是形状碰巧相容，不能说明归一化方向正确。

![keepdim=True 与 Bigram 概率的逐行广播](/assets/img/bigram-keepdim-broadcast.svg)

可以用下面这句检查结果：

```python
P[0].sum()
```

输出为 `tensor(1.)`，说明第 0 行已经是一个概率分布。更完整的检查可以写成：

```python
torch.allclose(P.sum(1), torch.ones(27))
```

### 5.3 `keepdim` 只影响形状，不改变求和结果

`keepdim=True` 并没有改变“求和”本身，只是保留了被压掉的维度。它相当于给广播机制留下了一个明确的方向提示：

> 这是每一行的一个分母，请沿列复制它。

## 6. 按概率逐字符采样

有了 `P`，就可以让模型生成名字：

```python
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(
            p, num_samples=1, replacement=True, generator=g
        ).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
```

每一步的形状和类型变化是：

```text
P             : (27, 27)
P[ix]         : (27,)       当前字符对应的下一字符分布
multinomial   : (1,)        采出的类别索引
.item()       : Python int   方便作为下一轮的 ix
```

`replacement=True` 表示有放回采样：抽到某个字符后，下一步仍然可以再次抽到它。`ix == 0` 时遇到结束边界，生成停止。

固定随机种子后，当前 notebook 的统计模型生成结果是：

```text
cexze.
momasurailezitynn.
konimittain.
llayn.
ka.
```

这些字符串看起来像名字，是因为局部拼写规律已经被保留下来；但它们不代表模型理解了词义、语法或真实姓名的语义结构。

![从训练后的开始边界出发，比较两种模型的首字符分布](/assets/img/day3-start-distribution.svg)

上图比较的是平滑计数模型与训练后神经网络在首字符上的分布。它不是在比较“谁更聪明”，只是把两种参数化方式放在同一个条件分布上观察。

## 7. 似然、对数似然与 NLL

### 7.1 一个名字的概率如何分解

对于字符序列：

```text
. → e → m → m → a → .
```

bigram 模型给出的整条序列概率是：

$$
P(w)=\prod_tP(c_{t+1}\mid c_t)
$$

很多小概率直接连乘容易发生数值下溢，所以取对数：

$$
\log P(w)=\sum_t\log P(c_{t+1}\mid c_t)
$$

然后取负号，把“概率越大越好”改写成“损失越小越好”：

$$
\mathrm{NLL}
=-\frac{1}{n}\sum_{t=1}^{n}
\log P(c_{t+1}\mid c_t)
$$

### 7.2 notebook 如何展开这个定义

```python
total_log_likelihood = 0.0
num_bigrams = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        total_log_likelihood += logprob
        num_bigrams += 1

nll = -total_log_likelihood
print(nll / num_bigrams)
```

当前输出为：

```text
total_log_likelihood = -559951.5625
nll                    =  559951.5625
average NLL            =  2.4543561935424805
```

`2.4543561935424805` 是全体字符转移事件的平均 NLL，使用自然对数，单位可以称为 **nats**。它不是“每个名字的 NLL”，因为分母 `num_bigrams` 统计的是所有名字贡献的字符转移数量，即 `228146`。

### 7.3 目标到底是什么

若把 `P` 看成模型参数化出来的概率，那么评价的目标是：

$$
\max\sum_{(x,y)\in D}\log P(y\mid x)
$$

训练时通常写成等价的最小化问题：

$$
\min -\frac{1}{n}\sum_{(x,y)\in D}\log P(y\mid x)
$$

因此：

- log-likelihood 越大越好；
- NLL 越小越好；
- 真实目标字符的概率越高，它对平均 NLL 的贡献越小。

## 8. 用一个名字拆开神经网络输入

### 8.1 `emma` 变成 5 个监督样本

为了逐样本观察计算过程，notebook 暂时只取 `words[:1]`：

```python
xs, ys = [], []
for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs, dtype=torch.long)
ys = torch.tensor(ys, dtype=torch.long)
```

当前第一个名字是 `emma`，所以：

```text
xs = [0, 5, 13, 13, 1]
ys = [5, 13, 13, 1, 0]
```

对应关系为：

```text
. → e
 e → m
 m → m
 m → a
 a → .
```

`xs[i]` 是输入字符，`ys[i]` 是真实的下一个字符。这 5 条样本只用于演示，不是最终训练集。

### 8.2 one-hot 的形状

```python
xenc = F.one_hot(xs, num_classes=27).float()
```

得到：

```text
xs.shape   == (5,)
xenc.shape == (5, 27)
```

每一行是一个当前字符的 one-hot 向量：只有对应字符的位置是 `1`，其余位置是 `0`。`.float()` 很重要，因为后面要和浮点权重矩阵相乘。

## 9. one-hot × W：把表变成可训练参数

### 9.1 初始化权重矩阵

```python
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
```

`W.shape == (27, 27)`：

- 行 `i` 对应当前字符；
- 列 `j` 对应候选下一个字符；
- `W[i, j]` 是一个未归一化的分数，也就是 logit；
- `requires_grad=True` 让它成为自动微分图中的可训练叶子节点。

### 9.2 one-hot 矩阵乘法其实是在选行

```python
logits = xenc @ W
```

形状是：

```text
(5, 27) @ (27, 27) = (5, 27)
```

如果 `xenc[i]` 是第 `r` 个字符的 one-hot 向量，那么：

$$
\mathrm{onehot}(r)W=W_{r,:}
$$

也就是说，矩阵乘法并没有把 27 行信息混合成一个神秘向量；它只是取出了当前字符对应的那一行。对于这个最小模型，`W` 看起来像神经网络参数，结构上却仍然是一张可学习的字符转移表。

### 9.3 从 logits 到概率

notebook 为了对应前面的“计数 → 归一化”过程，手动展开了 softmax：

```python
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
```

数学形式是：

$$
P(j\mid i)
=\frac{\exp(W_{ij})}{\sum_k\exp(W_{ik})}
$$

三步分别是：

1. `W` 提供未归一化分数；
2. `exp` 把任意实数变成正数；
3. 按行归一化，让每一行加起来等于 1。

这里的 `counts` 只能叫模型生成的“正值权重”或“伪计数”，不能直接说它等于原始计数矩阵 `N`。只有当 `W` 恰好等于某个计数矩阵的对数（再允许每一行加一个常数）时，softmax 后的分布才会和那个计数模型一致。

生产代码通常会写成：

```python
log_probs = F.log_softmax(logits, dim=1)
loss = -log_probs[torch.arange(xs.numel()), ys].mean()
```

或者直接使用：

```python
loss = F.cross_entropy(logits, ys)
```

这样可以避免先 `exp` 再取 `log` 带来的溢出风险。当前 notebook 保留展开写法，是为了让每一步都能和前面的概率模型对上。

## 10. 这次 `loss.backward()` 报错到底错在哪里

### 10.1 报错信息的直译

原来的错误是：

```text
RuntimeError:
element 0 of tensors does not require grad and does not have a grad_fn
```

它不是说 `loss` 的数值不能算，而是说：

> 这个 `loss` 没有连接到一个需要梯度的计算图，因此 PyTorch 不知道应该沿哪条路径求导。

### 10.2 真正的错误行

错误版本的初始化是：

```python
W = torch.rand((27, 27), generator=g)
```

默认情况下：

```python
W.requires_grad == False
```

于是后面的链路虽然能做前向数值计算：

```text
W → logits → counts → probs → loss
```

但整个链路都不会建立 autograd 记录。此时通常可以观察到：

```python
loss.requires_grad   # False
loss.grad_fn         # None
```

所以执行：

```python
loss.backward()
```

就没有反向路径可走。

![Bigram 模型的前向计算与 autograd 路径](/assets/img/bigram-autograd-chain.svg)

### 10.3 正确初始化与重新运行顺序

修复为：

```python
W = torch.randn(
    (27, 27),
    generator=g,
    requires_grad=True,
)
```

也可以先创建普通 Tensor，再显式开启：

```python
W = torch.rand((27, 27), generator=g)
W.requires_grad_(True)
```

但下面这一句**不能**修复问题：

```python
W.grad = None
```

它只是在清空上一轮已经算出的梯度，不会把 `requires_grad` 从 `False` 改成 `True`。

修复后不能只重新执行 `loss.backward()`，而要按顺序重新运行：

```text
重新创建 W
→ 重新计算 logits
→ 重新计算 counts / probs
→ 重新计算 loss
→ loss.backward()
```

因为旧的 `loss` 仍然连接着旧的、没有梯度记录的前向结果。

当前 notebook 中，教学 cell 会直接打印：

```text
W.shape=torch.Size([27, 27]), W.requires_grad=True
```

而训练前查看：

```python
(W ** 2).mean()
```

会得到带有 `grad_fn` 的结果，这也是 `W` 已经接入计算图的一个证据。

## 11. 单样本 NLL：高级索引如何选出正确答案

先逐个样本观察真实目标概率：

```python
nlls = torch.zeros(xs.numel())
for i in range(xs.numel()):
    x = xs[i].item()
    y = ys[i].item()
    p_true = probs[i, y]
    logp = torch.log(p_true)
    nll = -logp
    nlls[i] = nll
```

`probs[i]` 有 27 个候选字符的概率，但训练只关心第 `y` 个真实答案。于是：

```python
probs[torch.arange(xs.numel()), ys]
```

一次性选出：

```text
第 0 个样本的 probs[0, ys[0]]
第 1 个样本的 probs[1, ys[1]]
...
```

形状由 `(5, 27)` 变成 `(5,)`。最后：

```python
loss = -probs[torch.arange(xs.numel()), ys].log().mean()
```

就是 5 条样本的平均 NLL。当前随机初始化下，教学 batch 的平均 NLL 约为：

```text
3.4815192222595215
```

这个值只是随机参数下的起点，不代表完整数据集的训练效果。

## 12. 全量数据训练

### 12.1 回到全部名字

教学 batch 只包含 `emma` 的 5 个转移；真正训练时重新遍历全部名字：

```python
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs, dtype=torch.long)
ys = torch.tensor(ys, dtype=torch.long)
num = xs.nelement()
print('训练样本数：', num)
```

当前输出：

```text
训练样本数：228146
```

形状变成：

```text
xs.shape   == (228146,)
ys.shape   == (228146,)
xenc.shape == (228146, 27)
W.shape    == (27, 27)
logits     == (228146, 27)
probs      == (228146, 27)
```

### 12.2 训练目标

notebook 使用的完整目标是：

```python
nll = -probs[torch.arange(num), ys].log().mean()
reg_loss = reg_strength * (W ** 2).mean()
loss = nll + reg_loss
```

其中：

$$
L_{\mathrm{NLL}}
=-\frac{1}{n}\sum_i\log P(y_i\mid x_i)
$$

$$
L_{\mathrm{reg}}
=\lambda\cdot\mathrm{mean}(W^2)
$$

$$
L=L_{\mathrm{NLL}}+L_{\mathrm{reg}}
$$

当前超参数是：

| 超参数 | 数值 |
| --- | ---: |
| 训练轮数 | 100 |
| 学习率 | 50.0 |
| L2 正则系数 | 0.01 |
| 字符类别数 | 27 |

这里的学习率 `50.0` 看起来很大，但它是这个小模型、这个初始化和这个损失尺度下的实验超参数，不能直接搬到其他模型。

### 12.3 加一平滑和 L2 正则不是一回事

| 比较项 | 加一平滑 | L2 正则化 |
| --- | --- | --- |
| 作用对象 | 计数或概率 | 参数矩阵 `W` |
| 代码 | `N + 1` | `loss + 0.01 * mean(W**2)` |
| 直接作用 | 给零计数转移留下概率 | 惩罚过大的参数幅度 |
| 主要目的 | 避免 `log(0)` | 限制 logits 过度极端、缓解过拟合 |
| 是否等价 | 否 | 否 |

“加一”是概率层面的伪计数；`0.01` 是目标函数中的权重约束。它们都带有“平滑”意味，但不是同一个层面的东西。

### 12.4 清零、反传、更新

完整循环的核心是：

```python
for k in range(num_steps):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    nll = -probs[torch.arange(num), ys].log().mean()
    reg_loss = reg_strength * (W ** 2).mean()
    loss = nll + reg_loss

    W.grad = None
    loss.backward()

    with torch.no_grad():
        W -= learning_rate * W.grad
```

它对应深度学习里最小的四步：

1. **前向**：从当前字符得到 27 个候选下一个字符的概率；
2. **计算目标**：真实字符概率越小，NLL 越大；
3. **反向**：autograd 计算 `d(loss) / dW`；
4. **更新**：沿负梯度方向移动参数。

这里使用 `W.grad = None` 是为了避免梯度跨轮累加；使用 `torch.no_grad()` 是为了让参数更新本身不再被记录进下一轮计算图。

课程原始写法常见的是：

```python
W.data += -50 * W.grad
```

它能表达同样的更新方向，但 `.data` 会绕开 autograd 的安全检查。教学代码可以用来理解机制，正式代码更推荐 `torch.no_grad()`。

## 13. 训练结果

notebook 现在记录了三条曲线：总目标、纯 NLL、L2 正则项。

![Bigram 神经网络训练损失与正则项](/assets/img/day3-training-loss.svg)

当前输出的关键节点是：

| 训练轮次 | total loss | mean NLL | L2 正则项 |
| ---: | ---: | ---: | ---: |
| 0 | 3.768619 | 3.758954 | 0.009665 |
| 10 | 2.696506 | 2.688079 | 0.008426 |
| 50 | 2.509855 | 2.497399 | 0.012456 |
| 90 | 2.491893 | 2.477113 | 0.014781 |
| 99 | 2.490130 | 2.474973 | 0.015157 |

可以观察到：

- 总目标从 `3.768619` 降到 `2.490130`；
- 纯 NLL 从 `3.758954` 降到 `2.474973`；
- 正则项后期反而从 `0.009665` 增加到 `0.015157`。

这并不矛盾。模型为了更贴合数据，可能会拉开不同字符 logit 的差距，使 NLL 下降；同时参数平方均值上升，正则项增加。总目标是否下降，要看两部分合在一起的结果。

### 13.1 这组结果能说明什么

它能说明：在当前随机初始化、学习率、正则化和 100 轮训练的设定下，参数更新确实让训练目标下降了。

它不能单独说明：

- 模型已经充分收敛；
- 神经网络优于直接计数；
- 模型具有泛化能力；
- 字符串已经“理解”了语言。

因为这里没有划分验证集，也没有报告困惑度、准确率或更长上下文上的泛化结果。这个实验的主要价值是把训练闭环跑通并观察参数如何改变概率分布。

## 14. 训练后采样：为什么不保证和统计模型一样

训练完成后，每一步都重新从 `W` 计算概率：

```python
with torch.no_grad():
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    p = counts / counts.sum(1, keepdim=True)
```

当前神经网络采样结果是：

```text
cexze.
momasurailezityha.
konimittain.
llayn.
ka.
```

统计模型和神经网络的第一、第三、第四、第五个样本相同，第二个样本不同：

```text
统计模型：momasurailezitynn.
神经网络：momasurailezityha.
```

这不是代码必须产生完全一致结果的证明或反证，而是一个提醒：

- 统计模型直接使用加一平滑后的 `P`；
- 神经网络使用训练后的 `W` 再经过 softmax；
- 神经网络还有 L2 正则和有限步数优化；
- 固定随机种子只固定随机数消费顺序，不会把两套不同的概率分布变成同一套。

只有当两者在每一个当前字符上的条件分布完全一致，并且采样调用顺序也一致时，逐个字符串才有可能一致。当前实验更合理的结论是：两种模型都抓住了局部字符转移规律，但参数化过程不同。

## 15. 收工小结

1. **bigram 是最小字符语言模型**：只根据当前字符预测下一个字符，一个长度为 `L` 的名字贡献 `L + 1` 个转移。
2. **计数矩阵的每一行就是一个条件分布的原材料**：`N[i, j]` 是转移次数，按行归一化后得到 `P(j|i)`。
3. **加一平滑解决零概率**：它作用在计数/概率层面，让未见过的转移也能参与对数似然计算。
4. **NLL、autograd、梯度下降组成训练闭环**：NLL 衡量真实字符概率有多低，autograd 计算参数该往哪里改，梯度下降执行更新。
7. **局部规律不等于语言理解**：模型已经能生成“像名字”的字符串，但它只记住了相邻字符统计。

## 附 · 对应代码

完整 notebook 代码（uild_makemore_yay.ipynb）已在本地附近完成教学化注释与阶段划分，线上版在此按 Day2 的做法原样贴出关键逻辑便于复现。完整可执行 notebook 见仓库内  7_MyProject/03_makemore/build_makemore_yay.ipynb。

# # Day3：从 bigram 统计到可训练的字符语言模型

思路：把名字数据缩小成一个最小字符语言模型：给定当前字符，只预测下一个字符。前半段用计数直接得到概率，后半段把同一张转移表改写成 one-hot 乘 W 再接 softmax，并用 NLL、autograd、L2 正则和梯度下降训练它——从简单的数学推导到一个真正真正的“模型”。

```python

# 阶段 1：读取数据。names.txt 与本 notebook 放在同一目录，因此这里使用相对路径。
# 每行是一个英文名字；splitlines() 把文本拆成 list[str]，后面每个名字会贡献若干个字符转移。
words = open('names.txt', 'r', encoding='utf-8').read().splitlines()

```

```python

# 先查看前 10 个样本，确认文件按“每行一个名字”的格式读入。
words[:10]

```

```python

len(words)

```

```python

# 观察名字长度范围；长度为 L 的名字加上起止边界后会产生 L + 1 个 bigram。
min(len(w) for w in words)

```

```python

# 最长名字长度；这个统计帮助我们估算每个名字可能产生的转移数量。
max(len(w) for w in words)

```

# ## 阶段 2：统计字符 bigram

一个长度为 L 的名字会贡献 L + 1 个相邻字符转移。开始和结束边界让模型同时学习名字从哪里开始、何时停止。

```python

# 阶段 2：先用 Python 字典统计字符 bigram。
# bigram 只看相邻两个字符；<S> 是开始标记，<E> 是结束标记。
# 例如 emma -> <S>, e, m, m, a, <E>，会贡献 5 个转移。
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

```

```python

# 按出现次数从高到低查看转移，先观察数据中的局部字符规律。
# 这里的数字是计数，不是概率；同一个前字符对应的所有后字符计数还要再归一化。
sorted(b.items(), key=lambda x: -x[1])

```

# ## 阶段 3：从字典到矩阵

N[i, j] 是当前字符 i 到下一个字符 j 的计数。矩阵形式让后续的归一化、批量计算和矩阵乘法成为可能。

```python

# 阶段 3：把字典统计改写为 PyTorch Tensor，后续才能使用矩阵运算和 autograd。
import torch

```

```python

# N[i, j] 记录字符 i 后面出现字符 j 的次数。
# 26 个小写字母 + 1 个边界符号 . = 27 个类别，因此 N.shape == (27, 27)。
N = torch.zeros((27, 27), dtype=torch.int32)

```

```python

# 建立双向词表：stoi 把字符映射到整数，itos 把整数映射回字符。
# 矩阵版本用同一个 . 同时表示名字开始和结束：第 0 行主要表示起始转移，第 0 列主要表示结束转移。
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

```

```python

# 把每个名字的相邻字符映射为矩阵坐标并累加。
# 当前字符是行索引 ix1，下一个字符是列索引 ix2。
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

```

```python

# 可视化 27×27 的 bigram 计数矩阵。
# 纵轴是当前字符，横轴是下一个字符；颜色越深表示该转移出现得越多。
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
%matplotlib inline

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
plt.title('字符 Bigram 计数矩阵：行是当前字符，列是下一个字符')
plt.axis('off')

```

```python

# 查看从开始状态 . 出发的所有首字符计数；第 0 行表示起始转移。
N[0]

```

```python

# 把首字符计数除以总数，得到 P(next | start)。
# p.shape == [27]，p.sum() == 1；第 0 项对应再次采到边界符号。
p = N[0].float()
p = p / p.sum()
p

```

```python

# 固定随机种子，按首字符概率采样一次。
# 固定种子只保证同一段调用顺序可复现，不代表换一套概率分布后仍会得到相同字符串。
g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
ix
itos[ix]

```

```python

# 先用 3 类分布做一个最小例子，理解 multinomial 的输入格式。
# 输入只需是非负权重；这里先归一化成概率，便于直接解释。
g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()
p

```

```python

# replacement=True 表示有放回采样：每次抽完后仍可再次抽到同一类别。
torch.multinomial(p, num_samples=100, replacement=True, generator=g)

```

```python

# multinomial 接收的一维概率向量形状是 [类别数]。
p.shape

```

# ## 阶段 4：条件概率、平滑与采样

每一行归一化后就是给定当前字符时的下一个字符概率。加一平滑为未见过转移留下非零概率，multinomial 按该分布抽取字符。

```python

# 加一平滑：给每个可能的 bigram 先加一个伪计数，避免未见过的转移得到概率 0。
# N.shape == [27, 27]；按 dim=1 求和后保留维度，分母形状是 [27, 1]。
# [27, 27] / [27, 1] 会沿列方向广播：每一行除以自己的行和。
# 不要去掉 keepdim=True：得到 [27] 后，方阵虽然可能仍能运行，但广播语义会从最后一维对齐，容易把行统计量错配到列。
P = (N + 1).float()
P = P / P.sum(1, keepdim=True)

```

```python

# 检查每一行是否都是概率分布；这里检查第 0 行，正确结果应为 1。
P[0].sum()

```

```python

# 阶段 4：用完整的平滑矩阵逐字符生成名字。
# 当前 ix 是上一个字符；P[ix] 是下一个字符的条件概率向量。采到 0 就遇到结束边界。
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

```

# ## 阶段 5：用 NLL 评价概率模型

名字概率是每一步条件概率的连乘。取对数把连乘变成求和，取负号后就是训练中常见的平均负对数似然。

```python

# 阶段 5：用真实数据评价这个概率模型。
# 目标是让真实 bigram 的条件概率尽可能大：最大化总 log-likelihood。
# 实际训练通常改写成最小化平均负对数似然（NLL），这里先用循环把定义展开。
total_log_likelihood = 0.0
num_bigrams = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        total_log_likelihood += logprob
        num_bigrams += 1
        # print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{total_log_likelihood=}')
nll = -total_log_likelihood
print(f'{nll=}')
print(f'{nll / num_bigrams}')

```

# ## 阶段 6：把统计表改写成神经网络

one-hot 输入只会选中 W 的一行，所以 one-hot 乘 W 本质上仍是在查一张字符转移表；区别是每个表项从计数变成了可学习的 logit。

```python

# 阶段 6：只用第一个名字构造一个 5 条样本的教学 batch。
# emma -> .e, em, mm, ma, a.；xs 是当前字符，ys 是真实的下一个字符。
# 这只是为了逐样本观察 forward/loss/backward，不是最终训练集。
xs, ys = [], []
for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

# torch.tensor(data) 根据 Python list 创建 Tensor；这里显式指定 long，满足 one_hot 的整数索引要求。
# torch.Tensor(data) 是历史上的类构造写法，默认 dtype 行为不如 torch.tensor 清晰，实际代码优先使用后者。
xs = torch.tensor(xs, dtype=torch.long)
ys = torch.tensor(ys, dtype=torch.long)

```

```python

# 输入字符索引，shape == [5]。
xs

```

```python

# 目标字符索引，shape == [5]；xs[i] 和 ys[i] 组成一个监督样本。
ys

```

```python

# one-hot 编码：5 个样本、每个样本 27 个类别，因此 xenc.shape == [5, 27]。
# one-hot 本身是整数/布尔结构，转 float 后才能与浮点权重矩阵做矩阵乘法。
import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float()
xenc

```

```python

# 初始化教学用权重矩阵 W。W[i, j] 表示当前字符 i 选择下一个字符 j 的未归一化分数。
# requires_grad=True 是 loss.backward() 能工作的前提；W.grad = None 只能清空梯度，不能开启自动微分。
g = torch.Generator().manual_seed(2147483647)
W = torch.rand((27, 27), generator=g, requires_grad=True)
print(f'{W.shape=}, {W.requires_grad=}')

```

```python

# 教学版 forward：one-hot × W 得到 logits，再通过 exp 和逐行归一化得到概率。
# xenc.shape [5,27] @ W.shape [27,27] -> logits.shape [5,27]。
# one-hot 只有一个 1，所以每个样本实际上选出了 W 中对应当前字符的一行。
# counts 是模型生成的正值权重，不等于原始统计矩阵 N；只有在 logits 恰好等于 log 计数时才有对应关系。
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
probs

```

```python

# 逐样本拆解 NLL，观察“真实下一个字符的概率”如何进入 loss。
nlls = torch.zeros(xs.numel())
for i in range(xs.numel()):
    x = xs[i].item()  # 当前字符的整数索引
    y = ys[i].item()  # 真实下一个字符的整数索引
    print('-------')
    print(f'第 {i + 1} 个 bigram：{itos[x]}{itos[y]}（索引 {x}, {y}）')
    print('输入字符索引：', x)
    print('模型输出的 27 类概率：', probs[i])
    print('真实目标索引：', y)
    p_true = probs[i, y]
    print('真实下一个字符的概率：', p_true.item())
    logp = torch.log(p_true)
    print('log probability：', logp.item())
    nll = -logp
    print('negative log likelihood：', nll.item())
    nlls[i] = nll

print('=====')
print('平均 NLL =', nlls.mean().item())

```

```python

# 用高级索引一次取出每个样本对应的真实类别概率，再求平均 NLL。
# probs[torch.arange(5), ys] 的 shape 是 [5]，每个位置只保留一个正确答案概率。
loss = -probs[torch.arange(xs.numel()), ys].log().mean()
loss

```

```python

# 反向传播：先清空旧梯度，再沿 loss -> probs -> counts -> logits -> W 的计算图反传。
# PyTorch 默认累加梯度，所以每次 backward 前都要把 W.grad 设回 None。
# 如果 W 没有 requires_grad=True，loss 会没有 grad_fn，并触发“does not require grad”报错。
W.grad = None
loss.backward()

```

```python

# 查看教学 batch 的 loss；这次 loss 还没有加 L2 正则项。
print(loss.item())

```

```python

# 用一次最朴素的梯度下降更新 W。
# W.data 是课程里的简写；更安全的现代写法是放在 torch.no_grad() 中原地更新。
with torch.no_grad():
    W -= 0.1 * W.grad

```

# ## 阶段 7：全量训练

单个名字的 5 条样本只用于拆解计算图；真正训练时使用全部名字产生的字符转移，并用梯度下降调整 W。

```python

# 阶段 7：回到完整数据集，所有名字的相邻字符对都作为训练样本。
# 总样本数 = 每个名字的长度 + 1 的总和；xs[i] 是当前字符，ys[i] 是下一个真实字符。
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs, dtype=torch.long)
ys = torch.tensor(ys, dtype=torch.long)
num = xs.nelement()
print('训练样本数：', num)

# W.shape == [27,27]：每一行是一个当前字符，每一列是一个候选下一个字符。
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

```

```python

# W 的平方均值是 L2 正则项的核心：reg_strength * (W**2).mean()。
# 这里只观察正则项的未乘系数部分，不要把它误认为完整 loss。
(W ** 2).mean()

```

```python

# 完整训练循环：每一轮依次执行 forward、计算目标、backward、更新参数。
# 目标 = 平均 NLL + L2 正则；加一平滑作用于计数/概率，L2 作用于 W，二者不等价。
# 这里保留 exp -> normalize 的展开写法，方便对应前面的概率模型；数值更稳的生产写法是 F.log_softmax 或 F.cross_entropy。
num_steps = 100
learning_rate = 50.0
reg_strength = 0.01
loss_history, nll_history, reg_history = [], [], []

for k in range(num_steps):
    # ① forward：把所有当前字符一次性 one-hot 化。xenc.shape == [num,27]。
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W                              # [num,27]：每个样本对 27 个下一个字符的分数
    counts = logits.exp()                          # [num,27]：正值权重（教学展开，不是原始 N）
    probs = counts / counts.sum(1, keepdim=True)   # [num,27]：每一行归一化为条件概率

    # 高级索引逐行取出 probs[i, ys[i]]，只评价真实下一个字符的概率。
    nll = -probs[torch.arange(num), ys].log().mean()
    reg_loss = reg_strength * (W ** 2).mean()
    loss = nll + reg_loss
    loss_history.append(loss.item())
    nll_history.append(nll.item())
    reg_history.append(reg_loss.item())
    if k % 10 == 0 or k == num_steps - 1:
        print(f'第 {k:03d} 轮：total={loss.item():.6f}, nll={nll.item():.6f}, reg={reg_loss.item():.6f}')

    # ② backward：梯度默认累加，因此每轮先清零。
    W.grad = None
    loss.backward()

    # ③ update：沿负梯度方向移动；no_grad 防止更新操作再次进入计算图。
    with torch.no_grad():
        W -= learning_rate * W.grad

```

# ## 训练结果：总目标的组成

下面将总 loss、纯 NLL 和 L2 正则项分开画出来，避免把“拟合数据”和“约束权重”混成一个无法解释的数字。

```python

# 训练结果可视化：总目标 = 平均 NLL + L2 正则。
fig, ax = plt.subplots(figsize=(8, 4.5))
steps = range(1, len(loss_history) + 1)
ax.plot(steps, loss_history, label='total loss', color='#B31B1B', linewidth=2)
ax.plot(steps, nll_history, label='mean NLL', color='#2B6CB0', linewidth=1.8)
ax.plot(steps, reg_history, label='L2 regularization', color='#D97706', linewidth=1.8)
ax.set_xlabel('训练轮次 k')
ax.set_ylabel('loss')
ax.set_title('Bigram 神经网络：训练目标的三个组成')
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
print(f'初始 total loss = {loss_history[0]:.6f}')
print(f'最终 total loss = {loss_history[-1]:.6f}')

```

# ## 阶段 8：从训练后的参数采样

训练结束后，W 不再是随机表，而是一张经目标函数和梯度更新得到的可学习转移表。它能生成有名字外形的字符串，但仍然只看一个字符的局部上下文。

```python

# 阶段 8：用训练后的 W 生成名字。
# 每一步都把当前字符 ix 编成 one-hot，再经过同一个 W 得到下一个字符分布。
# 与统计模型相比，二者的条件概率应接近，但由于参数化、正则化和随机采样不同，字符串不保证逐个相同。
g = torch.Generator().manual_seed(2147483647)
with torch.no_grad():
    for i in range(5):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdim=True)
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))

```
