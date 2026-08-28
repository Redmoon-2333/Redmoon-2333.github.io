---
title: "Day5：字符级 MLP 实战——把 Bengio 的图跑通在名字上"
date: 2026-08-28 08:01:00 +0800
categories: [技术实践]
tags: [makemore, Bengio, MLP, Embedding, BatchNorm, PyTorch]
excerpt: 追随Karpathy大佬的脚步
math: true
---

## 前言

刚准备开始看卡帕西的MLP部分，结果视频打开上来就告诉我这篇论文讲的不赖，那我势必要去阅读一下了。
这篇 Bengio 2003 正是 本次要进行的makemore项目第二阶段的**前置论文**——它回答了“为什么要给词/字符配一个向量（Embedding）”，以及接下来我们的 MLP 到底该怎么搭（进行一个知识的迁移）。
简单阅读并总结一下，学它有三个理由：
1. **它是 Embedding 的出生证明**：词不再是离散的编号，而是一个可学习的 `m` 维实值向量 `C(i)`；
2. **它是 MLP 做语言模型的标准模板**：所有后来的 `Word2Vec / GPT` 都是它的变体；
3. **它是维度灾难的第一次正面攻克**：用“连续空间的平滑”打败了“离散空间的指数爆炸”。

## 0. 这篇论文讲了什么

```text
名字数据（names.txt, 32033 行）
→ block_size=3 滑窗构造 (X, Y) 样本（共 228146 行）
→ C 查表：字符 → 10 维向量（Embedding 的雏形）
→ 展平拼接：view(-1, 30)
→ 隐藏层：tanh(xW1 + b1) → (B, 200)
→ 输出层：hW2 + b2 → 27 个字符得分
→ softmax → 交叉熵损失
→ 小批量梯度 + 学习率扫描
→ 验证集评估与过拟合
→ 自回归采样生成新名字
```

一句话：**论文负责“为什么用向量”，MLP 负责“在字符上把这条链路亲手跑通”**。前半部分是 Bengio 精读（第 1-7 节），后半部分是 Karpathy 代码实战（第 8-17 节）。

## 0.1 预备：先把符号一次性讲清（看到后面不再卡住）

> 本节把论文里反复出现的 `V, m, C, g, R, n, h, U, H, W, b, d, x, y` 用一句话讲清含义、形状与来龙去脉，后面再出现就直接用，不再展开。

| 符号 | 怎么读 | 它是什么 | 形状/取值 | 一句话通俗理解 |
| --- | --- | --- | --- | --- |
| `V` | Vocabulary | **词表**：所有允许出现的词的集合 | 大小 &#124;V&#124;，如 17 000 或 100 000 | 一本词典的“目录页”，共 &#124;V&#124; 个词 |
| `R^{m}` | R 的 m 次方 | **m 维实数向量空间** | `m = 30/60/100`，远小于 &#124;V&#124; | 每个词的“坐标系”，`m` 是坐标轴根数 |
| `C` | C 矩阵 | **查表**：词 `i → 向量 C(i)` 的映射表 | &#124;V&#124;×m 矩阵，行是词，列是特征 | 一张“词→坐标”的大表格，第 `i` 行就是词 `i` 的向量 |
| `C(i)` | C of i | 词 `i` 的向量 | `m` 维，如 `[0.1, -0.3, ...]` | `cat` 的坐标 |
| `g` | g 函数 | **神经网络**：把上下文向量算成概率 | 输入 `(n-1)·m` 维，输出 &#124;V&#124; 维 | 一台“概率计算器”，输入旧词向量，输出下一个词的概率分布 |
| `n` | n | **上下文窗口** | 如 3/5/8 | 每次看前面 `n-1` 个词来猜第 `n` 个 |
| `h` | hidden | **隐藏单元数** | 如 50/100 | 隐藏层的“神经元个数” |
| `H` | H 矩阵 | 隐藏层权重 | `h × (n-1)m` | 把长向量 `x` 压进隐藏层的“投影仪” |
| `U` | U 矩阵 | 隐藏→输出权重 | &#124;V&#124;×h | 把隐藏层信号“翻译”回词表大小 |
| `W` | W 直连 | 输入→输出直连权重 | &#124;V&#124;×(n-1)m，可省略 | 一条“捷径”，让 `x` 直接对输出投票（论文称直连） |
| `b` | bias 输出偏置 | 输出偏置 | &#124;V&#124; 维 | 每个词的“基础人气分” |
| `d` | bias 隐藏偏置 | 隐藏偏置 | `h` 维 | 隐藏层的“起点偏移” |
| `x` | x 向量 | 上下文拼接向量 | `(n-1)·m` 维 | `n-1` 个词向量排成一排 |
| `y` | y 向量 | 未归一化分数（logits） | &#124;V&#124; 维 | 每个词的“原始票数”，进 softmax 前 |
| `y_i` | y 的第 i 维 | 词 `i` 的分数 | 标量 | 词 `i` 比别人高多少分 |
| &#124;V&#124;×m 类记号 | — | 矩阵形状 | 行×列 | 如 `17000×30` 就是 51 万个参数 |

**三条快速区分**

1.  **`C` vs `g`**：`C` 是**记忆**（查表得向量），`g` 是**计算**（向量→概率），合起来 `f = g∘C` 才是完整语言模型。
2.  **`C(i)` vs `R^{m}`**：`C(i)` 是一个具体向量（如 `cat` 的坐标），`R^{m}` 是所有这类向量住的空间（`m` 维实数空间）。
3.  **`H/U` vs `W`**：`H/U` 走“隐藏层弯路”（非线性，能力强），`W` 走“直连高速路”（线性，快但简单）；论文实测直连有益但非必须，核心是 `C→H→U` 的主干。

> 读法小贴士：`R^{m}` 读“R 的 m 次方”即 `m` 维实向量；`C ∈ R^{|V|×m}` 表示 `C` 住在 `|V|×m` 的矩阵空间里。

## 0.2 这篇论文解决了什么

n-gram 靠“背诵短语”泛化，神经模型靠“词的相似性”泛化。

```text
n-gram（离散查表）：
  训练集见过 "The cat is walking in the bedroom" →
  但 "A dog was running in a room" 算全新句子 → 概率直接归零或回退
  泛化方式：把句子切成 1-3 词的小块（trigram），用小块的计数硬拼

Bengio 2003（连续分布式）：
  把 the/a、cat/dog、bedroom/room 看成空间中相邻的点 →
  见过第一句，就等于见过了它的“语义邻居们”的指数级变体
  泛化方式：词向量相近 → 句子向量相近 → 概率自然相近（光滑函数）
```

![n-gram 靠拼接短块 vs 神经模型靠语义邻居的泛化对比](/assets/img/curse-vs-distributed.png)

> 某种意义上，相当于给模型真正加入了“语义”这一概念。

## 1. 维度灾难：为什么离散建模注定走不远

> **原文关键内容**：For example, if one wants to model the joint distribution of 10 consecutive words ... with |V|=100,000, there are potentially 100000^{10}−1 = 10^{50}−1 free parameters.

**翻译**：若想对词表 `|V|=100000` 的连续 10 个词的联合分布直接建模，自由参数数量是 `100000^{10}−1 = 10^{50}−1`。

**解释**：

想象词表是 10 万个抽屉，10 个词的句子就是从 10 万个抽屉里连续抽 10 次，有 `100000^{10}` 种抽法。为了给每种抽法都单独学一个概率，你需要 10^{50} 个独立旋钮——比地球原子数还多，根本学不完。

核心矛盾是：**离散空间没有“中间地带”**。连续世界里 `3.01` 和 `3.02` 很近，学了前者就能猜后者；但离散世界里 `cat` 和 `dog` 是两个毫不相干的编号，`cat` 的统计结果分不到 `dog` 半点好处。

**为什么重要**：这一段给出了全文的立论基石：只要还是离散 one-hot/计数表，参数就会随上下文长度 `n` 指数爆炸，后续所有 n-gram 技巧（back-off、平滑、聚类）都是在“修补”这个根因，但这个病因是所谓的**痼疾**，根本没法修正。

![计数表的指数墙：参数量随 n 指数飙升，神经模型用 Embedding 将其拉平](/assets/img/exponential-wall.png)

## 2. 核心方案

> **原文关键内容**：1. associate with each word ... a distributed word feature vector in R^m, 2. express the joint probability function ... in terms of the feature vectors, 3. learn simultaneously the word feature vectors and the parameters of that probability function.

**翻译**：1. 为每个词关联一个 `m` 维实值分布式特征向量；2. 用这些特征向量来表达词序列的联合概率；3. 同时学习词向量与概率函数的参数。

**解释**：

把每个词从“门牌号”变成“坐标”：`C(the) = [0.2, -1.1, ..., 0.5]`，`m` 通常 30/60/100，远小于 `|V|=17000`。然后所有概率都基于这些坐标来算，而坐标本身和算概率的神经网络一起被反向传播更新——词的意义是在预测下一个词的任务中“顺带”学会的。

注意“同时”二字的分量：**不是先用别的办法把词向量训好，再去训语言模型；而是端到端一起学**。这让词向量天生就对“预测下一个词”最有用，而不是对“统计共现”最有用。

**为什么重要**：这是实现 MLP 时 `nn.Embedding(vocab, m)` + `MLP` 的精确对应。

**关键思想提炼**：**表示学习与概率建模联合优化**。`C(|V|×m)` 是知识载体，`g`（MLP）是概率计算器，二者共享梯度，泛化源于 `向量近 ≈ 语义近 ≈ 概率近`。

## 3. 为什么能泛化：一个训练句照亮指数个邻居

> **原文关键内容**：if we knew that dog and cat played similar roles ... we could naturally generalize from "The cat is walking in the bedroom" to "A dog was running in a room" and many other combinations ... because the probability function is a smooth function of these feature values, a small change in the features will induce a small change in the probability.

**翻译**：若已知 `dog/cat`、`the/a`、`bedroom/room` 扮演相似角色，就能从第一句自然泛化到第二句及大量组合——因为概率函数是特征值的光滑函数，特征小变化只会引起概率小变化。

**解释**：

论文把句子看成“词向量序列在空间中的路径”：路径上每个点抖一点点，整条路径的概率就抖一点点。训练集只采了几条路径，但**路径周围的指数条“抖动的”路径**都被照亮了——这就是“一个训练句顶一群语义邻居”的来源。

**关键思想提炼**：泛化不靠“背更多短语”，靠 **向量空间的度量 + 光滑函数的连续性**，实现组合式（compositional）外推。

## 4. 模型全貌：C 表 + g 网络

> **原文关键内容**：We decompose f(w_t, ..., w_{t-n+1}) = \hat{P}(w_t|w_{t-n+1}...w_{t-1}) in two parts: 1. A mapping C from any element i of V to C(i) in R^m (|V|×m matrix). 2. The probability function g maps (C(w_{t-n+1}), ..., C(w_{t-1})) to a distribution over V.

**翻译**：把条件概率 `f` 拆成两部分：1. 查表 `C` 把词编号 `i` 映成向量 `C(i) ∈ R^m`；2. 函数 `g` 把上下文 `(n-1)` 个向量映射成词表上的概率分布。

**公式**：

$$
f(i, w_{t-1}, ..., w_{t-n+1}) = g(i, C(w_{t-1}), ..., C(w_{t-n+1}))
$$

约束：对任意上下文，$\sum_{i=1}^{|V|} f(i, \cdot) = 1$，$f > 0$。

### 4.1 逐层数据流（以 `n=3` 为例，`m=2` 可视化）

输入 `w_{t-2} w_{t-1} → w_t`，词表 `|V|=5`（a, cat, dog, in, bedroom）：

```
词 id:   w_{t-2}=cat(2)   w_{t-1}=in(4)
           │               │
           ▼               ▼
     ┌─────────────────────────────────┐
     │  C 表 (|V|×m)：查两次，复用同一张表  │  ← 参数共享的关键：cat 在位置1和位置2是同一个向量
     │  C(cat)=[0.1, 0.8]  C(in)=[-0.5,0.3] │
     └─────────────────────────────────┘
           │               │
           └──────┬────────┘
                  ▼
        x = concat(C(w_{t-2}), C(w_{t-1}))  ∈ R^{(n-1)m}  —— 例：4 维
                  │
                  ▼
        h = tanh(d + H x)               —— 隐藏层，H ∈ R^{h×(n-1)m}
                  │
                  ▼
        y = b + W x + U h               —— 输出层（含直连 W）
                  │
                  ▼
        p = softmax(y)  ∈ R^{|V|}  —— 第 i 维即 P(w_t=i | 上文)
```

![Bengio NNLM 前向结构：查表 C → 拼接 → tanh 隐藏层 → 仿射输出 → softmax](/assets/img/nnlm-forward.svg)

**Day3 vs NNLM**：

| | Day3 bigram `one-hot×W` | Bengio NNLM |
| --- | --- | --- |
| 查表 | `W ∈ R^{27×27}`，one-hot 选行，等价于查表 | `C ∈ R^{|V|×m}`，编号查表 |
| 上下文 | 只看前 1 个词 (`n=2`) | 看前 `n-1` 个词，拼接成长向量 |
| 非线性 | 无（线性+softmax） | `tanh` 隐藏层，学复杂交互 |
| 参数共享 | 无 | `C` 在所有位置、所有样本间共享 |

**为什么重要**：这幅图是全论文的“唯一架构图”，后来的 `Embedding → Concat/Sum/Attention → MLP → Softmax` 都在它的影子里。`C` 的共享是灵魂——`cat` 不管出现在句首还是句尾，都贡献同一个梯度方向。

**关键思想提炼**：**查表（记忆）+ 函数（推理）分离，但参数共享**。`C` 学“词是什么”，`g` 学“词怎么组合成句”，二者拼接即 `f = g∘C`。

### 4.2 维度与复杂度

- **参数量**：`C: |V|×m` 主导（如 17000×30=510k），`H: h×(n-1)m`，`U: |V|×h`，`W: |V|×(n-1)m`（直连可省）。总数百万级，2003 年已算“大模型”。
- **计算量**：每预测一次只算一次前向，与 `|V|^n` 无关，因此 `n` 可从 trigram 的 3 轻松提到 5/8。
- **训练目标**：最大化对数似然 $\sum_t \log f(w_t|w_{t-n+1}...w_{t-1})$，等价于最小化困惑度（perplexity）。

## 5. 与 n-gram、聚类、LSI 的边界

**n-gram**：离散拼接短块，`P(w_t|w_{t-n+1}...w_{t-1})` 直接查计数表 + 回退/插值平滑。泛化靠“多看短块”，天花板是数据稀疏与上下文长度。

**词聚类 n-gram**（Brown 等）：把词硬/软分到 `K` 个离散类，类内共享计数。已在利用相似性，但**类是离散跳变**，类内一视同仁、类间老死不相往来，不够细腻。

**LSI/信息检索**：也学连续词向量，但目标是“文档级共现”，且往往固定向量再训分类器。论文的差异在于**为“预测下一个词”而联合学习**，向量直接对困惑度负责，效果显著更好（作者试过冻住 LSI 向量去训，效果差）。

**神经文本压缩/语音合成**：已有用神经网络预测下一个字符/音素，但规模小、未证实可训动大词表语言模型。本文首次把规模做到“能跟 state-of-the-art trigram 打，且把 trigram + 模型插值后还能再降困惑度”。

![离散聚类 vs 连续分布式 vs 固定 LSI 向量的适用场景对比](/assets/img/cluster-vs-distributed.png)

## 6. 训练：“能训动”本身就是贡献

- **样本量**：百万到千万词，上下文滑窗每步一个样本，远超当时常见的 UCI 小表。
- **优化**：随机梯度下降 + 权重衰减（weight decay，岭回归思想），梯度需穿过 `softmax → U/W → tanh → H → C`，`C` 的梯度是**多个位置的和**（共享带来的聚合）。
- **提速技巧**：论文已讨论用重要性采样/层次 softmax 逼近归一化分母（完整 softmax 需对 `|V|` 求和，`O(|V|)` 是瓶颈），这为后来的层次 softmax、负采样埋下伏笔。

## 7. 关键公式

**联合概率分解（链式法则，n-gram 马尔可夫近似）**：

$$
\hat{P}(w_1^T) = \prod_{t=1}^{T} \hat{P}(w_t | w_1^{t-1}) \approx \prod_{t} f(w_t, w_{t-1}, ..., w_{t-n+1})
$$

**NNLM 前向（论文式）**：

$$
\begin{aligned}
x &= [C(w_{t-n+1}); ...; C(w_{t-1})] \in \mathbb{R}^{(n-1)m} \\
h &= \tanh(d + Hx), \quad y = b + Wx + U\tanh(d+Hx) \\
\hat{P}(w_t = i | \cdot) &= \frac{e^{y_i}}{\sum_j e^{y_j}}
\end{aligned}
$$

**训练损失**：

$$
L = -\frac{1}{T}\sum_t \log \hat{P}(w_t | w_{t-n+1}...w_{t-1}) + \lambda\|\theta\|^2
$$

> **过渡**：其实感觉这篇论文的核心概念就是“学习语义”——在学习某些语义相近的句子的过程中，模型自然而然的学会某些**语法**。但涉及到一些数学公式的推导可能还是看不太懂。没关系，因为卡帕西说了，下一节课虽然和这篇论文说的不一样，但整体思路完全一致：**从词的级别下降到字母的级别（降低维度），方便实操**。前面看不懂的，向后看着看着或许就看懂了。

---


## 8. 上下文窗口：block_size=3 

**类比：** 模型是“滑窗猜下一个字母”。`block_size=3` 就是窗口宽度——每次只看前 3 个字符来猜第 4 个。像玩填空：`... → e`、`..e → m`、`.em → m`，哪怕名字很长，也是一格一格滑过去的。

**公式：** 训练集构造

$$
\mathcal{D} = \{(X_i, Y_i)\}_{i=1}^{N},\quad X_i\in\{0..26\}^{3},\; Y_i\in\{0..26\},\quad N\approx 228146
$$

$$
\text{emma}: [\_,\_,\_]\!\to\!e,\; [\_,\_,e]\!\to\!m,\; [\_,e,m]\!\to\!m,\; [e,m,m]\!\to\!a,\; [m,m,a]\!\to\!.
$$

> **实现要点：** `context = [0]*block_size` 滑窗 → `X,Y`，全量 `build_dataset()` 再做 `Xtr/Xdev/Xte = 8:1:1` 切分——思路与 `Day3 §8.1` 的 `emma` 5 样本拆解一致。

![滑窗 block_size=3 构造 (X,Y) 的过程](/assets/img/mlp-block-context.svg)

---

## 9. Embedding 查表：C 是字典
Day3 的代码里，我们用 one-hot 乘 `W` 选行，本质上还是查表；MLP 里直接 `C[X]` 就是查表——`C` 是 `27×m` 的“字符字典”，`C[5]` 就是字母 `e` 的 10 维坐标。这样返回的梯度可以直接作用到这本"字典"里，也就是论文中说的，在训练神经网络的同时，把"语义"也顺便训练了。

**公式：** 查表前向（`m=10` 版更贴近实战）

$$
C \in \mathbb{R}^{27\times 10},\quad \text{emb}=C[X]\in\mathbb{R}^{B\times 3 \times 10},\quad \text{emb}_{b,t,:}=C[X_{b,t}]
$$

$$
\text{Day3: } \text{onehot}(X)W = W[X] \equiv C[X]\ \text{（选行等价）}
$$

> **实现要点：** `C[X]` 批量查表——`C` 是 `27×m` 字符字典，`C[5]` 即 `e` 的向量；与 `Day3 §9.2` 的 `one-hot @ W` 选行等价，只是此处的 `W` 有了 `m=10` 的瓶颈。

![Embedding 查表：C[X] 选行 vs one-hot 乘法等价](/assets/img/mlp-embedding-lookup.svg)

---

## 10. 拼接的艺术：cat 很直观，但view 才具有真正的高效率

 `cat` 有一个小小的性能坑。`emb` 是 `(B,3,10)`，要喂给 `W1(30×200)` 必须拼成 `(B,30)`。`torch.cat([emb[:,0,:],emb[:,1,:],emb[:,2,:]],dim=1)` 能得到合理的结构，但实际上运行过程中做了 3 次内存拷贝；`emb.view(B,-1)` / `emb.view(-1,30)` 只是换了个“看这块内存的透镜”。从这里我们可以看出在pytorch中，张量的底层存储并没动，变的只是 `shape/stride`，也就是我们看它的一种“逻辑结构”发生了改变。

**公式：** 形状变换

$$
\text{emb}\in\mathbb{R}^{B\times 3 \times 10}\ \xrightarrow{\text{view}}\ x\in\mathbb{R}^{B\times 30},\quad x_b = [\text{emb}_{b,0};\text{emb}_{b,1};\text{emb}_{b,2}]
$$

$$
\text{cat 版：拼接 }3\times(B\times10)\quad\text{vs}\quad\text{view 版：重解释 }(B\cdot3\cdot10)\text{ 连续内存}
$$

> **实现要点：** `emb` 为 `(B,3,10)`，`view(-1,30)` 零拷贝拼为 `(B,30)` 再进 `W1(30×200)`；`cat/unbind` 直观但多一次拷贝。

![cat 拷贝 vs view 透镜：同一块内存的两种视图](/assets/img/mlp-cat-vs-view.svg)

---

## 11. tanh 隐层

`x @ W1 + b1` 再线性到 logits，整体还是线性映射，3 个字符的交互学不出来。`tanh` 就是给模型一个“拐弯”的能力，让 `(e,m)` 和 `(m,m)` 这种组合能产生非线性响应。Bengio 论文里的 `h = tanh(d+Hx)` 在这里我们类比使用了 `h = tanh(xW1+b1)`，形状从 `(B,30) → (B,200)`。

**公式：** 隐层前向（论文 ↔ MLP 对照）

$$
\text{论文: } h = \tanh(d + Hx),\; y = b + Wx + Uh \qquad\Longleftrightarrow\qquad
\text{MLP: } h = \tanh(\text{emb.view}(B,30)W_1 + b_1),\; \text{logits}=hW_2+b_2
$$

$$
W_1\in\mathbb{R}^{30\times 200},\; b_1\in\mathbb{R}^{200},\; W_2\in\mathbb{R}^{200\times 27}
$$

> **实现要点：** `h = tanh(emb.view(B,30)W1 + b1)`，形状 `(B,30)→(B,200)`；去掉 `tanh` 则退化为 `Day3` 的线性 bigram。

<p class="fig-wide" markdown="1"><img src="/assets/img/mlp-tanh-hidden.png" alt="tanh 非线性门" style="max-width:78%;height:auto;display:block;margin:1rem auto;" /></p>

---

## 12. 小批量梯度

全量 `228k` 样本算一次梯度太慢，因而需要进行批量的训练。如果一步一步训练算作穿针引线，固然精细，训练成果可能细致入微。但往往批量训练的“生产线”才是我们需要的，可能会牺牲一部分的训练时的精度，但大幅加快了训练的速度。每步 `ix=torch.randint(0,Xtr.shape[0],(32,))` 随机捞 32 个名字片段，`C[Xtr[ix]]` 只更新这 32 个片段涉及的 Embedding 行，大大提高了训练速度。

**公式：** 小批量交叉熵

$$
\mathcal{L}_{\text{batch}} = -\frac{1}{32}\sum_{i\in\text{ix}}\log \frac{\exp(\text{logits}_{i,Y_i})}{\sum_j\exp(\text{logits}_{i,j})},\quad \text{ix}\sim\mathcal{U}[0,N)
$$

$$
\theta_{t+1}=\theta_t - \eta \nabla_{\theta}\mathcal{L}_{\text{batch}}(\theta_t)
$$

> **实现要点：** 小批量 `randint(0,N,(32,))` 随机捞 32 行，`F.cross_entropy` 等价手写 `log_softmax+NLL`——与 `Day3 §9.3` 同构。

![小批量采样：32 个随机索引撬动全量 228k](/assets/img/mlp-minibatch.svg)

---

## 13. 学习率与 loss 曲线

学习率也会影响学习成果，例如：`lr=1.0` 直接发散无法收敛、`lr=0.001` 进展太慢、`lr=0.1` 则刚好。Karpathy 教了我们一种测试学习率的方法：先用 `lre=linspace(-3,0,1000), lrs=10**lre` 做指数扫描，每种学习率进行一次训练的前向反向传播，画 `loss.log10()` 找拐点——先让 loss 动起来，找到看似最佳的学习率，再固定 `lr=0.1（推测的最佳学习率，也就是拐点）` 跑 5 万步，`stepi/lossi` 曲线从 `~2.5 → ~2.0` 平滑下降。

**公式：** 学习率扫描

$$
\text{lrs}=10^{\text{linspace}(-3,0,1000)},\quad \mathcal{L}(\eta)=\text{CrossEntropy}(\theta - \eta\nabla\mathcal{L}),\quad \eta^* \approx 0.1
$$


![学习率扫描与 loss 曲线：log10 loss 随 lr/步数下降](/assets/img/mlp-lr-loss-curve.svg)
> 以上为概念图，实际的曲线是一个先下后上的曲线，大概有点像对勾函数吧，然后最低点在0.1处左右:
---

## 14. 采样复盘：multinomial 如何“写”出新名字

训练完的模型像一个“接龙”：从 `[.,.,.]` 开始，每步把上下文喂进去得到 `probs`，再 `multinomial` 掷骰子抽下一个字符，滑窗 `context = context[1:]+[ix]` 继续。比起bigram 只能看 1 个字符，MLP 看 3 个字符后生成的名字明显更像人名。

**公式：** 自回归采样

$$
p(\cdot\mid c_{t-2},c_{t-1},c_t)=\text{softmax}(hW_2+b_2),\quad c_{t+1}\sim\text{Multinomial}(p),\quad \text{context}_{t+1}=[c_{t-1},c_t,c_{t+1}]
$$

> **实现要点：** `context=[0]*3 → emb=C[context] → tanh → softmax → multinomial` 接龙 20 行，与 `Day3 §6` 的 `multinomial` 采样同源。

<p class="fig-wide" markdown="1"><img src="/assets/img/mlp-sampling.png" alt="自回归采样" style="max-width:82%;height:auto;display:block;margin:1rem auto;" /></p>

---

## 17. 收工小结：N-gram → Bengio → MLP 的递进

1.  **N-gram**：`N[27,27]` 把转移次数摆在桌上，按行归一就是概率，窗口稍长参数会指数爆炸。
2.  **Bengio**：`C(|V|×m)` 把离散编号压进连续空间，`g=C→H→U` 的光滑函数让“相似词相似句”自动泛化，`C` 与 `g` 联合端到端优化。
3.  **MLP**：`C[X] → view 拼接 → tanh(30→200) → logits(200→27) → cross_entropy` 在 `32` 的小批量下用 `lr≈0.1` 跑 5 万步，能直观看到元音字母在 Embedding 空间里的聚类过程。

![Embedding 2D 投影：首次训练后 a/e/i/o/u 的聚类情况（成功）](/assets/img/mlp-embedding-scatter-try1.png)

> 注：同一份代码多次训练会有随机性。首次运行可得到清晰的元音聚类，另一次重试也可能出现聚类分散的情况，图示如下——这正是随机初始化与小批量随机性的体现。

![Embedding 2D 投影：重新训练后的聚类情况（较分散）](/assets/img/mlp-embedding-scatter-try2.png)

---

## 附 · 资料下载

本篇配套的 Bengio 精读与 MLP 字符级实战代码、图示，可直接下载。

### 完整实验代码与笔记

- [MLP.ipynb](https://github.com/Redmoon-2333/Redmoon-2333.github.io/releases/download/day5-assets/MLP.ipynb)：注释版 MLP（在原始课程跟写版上新增 123 行中文注释）
- [MLP_original.ipynb](https://github.com/Redmoon-2333/Redmoon-2333.github.io/releases/download/day5-assets/MLP_original.ipynb)：原版备份（无注释原版，含输出）





