---
layout: post
title: Day8：MLP的一些收尾工作
date: 2026-09-05 08:00:00 +0800
categories:
  - 技术实践
tags:
  - makemore
  - BatchNorm
  - Kaiming
  - 初始化
  - 梯度消失
  - PyTorch
excerpt: 先给 MLP 做体检（初始 loss、tanh 饱和），再开两味药（Kaiming、BatchNorm）。
math: true
---

## 前言
依旧听取一些越来越难听懂的课程。
Day5 把字符级 MLP 跑通了，但那版网络是"能训"而已，还有很多bug急需解决。于是卡帕西特意花了整整一节课来为上节课的MLP做一个收尾工作。
![Day8 路线图：从 Day5 的能训，到诊断、修复、模块化三步走](/assets/img/day8-roadmap.svg)


## 1. 基线复现：先确认"病"长什么样

把 Day5 的单隐层 MLP（`n_emb=10, n_hidden=200`，11897 个参数）原样重训 200k 步，同种子下面貌完全可复现：

**公式：** 训练配置

$$
\text{batch}=32,\quad \text{steps}=200000,\quad \text{lr}=\begin{cases}0.1 & i<10^5\\ 0.01 & i\ge 10^5\end{cases}
$$

![第一阶段 loss 曲线：初始约 27.9，迅速跌落后长期在 0.3~0.5（log10）震荡](/assets/img/day8-loss-phase1.svg)

> **实现要点：** 起点 loss≈27.9、终点 train 2.1256 / val 2.1674，与 Day5 一致。曲线抖是 batch=32 的正常噪声。但有两个"症状"藏在这条平静曲线下面：初始 loss 太大、tanh 大面积饱和——下面逐一拆解。

---

## 2. 症状一：初始 loss 违反数学预期

**类比：** 模型还没学任何东西时，对 27 个字符应该一视同仁，此时交叉熵的理论值是 $\ln 27 \approx 3.30$。可实际初始 loss 是 **27.88**——模型开局就"迷之自信"，要用几千步先学会"谦虚"（把 logits 压回均匀分布），白白浪费训练预算。

**公式：** 均匀分布下的期望损失

$$
\mathcal{L}_0 = -\ln\frac{1}{27} = \ln 27 \approx 3.30 \quad(\text{实测 }27.88\text{，偏离一个数量级})
$$

**病因在输出层的初始化：** `W2 = torch.randn(...)`、`b2 = torch.randn(...)` 都是标准正态，200 个隐层输出的加权和轻松把 logits 推到 ±20，softmax 一算就给出极端错误的置信度。

> **实现要点：** 修法是 `W2 *= 0.01`（打破对称性同时压低置信度）+ `b2 *= 0`。这里 `W2` 不能真设成 0——所有神经元输出相同、梯度相同，对称性永远破不开。

---

## 3. 症状二：tanh 饱和

**类比：** `tanh` 是一扇只在中间开启的门：输入在 ±0.5 以内时导数接近 1，信号畅通；输入一旦跑到 ±3 开外，门基本焊死，导数趋近 0——梯度传到这里就"消失"了。

![tanh 及其导数：中间是健康区，两端是梯度近似为零的饱和区](/assets/img/day8-tanh-saturation.svg)

**公式：** tanh 的局部梯度

$$
h=\tanh(x),\qquad \frac{\partial h}{\partial x}=1-\tanh^2(x)\xrightarrow{|x|\to\infty}0
$$

体检方式是两句 matplotlib：直方图看 `hpreact`/`h` 的分布，`imshow(h.abs()>0.99)` 把饱和神经元点成白点。结果：`hpreact` 横跨 ±20，`h` 两端堆出高墙，白色饱和点大面积出现。

> **实现要点：** 病根是 `embcat @ W1 + b1` 的方差随 `fan_in=30` 个输入累加放大（$\text{Var}(y)=30\,\text{Var}(w)\text{Var}(x)$），信号进 tanh 之前就爆了。修法：`b1 *= 0.01`、`W1 *= 0.1`，让 `hpreact` 回到健康区。同一组阈值图立刻全黑（零饱和）。

---

## 4. 药方一：Kaiming 初始化——缩放不是玄学

"权重除个 $\sqrt{\text{fan-in}}$"听起来像经验玄学，其实是一道方差算术。

**公式：** 线性层输出的方差

$$
y_i=\sum_{k=1}^{fan\text{-}in}w_kx_k,\qquad
\text{Var}(y_i)=fan\text{-}in\cdot\text{Var}(w)\cdot\text{Var}(x)
\;\xrightarrow{\;\text{Var}(w)=1/fan\text{-}in\;}\;\text{Var}(x)
$$

每个 $w_k$ 都和 $x_k$ 独立、零均值，所以方差直接连乘——输入有 30 个，方差就膨胀 30 倍。把 `Var(w)` 压到 $1/fan\text{-}in$，输出方差和输入一样大，信号就能一层层传下去不放大不缩小。

![左：不缩放时输出方差爆炸；右：按 1/√fan_in 缩放后方差稳定](/assets/img/day8-fanin-scaling.svg)

> **实现要点：** notebook 里做了两组对照（同分布输入，一组 `w` 缩放一组不缩放）：不缩放 `y.std()≈3.2`，缩放后 `y.std()≈1.0`。Kaiming 论文 (He et al., 2015) 给出的是针对 ReLU 的精确版（乘 $\sqrt{2/fan\text{-}in}$，因为 ReLU 砍掉一半信号），`5/3` 这个增益则是给 tanh 用的经验修正——但"除以 $\sqrt{\text{fan-in}}$"这个骨架是同一件事：**方差守恒**。

---

## 5. 药方二：BatchNorm——把每一层的输入"重新校准"

缩放初始化只能保证**第 0 步**健康，训练几步之后各层分布又会漂走（内部协变量偏移）。BatchNorm 的思路简单粗暴：**既然每层输入会漂，那就每层动手把它拉回标准正态。**

**公式：** BatchNorm 前向（论文原式）

$$
\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i,\quad
\sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2,\quad
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\quad
y_i=\gamma\hat{x}_i+\beta
$$

$\gamma$（初始全 1）和 $\beta$（初始全 0）是可学习参数：归一化先把信号强制拉回零均值单位方差，再交给 $\gamma,\beta$ 决定"要多宽、偏多少"——网络如果觉得归一化碍事，可以自己学回去。

**训练 / 推理的不对称**是 BN 最容易踩的坑：

![训练态用 batch 统计量并滚动更新 running 统计；推理态直接用 running 统计](/assets/img/day8-bn-train-infer.svg)

| | 训练态 | 推理态 |
| --- | --- | --- |
| 均值/方差 | 当前 mini-batch 的 $\mu_B,\sigma_B^2$ | `running_mean` / `running_var` |
| running 更新 | $\text{running}\leftarrow 0.999\,\text{running}+0.001\,\text{batch}$ | 冻结不动 |
| 副作用 | 同一样本随 batch 组成轻微抖动（附带正则化效果） | 确定性输出 |

> **实现要点：** running 统计是 `no_grad` 下的缓冲区，**不参与反向传播**；`bnmean_running/bnstd_running` 在训练中每步滚动更新，推理时用它们代替 batch 统计。另外归一化会减去均值，所以 `W1@embcat` 后面的 `b1` 恒被抵消——BN 层之前放 bias 是无效参数（BN 论文原话：可以去掉）。

**效果：** 加上 BN 后同样的 200k 步训练，初始 loss 直接落在 3.3 附近（不再需要"先学谦虚"），train/val 到 2.07/2.11。**BN 真正的价值不只是这次涨的点**：它让网络对初始化细节不敏感、允许更大的学习率——下一步的模块化实验会验证这一点。

---

## 6. Pytorch 化：Linear / BatchNorm1d / Tanh 三件套

散装代码堆到 4 个隐层就该模块化了。手写三个类，接口对齐 `nn.Module`（但不用 autograd 之外的框架设施）：

![模块化堆叠：Linear→BatchNorm1d→Tanh ×4 + 输出 Linear，36397 参数](/assets/img/day8-module-stack.svg)

**公式：** 模块接口约定

$$
$$
\$\$
\\text{Layer: }x\\mapsto \\text{call}(x),\\qquad \\theta\\leftarrow \\text{parameters}(),\\qquad \\text{out 挂中间量供诊断}
\$\$
$$
$$

> **实现要点：** 三个设计点——① `Linear` 构造时就做 `randn/fan_in**0.5` 的缩放初始化（药方一内置化）；② `BatchNorm1d` 内部区分 `training` 标志并维护 running 缓冲；③ 每层把输出存到 `self.out`，训练循环里 `retain_grad()` 一挂，诊断数据就有了。最后层权重 `*= 0.1` 压低初始置信度（症状一的修法），其余 Linear 层 `*= 5/3`（tanh 增益补偿）。

---

## 7. 诊断工具箱：训网络要会看三张图

深层网络训完不能只看一个 loss 数字。Karpathy 的做法是把网络切开看三样东西：激活、梯度、更新量。

![诊断总览：激活看饱和、中间梯度看消失、参数梯度看各层、更新比例看步长](/assets/img/day8-diagnostic-panel.svg)

### 7.1 激活分布——查饱和

![四层 Tanh 激活分布：U 形，两端堆积，饱和率 42%–56%](/assets/img/day8-activation-dist.svg)

**怎么读：** 横轴是 tanh 输出值，纵轴密度；四条曲线对应 4 个隐层。**理想形态是中间鼓的钟形**；这图是 U 形——大量激活压在 ±1 边界，说明每层都有四到五成神经元饱和。用 `5/3` 增益初始化后仍是 U 形，说明**深层堆叠本身就会加剧饱和**，想根治要靠 BN（每层重新校准）而不是只靠初始化。

### 7.2 中间层梯度——查消失

![四层 Tanh 输出梯度分布：0 附近尖峰，std 约 4~5e-3](/assets/img/day8-gradient-dist.svg)

**怎么读：** 曲线集中在 0 附近、std 在 $4\sim5\times10^{-3}$ 量级且**四层基本一致**——这是好现象：梯度没有逐层衰减（后面的层不比前面的层大一个量级）。若某层 std 比别层小 100 倍，才是梯度消失的实锤。

### 7.3 参数更新比例——查步长

![各参数的 log10 更新比例：黑色参考线 -3，100k 步处整体下移 1 个单位](/assets/img/day8-update-ratio.svg)

**怎么读：** 纵轴是 $\log_{10}\dfrac{\text{lr}\cdot\text{grad.std()}}{\text{data.std()}}$，即**每步更新量占参数尺度的比例**。黑色参考线 $-3$ 是经验法则：更新量约为参数的千分之一比较健康。

三个看点：
1. **100000 步处的整体断层**：所有曲线整齐下掉恰好 1 个单位——是学习率 `0.1→0.01` 造成的。纵轴公式里乘了 lr，$\log_{10}$ 坐标下 lr 缩小 10 倍正好是 $-1$；断层后曲线走平说明梯度本身没突变，训练连续，只是步子变小了。
2. **param 9（输出层权重）高居 -1.8**：它初始化时被乘了 0.1 压低置信度，分母 `data.std()` 小，比例显得大——衰减后回落到 -2.8，回到参考线附近，属于"设计如此"而非病态。
3. **开局毛刺**：前几百步梯度/参数尺度未稳，忽略即可。

---

## 8. 收尾对照实验：两个"如果"

### 8.1 如果不缩放初始化

![对比实验：不缩放时输出 std≈3.2（宽平），缩放后 std≈1.0（标准钟形）](/assets/img/day8-compare-init.svg)

同分布输入、同尺寸权重，只差一个 `1/√fan_in`：输出 std 从 1.0 膨胀到 3.2。单层 3.2 倍，十层就是 $3.2^{10}$——这就是没有方差守恒时深层网络"训不动"的算术本质。

### 8.2 如果加不加 BatchNorm

![对比实验：有无 BatchNorm1d 的 2000 步短训练，两条曲线交织](/assets/img/day8-compare-bn.svg)

相同结构、相同初始化、相同 minibatch 序列各训 2000 步：两条 loss 曲线基本交织，BN 没有立刻赢。**这不是 BN 没用**——在初始化已经修好的前提下，短训练里两者差距本来就小于 batch 噪声；BN 的价值在别处：对初始化不敏感、可以配更大的学习率、深网里免于分布漂移。

---

## 9. BatchNorm 全面上线：bias 退场、饱和率减半

§7 诊断图里的 U 形饱和是"只有初始化、没有 BN"的形态。这节把 BatchNorm 真正铺进网络，四个改动一次到位：

1. **每个 `Linear` 都 `bias=False`**——反正后面跟的 BN 会减均值，bias 是无效参数；
2. **每层 Linear 后面都跟一个 `BatchNorm1d`**（5 个隐层 + 输出层，共 6 个 BN，47024 参数）；
3. **增益归一**：`5/3` 的 tanh 增益不再需要（BN 每层重新校准），只有输出层 `gamma *= 0.1` 压低初始置信度；
4. **学习率衰减挪到 150k**（§7 的诊断图保持在 100k，两版不混）。

**公式：** 完整版前向（对比 §6 的无 BN 版）

$$
x \to \texttt{Linear}(\text{bias=False}) \to \texttt{BatchNorm1d} \to \texttt{Tanh} \;\times 5 \to \texttt{Linear} \to \texttt{BatchNorm1d} \to \text{logits}
$$

重训 200k 步后，三个结论：

1. **开局 loss = 3.3001**，紧贴理论值 $\ln 27 = 3.2958$——"先学谦虚"的几万步直接从第 0 步省掉，训练预算全部花在学习本身；
2. **train 2.0020 / val 2.0828，全场最佳**：比 Kaiming 初始化版（2.0377 / 2.1070）再进一步，也是整个 notebook 里第一个把 val 压到 2.1 以下的配置；
3. **饱和率从 42%–56% 降到 18.7%–24.3%**，激活 std 从 0.85–0.90 收回到 0.73–0.79：激活离开两极，梯度通路明显改善。

> **实现要点：** 评估时必须先把各层切到推理态（`layer.training = False`），BN 才会用 running 统计而不是当前 batch——评估完再切回来。

---

## 10. 小的记忆点

1. **初始 loss 的 sanity check**：$N$ 类分类问题开局应接近 $\ln N$；显著偏高说明输出层初始化太"自信"（`W2*=0.01, b2*=0` ）。
2. **tanh 饱和的因果链**：`fan_in` 累加方差 → `hpreact` 太宽 → tanh 两端饱和 → 局部梯度趋零。
3. **方差守恒**：$\text{Var}(y)=fan\text{-}in\cdot\text{Var}(w)\text{Var}(x)$，权重除 $\sqrt{fan\text{-}in}$ 即守恒；Kaiming 的 $\sqrt{2/fan\text{-}in}$ 是 ReLU 修正，`5/3` 是 tanh 增益。
4. **BN 四件套**：batch 统计归一化 → $\gamma,\beta$ 缩放平移 → running 统计滚动更新 → 推理用 running。
5. **三诊断**：激活看饱和（要钟形）、中间梯度看消失（四层 std 应同量级）、更新比例看步长（贴着 $-3$ 参考线；100k 断层 = lr 衰减 10 倍在 log 坐标下的 $-1$）。

---

## 附 · 资料下载

本篇配套的最终版 Notebook（已修复运行问题、补全 BatchNorm 与诊断单元，并从头完整执行保留全部输出）可直接下载：

- [MLP的一些收尾工作.ipynb](/assets/attach/day8/MLP%E7%9A%84%E4%B8%80%E4%BA%9B%E6%94%B6%E5%B0%BE%E5%B7%A5%E4%BD%9C.ipynb)：Day8 收尾版（初始化/BatchNorm/模块化 + 三张诊断图 + 有无 BN 对照实验，200k 步全量输出）

> 说明：Notebook 请放在 `names.txt` 同级目录运行；Python 3.11 + PyTorch。
