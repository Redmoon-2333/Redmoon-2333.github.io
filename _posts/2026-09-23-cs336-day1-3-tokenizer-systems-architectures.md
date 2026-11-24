---
topic: cs336
layout: post
title: "CS336 前三讲：从分词、张量核算到现代 Transformer"
date: 2026-09-23 02:00:00 +0800
categories: [技术实践]
tags: [CS336, BPE, 资源核算, RoPE, GQA, RMSNorm, SwiGLU]
excerpt: "CS336 前三讲笔记：解释 BPE 分词、训练计算量与显存估算，以及 Pre-Norm、RMSNorm、SwiGLU、RoPE 和 GQA 的作用，附注释完整的代码示例。"
math: true
---

> CS336 Lecture 1–3：文本如何分词，训练需要多少资源，以及 Transformer 各组件的作用。

## 前言

之前跟着 Karpathy 的课，我们已经能把一个小模型从分词到训练完整跑通。但跑通之后回头看，很多选择我说不出理由：为什么很多模型用 BPE 分词？训练一个模型需要多少显存、多少计算量？Pre-Norm、RMSNorm、SwiGLU 和 RoPE 各自解决了什么问题？

这次开始跟斯坦福 CS336，才刚听了 3 节课，正好讲到了这些问题。第一讲讨论怎样把文本切成模型能处理的单元；第二讲计算这些数据和模型需要多少存储与运算；第三讲再解释，资源限制和训练稳定性如何影响模型设计。

假设同一段文字用分词器 A 切出 1,000 个 token，用 B 切出 800 个。全注意力对每对位置算一个分数，分数矩阵从 `1000 × 1000` 变成 `800 × 800`，格子少 36%。推理时缓存则按历史位置数增长，其他配置相同就少 20%。若 B 的词表多出 2,000 项、每项向量有 2,048 维，输入嵌入表会多 `2,000×2,048=4,096,000` 个参数。一个分词器让正文变短，也可能让词表变大，估算时各算一笔。

BPE 代码按“先学合并表，再编码新词”的顺序运行。涉及张量的代码用 PyTorch，在 CPU 上就能观察形状和结果。

---

## 1. 分词：把文本转换成 token 序列

### 1.1 词、字符、字节与子词有什么区别

文本进入语言模型前，要先切成一个个 token（词元），再给每个 token 分配整数编号。token 可以是一个词、一个字符，也可以是词的一部分。词表保存 token 与编号的对应关系；模型通过 embedding（嵌入表）把编号转换成向量，再进行计算。

怎么切，会同时影响词表大小和序列长度：

| 分词粒度 | 怎样切 | 主要取舍 |
|---|---|---|
| 词级 | 把一个完整的词作为一个 token | 常见词表示紧凑，但新词可能不在词表中，只能用 UNK（未知词标记）表示 |
| 字符级 | 每个字符一个 token | 有限字符集的词表较小；覆盖更多语言时词表会增大，未收录的字符仍需处理 |
| 字节级 | 先转为字节，每个字节一个 token | 256 种字节值足以表示合法 UTF-8 文本，但一个汉字可能拆成多个 token |
| 子词级 | 常见片段合在一起，少见片段继续拆分 | 用较大的词表换取较短的序列，BPE 就属于这一类 |

序列越长，模型要处理的位置越多。对于全注意力，每个位置都要与其他可见位置计算相关性，这部分计算量随序列长度的平方增长。

BPE（Byte Pair Encoding，字节对编码）反复合并语料中出现次数最多的相邻片段。这里采用字节级 BPE：从 256 种字节值出发，让常见字节串逐渐成为一个 token。BPE 最早是 Gage 在 1994 年提出的数据压缩算法，Sennrich 等人在 2016 年将其用于神经机器翻译（arXiv:1508.07909）。BPE 可以从字符或字节开始；本节的 256 项初始词表对应字节级实现。

「字节级」意味着先看字符串的 UTF-8 字节表示：

```python
s = "A中文🙂"
b = s.encode("utf-8")  # 把字符串转为 UTF-8 字节串，一个字符可能占多个字节
print(len(s), len(b))  # 4 11：本例有 4 个 Unicode 码点、11 个字节
print(list(b))         # 逐个显示字节值，每个值都在 0 到 255 之间
# [65, 228, 184, 173, 230, 150, 135, 240, 159, 153, 130]
```

本例中，`A` 占 1 字节，`中` 和 `文` 各占 3 字节，`🙂` 占 4 字节。若每个字节都单独作为 token，就需要 11 个 token。BPE 会把常见字节串合并，从而减少 token 数。课程用“字节数 ÷ token 数”衡量压缩率：同一段文本下，这个比值越大，序列越短；具体计算成本还受词表大小和模型结构影响。

### 1.2 BPE 训练与编码是两件不同的事

BPE 训练的流程：

1. **预分词（pre-tokenization）**：先把语料粗切成若干片段，再在每个片段内部合并。A1 采用 GPT-2 式正则，讲义所用写法来自 tiktoken PR #234；下方教学例子为方便观察，直接按空格分词。SentencePiece 可以直接处理原始文本，不能简单等同于按空格切分。
2. **统计相邻 pair 的加权频次**：权重是该预词在语料中出现的次数，统计绝不跨预词。
3. **合并最高频 pair** 并加入词表。CS336 作业规定频次相同时取字典序更大的 pair；其他实现可能采用不同规则。
4. 重复，直到词表达到目标大小，或已经没有可合并的相邻对。按每次合并新增一个词条计算：最终词表大小 = 256 + 实际新增的合并词条数 + special token 数。

special token（特殊词元，如表示文档结束的 `<|endoftext|>`）有独立编号。训练 BPE 时，以它为界切开文本，各段分别统计和合并，特殊词元本身不参与统计。这样不会把前一篇文档的结尾与后一篇的开头合成一个 token。编码时则把它保留为一个完整 token。具体要求见 A1 讲义 §2.5。

下面这段代码演示官方讲义 §2.4 的 Sennrich 例子。`low×5` 表示 `low` 在语料中出现 5 次，因此其中每个相邻对也要累计 5 次。代码只展示六轮合并，输入已经是“单词及其频次”的表：

```python
from collections import defaultdict

corpus = {"low": 5, "lower": 2, "widest": 3, "newest": 6}
# w 是单词，c 是出现次数；如 "low" 变成 (b"l", b"o", b"w")。
# 遍历字节串得到的是整数，bytes([b]) 将它重新包装成可拼接的字节串。
words = {tuple(bytes([b]) for b in w.encode()): c for w, c in corpus.items()}

def count_pairs(words):
    counts = defaultdict(int)  # 尚未出现的相邻对从 0 开始计数
    for w, c in words.items():
        for a, b in zip(w, w[1:]):  # 错开一位配对，只统计当前单词内部
            counts[(a, b)] += c     # 同一单词出现 c 次，就贡献 c 次
    return counts

merges = []  # 按学习顺序保存规则；编码新文本时还要用到这个顺序
for _ in range(6):
    # kv[0] 是相邻对，kv[1] 是频次：先比频次，再用字典序打破平局。
    best = max(count_pairs(words).items(), key=lambda kv: (kv[1], kv[0]))[0]
    merges.append(best)
    new = {}  # 保存本轮合并后的单词表示及频次
    for w, c in words.items():
        out, i = [], 0
        while i < len(w):
            # 先检查右边还有元素，再判断当前位置能否应用本轮规则。
            if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                out.append(w[i] + w[i + 1])  # 拼成一个新的字节串
                i += 2                     # 两个旧片段都已处理，跳过它们
            else:
                out.append(w[i])  # 未匹配就保留原片段
                i += 1
        new[tuple(out)] = new.get(tuple(out), 0) + c
    words = new  # 下一轮基于新片段重新统计相邻对

print([(a.decode(), b.decode()) for a, b in merges])
# [('s', 't'), ('e', 'st'), ('o', 'w'), ('l', 'ow'), ('w', 'est'), ('n', 'e')]
```

六次合并的加权频次依次是 9、9、7、7、6、6。第一轮中，`widest` 出现 3 次、`newest` 出现 6 次，所以 `(e,s)` 与 `(s,t)` 都出现 9 次。按作业的平局规则，先合并 `(s,t)`；第二轮再把 `e` 与刚得到的 `st` 合成 `est`。

训练结束后，`merges` 固定下来。编码新文本时，不再统计新文本里哪个相邻对最常见，而是按已有规则的优先级合并。A1 讲义 §2.6.1 规定按规则的创建顺序应用。下面接着使用上段代码得到的六条规则，从前到后检查，每条规则都在当前单词中从左到右匹配：

```python
def encode_word(word, merges):
    # 新单词也从单字节片段开始，和训练时使用相同的表示。
    seq = tuple(bytes([b]) for b in word.encode())
    for a, b in merges:  # 规则顺序固定，不按新单词的频次重新排序
        out, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == (a, b):
                out.append(a + b)  # 命中规则后用合并片段替换原来的两个
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = tuple(out)  # 本条规则处理完，再应用下一条规则
    return seq

for w in ["lowest", "newest", "stone"]:
    # 这里的例子全是 ASCII，逐片段解码是为了显示合并结果。
    print(w, [t.decode() for t in encode_word(w, merges)])
# lowest ['low', 'est']
# newest ['ne', 'west']
# stone ['st', 'o', 'ne']
```

以 `lowest` 为例，六条规则逐步得到 `l,o,w,e,st`、`l,o,w,est`、`l,ow,est`，最后得到 `low,est`。示例返回的是字节片段；完整编码还要查词表，把每个片段换成整数 ID。解码时反过来查出字节串，先拼接，再统一做 UTF-8 解码，因为中文的单个 token 可能只包含某个字符的一部分字节。

规则顺序解决的是重叠冲突。假设规则先学到 `b+c→bc`，再学到 `a+b→ab`。编码 `abc` 时，两个候选都包含 `b`，先按旧规则合并成 `a,bc`；此时已没有相邻的 `a,b`，所以它不会变成 `ab,c`。这就是为什么保存分词器时，光保存词表还不够。

词表越大，常见片段越可能合成单个 token，序列通常越短；但输入嵌入表和输出投影矩阵也会变大。例如词表有 $V$ 项、模型向量维度为 $d$，嵌入表就有 $Vd$ 个参数。低频 token 能得到的训练样本也更少。

Lecture 3 第 47 页列举的词表大小包括 GPT-2 的 50257、LLaMA 的 32000、GPT-4 的 100276 和 PaLM 的 256000。课件用这些案例说明：覆盖更多语言的模型往往采用更大的词表。这些是具体模型的配置，不能作为所有同系列模型的固定值。

---

## 训练成本：从张量字节到计算时间

计算量看乘加次数，显存看每份张量占多少字节。这里从一层矩阵乘法算到 Adam、梯度累积与重计算，再区分 GPU 是算不过来还是读不过来。

[阅读全文](/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/training-cost/)

<ul>
<li id="2-资源核算张量计算量与显存"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/training-cost/#2-资源核算张量计算量与显存">2. 资源核算：张量、计算量与显存</a></li>
<li id="21-张量shapestridestoragedtype"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/training-cost/#21-张量shapestridestoragedtype">2.1 张量：shape、stride、storage、dtype</a></li>
<li id="22-从矩阵乘法到-6nd"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/training-cost/#22-从矩阵乘法到-6nd">2.2 从矩阵乘法到 6ND</a></li>
<li id="23-训练显存参数梯度优化器状态与激活"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/training-cost/#23-训练显存参数梯度优化器状态与激活">2.3 训练显存：参数、梯度、优化器状态与激活</a></li>
<li id="24-算术强度与-roofline判断计算速度的瓶颈"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/training-cost/#24-算术强度与-roofline判断计算速度的瓶颈">2.4 算术强度与 Roofline：判断计算速度的瓶颈</a></li>
<li id="25-梯度累积与激活重计算降低激活显存"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/training-cost/#25-梯度累积与激活重计算降低激活显存">2.5 梯度累积与激活重计算：降低激活显存</a></li>
</ul>

## 现代 Transformer：从归一化到 KV Cache

按一个 token 进入模型后的顺序理解部件：先认识归一化、位置与前馈层，再算生成时重复读取哪些数据、GQA 又能省下多少缓存。

[阅读全文](/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/transformer/)

<ul>
<li id="3-transformer-架构各组件解决什么问题"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/transformer/#3-transformer-架构各组件解决什么问题">3. Transformer 架构：各组件解决什么问题</a></li>
<li id="31-pre-norm归一化放在子层之前"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/transformer/#31-pre-norm归一化放在子层之前">3.1 Pre-Norm：归一化放在子层之前</a></li>
<li id="32-rmsnorm-与去偏置简化归一化和线性层"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/transformer/#32-rmsnorm-与去偏置简化归一化和线性层">3.2 RMSNorm 与去偏置：简化归一化和线性层</a></li>
<li id="33-swiglu前馈网络中的门控与参数量"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/transformer/#33-swiglu前馈网络中的门控与参数量">3.3 SwiGLU：前馈网络中的门控与参数量</a></li>
<li id="34-rope用旋转表达相对位置"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/transformer/#34-rope用旋转表达相对位置">3.4 RoPE：用旋转表达相对位置</a></li>
<li id="35-超参数前馈维度注意力头数与层数"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/transformer/#35-超参数前馈维度注意力头数与层数">3.5 超参数：前馈维度、注意力头数与层数</a></li>
<li id="36-训练稳定性z-lossqk-norm-与-soft-capping"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/transformer/#36-训练稳定性z-lossqk-norm-与-soft-capping">3.6 训练稳定性：z-loss、QK-Norm 与 soft-capping</a></li>
<li id="37-推理效率kv-cachegqa-与滑动窗口"><a href="/2026/09/23/cs336-day1-3-tokenizer-systems-architectures/transformer/#37-推理效率kv-cachegqa-与滑动窗口">3.7 推理效率：KV Cache、GQA 与滑动窗口</a></li>
</ul>

## 4. 分词、资源与架构之间的关系

分词决定同一段文本需要多少 token，也决定嵌入表有多大。序列长度和参数量进一步影响计算量与显存。架构设计则在这些约束下，选择归一化、前馈层和注意力的具体形式：

| 内容 | 要回答的问题 | 代表结论 |
|---|---|---|
| 文本表示 | 文本怎样切分，词表需要多大 | BPE 在词表规模与序列长度之间取舍；特殊词元单独处理 |
| 资源核算 | 需要多少计算和存储，速度受什么限制 | $6ND$ 估算稠密训练量；按 dtype 逐项计算显存；用 Roofline 分析算力与带宽限制 |
| 架构选择 | 怎样稳定训练并控制推理成本 | Pre-Norm 改变梯度路径；RMSNorm 简化归一化；GQA 减少 KV 缓存；滑窗减少注意力交互 |

例如，更大的词表可以缩短输入序列，却会增加嵌入表和输出层参数；更多 KV 共享可以减少缓存，却会改变注意力的表达方式；激活重计算可以减少显存，却会增加计算量。比较配置时，需要同时说明改变了哪个量，以及其他条件是否保持一致。

CS336 A1（Version 26.0.3，Spring 2026）将这些内容对应到 BPE 分词器、Transformer、交叉熵和 AdamW 等实现任务。前三讲提供的依据分别是：文本表示规则、资源计算方法，以及组件选择的原理和实验经验。

## 附 · 参考与延伸

**课程与官方材料**

- [CS336 课程主页（2026 春季）](https://cs336.stanford.edu/)，Lecture 1/2/3 分别为 Overview+Tokenization、PyTorch+资源核算、架构与超参数
- [官方讲义仓库 stanford-cs336/lectures](https://github.com/stanford-cs336/lectures)（lecture_03.pdf）与 [assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)（A1 讲义 Version 26.0.3）

**论文（正文引用处已标注）**

- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*（BPE 引入 NLP）: [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Shazeer, *GLU Variants Improve Transformer*: [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*（RoPE）: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*（GQA）: [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Kaplan et al., *Scaling Laws for Neural Language Models*: [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)；Xiong et al. 2020（Pre-LN 收敛）、Andriushchenko et al. 2023（权重衰减）、Ivanov et al. 2023（归一化运行时间占比）见课件引用
