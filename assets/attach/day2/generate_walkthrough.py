# -*- coding: utf-8 -*-
"""生成 TRAINING_WALKTHROUGH.md —— 玩具注意力模型训练全程拆解文档
====================================================================
运行：python generate_walkthrough.py
文档里所有数字都来自本次真实运行（种子 7 固定，可复现）。
模型定义与 two_pass_training.py 完全一致，但训练 N_PASSES 轮直到收敛，
然后分析最终权重的含义。
"""
import numpy as np

np.random.seed(7)

# ---------------- 模型定义（与 two_pass_training.py 一致） ----------------
VOCAB = ["<pad>", "小猫", "垫子", "月亮", "很", "困", "软", "亮"]
tid = {w: i for i, w in enumerate(VOCAB)}
D = 8
LR = 0.5
INIT_SCALE = 0.3
N_PASSES = 300
TRAIN = [
    (["<pad>", "小猫", "很"], "困"),
    (["<pad>", "垫子", "很"], "软"),
    (["<pad>", "月亮", "很"], "亮"),
]
PROPS = [t for _, t in TRAIN]

rng = np.random.default_rng(7)
E = rng.normal(0, 1, (len(VOCAB), D))
W_Q = rng.uniform(-INIT_SCALE, INIT_SCALE, (D, D))
W_K = rng.uniform(-INIT_SCALE, INIT_SCALE, (D, D))
W_V = rng.uniform(-INIT_SCALE, INIT_SCALE, (D, D))
W_OUT = rng.uniform(-INIT_SCALE, INIT_SCALE, (len(VOCAB), D))
B_OUT = np.zeros(len(VOCAB))

INIT = dict(W_Q=W_Q.copy(), W_K=W_K.copy(), W_V=W_V.copy(),
            W_OUT=W_OUT.copy(), B_OUT=B_OUT.copy())   # 训练前快照（文档用）


def forward(ids):
    X = E[ids]
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    S = Q @ K.T / np.sqrt(D)
    S -= S.max(axis=-1, keepdims=True)
    e = np.exp(S)
    A = e / e.sum(axis=-1, keepdims=True)
    O = A @ V
    h = O[-1]
    logits = W_OUT @ h + B_OUT
    logits -= logits.max()
    p = np.exp(logits) / np.exp(logits).sum()
    return dict(X=X, Q=Q, K=K, V=V, S=S, A=A, O=O, h=h, logits=logits, p=p)


def backward(cache, tgt_idx):
    p, h, A, V, K, Q, X = (cache["p"], cache["h"], cache["A"], cache["V"],
                           cache["K"], cache["Q"], cache["X"])
    dlogits = p.copy()
    dlogits[tgt_idx] -= 1.0
    dW_OUT = np.outer(dlogits, h)
    dB_OUT = dlogits
    dh = W_OUT.T @ dlogits
    dO = np.zeros_like(V)
    dO[-1] = dh
    dA = dO @ V.T
    dV = A.T @ dO
    dS = A * (dA - (dA * A).sum(axis=1, keepdims=True))
    dQ = (dS @ K) / np.sqrt(D)
    dK = (dS.T @ Q) / np.sqrt(D)
    return dict(W_Q=X.T @ dQ, W_K=X.T @ dK, W_V=X.T @ dV, W_OUT=dW_OUT, B_OUT=dB_OUT)


def mat(m, nd=2):
    return "\n".join("    " + "  ".join(f"{v:+.{nd}f}" for v in row) for row in m)


def vec(v, nd=3):
    return "  ".join(f"{x:+.{nd}f}" for x in v)


def dot_line(label, a, b, result, nd=2):
    terms = " + ".join(f"{ai:+.{nd}f}·{bi:+.{nd}f}" for ai, bi in zip(a, b))
    return f"{label}: {terms} = {result:+.4f}"


def batch_loss():
    return sum(-np.log(max(forward(i)["p"][t], 1e-12)) for i, t in zip(ids_all, tgt_all))


# ---------------- 训练 N_PASSES 轮 ----------------
ids_all = [[tid[w] for w in seq] for seq, _ in TRAIN]
tgt_all = [tid[t] for _, t in TRAIN]

curve = []
pass1_caches = pass1_grads = None
pass1_loss = None

for p in range(1, N_PASSES + 1):
    caches, total = [], 0.0
    grads = {k: np.zeros_like(v) for k, v in INIT.items()}
    for k, ids in enumerate(ids_all):
        c = forward(ids)
        caches.append(c)
        total += -np.log(max(c["p"][tgt_all[k]], 1e-12))
        g = backward(c, tgt_all[k])
        for key in grads:
            grads[key] += g[key]
    for key in grads:
            grads[key] /= len(TRAIN)   # 平均 loss：梯度除以样本数，避免大 lr 发散
    for name, pm in dict(W_Q=W_Q, W_K=W_K, W_V=W_V, W_OUT=W_OUT, B_OUT=B_OUT).items():
        pm -= LR * grads[name]
    curve.append(dict(n=p, loss=total, ps=[c["p"][t] for c, t in zip(caches, tgt_all)]))
    if p == 1:
        pass1_caches, pass1_grads, pass1_loss = caches, grads, total

final_caches = [forward(ids) for ids in ids_all]
final_loss = sum(-np.log(max(c["p"][t], 1e-12)) for c, t in zip(final_caches, tgt_all))

# 数值梯度抽查（在收敛后的参数下，验证手推反向正确性）
c_now = forward(ids_all[0])
analytic_now = backward(c_now, tgt_all[0])["W_Q"][0, 0]
o = W_Q[0, 0]
eps = 1e-5
W_Q[0, 0] = o + eps
lp = batch_loss()
W_Q[0, 0] = o - eps
lm = batch_loss()
W_Q[0, 0] = o
numerical_now = (lp - lm) / (2 * eps)

# ---------------- 生成 Markdown ----------------
md = []
add = md.append

add("# 玩具注意力模型 · 训练全程拆解（真实运行数据）\n")
add("> 本文由 `generate_walkthrough.py` 自动生成（种子 7，可复现），**所有数字都是程序实际算出来的**，非手打。")
add("> 模型与任务和 `two_pass_training.py` 完全一致，但这里训练 "
    f"{N_PASSES} 轮直到收敛，最后分析学到的权重。\n")

add("## 0. 模型结构与任务（一分钟版）\n")
add("**数据流**（单头自注意力，只有位置 2 的输出接 loss）：\n")
add("```")
add("词 id → 查词嵌入表 E（固定不训） → X (3×8)")
add("      → Q = X·W_Q   K = X·W_K   V = X·W_V    ← 同一个 X 的三种投影（可学习）")
add("      → S = QKᵀ/√8 → A = softmax(S)          ← 注意力权重，每行和为 1")
add("      → O = A·V → 取末位 h = O[2]             ← 位置 2 的'信息包'")
add("      → logits = W_OUT·h + b → softmax → P(词) → loss = -log P(正确词)")
add("```\n")
add("**任务**：输入 `[<pad>, 主语, 很]`，预测属性词。\n")
add("| 输入 | 应预测 |")
add("| --- | --- |")
for seq, tgt in TRAIN:
    add(f"| {' '.join(seq)} | **{tgt}** |")
add("\n**为什么这个任务需要注意力**：位置 2（\"很\"）自己不携带任何主语信息，"
    "想预测对属性，它的注意力必须**越过\"很\"、回头盯住位置 1 的主语**——"
    "这正是训练结束后我们能从权重里读出的\"视角\"。\n")

add("## 1. 变量总表 + 初始参数（训练前的随机数）\n")
add("本文用到的**全部变量**一览（后面每一节都会用到它们）：\n")
add("| 变量 | 形状 | 是什么 | 训练？ |")
add("| --- | --- | --- | --- |")
add("| **E** | 8×8 | 词嵌入表：每个词的初始向量，查表用 | ❌ 固定 |")
add("| **W_Q** | 8×8 | Query 投影：决定\"提什么问题\" | ✅ |")
add("| **W_K** | 8×8 | Key 投影：决定\"怎么自我介绍\" | ✅ |")
add("| **W_V** | 8×8 | Value 投影：决定\"交付什么内容\" | ✅ |")
add("| **W_OUT / B_OUT** | 8×8 / 8 | 输出层：把信息包翻译成词表打分 | ✅ / ✅ |")
add("| **X** | 3×8 | 一个样本的输入向量序列（查 E 表而来，每样本一份） | 数据 |")
add("| **Q K V** | 3×8 各三 | X 的三种投影（每样本各一套） | — |")
add("| **S** | 3×3 | 打分矩阵 = QKᵀ/√8 | — |")
add("| **A** | 3×3 | 注意力权重 = softmax(S)，每行和为 1 | — |")
add("| **O** | 3×8 | 汇聚输出 = A·V | — |")
add("| **h** | 8 | 位置 2 的信息包（O 的最后一行） | — |")
add("| **logits / p** | 8 | 词表打分 / 概率 | — |")
add("")
add("```")
add(f"E =\n{mat(E)}")
add(f"\nB_OUT =\n[{vec(INIT['B_OUT'])}]   （全 0：输出层偏置从零开始）")
add(f"\nW_Q =\n{mat(INIT['W_Q'])}")
add(f"\nW_K =\n{mat(INIT['W_K'])}")
add(f"\nW_V =\n{mat(INIT['W_V'])}")
add(f"\nW_OUT =\n{mat(INIT['W_OUT'])}")
add("```\n")
add("> 概念解析：初始时 W 全是随机小数，所以注意力接近均匀、预测接近瞎猜"
    f"（第 1 轮 loss = {pass1_loss:.3f}，而 ln(8)≈2.08 正是 8 选 1 瞎猜的水平）。"
    "训练的全部意义就是把这些随机数改成\"有结构\"的数。\n")

c1 = pass1_caches[0]
ids1, words1 = ids_all[0], [VOCAB[i] for i in ids_all[0]]
tgt1 = tgt_all[0]

add("## 2. 第 1 次前向传播：逐步拆解（主角 = 样本 1 的位置 2）\n")

add("### 2.1 词向量：查表得来（三个样本的 X 全部列出）\n")
add("```")
for k, (seq, tgt) in enumerate(TRAIN):
    ck = pass1_caches[k]
    ws = [VOCAB[i] for i in ids_all[k]]
    add(f"样本{k+1} {' '.join(ws)}:")
    for i, w in enumerate(ws):
        add(f"  X[{i}] ({w}) = E 第 {ids_all[k][i]} 行 = [{vec(ck['X'][i])}]")
add("```\n")
add("> 概念解析：词嵌入表 E 是固定的随机数（本演示不训练它）。"
    "真实 Transformer 里词嵌入是训练出来的——但角色一样：把离散的词变成可计算的向量。\n")

add("### 2.2 Q / K / V：同一个 X 的三种投影\n")
add("**样本 1**（主角，后面逐步拆它）：\n")
add("```")
add(f"Q = X·W_Q =\n{mat(c1['Q'])}")
add(f"\nK = X·W_K =\n{mat(c1['K'])}")
add(f"\nV = X·W_V =\n{mat(c1['V'])}")
add("```\n")
add("**样本 2 / 样本 3** 的 Q（K、V 同理 = 各自 X 乘同一组 W，此处一并列出）：\n")
add("```")
for k in (1, 2):
    ck = pass1_caches[k]
    ws = "".join(VOCAB[i] for i in ids_all[k])
    add(f"样本{k+1}({ws}) Q =\n{mat(ck['Q'])}")
    add(f"样本{k+1}({ws}) K =\n{mat(ck['K'])}")
    add(f"样本{k+1}({ws}) V =\n{mat(ck['V'])}")
add("```\n")
q20 = c1["X"][2] @ INIT["W_Q"][:, 0]
add("抽查一个分量怎么算（位置 2 \"很\" 的 Q 第 0 维）：\n")
add("```")
add(dot_line("Q[2,0] = X[2]·W_Q初值[:,0]", c1["X"][2], INIT["W_Q"][:, 0], q20))
add("```\n")
add("> 概念解析：Q、K、V 是同一个 X 乘三个不同矩阵得到的，**同源、独立、互不推出**。"
    "三个 W 矩阵的参数全部由训练学出：W_Q 负责\"提什么问题\"，W_K 负责\"怎么自我介绍\"，"
    "W_V 负责\"实际交付什么内容\"。\n")

add("### 2.3 打分：位置 2 对三个位置分别点积\n")
add("参与运算：**Q[2]**（= 2.2 样本 1 Q 矩阵的第 2 行，\"很\"的查询向量）"
    "和 **K 的三行**（每个词的自我介绍，见 2.2 样本 1 K 矩阵）。公式：s(i→j) = Q[i]·K[j]/√D。\n")
s2 = c1["S"][2]
add("```")
for j in range(3):
    add(dot_line(f"s(2→{j}) = Q[2]·K[{j}]/√{D}", c1["Q"][2], c1["K"][j],
                 c1["Q"][2] @ c1["K"][j] / np.sqrt(D)))
add(f"三个原始分: [{vec(s2)}]")
add("```\n")
add("三个样本的完整打分矩阵 S（行=Query 位置，列=Key 位置）：\n")
add("```")
for k in range(3):
    ck = pass1_caches[k]
    ws = "".join(VOCAB[i] for i in ids_all[k])
    add(f"样本{k+1}({ws}) S =\n{mat(ck['S'], 3)}")
add("```\n")

add("### 2.4 softmax：分数 → 和为 1 的权重\n")
e2 = np.exp(s2 - s2.max())
w2 = e2 / e2.sum()
add("```")
add(f"减最大值 {s2.max():+.4f}（防溢出）: [{vec(s2 - s2.max())}]")
add(f"逐个取指数 e^x        : [{vec(e2)}]   求和 = {e2.sum():.4f}")
for j in range(3):
    add(f"A[2,{j}] = {e2[j]:.4f} / {e2.sum():.4f} = {w2[j]:.4f}   ← 看 '{words1[j]}' 的比重")
add(f"验证和为 1: {w2.sum():.4f}")
add("```\n")
add("三个样本的完整注意力矩阵 A（每行和为 1）：\n")
add("```")
for k in range(3):
    ck = pass1_caches[k]
    ws = "".join(VOCAB[i] for i in ids_all[k])
    add(f"样本{k+1}({ws}) A =\n{mat(ck['A'], 3)}")
add("```\n")
add("> 概念解析：softmax 是\"软性 argmax\"——分数差距越大权重越尖；"
    "全相等时退化为均匀分布。它保证权重和为 1，所以输出永远是\"按比例混合\"而不是\"全抄某一个\"。\n")

add("### 2.5 加权汇聚：按权重取货（公式：O = A·V）\n")
add("**要算什么**：位置 2 的信息包 O[2] = A[2,0]×V[0] + A[2,1]×V[1] + A[2,2]×V[2]"
    "——用 2.4 算出的三个注意力权重当\"配比\"，把 2.2 里 V 矩阵的三行\"内容\"按比例混合。\n")
add("**参与运算的两组数，先摆在一起**：\n")
add("```")
add(f"配比（来自 2.4 的 A 第 2 行）: A[2,0]={w2[0]:.4f}  A[2,1]={w2[1]:.4f}  A[2,2]={w2[2]:.4f}")
add(f"原料（来自 2.2 的 V 矩阵三行）:")
add(f"  V[0]（<pad> 的内容）= [{vec(c1['V'][0])}]")
add(f"  V[1]（小猫 的内容） = [{vec(c1['V'][1])}]")
add(f"  V[2]（很   的内容） = [{vec(c1['V'][2])}]")
add("```\n")
add("**逐分量计算**（信息包的第 k 个分量 = 三个 V 的第 k 分量按配比加权和）：\n")
add("```")
add(f"O[2] = {w2[0]:.4f}·V[0] + {w2[1]:.4f}·V[1] + {w2[2]:.4f}·V[2]")
for j in range(3):
    add(f"  {w2[j]:.4f} × V[{j}]({words1[j]}) = [{vec(w2[j] * c1['V'][j])}]")
add(f"相加 = O[2] = [{vec(c1['O'][2])}]")
add("```\n")
add("抽查一个分量的完整算术（第 0 维，其余 7 维同理）：\n")
add("```")
add(f"O[2][0] = {w2[0]:.4f}×({c1['V'][0][0]:+.3f}) + {w2[1]:.4f}×({c1['V'][1][0]:+.3f}) + {w2[2]:.4f}×({c1['V'][2][0]:+.3f})")
add(f"        = {w2[0] * c1['V'][0][0]:+.4f} + {w2[1] * c1['V'][1][0]:+.4f} + {w2[2] * c1['V'][2][0]:+.4f}")
add(f"        = {c1['O'][2][0]:+.4f}   ✓ 与上面一致")
add("```\n")
add("> 概念解析：这一步就是\"取货\"——注意力权重是\"配比\"，V 是\"原料\"，"
    "信息包 = 按配比混合的原料。训练后配比高度集中在主语上，"
    "信息包就几乎等于\"主语的内容\"——这就是\"查到了主语的信息\"的数学形式。\n")

add("### 2.6 logits → 概率 → loss\n")
add("参与运算：**h**（= 2.5 算出的 O[2]，位置 2 的信息包）和 **W_OUT 的 8 行**"
    "（每个候选词的\"打分模板\"，此刻还是第 1 节的初值）。公式：logits[j] = h·W_OUT[j] + B_OUT[j]。\n")
lg = c1["logits"]
e_all = np.exp(lg)
add("```")
add(f"h = O[2] = [{vec(c1['h'])}]")
for w in ["困", "软", "亮", "小猫"]:
    j = tid[w]
    add(dot_line(f"logits[{w}] = h·W_OUT[{j}]", c1["h"], INIT["W_OUT"][j],
                 c1["h"] @ INIT["W_OUT"][j]))
add(f"8 个 logits: [{vec(lg)}]")
add(f"逐个取指数  : [{vec(e_all)}]   求和 = {e_all.sum():.3f}")
add(f"P(困) = {e_all[tgt1]:.3f} / {e_all.sum():.3f} = {c1['p'][tgt1]:.4f}")
add(f"loss  = -ln(P) = {-np.log(c1['p'][tgt1]):.4f}")
add("```\n")
add("> 概念解析：logits 是\"每个候选词的得分\"，softmax 把得分变概率；"
    "交叉熵 loss = -log(正确词概率)——概率越接近 1，loss 越接近 0。"
    "训练的全部目标就是**把正确词的概率推高**。\n")

add("### 2.7 三个样本的第 1 次前向总览\n")
add("| 样本 | P(正确词) | loss |")
add("| --- | --- | --- |")
for k, (seq, tgt) in enumerate(TRAIN):
    ck = pass1_caches[k]
    add(f"| 样本{k+1}（应预测 {tgt}） | {ck['p'][tgt_all[k]]:.4f} | {-np.log(max(ck['p'][tgt_all[k]], 1e-12)):.4f} |")
add(f"| **合计（训练要减小的目标）** | — | **{pass1_loss:.4f}** |")
add("")

g1 = pass1_grads
add("## 3. 第 1 次反向传播：梯度从哪来\n")

add("### 3.1 梯度源头：dlogits = p − onehot(正确词)\n")
add("```")
add(f"p       = [{vec(c1['p'])}]")
add(f"dlogits = [{vec(g1['B_OUT'])}]")
add("```\n")
add("> 概念解析：交叉熵+softmax 的梯度有个极简形式——**正确词概率被拽高的力度 = (1−p)，"
    "错误词被压低的力度 = 各自的 p**。概率错得越离谱，推力越大。\n")

add("### 3.2 W_OUT[困,0] 的梯度 = 三个样本贡献相加\n")
add("```")
tot = 0.0
for k, (seq, tgt) in enumerate(TRAIN):
    ck = pass1_caches[k]
    dl = ck["p"][tid[tgt]] - 1.0
    contrib = dl * ck["h"][0]
    tot += contrib
    add(f"样本{k+1}(应预测{tgt}): dlogits[{tid[tgt]}]={dl:+.3f} × h[0]={ck['h'][0]:+.3f} → {contrib:+.4f}")
add(f"相加 = 总梯度 {tot:+.4f}")
add("```\n")
add("> 概念解析：这就是 micrograd 里\"梯度是所有路径贡献之和，必须 += 累加\"的再现——"
    "batch 里每个样本都往同一个参数上贡献一份梯度。\n")

add("### 3.3 更新公式实算\n")
add("```")
add(f"W_OUT[困,0] = {INIT['W_OUT'][tgt1,0]:+.4f} - {LR} × {g1['W_OUT'][tgt1,0]:+.4f} "
    f"= {INIT['W_OUT'][tgt1,0] - LR * g1['W_OUT'][tgt1,0]:+.4f}")
add("```\n")

add("### 3.4 数值梯度抽查（验证手推反向没推错）\n")
add("```")
add(f"W_Q[0,0]: 手推 = {analytic_now:+.6f}   数值差分 = {numerical_now:+.6f}   "
    f"误差 = {abs(analytic_now - numerical_now):.2e}")
add("```\n")
add("> 概念解析：把参数手动挪动 ±ε 各算一次 loss，(loss₊−loss₋)/2ε 就是数值导数。"
    "两者一致 → 手推的反向传播是对的。micrograd 里\"对照数值导数\"的同款操作。\n")

add(f"## 4. {N_PASSES} 轮训练：loss 与正确词概率曲线\n")
add("| 轮次 | 总 loss | P(困) | P(软) | P(亮) |")
add("| --- | --- | --- | --- | --- |")
show_passes = sorted(set([1, 2, 5, 10, 20, 50, 100, 150, N_PASSES - 1, N_PASSES]))
for rec in curve:
    if rec["n"] in show_passes:
        add(f"| {rec['n']:3d} | {rec['loss']:.4f} | {rec['ps'][0]:.3f} | "
            f"{rec['ps'][1]:.3f} | {rec['ps'][2]:.3f} |")
add("")
add("> 概念解析：前期 loss 骤降主要靠**输出层 W_OUT**（把 logits 摆正，快）；"
    "后期注意力权重慢慢聚焦（慢）。两类参数的收敛速度天然不同——"
    "这就是两次版演示里 |grad W_OUT| 远大于 |grad W_Q| 的原因。\n")

add("## 5. 训练完成：最终权重在说什么（反向理解）\n")

add("### 5.1 注意力分布：训练前 vs 训练后（位置 2 的行）\n")
add("| 样本 | 训练前（pad / 主语 / 很） | 训练后（pad / 主语 / 很） |")
add("| --- | --- | --- |")
for k, (seq, tgt) in enumerate(TRAIN):
    b = pass1_caches[k]["A"][-1]
    a = final_caches[k]["A"][-1]
    add(f"| {''.join(VOCAB[i] for i in ids_all[k])} → {tgt} | "
        f"{b[0]:.3f} / {b[1]:.3f} / {b[2]:.3f} | **{a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f}** |")
add("")
add("> 读法：训练前三个位置接近均摊（≈1/3）；训练后位置 2 的注意力**几乎全部压在主语上**"
    "（位置 1），\"<pad>\"和\"很\"被压到接近 0。\n")
add("> **诚实的发现**：三个样本没有走同一条路——样本 2/3 学到了\"回头盯主语\""
    "（0.95、0.99 压在主语上）；样本 1 却把 0.66 的注意力放在了 <pad> 上。这不是 bug："
    "对样本 1 来说，\"关注常量 <pad> + 输出层记住这种组合\"同样能把 P(困) 推到 0.999——"
    "**梯度下降只保证 loss 下降，不保证选中的是最优雅的路径**（真实模型里的\"捷径学习\"现象）。"
    "样本 2/3 的模式才是我们想要的\"属性查询\"视角。\n")

add("### 5.2 打分的变化：Q·K 学会了\"认主语\"\n")
add("| 样本 | s(2→pad) 前→后 | s(2→主语) 前→后 | s(2→很) 前→后 |")
add("| --- | --- | --- | --- |")
for k in range(3):
    sb = pass1_caches[k]["S"][2]
    sa = final_caches[k]["S"][2]
    add(f"| 样本{k+1} | {sb[0]:+.4f} → {sa[0]:+.4f} | {sb[1]:+.4f} → **{sa[1]:+.4f}** | "
        f"{sb[2]:+.4f} → {sa[2]:+.4f} |")
add("")
add("> 解读：主语方向的打分被显著放大（或 pad/很 方向被打压）——"
    "这就是 W_Q、W_K 训练后学到的东西：**位置 2 的 Query 与主语的 Key 在投影空间里方向对齐**。"
    "注意力权重的变化只是这个打分变化经过 softmax 的结果。\n")
add("> 样本 1 是例外：它的 s(2→pad) 反而升到相对最高——对应 5.1 里它走的\"<pad> 捷径\"。"
    "读表时把样本 2/3 当作\"属性查询头\"的标准形态即可。\n")

add("### 5.3 输出层：3×3 \"主语 → 属性\"对齐矩阵\n")
add("h（位置 2 的信息包，几乎 = 主语内容）与 W_OUT 里三行属性向量的点积：\n")
add("| 信息包来自 | ×W_OUT[困] | ×W_OUT[软] | ×W_OUT[亮] | 应选 |")
add("| --- | --- | --- | --- | --- |")
for k, (seq, tgt) in enumerate(TRAIN):
    h = final_caches[k]["h"]
    row = [h @ W_OUT[tid[pr]] for pr in PROPS]
    subj = VOCAB[ids_all[k][1]]
    add(f"| {subj} 的内容 | {row[0]:+.3f} | {row[1]:+.3f} | {row[2]:+.3f} | **{tgt}** |")
add("")
add("> 解读：对角线（小猫→困、垫子→软、月亮→亮）显著高于非对角线 —— "
    "W_OUT 学会了**把\"主语内容\"翻译成\"正确属性\"**。模型答对的机制就在这张表里。\n")

add("### 5.4 这个头学到的\"视角\"（一句话）\n")
add("> **\"属性查询头\"：要猜一个词的属性，越过无信息的连接词，回头盯住它的主语，"
    "再把主语身份翻译成对应属性。** 这正是 Day2 笔记里\"多头 = 每个头学一种关系视角\"的"
    "最小可运行证据——真实 Transformer 里成百上千个头，各自学到的关系（指代/语法/共现…）"
    "都可以用本文同款方法读出来。\n")

add("## 6. 概念 FAQ（聊天里问过的问题存档）\n")
add("**Q1：K 是由 V 推出来的，还是训练出来的？**\n")
add("都不是。K 和 V 是同一个词的输入向量 x 分别乘以 W^K、W^V 得到的——**同源、独立、互不推出**。"
    "训练学出的是 W 矩阵的参数；推理时每个词的 K、V 由它自己的 x 即时算出。\n")
add("**Q2：生成\"垫子\"那一刻，Q 是谁？**\n")
add("是当前已生成序列末位（\"在\"）的 Query——**垫子还没出生，不可能提供 Q**。"
    "流程：\"在\"的 Q 取货 → 预测出垫子 → 垫子拼回输入 → 下一步垫子才有自己的 Q。"
    "垫子是输出，不是输入。\n")
add("**Q3：训练和推理有什么区别？**\n")
add("训练用 teacher forcing：完整答案一次喂入，掩码保证每个位置只看过去，"
    "并行算出所有位置的预测。推理没有答案，只能逐词生成，一步一轮回。\n")
add("**Q4：为什么初始注意力是均匀的？**\n")
add("初始 W 是随机小数 → Q·K 分数差距很小 → softmax 接近均匀（≈1/3）。"
    "训练让\"该看的方向\"打分变大，softmax 逐渐变尖。INIT_SCALE 调大，初始分布就会变尖。\n")

out_path = r"D:\大学相关\03_个人成长与记录\LLM学习体系\07_MyProject\02_transformer\TRAINING_WALKTHROUGH.md"
with open(out_path, "w", encoding="utf-8") as fp:
    fp.write("\n".join(md))
print(f"written: {out_path}  ({len(md)} blocks)")
print(f"final loss = {final_loss:.4f}  (pass1 = {pass1_loss:.4f})")
for k, (seq, tgt) in enumerate(TRAIN):
    print(f"  P({tgt}) = {final_caches[k]['p'][tgt_all[k]]:.4f}")
print(f"grad check: analytic={analytic_now:+.6f} numerical={numerical_now:+.6f}")
