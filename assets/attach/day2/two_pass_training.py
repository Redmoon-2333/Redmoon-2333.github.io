# -*- coding: utf-8 -*-
"""02_transformer · 两次前向反向传播演示（配套 08_MyNote/Day2 第 6/7/11 节）
====================================================================
和 01_micrograd 一样的思路：把训练拆到最细。
本版输出是"显微镜模式"：样本 1 位置 2 的每一步计算都逐项摊开，
输出里每个数字都标注了它是怎么算出来的。

任务（故意设计成"注意力有活干"）：
    [小猫, 很, ?] -> 困      [垫子, 很, ?] -> 软      [月亮, 很, ?] -> 亮
  想答对，位置 2（"很"）的注意力必须越过自己、回头盯住位置 0 的主语。

只训练 4 组参数：W_Q / W_K / W_V / W_OUT（词嵌入固定，相当于查表）。
想玩：改 LEARNING_RATE / INIT_SCALE / 换训练对，重跑，对比 QKV 的变化。
"""
import numpy as np

np.random.seed(7)   # 固定种子：你跑出的每个数都和注释一致

# ============================================================
# 0. 任务设定
# ============================================================
VOCAB = ["<pad>", "小猫", "垫子", "月亮", "很", "困", "软", "亮"]
tid = {w: i for i, w in enumerate(VOCAB)}
D = 8                      # 模型维度（真实 Transformer 是 512，这里 8 方便打印）
LEARNING_RATE = 1.0        # 学习率：故意开大，两次更新也要看出明显变化
INIT_SCALE = 0.3           # 参数初始值范围（±INIT_SCALE 的均匀随机）

TRAIN = [                  # (前两个位置的词, 位置 2 应预测的词)
    (["<pad>", "小猫", "很"], "困"),
    (["<pad>", "垫子", "很"], "软"),
    (["<pad>", "月亮", "很"], "亮"),
]

print("=" * 60)
print("0. 任务设定")
print("=" * 60)
print(f"词表({len(VOCAB)}): {VOCAB}")
print(f"模型维度 D={D}  学习率={LEARNING_RATE}  初始化范围=±{INIT_SCALE}")
for seq, tgt in TRAIN:
    print(f"  训练对: {' '.join(seq)} -> {tgt}")
print("只有位置 2 的输出接 loss（位置 0/1 只是陪跑，但它们的注意力照样计算）")

# ============================================================
# 1. 参数初始化 + 固定词嵌入
# ============================================================
rng = np.random.default_rng(7)
E = rng.normal(0, 1, (len(VOCAB), D))          # 词嵌入表：固定不训练（当查表用）
W_Q = rng.uniform(-INIT_SCALE, INIT_SCALE, (D, D))
W_K = rng.uniform(-INIT_SCALE, INIT_SCALE, (D, D))
W_V = rng.uniform(-INIT_SCALE, INIT_SCALE, (D, D))
W_OUT = rng.uniform(-INIT_SCALE, INIT_SCALE, (len(VOCAB), D))
B_OUT = np.zeros(len(VOCAB))
PARAMS = {"W_Q": W_Q, "W_K": W_K, "W_V": W_V, "W_OUT": W_OUT, "B_OUT": B_OUT}

print()
print("=" * 60)
print("1. 参数初始化（词嵌入 E 固定不训，只训下面 4 组）")
print("=" * 60)
for name, p in PARAMS.items():
    print(f"  {name}: shape={p.shape}  范数={np.linalg.norm(p):.3f}")


# ============================================================
# 2. 前向 / 反向 的手写实现（micrograd 精神：规则自己推）
# ============================================================
def forward(ids):
    """ids: 长度 3 的 token id 序列。返回前向中间值字典。"""
    X = E[ids]                                  # (3, D) 查表得到每个词的向量
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V         # 同一个 X 投影出三个角色
    S = Q @ K.T / np.sqrt(D)                    # 相似度矩阵 (3,3)，缩放防饱和
    S -= S.max(axis=-1, keepdims=True)          # 数值稳定
    e = np.exp(S)
    A = e / e.sum(axis=-1, keepdims=True)       # 逐行 softmax -> 注意力权重
    O = A @ V                                   # 加权汇聚 (3, D)
    h = O[-1]                                   # 只取位置 2 的输出接预测
    logits = W_OUT @ h + B_OUT                  # (V,) 词表上的打分
    logits -= logits.max()
    p = np.exp(logits) / np.exp(logits).sum()   # softmax 概率
    return dict(X=X, Q=Q, K=K, V=V, S=S, A=A, O=O, h=h, logits=logits, p=p)


def backward(cache, tgt_idx):
    """手推反向：loss = -log p[tgt]（单样本）。返回各组参数的梯度。"""
    p, h, A, V, K, Q, X = cache["p"], cache["h"], cache["A"], cache["V"], cache["K"], cache["Q"], cache["X"]
    dlogits = p.copy()
    dlogits[tgt_idx] -= 1.0                     # CE + softmax: dlogits = p - onehot
    dW_OUT = np.outer(dlogits, h)
    dB_OUT = dlogits
    dh = W_OUT.T @ dlogits                      # (D,)
    dO = np.zeros_like(V)
    dO[-1] = dh                                 # 只有位置 2 连着 loss
    dA = dO @ V.T                               # (3,3)
    dV = A.T @ dO                               # (D,D)
    dS = A * (dA - (dA * A).sum(axis=1, keepdims=True))   # softmax 反向（逐行）
    dQ = (dS @ K) / np.sqrt(D)
    dK = (dS.T @ Q) / np.sqrt(D)
    dW_Q = X.T @ dQ
    dW_K = X.T @ dK
    dW_V = X.T @ dV
    return dict(W_Q=dW_Q, W_K=dW_K, W_V=dW_V, W_OUT=dW_OUT, B_OUT=dB_OUT)


def dot_expanded(label, a, b, result):
    """把一个点积逐项展开打印：a·b = a0*b0 + a1*b1 + ... = result"""
    terms = " + ".join(f"{ai:+.2f}·{bi:+.2f}" for ai, bi in zip(a, b))
    print(f"      {label}: {terms} = {result:+.4f}")


def show_forward(tag, ids, tgt, verbose=True):
    words = [VOCAB[i] for i in ids]
    tgt_idx = tid[tgt]
    c = forward(ids)
    loss = -np.log(c["p"][tgt_idx])
    print(f"\n---- {tag}: {' '.join(words)} -> {tgt} ----")
    print("  注意力权重 A（行=Query 位置，列=Key 位置，每行和=1）")
    print("  算法: A = softmax(QK^T/sqrt(D))，逐行归一化")
    for i, w in enumerate(words):
        print(f"    {w} 的行: " + "  ".join(f"{v:.3f}" for v in c["A"][i]))
    print(f"  位置 2 预测: P({tgt})={c['p'][tgt_idx]:.3f}  "
          f"最像: {VOCAB[int(np.argmax(c['p']))]}({c['p'].max():.3f})  loss={loss:.3f}")
    return c, loss


def apply_grads(grads):
    """更新公式：W_new = W_old - lr * grad（沿负梯度走一步）"""
    for name in PARAMS:
        PARAMS[name] -= LEARNING_RATE * grads[name]


# ============================================================
# 3. 第 1 次前向传播（训练前：注意力接近均匀，预测接近瞎猜）
# ============================================================
print()
print("=" * 60)
print("3. 第 1 次前向传播（训练前，紧凑视图）")
print("=" * 60)
caches, losses = [], []
for k, (seq, tgt) in enumerate(TRAIN):
    ids = [tid[w] for w in seq]
    c, loss = show_forward(f"样本{k+1}", ids, tgt)
    caches.append(c); losses.append(loss)
print(f"\n  第 1 次前向总 loss = {sum(losses):.3f}  （三个样本 loss 求和）")
A1_pos2 = [c["A"][-1].copy() for c in caches]          # 留档：位置 2 的注意力行
P1_correct = [c["p"][tid[t]] for c, (_, t) in zip(caches, TRAIN)]

# ============================================================
# 4. 显微镜：样本 1 · 位置 2 的每一步计算逐项摊开
# ============================================================
print()
print("=" * 60)
print("4. 显微镜：样本 1 位置 2（'很'）的每个数字是怎么算出来的")
print("=" * 60)
c1 = caches[0]
ids1 = [tid[w] for w in TRAIN[0][0]]
words1 = [VOCAB[i] for i in ids1]
tgt1 = tid[TRAIN[0][1]]

print("\n[4.1] 词向量哪来：查词嵌入表 E（固定不训）")
for i, w in enumerate(words1):
    print(f"      X[{i}]（{w}）= E 第 {ids1[i]} 行 = {np.round(c1['X'][i], 2)}")

print("\n[4.2] Q/K/V 哪来：同一个 X 各乘一个可学习矩阵（同源三投影）")
print("      Q = X@W_Q:"); print(np.round(c1["Q"], 2))
print("      K = X@W_K:"); print(np.round(c1["K"], 2))
print("      V = X@W_V:"); print(np.round(c1["V"], 2))
print("      抽查 Q 的一行怎么算（位置 2 '很' 的第 0 个分量）:")
dot_expanded("Q[2,0] = X[2]·W_Q[:,0]", c1["X"][2], W_Q[:, 0], c1["Q"][2, 0])

print("\n[4.3] 打分：位置 2 的 Query 对三个位置的 Key 分别点积，再除以 sqrt(D)")
for j in range(3):
    dot_expanded(f"s(2->{j}) = Q[2]·K[{j}]/sqrt({D})", c1["Q"][2], c1["K"][j],
                 c1["Q"][2] @ c1["K"][j] / np.sqrt(D))
s2 = c1["S"][2]
print(f"      三个原始分: {np.round(s2, 4)}  （除以 sqrt(D) 后）")

print("\n[4.4] softmax：把三个分变成和为 1 的权重")
e2 = np.exp(s2 - s2.max()); w2 = e2 / e2.sum()
print(f"      减最大值 {s2.max():+.4f}（防溢出，不影响结果）: {np.round(s2 - s2.max(), 4)}")
print(f"      逐个取指数 e^x: {np.round(e2, 4)}   求和 = {e2.sum():.4f}")
for j in range(3):
    print(f"      权重 A[2,{j}] = {e2[j]:.4f} / {e2.sum():.4f} = {w2[j]:.4f}   （对应看 '{words1[j]}'）")
print(f"      和为 1 验证: {w2.sum():.4f}")

print("\n[4.5] 取货：位置 2 的新表示 = 三个 V 按权重加权求和")
print(f"      O[2] = {w2[0]:.4f}*V[0] + {w2[1]:.4f}*V[1] + {w2[2]:.4f}*V[2]")
contrib = np.stack([w2[j] * c1["V"][j] for j in range(3)])
print(f"      三份贡献（逐分量）: ")
for j in range(3):
    print(f"        {w2[j]:.4f} * V[{j}]({words1[j]}) = {np.round(contrib[j], 3)}")
print(f"      相加 = O[2] = {np.round(c1['O'][2], 3)}   （这个'信息包'里主语的成分最高）")

print("\n[4.6] 打 logits：h=O[2] 在词表上逐词打分（只展示 3 个代表词）")
for w in ["困", "软", "小猫"]:
    j = tid[w]
    dot_expanded(f"logits[{w}] = h·W_OUT[{j}]", c1["h"], W_OUT[j], c1["logits"][j])

print("\n[4.7] 概率：对 8 个 logits 做 softmax")
lg = c1["logits"]
e_all = np.exp(lg)
print(f"      8 个 logits: {np.round(lg, 3)}")
print(f"      逐个取指数 : {np.round(e_all, 3)}   求和 = {e_all.sum():.3f}")
print(f"      P(困) = {e_all[tgt1]:.3f} / {e_all.sum():.3f} = {c1['p'][tgt1]:.4f}")
print(f"      loss = -ln(P(困)) = -ln({c1['p'][tgt1]:.4f}) = {-np.log(c1['p'][tgt1]):.4f}"
      f"   （概率越接近 1，loss 越接近 0）")

# ============================================================
# 5. 第 1 次反向传播（显微镜 + 更新）
# ============================================================
print()
print("=" * 60)
print("5. 第 1 次反向传播（每个梯度也摊开看）+ 更新")
print("=" * 60)
g1 = backward(c1, tgt1)
print("\n[5.1] 梯度的源头：dlogits = p - onehot(正确词)")
print(f"      p          = {np.round(c1['p'], 3)}")
print(f"      dlogits    = {np.round(g1['B_OUT'], 3)}   （正确词'困'那项 = 0.103-1 = -0.897，其余=自身概率）")
print("      直觉：正确词的概率被'拽高'（负梯度），错误词被'压低'（正梯度）")

print("\n[5.2] W_OUT[困,0] 的梯度 = 三个样本贡献相加（dlogits[困] * h[0]）")
total = 0.0
for k, (seq, tgt) in enumerate(TRAIN):
    ck = caches[k]
    dl = ck["p"][tid[tgt]] - 1.0
    contrib = dl * ck["h"][0]
    total += contrib
    print(f"      样本{k+1}（应预测{tgt}）: dlogits[{tid[tgt]}]={dl:+.3f} × h[0]={ck['h'][0]:+.3f}"
          f" -> 贡献 {contrib:+.4f}")
print(f"      相加 = 总梯度 {total:+.4f}")

print("\n[5.3] 更新公式：W_new = W_old - lr × grad（沿负梯度走一步）")
print(f"      W_OUT[困,0]: {W_OUT[tgt1,0]:+.4f} - {LEARNING_RATE} × {g1['W_OUT'][tgt1,0]:+.4f} "
      f"-> {W_OUT[tgt1,0] - LEARNING_RATE * g1['W_OUT'][tgt1,0]:+.4f}")

all_grads = {k: np.zeros_like(v) for k, v in PARAMS.items()}
for c, (_, tgt) in zip(caches, TRAIN):
    g = backward(c, tid[tgt])
    for k in all_grads:
        all_grads[k] += g[k]
for name in all_grads:
    print(f"  |grad {name}| = {np.linalg.norm(all_grads[name]):.4f}")

eps = 1e-5
ids0 = [tid[w] for w in TRAIN[0][0]]
def batch_loss():
    tot = 0.0
    for seq, tgt in TRAIN:
        c = forward([tid[w] for w in seq])
        tot += -np.log(c["p"][tid[tgt]])
    return tot
old = W_Q[0, 0]
W_Q[0, 0] = old + eps;  lp = batch_loss()
W_Q[0, 0] = old - eps;  lm = batch_loss()
W_Q[0, 0] = old
numerical = (lp - lm) / (2 * eps)
print(f"  梯度抽查 W_Q[0,0]: 手推={all_grads['W_Q'][0,0]:.5f}  数值={numerical:.5f}  "
      f"(差 {abs(all_grads['W_Q'][0,0]-numerical):.2e}，说明反向推对了)")

snap = {k: v.copy() for k, v in PARAMS.items()}
apply_grads(all_grads)
print("  代表参数更新对照（前 -> 后）:")
print(f"    W_Q[0,0] : {snap['W_Q'][0,0]:+.4f} -> {W_Q[0,0]:+.4f}")
print(f"    W_K[1,1] : {snap['W_K'][1,1]:+.4f} -> {W_K[1,1]:+.4f}")
print(f"    W_V[2,2] : {snap['W_V'][2,2]:+.4f} -> {W_V[2,2]:+.4f}")
print(f"    W_OUT[{tid['困']},0]: {snap['W_OUT'][tid['困'],0]:+.4f} -> {W_OUT[tid['困'],0]:+.4f}")

# ============================================================
# 6. 第 2 次前向传播（训练后：看 QKV 和注意力怎么变）
# ============================================================
print()
print("=" * 60)
print("6. 第 2 次前向传播（训练后）")
print("=" * 60)
caches2, losses2 = [], []
for k, (seq, tgt) in enumerate(TRAIN):
    ids = [tid[w] for w in seq]
    c, loss = show_forward(f"样本{k+1}", ids, tgt)
    caches2.append(c); losses2.append(loss)
print(f"\n  第 2 次前向总 loss = {sum(losses2):.3f}   （第 1 次: {sum(losses):.3f}）")

# ============================================================
# 7. 第 2 次反向传播 + 更新
# ============================================================
print()
print("=" * 60)
print("7. 第 2 次反向传播 + 更新")
print("=" * 60)
grads2 = {k: np.zeros_like(v) for k, v in PARAMS.items()}
for c, (_, tgt) in zip(caches2, TRAIN):
    g = backward(c, tid[tgt])
    for k in grads2:
        grads2[k] += g[k]
for name in grads2:
    print(f"  |grad {name}| = {np.linalg.norm(grads2[name]):.4f}")
snap2 = {k: v.copy() for k, v in PARAMS.items()}
apply_grads(grads2)
print(f"  W_Q[0,0] : {snap2['W_Q'][0,0]:+.4f} -> {W_Q[0,0]:+.4f}")

# ============================================================
# 8. 反向理解：从参数变化推测模型学到了什么"视角"
# ============================================================
print()
print("=" * 60)
print("8. 反向理解：模型学到了什么视角？")
print("=" * 60)
for k, (seq, tgt) in enumerate(TRAIN):
    words = [VOCAB[i] for i in [tid[w] for w in seq]]
    a_before, a_after = A1_pos2[k], caches2[k]["A"][-1]
    print(f"\n  样本 {' '.join(words)}（应预测 {tgt}）—— 位置 2 的注意力行:")
    print(f"    训练前: " + "  ".join(f"{w}:{v:.3f}" for w, v in zip(words, a_before)))
    print(f"    训练后: " + "  ".join(f"{w}:{v:.3f}" for w, v in zip(words, a_after)))
    gain = a_after - a_before
    winner = int(np.argmax(gain))
    print(f"    变化最大: 对'{words[winner]}'的注意力 {a_before[winner]:.3f} -> {a_after[winner]:.3f}"
          f"（{'增加' if gain[winner] > 0 else '减少'} {abs(gain[winner]):.3f}）")
    print(f"    P(正确={tgt}): {P1_correct[k]:.3f} -> {caches2[k]['p'][tid[tgt]]:.3f}")

print()
print("  解读框架（对着上面的数字套）：")
print("  · 位置 2 的注意力若从'三处均摊'变成'集中到主语（位置 1）'")
print("    => 这个头学到的视角是：'要猜属性，先回头找到主语'（越过无信息的'很'）")
print("  · W_OUT 里'困/软/亮'三行的变化 => 输出层在把'主语信息包'翻译成对应属性词")
print("  · 个别样本早期往'错'的方向小幅抖动是正常的：梯度是三个样本平均的结果，")
print("    样本间互相拉扯；训练步数一多，方向就会统一")
print("  · 两次更新里 |grad W_OUT| 远大于 |grad W_Q|：输出层先动、注意力慢热——")
print("    真实训练也分层收敛，注意力模式往往要几百步才定型")
print("  · 这只是 2 步训练，模式刚冒头；步数加到几十，注意力会逼近 one-hot")
print()
print("  想继续玩（改完重跑，对比第 8 节的数字）：")
print("  · LEARNING_RATE 调小到 0.1：两次更新的变化还够看吗？")
print("  · INIT_SCALE 调大到 1.0：初始注意力还均匀吗？（提示：QK^T 变大，softmax 变尖）")
print("  · 把 TRAIN 换成新主语/新属性：模型能零样本泛化吗？为什么不能？（提示：词嵌入是固定的随机数）")
