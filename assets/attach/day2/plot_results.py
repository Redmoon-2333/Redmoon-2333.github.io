# -*- coding: utf-8 -*-
"""生成 Day2 笔记第 11 节的三张实验成果图
====================================================================
运行：python plot_results.py
训练与 generate_walkthrough.py 完全一致（种子 7，300 轮，lr=0.5 平均 loss），
图输出到 08_MyNote/Day2/assets/img/：
  exp-loss-curve.png        loss 与正确词概率曲线
  exp-attention-shift.png   位置 2 注意力：训练前 vs 训练后（3 样本）
  exp-alignment-matrix.png  输出层 3x3 "主语->属性"对齐矩阵
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = r"D:\大学相关\03_个人成长与记录\LLM学习体系\08_MyNote\Day2\assets\img"
os.makedirs(OUT, exist_ok=True)

# ---------------- 模型（与 generate_walkthrough.py 一致） ----------------
VOCAB = ["<pad>", "小猫", "垫子", "月亮", "很", "困", "软", "亮"]
tid = {w: i for i, w in enumerate(VOCAB)}
D, LR, INIT_SCALE, N_PASSES = 8, 0.5, 0.3, 300
TRAIN = [(["<pad>", "小猫", "很"], "困"),
         (["<pad>", "垫子", "很"], "软"),
         (["<pad>", "月亮", "很"], "亮")]
PROPS = [t for _, t in TRAIN]
SUBJ = ["小猫", "垫子", "月亮"]

rng = np.random.default_rng(7)
E = rng.normal(0, 1, (len(VOCAB), D))
W_Q = rng.uniform(-INIT_SCALE, INIT_SCALE, (D, D))
W_K = rng.uniform(-INIT_SCALE, INIT_SCALE, (D, D))
W_V = rng.uniform(-INIT_SCALE, INIT_SCALE, (D, D))
W_OUT = rng.uniform(-INIT_SCALE, INIT_SCALE, (len(VOCAB), D))
B_OUT = np.zeros(len(VOCAB))


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
    dlogits = p.copy(); dlogits[tgt_idx] -= 1.0
    dW_OUT = np.outer(dlogits, h); dB_OUT = dlogits
    dh = W_OUT.T @ dlogits
    dO = np.zeros_like(V); dO[-1] = dh
    dA = dO @ V.T; dV = A.T @ dO
    dS = A * (dA - (dA * A).sum(axis=1, keepdims=True))
    dQ = (dS @ K) / np.sqrt(D); dK = (dS.T @ Q) / np.sqrt(D)
    return dict(W_Q=X.T @ dQ, W_K=X.T @ dK, W_V=X.T @ dV, W_OUT=dW_OUT, B_OUT=dB_OUT)


ids_all = [[tid[w] for w in seq] for seq, _ in TRAIN]
tgt_all = [tid[t] for _, t in TRAIN]

# ---------------- 训练 300 轮，记录曲线与前后快照 ----------------
curve_loss, curve_ps = [], []
A_pos2_before = None
grad_norms = {"W_OUT": [], "W_Q": []}
for p in range(1, N_PASSES + 1):
    caches, total = [], 0.0
    grads = {k: np.zeros_like(v) for k, v in
             dict(W_Q=W_Q, W_K=W_K, W_V=W_V, W_OUT=W_OUT, B_OUT=B_OUT).items()}
    for k, ids in enumerate(ids_all):
        c = forward(ids)
        caches.append(c)
        total += -np.log(max(c["p"][tgt_all[k]], 1e-12))
        g = backward(c, tgt_all[k])
        for key in grads:
            grads[key] += g[key]
    for key in grads:
        grads[key] /= len(TRAIN)
    if p == 1:
        A_pos2_before = [c["A"][-1].copy() for c in caches]
        P_before = [c["p"][t] for c, t in zip(caches, tgt_all)]
    grad_norms["W_OUT"].append(np.linalg.norm(grads["W_OUT"]))
    grad_norms["W_Q"].append(np.linalg.norm(grads["W_Q"]))
    for name, pm in dict(W_Q=W_Q, W_K=W_K, W_V=W_V, W_OUT=W_OUT, B_OUT=B_OUT).items():
        pm -= LR * grads[name]
    curve_loss.append(total)
    curve_ps.append([c["p"][t] for c, t in zip(caches, tgt_all)])

final_caches = [forward(ids) for ids in ids_all]
A_pos2_after = [c["A"][-1] for c in final_caches]
P_after = [c["p"][t] for c, t in zip(final_caches, tgt_all)]
print(f"final loss={curve_loss[-1]:.4f}  P={np.round(P_after, 4)}")

# ---------------- 图 1：loss 与概率曲线 ----------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(curve_loss, color="#b31b1b", lw=2, label="总 loss（越低越好）")
ax.set_xlabel("训练轮次"); ax.set_ylabel("loss", color="#b31b1b")
ax.tick_params(axis="y", labelcolor="#b31b1b")
ax2 = ax.twinx()
for i, pr in enumerate(PROPS):
    ax2.plot([row[i] for row in curve_ps], lw=1.8, label=f"P(正确={pr})")
ax2.set_ylabel("正确词概率"); ax2.set_ylim(0, 1.05)
ax.set_title("300 轮训练：loss 下降与正确词概率上升")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="center right")
ax.grid(alpha=0.25)
fig.savefig(os.path.join(OUT, "exp-loss-curve.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- 图 2：位置 2 注意力 训练前 vs 训练后 ----------------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
labels = ["<pad>", "主语", "很"]
x = np.arange(3)
for k, ax in enumerate(axes):
    b, a = A_pos2_before[k], A_pos2_after[k]
    ax.bar(x - 0.18, b, 0.36, label="训练前", color="#b8c4d9")
    ax.bar(x + 0.18, a, 0.36, label="训练后", color="#b31b1b")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"样本{k+1}：应预测「{PROPS[k]}」", fontsize=11)
    for xi, (vb, va) in enumerate(zip(b, a)):
        ax.text(xi - 0.18, vb + 0.02, f"{vb:.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.18, va + 0.02, f"{va:.2f}", ha="center", fontsize=8, color="#b31b1b")
    ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("注意力权重"); axes[0].legend(fontsize=9)
fig.suptitle("位置 2（\"很\"）的注意力：训练前均摊 → 训练后盯住主语（个别样本走捷径）", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "exp-attention-shift.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- 图 3：3x3 主语->属性 对齐矩阵 ----------------
M = np.zeros((3, 3))
for k in range(3):
    h = final_caches[k]["h"]
    for j, pr in enumerate(PROPS):
        M[k, j] = h @ W_OUT[tid[pr]]
fig, ax = plt.subplots(figsize=(7.2, 5.2))
im = ax.imshow(M, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels([f"W_OUT[{p}]" for p in PROPS], fontsize=10)
ax.set_yticks(range(3)); ax.set_yticklabels([f"来自{s}" for s in SUBJ], fontsize=10)
ax.set_xlabel("输出层的属性打分行", fontsize=10)
ax.set_ylabel("位置 2 信息包的来源（主语）", fontsize=10)
ax.set_title("训练后的 3×3 对齐矩阵：对角线 = 正确的\"主语→属性\"翻译", fontsize=12)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=11,
                color="white" if M[i, j] > M.max() * 0.6 else "#333")
fig.colorbar(im, ax=ax, shrink=0.85)
fig.savefig(os.path.join(OUT, "exp-alignment-matrix.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("figures written:")
for n in ["exp-loss-curve.png", "exp-attention-shift.png", "exp-alignment-matrix.png"]:
    print("  ", os.path.join(OUT, n))
print(f"attention before: {[np.round(a, 3).tolist() for a in A_pos2_before]}")
print(f"attention after : {[np.round(a, 3).tolist() for a in A_pos2_after]}")
print(f"P before: {np.round(P_before, 3).tolist()}  after: {np.round(P_after, 4).tolist()}")
print(f"loss: first={curve_loss[0]:.3f} @10={curve_loss[9]:.3f} @50={curve_loss[49]:.3f} final={curve_loss[-1]:.4f}")


# ---------------- 图 4：前向/反向传播路径示意图（双车道版） ----------------
def fig_forward_backward_path():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(15.5, 7.6))
    ax.set_xlim(0, 17.5); ax.set_ylim(0, 8.6); ax.axis("off")

    def box(cx, cy, w, h, text, fc, ec="#333", fs=10, lw=1.5, ls="-", tc="#333"):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.03,rounding_size=0.1",
                                    fc=fc, ec=ec, lw=lw, ls=ls))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2, color, lw=2, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls))

    FY = 6.6         # 前向行
    GY = 3.6         # 梯度行
    DW = 1.5         # dW 行

    # ---- 上排：前向链（蓝） ----
    fb = "#dbe7f7"; fe = "#2a6fb3"
    box(1.1, FY, 1.5, 0.85, "词 id", "#f8cecc", fs=10.5)
    box(3.2, FY, 2.1, 0.85, "X = E[词id]\n查表", "#f8cecc", fs=9.5)
    box(6.0, FY, 3.0, 0.95, "Q, K, V = X·(W_Q, W_K, W_V)", fb, fe, fs=9.5)
    box(9.0, FY, 2.0, 0.85, "S = QK^T/√d", fb, fe, fs=10)
    box(11.4, FY, 2.3, 0.85, "A = softmax(S)", fb, fe, fs=10)
    box(13.5, FY, 1.6, 0.85, "O = A·V", fb, fe, fs=10)
    box(15.2, FY, 1.6, 0.85, "h = O[末位]", fb, fe, fs=9.5)
    box(16.6, 7.7, 1.7, 0.8, "loss\n=-log P", "#f4cccc", fs=9.5)

    arrow(1.85, FY, 2.15, FY, "#2a6fb3")
    arrow(4.25, FY, 4.5, FY, "#2a6fb3")
    arrow(7.5, FY, 8.0, FY, "#2a6fb3")
    arrow(10.0, FY, 10.25, FY, "#2a6fb3")
    arrow(12.55, FY, 12.7, FY, "#2a6fb3")
    arrow(14.3, FY, 14.4, FY, "#2a6fb3")
    arrow(15.9, FY + 0.43, 16.3, 7.35, "#2a6fb3")

    # ---- 中排：参数（橙） ----
    box(3.2, 4.9, 2.1, 0.75, "E 词嵌入（固定，不训练）", "#e8e8e8", ec="#777", fs=9)
    box(5.2, 4.9, 1.1, 0.7, "W_Q", "#ffe6cc", fs=9.5)
    box(6.4, 4.9, 1.1, 0.7, "W_K", "#ffe6cc", fs=9.5)
    box(7.6, 4.9, 1.1, 0.7, "W_V", "#ffe6cc", fs=9.5)
    box(15.2, 4.9, 1.7, 0.75, "W_OUT", "#ffe6cc", fs=10)
    arrow(3.2, 5.28, 3.2, FY - 0.45, "#2a6fb3", 1.2)
    arrow(5.2, 5.26, 5.5, FY - 0.5, "#2a6fb3", 1.2)
    arrow(6.4, 5.26, 6.0, FY - 0.5, "#2a6fb3", 1.2)
    arrow(7.6, 5.26, 6.6, FY - 0.5, "#2a6fb3", 1.2)
    arrow(15.2, 5.28, 15.2, FY - 0.45, "#2a6fb3", 1.2)

    # ---- 下排：梯度链（红，右->左） ----
    gb = "#f9dcd8"; ge = "#b31b1b"
    box(16.6, GY, 1.7, 0.75, "dlogits\n= p − onehot", gb, ge, fs=8.5, tc="#b31b1b")
    box(14.6, GY, 1.4, 0.75, "dh", gb, ge, fs=10, tc="#b31b1b")
    box(13.0, GY, 1.4, 0.75, "dO", gb, ge, fs=10, tc="#b31b1b")
    box(11.2, GY, 1.6, 0.75, "dA", gb, ge, fs=10, tc="#b31b1b")
    box(8.9, GY, 1.7, 0.75, "dS", gb, ge, fs=10, tc="#b31b1b")
    box(5.9, GY, 2.8, 0.75, "dQ, dK, dV", gb, ge, fs=10, tc="#b31b1b")
    box(16.6, DW, 1.7, 0.7, "dW_OUT ★", gb, ge, fs=9, tc="#b31b1b")
    box(5.9, DW, 2.8, 0.7, "dW_Q ★  dW_K ★  dW_V ★", gb, ge, fs=8.5, tc="#b31b1b")

    # 反向链条（红实线，右->左）
    arrow(17.2, 7.28, 17.2, GY + 0.42, "#b31b1b", 1.8)        # loss -> dlogits（绕开 W_OUT）
    ax.text(16.4, 5.75, "dlogits = p − onehot\n（loss 反向起点）", fontsize=8.5,
            color="#b31b1b", ha="center")
    arrow(15.75, GY, 15.3, GY, "#b31b1b", 1.8)                # dlogits -> dh
    arrow(13.9, GY, 13.7, GY, "#b31b1b", 1.8)                 # dh -> dO
    arrow(12.3, GY, 12.0, GY, "#b31b1b", 1.8)                 # dO -> dA
    arrow(10.4, GY, 9.75, GY, "#b31b1b", 1.8)                 # dA -> dS
    arrow(8.05, GY, 7.3, GY, "#b31b1b", 1.8)                  # dS -> dQKV
    arrow(5.9, GY - 0.38, 5.9, DW + 0.37, "#b31b1b", 1.8)     # dQKV -> dW
    arrow(16.6, GY - 0.38, 16.6, DW + 0.37, "#b31b1b", 1.8)   # dlogits -> dW_OUT

    # 对应关系（灰虚线：前向节点 与 它的梯度）
    for xf, xg in [(15.2, 14.6), (13.5, 13.0), (11.4, 11.2), (9.0, 8.9), (6.0, 6.0)]:
        ax.plot([xf, xg], [FY - 0.45, GY + 0.4], color="#999", lw=0.9, ls=":")

    # 箭头含义小标签（放在箭头上方的空白区）
    ax.text(15.5, GY + 0.55, "×W_OUT^T", fontsize=8, color="#b31b1b", ha="center")
    ax.text(13.45, GY - 0.62, "取末位", fontsize=8, color="#b31b1b", ha="center")
    ax.text(11.75, GY + 0.55, "·V^T", fontsize=8, color="#b31b1b", ha="center")
    ax.text(10.05, GY + 0.55, "softmax 反向", fontsize=8, color="#b31b1b", ha="center")
    ax.text(7.65, GY + 0.55, "·K / ·Q / ·A^T", fontsize=8, color="#b31b1b", ha="center")

    # ---- 图例 ----
    ax.plot([1.0, 2.3], [0.55, 0.55], color="#2a6fb3", lw=2.2)
    ax.text(2.5, 0.55, "前向传播（左→右）", fontsize=10.5, va="center")
    ax.plot([5.6, 6.9], [0.55, 0.55], color="#b31b1b", lw=2)
    ax.text(7.1, 0.55, "反向传播（右→左，红盒 = 梯度）", fontsize=10.5, va="center")
    ax.plot([11.3, 12.3], [0.55, 0.55], color="#999", lw=1, ls=":")
    ax.text(12.5, 0.55, "灰虚线 = 前向节点 与 它对应的梯度", fontsize=10.5, va="center")
    ax.text(1.0, 7.9, "上排 = 前向（蓝，左→右算出 loss）；下排 = 反向（红，右→左把梯度送回每个参数）",
            fontsize=11, color="#555")
    ax.set_title("一次前向 + 一次反向的完整路径（玩具注意力模型）", fontsize=13.5)
    fig.savefig(os.path.join(OUT, "exp-forward-backward-path.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


fig_forward_backward_path()
print("  ", os.path.join(OUT, "exp-forward-backward-path.png"))
