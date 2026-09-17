# -*- coding: utf-8 -*-
"""Day11 从 Bigram 到 GPT — matplotlib 技术示意图

一键产出全部 6 张 SVG（从仓库根目录运行）：
  KMP_DUPLICATE_LIB_OK=TRUE python 08_MyNote/Day11/assets/make_figures.py
输出：08_MyNote/Day11/assets/img/*.svg
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = Path("08_MyNote/Day11/assets/img")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ACCENT = "#B31B1B"   # arXiv 红（v2.py 部分）
NOTEBOOK = "#1B5E20"  # 绿（notebook 部分）
BLUE = "#1565C0"
GRAY = "#666666"


def _save(fig, name):
    fp = OUT / name
    fig.savefig(fp, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {fp} {fp.stat().st_size//1024}KB")


def _box(ax, x, y, w, h, text, ec, fc="white", fs=8, lw=1.4, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#222", linespacing=1.45, weight=weight)


def _arrow(ax, p1, p2, color=GRAY, lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                                 lw=lw, shrinkA=1, shrinkB=1, mutation_scale=11))


# ================= 1. 学习路线图 =================

def fig_roadmap():
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 4.8); ax.axis("off")

    stages = [
        ("文本加载\n1,115,394 字符", NOTEBOOK),
        ("字符分词\n65 字符词表", NOTEBOOK),
        ("滑动窗口\n(B, T) = (4, 8)", NOTEBOOK),
        ("Bigram 查表\nloss 4.88 → 3.70", NOTEBOOK),
        ("玩具注意力\n累积平均三版本", NOTEBOOK),
        ("单头自注意力\nQ / K / V + 掩码", NOTEBOOK),
        ("多头 + FFN + Block\n残差 + Pre-LN", ACCENT),
        ("完整 GPT\n6 层 ≈10.8M", ACCENT),
    ]
    w, gap = 1.32, 0.22
    for i, (txt, col) in enumerate(stages):
        x = 0.18 + i * (w + gap)
        _box(ax, x, 2.15, w, 1.35, txt, col, fs=7.2, lw=1.6)
        if i < len(stages) - 1:
            _arrow(ax, (x + w + 0.01, 2.82), (x + w + gap - 0.01, 2.82))

    # 括号标注
    x_nb0, x_nb1 = 0.18, 0.18 + 5 * (w + gap) + w
    x_py0, x_py1 = 0.18 + 6 * (w + gap), 0.18 + 7 * (w + gap) + w
    ax.plot([x_nb0, x_nb0, x_nb1, x_nb1], [1.75, 1.45, 1.45, 1.75], color=NOTEBOOK, lw=1.6)
    ax.text((x_nb0 + x_nb1) / 2, 1.05, "gpt-dev.ipynb（跟敲进行时：1–6 步已完成）",
            ha="center", fontsize=9, color=NOTEBOOK, weight="bold")
    ax.plot([x_py0, x_py0, x_py1, x_py1], [1.75, 1.45, 1.45, 1.75], color=ACCENT, lw=1.6)
    ax.text((x_py0 + x_py1) / 2, 1.05, "v2.py（完成版：7–8 步补齐）",
            ha="center", fontsize=9, color=ACCENT, weight="bold")

    ax.text(6.25, 4.25, "从 Bigram 到完整 GPT：一趟「零件 → 组装」的旅程",
            ha="center", fontsize=12, weight="bold", color="#222")
    ax.text(6.25, 0.35, "每步只加一个新零件：查表 → 加权平均 → 数据依赖的权重（注意力）→ 并行多头 → 组装成层",
            ha="center", fontsize=8.5, color=GRAY)
    _save(fig, "day11-roadmap.svg")


# ================= 2. 分词 + 滑动窗口 =================

def fig_tokenize_window():
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4))
    fig.subplots_adjust(hspace=0.55)

    # 上：字符 → 编号
    ax = axes[0]
    ax.set_xlim(0, 11); ax.set_ylim(0, 2.6); ax.axis("off")
    chars = ["F", "i", "r", "s", "t", "空格", "C", "i"]
    ids = [18, 47, 56, 57, 58, 1, 15, 47]
    for j, (ch, ix) in enumerate(zip(chars, ids)):
        x = 1.3 + j * 1.05
        _box(ax, x, 1.35, 0.9, 0.7, ch, BLUE, fs=11, lw=1.2)
        _arrow(ax, (x + 0.45, 1.33), (x + 0.45, 1.05), color=GRAY, lw=1.0)
        _box(ax, x, 0.35, 0.9, 0.7, str(ix), GRAY, fs=10, lw=1.0)
    ax.text(0.15, 1.7, "① 字符级分词：", fontsize=10, weight="bold", color="#222")
    ax.text(0.15, 0.75, "文字 → 整数\n(encode)", fontsize=8, color=GRAY, ha="left", va="center")
    ax.set_title("「First Ci」→ [18, 47, 56, 57, 58, 1, 15, 47]：每个字符对应一个编号，65 个字符就是全部词表",
                 fontsize=9, color="#222", pad=6)

    # 下：x / y 错位
    ax = axes[1]
    ax.set_xlim(0, 11); ax.set_ylim(0, 2.7); ax.axis("off")
    x = [18, 47, 56, 57, 58, 1, 46, 43]
    y = [47, 56, 57, 58, 1, 46, 43, 39]
    for j in range(8):
        xx = 1.35 + j * 1.08
        _box(ax, xx, 1.55, 0.95, 0.62, str(x[j]), "#333", fc="#F5F5F5", fs=10)
        _box(ax, xx, 0.28, 0.95, 0.62, str(y[j]), ACCENT, fc="#FFF5F5", fs=10)
    for j in range(7):
        x_from = 1.35 + (j + 1) * 1.08 + 0.475   # 下一格的 x
        x_to = 1.35 + j * 1.08 + 0.475           # 本格的 y
        _arrow(ax, (x_from, 1.53), (x_to, 0.92), color=ACCENT, lw=0.85, style="-|>")
    ax.text(0.12, 2.0, "x（输入）", fontsize=9.5, weight="bold", color="#333")
    ax.text(0.12, 0.6, "y（目标）", fontsize=9.5, weight="bold", color=ACCENT)
    ax.text(9.85, 2.32, "y[i] = x[i+1]（箭头即右移对应）", fontsize=8.5, color=GRAY, ha="right")
    ax.set_title("② 滑动窗口：y 是 x 右移一位 —— 一个长度 T 的窗口里藏着 T 个「看前文猜下一个」的训练样本",
                 fontsize=9, color="#222", pad=6)
    _save(fig, "day11-tokenize-window.svg")


# ================= 3. 玩具注意力三版本 =================

def fig_attention_toy():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    T = 4
    tril = np.tril(np.ones((T, T)))
    wei2 = tril / tril.sum(1, keepdims=True)
    mask = np.where(tril == 1, 0.0, -np.inf)
    e = np.exp(mask - mask.max(axis=1, keepdims=True))
    wei3 = e / e.sum(axis=1, keepdims=True)

    titles = ["① 下三角掩码 tril", "② 行归一化 = 累积平均", "③ -∞ + softmax = 同一结果"]
    mats = [tril, wei2, wei3]
    for ax, title, m in zip(axes, titles, mats):
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        for i in range(T):
            for j in range(T):
                v = m[i, j]
                col = "#999" if v == 0 else "white" if v > 0.5 else "#222"
                txt = "0" if v == 0 else f"{v:.2f}".rstrip("0").rstrip(".")
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)
        ax.set_xticks(range(T)); ax.set_yticks(range(T))
        ax.set_title(title, fontsize=9.5, color="#222", pad=6)
        ax.tick_params(length=0, labelsize=7)
        for s in ax.spines.values():
            s.set_color("#ccc")

    fig.suptitle("注意力的前传：三个版本做同一件事（torch.allclose 全部为 True）", fontsize=11, weight="bold", y=1.04)
    fig.text(0.5, -0.06, "每个位置只能看「自己和左边」→ 权重归一化 → 加权平均 value；把固定权重换成可训练的 q·k.T 就是真正的注意力",
             ha="center", fontsize=8.5, color=GRAY)
    fig.tight_layout()
    _save(fig, "day11-attention-toy.svg")


# ================= 4. 单头自注意力数据流 =================

def fig_self_attention():
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 5.6); ax.axis("off")

    _box(ax, 0.15, 2.35, 1.15, 0.9, "x\n(B,T,C)", "#333", fs=9)

    _box(ax, 2.0, 4.05, 1.35, 0.62, "query = x Wq\n(B,T,hs)", BLUE, fs=7.8)
    _box(ax, 2.0, 2.55, 1.35, 0.62, "key = x Wk\n(B,T,hs)", BLUE, fs=7.8)
    _box(ax, 2.0, 1.05, 1.35, 0.62, "value = x Wv\n(B,T,hs)", ACCENT, fs=7.8)
    for y in (4.36, 2.86, 1.36):
        _arrow(ax, (1.32, 2.8), (1.98, y))

    _box(ax, 4.05, 3.1, 1.75, 1.05, "wei = q @ k.T / √hs\n(B,T,T)\n相似度分数", "#333", fs=7.8)
    _arrow(ax, (3.37, 4.36), (4.03, 4.0))
    _arrow(ax, (3.37, 2.86), (4.03, 3.35))

    _box(ax, 6.35, 3.3, 1.6, 0.85, "因果掩码\ntril==0 → -inf", ACCENT, fs=8)
    _arrow(ax, (5.82, 3.62), (6.33, 3.72))

    _box(ax, 8.5, 3.3, 1.35, 0.85, "softmax\n(dim=-1)", "#333", fs=8)
    _arrow(ax, (7.97, 3.72), (8.48, 3.72))

    _box(ax, 10.35, 1.75, 1.75, 1.05, "out = wei @ v\n(B,T,hs)\n加权聚合", BLUE, fs=7.8)
    _arrow(ax, (9.17, 3.28), (10.35, 2.75))
    _arrow(ax, (3.37, 1.36), (10.33, 2.0), color=ACCENT)

    ax.text(6.1, 5.15, "单头自注意力：权重不再固定，而是 q·k.T 算出来的「谁该关注谁」",
            ha="center", fontsize=12, weight="bold", color="#222")
    ax.text(6.1, 0.55, "除以 √hs 防止点积过大、softmax 过尖；掩码保证「只能看左边」；最后按权重把 value 加权求和",
            ha="center", fontsize=8.5, color=GRAY)
    _save(fig, "day11-self-attention.svg")


# ================= 5. 多头 + Block 结构 =================

def fig_multihead_block():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # 左：多头
    ax1.set_xlim(0, 6); ax1.set_ylim(0, 6); ax1.axis("off")
    _box(ax1, 0.15, 2.6, 1.0, 0.8, "x", "#333", fs=10)
    for i in range(3):
        _box(ax1, 1.85, 4.3 - i * 1.35, 1.5, 0.95, f"Head {i}\nq·k.T → softmax", BLUE, fs=7.2)
    _box(ax1, 1.85, 0.6, 1.5, 0.75, "…（共 6 个）", GRAY, fs=7)
    for y in (4.75, 3.4, 2.05, 1.0):
        _arrow(ax1, (1.17, 3.0), (1.83, y))
    _box(ax1, 3.75, 2.55, 1.15, 0.9, "concat\n(B,T,384)", "#333", fs=7.5)
    for y in (4.75, 3.4, 2.05):
        _arrow(ax1, (3.37, y), (3.73, 3.05))
    _box(ax1, 3.75, 0.75, 1.15, 0.75, "proj\n384→384", ACCENT, fs=7.5)
    _arrow(ax1, (4.32, 2.53), (4.32, 1.52))
    ax1.set_title("MultiHeadAttention：6 个头并行\n各自关注不同的关系，再混合回 384 维", fontsize=9.5, pad=8)

    # 右：Block（Pre-LN + 残差）
    ax2.set_xlim(0, 6.5); ax2.set_ylim(0, 6); ax2.axis("off")
    ax2.plot([0.3, 6.1], [3.0, 3.0], color="#333", lw=3, alpha=0.25, solid_capstyle="round")
    ax2.text(0.35, 3.18, "残差主干（x 一路直通）", fontsize=8, color=GRAY)
    _box(ax2, 1.05, 4.0, 1.6, 0.75, "LayerNorm\nln1", "#555", fs=7.5)
    _box(ax2, 1.05, 1.1, 1.6, 0.75, "LayerNorm\nln2", "#555", fs=7.5)
    _box(ax2, 2.95, 4.0, 1.7, 0.75, "多头注意力\n（通信）", BLUE, fs=7.5)
    _box(ax2, 2.95, 1.1, 1.7, 0.75, " FFWD\n（计算）", ACCENT, fs=7.5)
    _arrow(ax2, (2.67, 4.38), (2.93, 4.38))
    _arrow(ax2, (2.67, 1.48), (2.93, 1.48))
    for x0, y0 in ((2.0, 3.0), (3.95, 3.0)):
        _arrow(ax2, (x0, y0), (x0, 4.0 if y0 == 3.0 and x0 == 2.0 else 1.85))
    _arrow(ax2, (4.65, 4.38), (4.95, 3.1))
    _arrow(ax2, (4.65, 1.48), (4.95, 2.9))
    for cx in (5.1,):
        ax2.add_patch(plt.Circle((cx, 3.0), 0.16, fc="white", ec="#333", lw=1.4, zorder=5))
        ax2.text(cx, 3.0, "+", ha="center", va="center", fontsize=11, zorder=6)
    ax2.text(3.7, 5.35, "Block = 通信 + 计算，各带残差", ha="center", fontsize=9.5, weight="bold")
    ax2.text(3.7, 0.35, "x = x + sa(ln1(x)) ;  x = x + ffwd(ln2(x))", ha="center", fontsize=9, color=GRAY, family="monospace")
    fig.suptitle("v2.py 的两个核心零件：多头注意力 & Transformer Block（Pre-LN）", fontsize=11.5, weight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "day11-multihead-block.svg")


# ================= 6. 完整 GPT 架构 =================

def fig_gpt_full():
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 9.4); ax.axis("off")

    _box(ax, 3.2, 8.55, 2.4, 0.65, "idx（B,T）字符编号", "#333", fs=9)
    _box(ax, 1.55, 7.35, 2.3, 0.8, "Token Embedding\n65×384", BLUE, fs=8, fc="#F5F9FF")
    _box(ax, 4.95, 7.35, 2.3, 0.8, "Position Embedding\n256×384", BLUE, fs=8, fc="#F5F9FF")
    _arrow(ax, (4.1, 8.53), (2.9, 8.18))
    _arrow(ax, (4.7, 8.53), (5.95, 8.18))
    ax.add_patch(plt.Circle((4.4, 6.85), 0.17, fc="white", ec="#333", lw=1.4, zorder=5))
    ax.text(4.4, 6.85, "+", ha="center", va="center", fontsize=11, zorder=6)
    _arrow(ax, (2.7, 7.33), (4.25, 6.95))
    _arrow(ax, (6.1, 7.33), (4.55, 6.95))

    _box(ax, 2.3, 4.65, 4.2, 1.8,
         "Transformer Block × 6 层\n（多头注意力 + FFN，残差 + Pre-LN）\n每层参数 ≈1.77M",
         ACCENT, fs=9.5, lw=1.8, fc="#FFF8F8")
    _arrow(ax, (4.4, 6.65), (4.4, 6.47))

    _box(ax, 3.3, 3.75, 2.2, 0.6, "LayerNorm（ln_f）", "#555", fs=8.5)
    _arrow(ax, (4.4, 4.63), (4.4, 4.37))
    _box(ax, 3.05, 2.75, 2.7, 0.65, "lm_head：Linear 384 → 65", BLUE, fs=8.5)
    _arrow(ax, (4.4, 3.73), (4.4, 3.42))
    _box(ax, 3.55, 1.75, 1.7, 0.62, "logits（B,T,65）", "#333", fs=8.5)
    _arrow(ax, (4.4, 2.73), (4.4, 2.39))
    _box(ax, 3.35, 0.7, 2.1, 0.62, "softmax → 采样", NOTEBOOK, fs=8.5)
    _arrow(ax, (4.4, 1.73), (4.4, 1.34))
    ax.text(4.4, 0.28, "拼回序列 → 循环（generate）", ha="center", fontsize=8.5, color=GRAY)

    # 右侧参数账
    ax.text(7.3, 7.15, "超参数（v2.py）", fontsize=10, weight="bold", color="#222")
    cfg = ("batch_size = 64\nblock_size = 256\nn_embd = 384 · n_head = 6\n"
           "n_layer = 6 · dropout = 0.2\nlr = 3e-4 · max_iters = 5000")
    _box(ax, 7.05, 5.0, 3.15, 2.0, cfg, GRAY, fs=8, lw=1.1, fc="#FAFAFA")
    ax.text(7.3, 4.7, "参数账（实测）", fontsize=10, weight="bold", color="#222")
    params = ("Block × 6       10.64 M\n嵌入层           0.12 M\nln_f + lm_head   0.03 M\n"
              "─────────────────\n合计           10.79 M")
    _box(ax, 7.05, 2.75, 3.15, 1.7, params, ACCENT, fs=8, lw=1.1, fc="#FFF8F8")
    ax.text(8.62, 2.1, "对比：GPT-2 small = 124M\n（12 层 × 768 维）", ha="center", fontsize=8, color=GRAY)

    ax.text(4.4, 9.15, "完整 GPT：字符编号 → 词义+位置 → 6 层通信与计算 → 下一个字符的概率分布",
            ha="center", fontsize=11.5, weight="bold", color="#222")
    _save(fig, "day11-gpt-arch.svg")


if __name__ == "__main__":
    fig_roadmap()
    fig_tokenize_window()
    fig_attention_toy()
    fig_self_attention()
    fig_multihead_block()
    fig_gpt_full()
    print("all day11 figures done")
