# -*- coding: utf-8 -*-
"""Day2 实验：用 NumPy 把注意力跑起来
配套笔记：08_MyNote/Day2/Day2.md 第 11 节
运行：python attention_experiments.py   （仅需 numpy）

三个实验：
  1. 自注意力前向 —— 打印 4x4 权重矩阵，验证每行和为 1
  2. 因果掩码     —— 解码器不许偷看未来，右上角精确归零
  3. 多头切分     —— 4 维拆 2 个头各自算再拼接，输出形状不变
"""
import numpy as np

np.random.seed(0)  # 固定随机种子，保证你跑出的数和笔记一致


def softmax(x):
    """数值稳定的 softmax：沿最后一维归一化"""
    e = np.exp(x - x.max(axis=-1, keepdims=True))  # 减最大值，防 e^大数 溢出
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q, K, V, mask=None):
    """缩放点积注意力
    Q: (n_q, d_k)  K: (n_k, d_k)  V: (n_k, d_v)
    mask: 与 (n_q, n_k) 同形的布尔阵，True=允许看，False=屏蔽
    返回: (注意力权重矩阵, 加权求和后的新表示)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)           # 相似度矩阵，除 sqrt(d_k) 防饱和
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # 不许看的位置压成 -inf
    A = softmax(scores)                        # 每行归一化成权重
    return A, A @ V                            # 权重、加权求和后的新表示


words = ["小猫", "坐", "在", "垫子"]
X = np.random.randn(4, 4)        # 4 个词，每个 4 维（代替真实词向量）
Q, K, V = X, X, X                # 自注意力：QKV 同源

# ------------------------------------------------------------
# 实验 1：自注意力前向
# ------------------------------------------------------------
print("=" * 52)
print("实验 1：自注意力前向")
print("=" * 52)
A, out = attention(Q, K, V)
print("注意力权重（每行和为 1）：")
for w, name in zip(A, words):
    print(f"  {name}: " + "  ".join(f"{v:.2f}" for v in w))
print(f"每行和验证：{np.round(A.sum(axis=1), 2)}")

# ------------------------------------------------------------
# 实验 2：因果掩码——解码器不许偷看未来
# ------------------------------------------------------------
print()
print("=" * 52)
print("实验 2：因果掩码（下三角 True = 允许看）")
print("=" * 52)
mask = np.tril(np.ones((4, 4), dtype=bool))
A_masked, _ = attention(Q, K, V, mask)
print("加掩码后（右上角全部归零）：")
for w, name in zip(A_masked, words):
    print(f"  {name}: " + "  ".join(f"{v:.2f}" for v in w))
print("验证上三角全为 0：", np.all(np.triu(A_masked, 1) == 0))
# 观察：'垫子' 那一行只剩自己和之前的词有权重

# ------------------------------------------------------------
# 实验 3：多头 = 切开分头算，再拼回来
# ------------------------------------------------------------
print()
print("=" * 52)
print("实验 3：多头切分（4 维 -> 2 个头 x 2 维）")
print("=" * 52)
h = 2
d_h = 4 // h
outs = []
for i in range(h):
    sl = slice(i * d_h, (i + 1) * d_h)    # 头 i 拿第 i 段维度
    _, o_i = attention(Q[:, sl], K[:, sl], V[:, sl], mask)
    outs.append(o_i)
concat = np.hstack(outs)                   # 拼回 (4, 4)
print("多头拼接输出形状：", concat.shape, "（与单头输出相同）")
print("结论：多头不改变维度，只改变'看的方式'")
