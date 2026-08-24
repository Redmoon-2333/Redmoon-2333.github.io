---
title: "示例文章：这套排版支持的一切"
date: 2026-08-24 12:00:00 +0800
categories: [技术实践]
tags: [示例]
excerpt: 一篇可以随时删除的示例：代码块、图片、公式、表格、引用、脚注——确认渲染无误后，整篇删掉即可。
math: true
---

> **提示**：这是一篇示例文章，用来验证主题的所有排版元素。确认没问题后可以直接删除 `_posts/2026-08-24-sample-post.md`，开始写你自己的文章。

## 中文正文长这样

标题用衬线字体（Newsreader × Noto Serif SC），正文用系统黑体并放宽了行高，中文阅读体验优先。行内元素比如 `inline code`、**加粗**、*斜体*、以及[一个链接](https://pages.github.com/)的样式都已在设计系统里定义好。

## 代码块

```python
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    """手写一遍才叫真的会。"""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v
```

## 图片

把图片放进 `assets/img/` 目录，然后在 Markdown 里引用：

![占位图：替换成你的截图](/assets/img/placeholder.svg)

## 公式

在 front matter 里写 `math: true` 就能使用 KaTeX：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

## 表格

| 显存预算 | 能做的事 |
| :-- | :-- |
| 全参微调 | 约 0.5B 以内的模型 |
| LoRA / QLoRA | 1.7B 舒适，4B 需量化 |

## 引用与脚注

> 精读标准：能画出架构图，能说清它解决了什么问题、牺牲了什么。[^1]

[^1]: 脚注会被渲染在文末，适合放参考链接。
