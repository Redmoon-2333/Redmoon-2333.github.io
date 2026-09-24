---
title: 算法设计与分析 Day2：插入排序、归并排序与渐进分析
date: 2026-09-24 18:00:00 +0800
categories:
- 课堂笔记
topic: algorithms
lesson_day: 2
tags:
- 算法设计与分析
- 插入排序
- 归并排序
- 渐进分析
excerpt: 插入排序、归并排序和渐进符号的课堂整理，并给出 PDF 课后作业的中英文结合解答。
math: true
---

> RedMoon · 2026-09-24 · 课堂阅读笔记

## 这节课解决了什么问题

排序问题的输入是一组数，输出仍然是这组数，只是排列成非降序：

$$
a'_1 \leq a'_2 \leq \cdots \leq a'_n
$$

这里有两个容易被忽略的要求：

1. 输出中的元素必须和输入一一对应，不能凭空增加、删除或修改元素。
2. 算法要对所有合法输入成立，不能只在一个样例上得到正确结果。

Day2 的主线是：先用插入排序把“把一个元素放到已有序部分”的想法写清楚，再用归并排序把问题拆成更小的部分，最后通过渐进符号比较算法随输入规模增长时的效率。

课堂还用“买卖股票的最佳时机 II”复习了贪心观察：当一段上涨区间可以拆成相邻上涨的差值时，直接累加所有正差值就能得到最大利润。这个例子和排序的共同点不是“都在找一个答案”，而是都要先找到输入中可以利用的结构。

## 本文的记号约定

为与本次算法课作业的解答保持一致，本文统一取：

$$
\lg n=\log_2 n
$$

因此：

$$
n^{1/\lg n}=2,\qquad
2^{\lg n}=n,\qquad
4^{\lg n}=n^2
$$

若其他教材把 `lg` 约定为别的底数，常数值会变化，但这些表达式的渐进增长级别不变。

## 先把渐进符号分清楚

设 $f(n)$ 和 $g(n)$ 在足够大的 $n$ 上为正。

### Big-O：不超过某个增长级别

$$
f(n)=O(g(n))
$$

表示存在常数 $c>0$ 和 $n_0$，使得当 $n\geq n_0$ 时：

$$
0\leq f(n)\leq c g(n)
$$

它描述的是渐进上界。这里的“上界”不是说 $f(n)$ 在每一个小规模输入上都小于 $g(n)$，而是说从某个规模开始，可以用常数倍的 $g(n)$ 控住它。

### Big-Omega：至少达到某个增长级别

$$
f(n)=\Omega(g(n))
$$

表示存在常数 $c>0$ 和 $n_0$，使得当 $n\geq n_0$ 时：

$$
f(n)\geq c g(n)
$$

### Theta：同一个增长级别

$$
f(n)=\Theta(g(n))
$$

当且仅当同时满足：

$$
f(n)=O(g(n)),\qquad f(n)=\Omega(g(n))
$$

因此，$\Theta$ 比单独写 $O$ 更强。若只知道 $f(n)=O(g(n))$，不能直接推出两者增长速度相同。

### 小 $o$ 与小 $\omega$

$$
f(n)=o(g(n))
\Longleftrightarrow
\lim_{n\to\infty}\frac{f(n)}{g(n)}=0
$$

$$
f(n)=\omega(g(n))
\Longleftrightarrow
\lim_{n\to\infty}\frac{f(n)}{g(n)}=\infty
$$

例如 $n^2=o(n^3)$，而 $n^3=\omega(n^2)$。如果比值趋近于一个正常数，则是 $\Theta$ 关系，而不是小 $o$ 或小 $\omega$。

### 比较增长率时的几个基本事实

在常见函数中，增长速度大致遵循：

$$
1
\prec
\lg^* n
\prec
\sqrt{\lg n}
\prec
\lg n
\prec
n
\prec
n\lg n
\prec
n^2
\prec
c^n
\prec
n!
\prec
2^{2^n}
$$

其中 $\lg^* n$ 是迭代对数，增长极慢；$c>1$ 是固定常数。具体题目仍需根据函数形式判断，不能只背这一行。

## 插入排序：把新元素放进有序前缀

### 核心想法

插入排序可以类比整理手牌，但真正需要记住的是它维护的结构：

> 在处理第 `idx` 个元素时，位置 `1` 到 `idx - 1` 已经按非降序排列。

取出 `A[idx]`，从右向左扫描前面的有序部分。比 `key` 大的元素向右移动一格，直到找到第一个不大于 `key` 的位置，再把 `key` 放进去。

例如：

```text
[2, 5, 7, 3, 6]
             key = 3
```

从右向左移动：

```text
[2, 5, 7, 7, 6]
[2, 5, 5, 7, 6]
[2, 3, 5, 7, 6]
```

插入后，前四个元素 `[2, 3, 5, 7]` 有序。下一轮再处理 `6`。

### 伪代码

下面使用 PDF 中的 1-based 下标：

```text
INSERTION-SORT(A)
    for idx = 2 to A.length
        key = A[idx]                 // 暂存待插入元素
        i = idx - 1

        while i > 0 and A[i] > key
            A[i + 1] = A[i]          // 大元素向右移动
            i = i - 1

        A[i + 1] = key               // 放入空出的位置
```

### 循环不变量

外层循环每一轮开始时，循环不变量是：

> 子数组 `A[1..idx-1]` 恰好包含输入数组前 `idx - 1` 个元素，且这些元素按非降序排列。

证明分三步：

1. **初始化**：第一次循环开始时 `idx = 2`，`A[1]` 只有一个元素，单元素序列天然有序。
2. **保持**：假设 `A[1..idx-1]` 有序。向右移动所有大于 `key` 的元素后，把 `key` 放到第一个不大于它的元素后面，得到的 `A[1..idx]` 仍然有序。
3. **终止**：外层循环结束时，`idx = n + 1`，因此 `A[1..n]` 已经有序。

这个证明没有依赖某个具体数组，所以它适用于所有合法输入。

### 时间和空间

- 已经有序时，每个 `key` 只需要比较一次，时间复杂度为 $\Theta(n)$。
- 逆序时，第 `idx` 个元素几乎要向前移动 `idx - 1` 次：

$$
1+2+\cdots+(n-1)=\frac{n(n-1)}2
$$

因此最坏时间复杂度为 $\Theta(n^2)$。
- 算法只使用 `key`、`i`、`idx` 等常数个辅助变量，额外空间复杂度为 $O(1)$。
- 插入排序是稳定排序：当判断条件使用 `A[i] > key` 而不是 `A[i] >= key` 时，相等元素不会被无故交换位置。

### 为什么二分查找不能把插入排序变成 $O(n\log n)$

有序前缀确实允许用二分查找更快地找到插入位置，比较次数可以降到 $O(\log n)$。但找到位置后，数组中的元素仍然需要整体向右移动，单轮移动可能是 $O(n)$。

因此，普通数组上的插入排序仍然可能需要 $\Theta(n^2)$ 的移动。二分查找减少了比较，不足以消除移动成本。

## 归并排序：先分别排好，再线性合并

### 分治结构

归并排序使用分治法：

1. **Divide**：把数组分成左右两个子数组。
2. **Conquer**：递归地把左右子数组排好。
3. **Combine**：把两个已经有序的子数组合并成一个有序数组。

当区间长度为 $0$ 或 $1$ 时，不需要继续排序，因为它天然有序。

### 合并两个有序序列

设左边是：

```text
[2, 5, 9]
```

右边是：

```text
[1, 6, 8]
```

比较两个序列当前指针所指的元素：

1. 比较 `2` 和 `1`，取 `1`。
2. 比较 `2` 和 `6`，取 `2`。
3. 比较 `5` 和 `6`，取 `5`。
4. 比较 `9` 和 `6`，取 `6`。
5. 比较 `9` 和 `8`，取 `8`。
6. 右侧用完，把左侧剩下的 `9` 放入结果。

得到：

```text
[1, 2, 5, 6, 8, 9]
```

关键在于：两个子数组已经有序，所以当前最小元素只可能出现在两个指针指向的位置。每次取出一个元素，指针只向前移动，不会回退。

### 参考实现

下面使用 0-based 下标和一个共享辅助数组。代码中的注释说明每个边界判断的作用。

```cpp
#include <iostream>
#include <vector>
using namespace std;

void mergeRange(vector<int>& a, vector<int>& buffer, int left, int mid, int right) {
    // 先复制当前区间，避免边合并边覆盖还没有读取的元素。
    for (int i = left; i <= right; ++i) {
        buffer[i] = a[i];
    }

    int p1 = left;      // 左半部分当前最小元素的位置
    int p2 = mid + 1;   // 右半部分当前最小元素的位置

    for (int i = left; i <= right; ++i) {
        if (p1 > mid) {
            // 左半部分已经取完，只能取右半部分。
            a[i] = buffer[p2++];
        } else if (p2 > right) {
            // 右半部分已经取完，只能取左半部分。
            a[i] = buffer[p1++];
        } else if (buffer[p1] <= buffer[p2]) {
            // 相等时优先取左边，保留稳定性。
            a[i] = buffer[p1++];
        } else {
            a[i] = buffer[p2++];
        }
    }
}

void mergeSort(vector<int>& a, vector<int>& buffer, int left, int right) {
    // 区间长度不超过 1 时已经有序。
    if (left >= right) {
        return;
    }

    int mid = left + (right - left) / 2;
    mergeSort(a, buffer, left, mid);
    mergeSort(a, buffer, mid + 1, right);
    mergeRange(a, buffer, left, mid, right);
}

int main() {
    int n;
    cin >> n;

    vector<int> a(n);
    for (int& x : a) {
        cin >> x;
    }

    vector<int> buffer(n);
    if (n > 0) {
        mergeSort(a, buffer, 0, n - 1);
    }

    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            cout << ' ';
        }
        cout << a[i];
    }
    cout << '\n';
}
```

### 归并过程的正确性

合并前，左右两段分别有序。合并循环开始时，可以维护下面的不变量：

> `a[left..i-1]` 已经有序，并且正好包含左右两段中最小的 `i-left` 个元素；如果某一侧还没有取完，该侧的指针指向该侧剩余元素中的最小值。

每次选择两个指针中较小的元素，就把剩余元素中的最小值放到了结果末尾。因此不变量保持成立。循环结束时，所有元素都被取出，整个区间有序。

在此基础上，再对递归规模做归纳：

1. 长度为 $0$ 或 $1$ 的区间直接正确返回。
2. 假设更小的区间都能正确排序。
3. 当前区间分成两个更小的区间，递归调用根据归纳假设将它们分别排好。
4. `merge` 正确合并两个有序区间，因此当前区间也正确。

### 时间和空间

一次合并会让两个指针总共向前移动至多 $n$ 次，因此单次合并是：

$$
\Theta(n)
$$

设 $T(n)$ 是排序 $n$ 个元素的时间：

$$
T(n)=2T\left(\frac n2\right)+\Theta(n)
$$

递归树每层处理的元素总数都是 $\Theta(n)$，树高是 $\Theta(\log n)$，所以：

$$
T(n)=\Theta(n\log n)
$$

标准数组实现需要大小为 $n$ 的辅助数组，额外空间复杂度为 $O(n)$；递归栈另占 $O(\log n)$，被 $O(n)$ 吸收。

归并排序在最好、平均和最坏情况下都保持 $\Theta(n\log n)$，并且是稳定排序。

## 插入排序、归并排序与选择排序

| 算法 | 最好时间 | 最坏时间 | 额外空间 | 稳定性 | 主要结构 |
| --- | --- | --- | --- | --- | --- |
| 插入排序 | $\Theta(n)$ | $\Theta(n^2)$ | $O(1)$ | 稳定 | 维护有序前缀 |
| 选择排序 | $\Theta(n^2)$ | $\Theta(n^2)$ | $O(1)$ | 通常不稳定 | 每轮选出剩余部分最大值 |
| 归并排序 | $\Theta(n\log n)$ | $\Theta(n\log n)$ | $O(n)$ | 稳定 | 分治后线性合并 |

三者的差异不只是代码写法：

- 插入排序利用了“前缀已经有序”，所以输入接近有序时很有优势。
- 归并排序利用了“两个有序序列可以线性合并”，把最坏情况从平方级降到 $n\log n$。
- 选择排序每一轮都要在剩余区间中寻找最大值，输入是否有序并不能减少扫描次数，所以最好情况仍然是平方级。

## PDF 最后作业：中英文结合解答

### 1-1 Relative asymptotic growths

**题目 / Problem**

对于表格中的每一对函数 $(A,B)$，判断 $A$ 是否是 $O(B)$、$o(B)$、$\Omega(B)$、$\omega(B)$ 或 $\Theta(B)$。假设 $k\geq 1$、$\varepsilon>0$、$c>1$ 都是常数。

记号 $\lg^k n$ 表示 $(\lg n)^k$；$\lg^* n$ 表示迭代对数。上标 $k$ 与上标星号含义不同。

| 小题 | $A$ | $B$ | $O$ | $o$ | $\Omega$ | $\omega$ | $\Theta$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a | $(\lg n)^k$ | $n^\varepsilon$ | yes | yes | no | no | no |
| b | $n^k$ | $c^n$ | yes | yes | no | no | no |
| c | $\sqrt n$ | $n^{\sin n}$ | no | no | no | no | no |
| d | $2^n$ | $2^{n/2}$ | no | no | yes | yes | no |
| e | $n^{\lg c}$ | $c^{\lg n}$ | yes | no | yes | no | yes |
| f | $\lg(n!)$ | $\lg(n^n)$ | yes | no | yes | no | yes |

**Explanation / 说明**

- a. 任意固定次幂的对数都比任意正次幂增长慢，因此 $(\lg n)^k=o(n^\varepsilon)$。
- b. 固定底数的指数函数比任意多项式增长快，因此 $n^k=o(c^n)$。
- c. 整数 $n$ 对 $2\pi$ 取模后在圆周上稠密，因此有无穷多个 $n$ 满足 $\sin n>3/4$，也有无穷多个满足 $\sin n<1/4$。比较两函数可看比值：

$$
\frac{\sqrt n}{n^{\sin n}}=n^{1/2-\sin n}
$$

在前一类整数上，比值趋于 $0$；在后一类整数上，比值趋于 $\infty$。所以它既不有界，也不趋于 $0$ 或 $\infty$，五种渐进关系均不成立。
- d.

$$
\frac{2^n}{2^{n/2}}=2^{n/2}\to\infty
$$

因此 $2^n=\omega(2^{n/2})$，也必然是 $\Omega(2^{n/2})$。
- e. 利用换底公式：

$$
c^{\lg n}=2^{(\lg c)(\lg n)}=n^{\lg c}
$$

两者相等，所以是 $\Theta$，同时满足 $O$ 和 $\Omega$，但不是小 $o$ 或小 $\omega$。
- f. $\lg(n^n)=n\lg n$。另一方面，Stirling 公式给出：

$$
\lg(n!)=\Theta(n\lg n)
$$

所以两者同阶。

### 1-2 Ordering by asymptotic growth rates

**题目 / Problem**

将图片中的 30 个函数按增长速度从小到大排列，并按 $\Theta$ 等价关系分组。

**答案 / Answer**

从慢到快排列如下。每组花括号内的函数互为 $\Theta$；相邻组之间增长严格变快：

$$
\begin{aligned}
&\{1,\ n^{1/\lg n}\}
\prec \{\lg(\lg^* n)\}
\prec \{\lg^*(\lg n),\ \lg^* n\}
\prec \{2^{\lg^* n}\}
\prec \{\ln\ln n\}
\prec \{\sqrt{\lg n}\}
\prec \{\ln n\}
\prec \{(\lg n)^2\}
\prec \{2^{\sqrt{2\lg n}}\}
\prec \{(\sqrt 2)^{\lg n}\}
\prec \{2^{\lg n},\ n\}
\prec \{n\lg n,\ \lg(n!)\}
\prec \{4^{\lg n},\ n^2\}
\prec \{n^3\}
\prec \{(\lg n)!\}
\prec \{(\lg n)^{\lg n},\ n^{\lg\lg n}\}
\prec \{(3/2)^n\}
\prec \{2^n\}
\prec \{n\cdot 2^n\}
\prec \{e^n\}
\prec \{2^{2n+1}\}
\prec \{n!\}
\prec \{(n+1)!\}
\prec \{2^{2^n}\}.
\end{aligned}
$$

这里的几个关键化简是：

$$
n^{1/\lg n}=2
$$

$$
(\sqrt 2)^{\lg n}=n^{1/2}
$$

在完整顺序中，$(\sqrt 2)^{\lg n}=\sqrt n$，所以它要放在 $2^{\sqrt{2\lg n}}$ 之后、$n$ 之前；不能与 $n$ 放在同一组。

$$
4^{\lg n}=n^2
$$

$$
(\lg n)^{\lg n}=n^{\lg\lg n}
$$

以及：

$$
\lg(n!)=\Theta(n\lg n)
$$

需要特别区分以下几处：

- $(\lg n)!$ 位于 $n^3$ 与 $(\lg n)^{\lg n}$ 之间。令 $x=\lg n$，由 Stirling 公式，$\lg(x!)=x\lg x-\Theta(x)$：它最终超过任意固定次幂 $n^k$，但 $x!/(x^x)\to 0$。
- $n\cdot 2^n$ 位于 $2^n$ 与 $e^n$ 之间，因为 $n(2/e)^n\to 0$；而 $n!$ 比任意固定底数的指数函数都快。
- $(n+1)!$ 与 $n!$ 不是同一个 $\Theta$ 类，因为它们的比值是 $n+1$，会趋于无穷。

### 1-3 Asymptotic notation properties

**题目 / Problem**

设 $f(n)$ 和 $g(n)$ 是渐进为正的函数，证明或否定下列猜想。

#### a. $f(n)=O(g(n))$ implies $g(n)=O(f(n))$

**结论：False / 错误。**

反例：

$$
f(n)=n,\qquad g(n)=n^2
$$

有 $f(n)=O(g(n))$，但：

$$
\frac{g(n)}{f(n)}=n\to\infty
$$

所以 $g(n)\not=O(f(n))$。$O$ 关系不是对称关系。

#### b. $f(n)+g(n)=\Theta(\min(f(n),g(n)))$

**结论：False / 错误。**

反例：

$$
f(n)=n^2,\qquad g(n)=n
$$

此时：

$$
f(n)+g(n)=n^2+n=\Theta(n^2)
$$

但：

$$
\min(f(n),g(n))=n
$$

两者不可能是 $\Theta$ 关系。对于渐进为正的函数，一般有：

$$
f(n)+g(n)=\Theta(\max(f(n),g(n)))
$$

#### c. $f(n)=O(g(n))$ implies $\lg(f(n))=O(\lg(g(n)))$

题目还给出了 $\lg(g(n))\geq 1$，并且对足够大的 $n$ 有 $f(n)\geq 1$。

**结论：True / 正确。**

由 $f(n)=O(g(n))$，存在常数 $C>0$，使得足够大时：

$$
f(n)\leq Cg(n)
$$

两边取对数：

$$
\lg f(n)\leq \lg C+\lg g(n)
$$

由于 $\lg g(n)\geq 1$，常数项 $\lg C$ 可以被常数倍的 $\lg g(n)$ 吸收，因此：

$$
\lg f(n)=O(\lg g(n))
$$

注意，结论是对数后的 $O$ 上界，而不是说 $\lg f(n)$ 和 $\lg g(n)$ 一定是 $\Theta$ 关系。

#### d. $f(n)=O(g(n))$ implies $2^{f(n)}=O(2^{g(n)})$

**结论：False / 错误。**

反例：

$$
f(n)=2n,\qquad g(n)=n
$$

显然 $f(n)=O(g(n))$，但是：

$$
\frac{2^{f(n)}}{2^{g(n)}}=2^n\to\infty
$$

所以 $2^{f(n)}\not=O(2^{g(n)})$。指数函数会放大常数倍差异，不能把普通的 $O$ 关系直接放进指数中。

### 1-4 Selection sort

**题目 / Problem**

对存储在数组 `A` 中的 $n$ 个数，先找到最大元素并将其与 `A[n]` 交换；再在剩余部分中找到第二大元素并将其与 `A[n-1]` 交换；如此继续。要求：

1. 写出该算法的伪代码；
2. 写出算法维护的循环不变量；
3. 给出选择排序最好和最坏情况下的渐进运行时间。

**Pseudocode / 伪代码**

```text
SELECTION-SORT(A)
    n = A.length

    for end = n downto 2
        maxIndex = 1

        // 在 A[1..end] 中寻找最大元素
        for i = 2 to end
            if A[i] > A[maxIndex]
                maxIndex = i

        // 把当前最大值放到它最终的位置 A[end]
        exchange A[maxIndex] with A[end]
```

若使用 0-based 下标，可以写成：

```text
SELECTION-SORT(A)
    for end = A.length - 1 downto 1
        maxIndex = 0

        for i = 1 to end
            if A[i] > A[maxIndex]
                maxIndex = i

        swap(A[maxIndex], A[end])
```

**Loop invariant / 循环不变量**

外层循环每轮开始时：

> 区间 `A[end+1..n]` 已经包含原数组中最大的 `n-end` 个元素，并且这些元素已经按非降序排列；它们的位置已经确定，不再参与后续寻找。

等价地说，当前还需要处理的 `A[1..end]` 中的最大元素，会被放到 `A[end]`。

**Correctness / 正确性**

1. **Initialization / 初始化**：第一次外层循环开始时，`end = n`，后缀 `A[n+1..n]` 为空，循环不变量成立。
2. **Maintenance / 保持**：内层循环找到 `A[1..end]` 的最大元素，并将它交换到 `A[end]`。因此新的后缀 `A[end..n]` 包含当前应该放置的最大元素以及之前已经确定的更大元素，仍然有序。
3. **Termination / 终止**：外层循环处理完 `end = 2` 后，`A[2..n]` 已经按非降序排列并包含最大的 `n - 1` 个元素；剩下的 `A[1]` 是最小元素，因此整个数组有序。

**Running time / 运行时间**

第 `end` 轮需要扫描 `end` 个元素，比较次数为：

$$
(n-1)+(n-2)+\cdots+1
=\frac{n(n-1)}2
=\Theta(n^2)
$$

即使数组已经有序，也仍然要扫描每个未确定区间来确认最大值，所以：

- Best case / 最好情况：$\Theta(n^2)$；
- Worst case / 最坏情况：$\Theta(n^2)$；
- Extra space / 额外空间：$O(1)$。

## 课后复习时最容易混淆的地方

1. 一次 `merge` 不是 $O(\log n)$，而是 $\Theta(n)$；$\log n$ 来自递归层数。
2. 二分查找可以减少插入位置的比较次数，但不能消除数组搬移，所以普通插入排序仍可能是 $\Theta(n^2)$。
3. $O$ 不是“恰好等于”，也不是对称关系；只有同时有 $O$ 和 $\Omega$ 才能写 $\Theta$。
4. 选择排序的最好情况仍是 $\Theta(n^2)$，因为它不根据已有顺序提前结束。
5. 复杂度中的额外空间不把输入数组本身重复计算；插入排序和选择排序是 $O(1)$，标准归并排序是 $O(n)$。
