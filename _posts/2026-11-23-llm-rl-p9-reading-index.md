---
title: LLM 与强化学习 P9：当前研究坐标与分层论文阅读目录
date: 2026-11-23 00:09:00 +0800
categories:
- 论文精读
topic: llm-rl
series_order: 9
series_label: P9 · 阅读目录
tags:
- 文献目录
- 研究地图
- 前期探索
excerpt: 182 项文献的主题导航：从验证器、记忆、协作和工具入手，按问题找到原文。
math: false
slug: llm-rl-p9-reading-index
reading_prev:
  title: E11 · 当前三个候选，下一组实验各要区分什么
  url: /2026/11/23/llm-rl-e11-next-experiments/
---

想查奖励为什么会鼓励错答案，先翻验证器页；想查模型怎么保存改签后的日期，先翻记忆页。这份目录收录 182 项，按 11 类问题分到 7 个查阅页。

## 1. 先读哪些论文

- 第一次接触语言模型的强化学习：[P1：谁在学习](/2026/11/23/llm-rl-p1-learning-object/) → [P2：从概率到策略优化](/2026/11/23/llm-rl-p2-policy-optimization/)。原文先找 PPO 与 DeepSeekMath，再读 DeepSeek-R1、DAPO。
- 想测某位队友有没有帮忙：[P3：团队得分与个体信用](/2026/11/23/llm-rl-p3-credit-assignment/)。先读 COMA、LLM-MCA 或 LCA，再按问题查 CREDO、MARFT 与 TeamTR。
- 检查评分会不会误导训练：[P4：验证器为什么会误判](/2026/11/23/llm-rl-p4-verifier/) → [P5：跨题共享](/2026/11/23/llm-rl-p5-cross-prompt/)。先找 Optimistic Verifiers，再对照 Reward Suite Is Leaky 与 Wrong the Same Way。
- 做长程记忆或步骤评分：[P6：摘要与记忆](/2026/11/23/llm-rl-p6-memory/)先读 CompactionRL、ECHO、Supersede；[P7：聚焦和覆盖](/2026/11/23/llm-rl-p7-coverage/)再读 Coverage, Not Targeting 与 TACA。
- 接工具、机器人或图像：[P8：工具与多模态](/2026/11/23/llm-rl-p8-agent-routes/)先看 ReAct、Search-R1、SayCan，再查 Eureka、ReMAC 与 cooperative VLA。

例如，想查同一个错误答案为什么换题还能得分，可到 A 页找 `L09`（小模型的可控乘法实验），再看 `L12`（自然代码测试遗漏）。记忆方向可到 B 页找 `L01` 和 `L02`，比较前者怎么训练摘要、后者怎么追踪摘要里的来源。

## 按主题翻目录

- <span id="不完美验证器与-rlvr8项"></span><span id="推理-rl测试时学习与自博弈24项"></span>[P9A：验证器与推理训练](/papers/llm-rl/library/verifiers-and-reasoning/)（32 项）。从训练判分器可能错奖哪些答案读起，再看推理训练怎样利用这些分数。

- <span id="长程记忆与信用23项"></span>[P9B：长程记忆与信用](/papers/llm-rl/library/memory-and-credit/)（23 项）。模型只能带走一部分对话时，要决定保留哪些事实，还要决定哪些写入动作得到训练信号。

- <span id="多-llm-协作训练25项"></span>[P9C：多 LLM 协作训练](/papers/llm-rl/library/multi-llm-cooperation/)（25 项）。同一局有多个语言角色时，团队分数怎样交给每位成员，成员更新后又怎样继续一起采样？

- <span id="工具检索与多轮-agent25项"></span><span id="过程反馈安全与多模态14项"></span>[P9D：工具、过程反馈与多模态](/papers/llm-rl/library/tools-and-multimodal/)（39 项）。查询、工具结果和图像都能进入对话。先分清哪些内容由模型生成、哪些只是它读到的观察。

- <span id="llm--vlm-辅助奖励设计21项"></span>[P9E：LLM 与 VLM 设计奖励](/papers/llm-rl/library/reward-design/)（21 项）。把语言模型放在裁判或奖励设计者的位置，观察下层策略用这些评分能学到什么。

- <span id="小控制器编排冻结-llm13项"></span><span id="先验信用与层次控制22项"></span>[P9F：冻结模型、控制器与分层决策](/papers/llm-rl/library/frozen-controllers/)（35 项）。把语言模型当成可调用的帮手。一个小策略可以学习何时检索、复核、发消息或换技能。

- <span id="直接-llmrl-的综述与基准2项"></span><span id="llm-与-marl-交叉综述5项"></span>[P9G：综述与基准](/papers/llm-rl/library/surveys-and-benchmarks/)（7 项）。已经有研究问题时，用综述定位方法类别，用基准确认观测、动作和评分是否适合自己的问题。

## 2. 这份目录怎样计数

原合并库 307 项与新增 28 项合在一起，`L27` 和 `U071` 是同一作品，去重后共 334 项。这里列出其中 121 项直接使用 LLM 强化学习的工作，以及 61 项研究奖励、控制器或相关机制的工作。其余 152 项是基础、相邻方向或撤回留史材料。同一作品的不同 PDF 版本按一项计算。

`L01 / 2026` 中，`L01` 是资料编号，`2026` 是归档年份。各条目附原文入口和阅读提示。

## 当前待选问题

[实践 E11](/2026/11/23/llm-rl-e11-next-experiments/)列出三项待验证设计：C02 查跨题筛查会不会错伤正确复用，C01 改查记忆里不留痕迹的保护动作怎样学习，C03 查同一批数据在不同可验证程度下该怎样分配步骤权重。

[返回系列导读](/papers/llm-rl/)。
