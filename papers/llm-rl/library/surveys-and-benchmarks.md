---
layout: post
title: P9G：综述与基准
permalink: /papers/llm-rl/library/surveys-and-benchmarks/
date: 2026-11-24 00:00:00 +0800
topic: llm-rl
categories:
- 论文精读
math: false
reading_prev:
  title: P9F：冻结模型、控制器与分层决策
  url: /papers/llm-rl/library/frozen-controllers/
---

已经有研究问题时，用综述定位方法类别，用基准确认观测、动作和评分是否适合自己的问题。

[返回 P9 文献导航](/2026/11/23/llm-rl-p9-reading-index/)。可以从 [E1 的实验地图](/2026/11/23/llm-rl-e1-experiment-map/) 起步，区分数据复算、冻结推理、策略训练和待运行协议。

## 直接 LLM＋RL 的综述与基准（2项）

这组入口先列系统和任务，再追溯各自的原论文。

- `U146 / 2026` [A Survey on Reinforcement Learning Optimization Methods for Multi-Agent Collaboration of Large Language Models](https://doi.org/10.54254/2755-2721/2026.36623)<br />
  按个体奖励微调、联合奖励微调和推理期交互三类组织多语言模型协作研究。
- `U026 / 2025` [LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language Models](https://proceedings.mlr.press/v267/abdulhai25a.html)<br />
  提供多轮语言交互 RL 环境与离线/在线方法接口，作为长程代理训练评测入口。

## LLM 与 MARL 交叉综述（5项）

查找同类论文时，用通信、协作训练与任务类型做筛选。

- `U054 / 2026` [The Five Ws of Multi-Agent Communication: Who Talks to Whom, When, What, and Why -- A Survey from MARL to Emergent Language and LLMs](https://arxiv.org/abs/2602.11583v1)<br />
  以通信的对象、内容、时机和目的连接传统 MARL、涌现语言及 LLM 团队研究。
- `U173 / 2026` [From Multi-Agent Reinforcement Learning to Agentic AI: A Comprehensive Literature Review of Algorithmic Advances and Decision-Analytic Implications (2020-2025)](https://doi.org/10.66972/ada202734)<br />
  从价值分解、合作策略优化和信用分配等方法梳理 MARL，并讨论与语言智能体决策架构的联系。
- `U186 / 2026` [Integration of Multi-agent Reinforcement Learning (MARL) and Large Language Models (LLMs) in Agentic AI Systems: A Narrative Review](https://doi.org/10.1007/s42979-026-05256-2)<br />
  以叙述性综述整理语言推理和 MARL 相结合的分层控制、规划、记忆及多领域案例。
- `U281 / 2026` [Toward the Science of Multi-Agent Communication: A Comprehensive Survey](https://yifang99.github.io/Towards-The-Science-of-Agent-Communication/)<br />
  按通信格式和消息内容整理 MARL、涌现语言以及 LLM 协作，在同一分类下考察不同消息所携带的信息。
- `U126 / 2024` [LLM-based Multi-Agent Reinforcement Learning: Current and Future Directions](https://arxiv.org/abs/2405.11106)<br />
  早期 LLM 与 MARL 分类入口，串联语言规划、通信和合作决策。
