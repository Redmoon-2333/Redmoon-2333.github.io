---
layout: post
title: P9D：工具、过程反馈与多模态
permalink: /papers/llm-rl/library/tools-and-multimodal/
date: 2026-11-24 00:00:00 +0800
topic: llm-rl
categories:
- 论文精读
math: false
reading_prev:
  title: P9C：多 LLM 协作训练
  url: /papers/llm-rl/library/multi-llm-cooperation/
reading_next:
  title: P9E：LLM 与 VLM 设计奖励
  url: /papers/llm-rl/library/reward-design/
---

查询、工具结果和图像都能进入对话。先分清哪些内容由模型生成、哪些只是它读到的观察。

[返回 P9 文献导航](/2026/11/23/llm-rl-p9-reading-index/)。从 [P8](/2026/11/23/llm-rl-p8-agent-routes/) 的 Search-R1 例子入手，再看各篇是改了工具动作、增加过程反馈，还是换了输入模态。

## 工具、检索与多轮 Agent（25项）

把工具查询、返回内容与最终答案分开，核对预算和参与损失的动作。

- `U006 / 2026` [D-CORE: Incentivizing Task Decomposition in Large Reasoning Models for Complex Tool Use](https://proceedings.mlr.press/v306/xu26bn.html)<br />
  D-CORE 先通过自蒸馏强化子任务分解，再用多样性相关 RL 改善复杂工具推理。
- `U007 / 2026` [Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](https://aclanthology.org/2026.acl-long.318/)<br />
  STEP-HRL 用全局子任务完成信息与局部进度摘要构成紧凑的步骤级转移。
- `U009 / 2026` [ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration](https://proceedings.mlr.press/v306/su26d.html)<br />
  训练小型语言调度器选择模型和工具，奖励同时考虑结果、效率与用户偏好。
- `U032 / 2026` [Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning](https://proceedings.mlr.press/v306/wang26jh.html)<br />
  AWM 生成由代码和数据库驱动的可执行工具环境，为多轮 RL 提供状态与反馈。
- `U052 / 2026` [EvoRouter: Cost-Efficient Multi-Agent Routing via Insight-Guided On-Policy Distillation](https://arxiv.org/abs/2610.35690v1)<br />
  用跨任务执行反馈提炼路由经验，再结合特权教师蒸馏与成本感知 RL 训练路由器。
- `U063 / 2026` [From Proprietary to Open-Source: Bridging the Distribution Gap via Multi-Agent Protocol Distillation in Agentic Search](https://arxiv.org/abs/2607.24280v1)<br />
  MAPD 用结构化、风格归一的协议连接闭源教师与开放学生，再结合蒸馏和 RL。
- `U064 / 2026` [One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL](https://arxiv.org/abs/2608.12253v2)<br />
  分析单一 LLM 模拟器的行为模式收缩，并比较行为采样与可训练模拟器种群共同训练。
- `U068 / 2026` [Policy and World Modeling Co-Training for Language Agents](https://arxiv.org/abs/2606.02388v2)<br />
  PaW 利用 on-policy 转移的下一观测作世界建模辅助监督，并平衡 RL 与预测损失。
- `U117 / 2026` [DLM: Unified Decision Language Model for Offline Multi-Agent Sequential Decision Making](https://proceedings.mlr.press/v306/zhang26v.html)<br />
  将离线多智能体轨迹转为对话序列，经过监督训练与轻量奖励 GRPO 两阶段学习。
- `U132 / 2026` [AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning](https://openreview.net/forum?id=ZgCCDwcGwn)<br />
  在交互训练早期限制行动轮数，随后逐步延长允许的交互跨度，用同一框架训练多种环境中的语言策略。
- `U181 / 2026` [HiRS-Agent: A Hierarchical Multi-Agent System for Reliable Long-Horizon Remote Sensing Task Solving](https://doi.org/10.1145/3767308.3835311)<br />
  遥感长程工作流中，管理角色负责规划、路由和核验，专家角色操作工具，分别接收轨迹奖励和步骤奖励。
- `U243 / 2026` [SLMs as Multi-Agent Routers: A Progressive SFT and Reinforcement Learning Approach](https://arxiv.org/abs/2608.00030v1)<br />
  小模型为查询选择专门检索代理并输出调用参数，奖励同时考查主题匹配与实际检索内容相关性。
- `U270 / 2026` [Does Reinforcement Fine-Tuning Improve Generalization of LLM Agents? An Empirical Study](https://proceedings.mlr.press/v306/xi26b.html)<br />
  在任务难度、跨环境和连续训练三个设置下比较智能体强化微调的迁移，按观察与行动接口分析环境差异。
- `U271 / 2026` [GLARE: Scalable Neuro-Symbolic Reward Shaping for LLM Agents via Group-Level Automata](https://proceedings.mlr.press/v306/yan26k.html)<br />
  将轨迹事件符号化，借助时序逻辑构造自动机，以状态迁移驱动进度奖励，并用组相对策略更新语言智能体。
- `U005 / 2025` [Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](https://proceedings.mlr.press/v267/hu25q.html)<br />
  GLIDER 用离线层次 RL 连接高层抽象计划与低层控制，研究长程决策和迁移。
- `U048 / 2025` [CoDA: A Context-Decoupled Hierarchical Agent with Reinforcement Learning](https://arxiv.org/abs/2512.12716v1)<br />
  同一 LLM 分别承担上下文隔离的 Planner 与 Executor，以轨迹奖励联合优化。
- `U049 / 2025` [Learning to Orchestrate Agents in Natural Language with the Conductor](https://arxiv.org/abs/2512.04388v5)<br />
  Conductor 用 RL 学习工作池中的通信关系和定向指令，并适配变化的成员集合。
- `U050 / 2025` [Controlling Performance and Budget of a Centralized Multi-agent LLM System with Reinforcement Learning](https://arxiv.org/abs/2511.02755v1)<br />
  中央语言控制器按预算选择专家，联合优化任务质量和推理费用。
- `U076 / 2025` [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516v5)<br />
  让模型在推理中学习多次检索查询，以结果奖励优化并屏蔽检索内容的训练 token。
- `U077 / 2025` [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536v2)<br />
  通过多轮实时代码执行与结果反馈，学习何时、如何将工具插入长推理。
- `U078 / 2025` [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958v1)<br />
  系统比较工具使用奖励的类型、尺度、粒度和时间结构，再用 GRPO 训练工具策略。
- `U228 / 2025` [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2504.20073v2)<br />
  在数种交互环境研究语言策略训练中奖励变异收缩、梯度尖峰与行为退化，比较轨迹筛选等稳定化方法。
- `U265 / 2025` [Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning](https://arxiv.org/abs/2511.14460v2)<br />
  以交互步表征轨迹，将上下文管理、环境接口和策略优化分开，使工具反馈下的多轮训练可以采用不同的信用分配粒度。
- `U203 / 2024` [LM2: A Simple Society of Language Models Solves Complex Reasoning](https://arxiv.org/abs/2404.02255v1)<br />
  分解器逐步提出子问题，求解器回答，验证器分类错误。错误类型及其发生位置形成逐步奖励，再用 PPO 更新分解器。
- `U220 / 2023` [OpenAGI: When LLM Meets Domain Experts](https://arxiv.org/abs/2304.04370v6)<br />
  由 LLM 按自然语言任务选择并组合专门模型，再利用任务结果提供反馈，强化 LLM 的组合决策。

## 过程反馈、安全与多模态（14项）

过程分数可以来自规则、语言裁判或视觉输入。先看它能验证哪个步骤。

- `U069 / 2026` [Verifiable Process Rewards for Agentic Reasoning](https://arxiv.org/abs/2605.10325v3)<br />
  把搜索、约束或后验 oracle 转为动作级过程验证器，缓解长程任务的稀疏反馈。
- `U070 / 2026` [Verifiable Counterfactual Supervision for Process Reward Models](https://arxiv.org/abs/2605.02395v3)<br />
  在符号正确链中注入受控错误并重算后续状态，构造可验证的首错位置监督。
- `U115 / 2026` [Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models](https://proceedings.mlr.press/v306/liu26bn.html)<br />
  同一策略以攻防角色在线自博弈，共同适应变化的对手。
- `U116 / 2026` [AdvEvo-MARL: Shaping Internalized Safety through Adversarial Co-Evolution in Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/pan26p.html)<br />
  让攻击者与任务防守者共同演进，并用功能组基线稳定更新。
- `U120 / 2026` [MARCH: Multi-Agent Reinforced Check for Hallucination](https://aclanthology.org/2026.acl-long.1828/)<br />
  利用 Solver、Proposer、Checker 间的信息隔离改善证据核验闭环。
- `U124 / 2026` [Cooperative Multi-Agent Vision-Language-Action Models via Reinforced Fine Tuning](https://arxiv.org/abs/2609.36588)<br />
  以示范采集、离线个体信用过滤和在线潜空间优化改善合作 VLA。
- `U125 / 2026` [MRC-MMAgent: Reinforcement Learning for Multi-Role Collaboration in Multimodal Tool-Using Agents](https://arxiv.org/abs/2610.39324)<br />
  把求解、检查和继续执行组织为完整工具交互轨迹，以角色耦合目标优化。
- `U172 / 2026` [Enhancing Social Intelligence in LLMs with Hierarchical Reasoning and Utterance-Level Goal Rewarding](https://arxiv.org/abs/2608.05832v1)<br />
  先生成对话策略再生成话语，结合候选目标得分的方差，决定话语奖励强调目标达成还是策略遵循。
- `U253 / 2026` [ToMAS: A Pilot Failure-Grounded Theory-of-Mind Benchmark from Multi-Agent LLM Failures](https://arxiv.org/abs/2609.16986v1)<br />
  从真实团队出错轨迹提取需推断队友状态的题目，设置人工筛选、标注一致性与隔离留出集。
- `U034 / 2025` [Teaching Language Models to Critique via Reinforcement Learning](https://proceedings.mlr.press/v267/xie25a.html)<br />
  CTRL 用固定代码生成器的后续修复效果训练批评器，优化反馈的实际用途。
- `U035 / 2025` [S2R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning](https://aclanthology.org/2025.acl-long.1104/)<br />
  S2R 先初始化自验证与自修正行为，再用结果级和过程级 RL 强化这些行为。
- `U036 / 2025` [RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning](https://proceedings.mlr.press/v305/boyle25a.html)<br />
  将 R1 风格训练扩展到机器人闭环环境，研究小参数语言模型的具身决策。
- `U122 / 2025` [MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](https://arxiv.org/abs/2506.00555)<br />
  训练分诊和主治两个 GP 角色，组织多专科意见并用课程处理意见不一致。
- `U123 / 2025` [VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ab16cc8f472cf94b282dbff7b99f5f8e-Abstract-Datasets_and_Benchmarks_Track.html)<br />
  以层次具身合作任务评测代理激活、任务规划和轨迹感知，并进行强化微调。

下一个查阅页：[P9E：LLM 与 VLM 设计奖励](/papers/llm-rl/library/reward-design/)。
