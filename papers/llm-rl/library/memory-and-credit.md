---
layout: post
title: P9B：长程记忆与信用
permalink: /papers/llm-rl/library/memory-and-credit/
date: 2026-11-24 00:00:00 +0800
topic: llm-rl
categories:
- 论文精读
math: false
reading_prev:
  title: P9A：验证器与推理训练
  url: /papers/llm-rl/library/verifiers-and-reasoning/
reading_next:
  title: P9C：多 LLM 协作训练
  url: /papers/llm-rl/library/multi-llm-cooperation/
---

模型只能带走一部分对话时，要决定保留哪些事实，还要决定哪些写入动作得到训练信号。

[返回 P9 文献导航](/2026/11/23/llm-rl-p9-reading-index/)。从 L01 的摘要与执行联合训练读起，再对照 L02 的来源记录和 L03 的事实覆写。三篇的做法见 [P6](/2026/11/23/llm-rl-p6-memory/)。

## 长程记忆与信用（23项）

先看记忆允许保留几条旧信息，再看摘要、执行与来源选择各收到什么信号。

- `L01 / 2026` [CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents](https://arxiv.org/abs/2607.05378v2)<br />
  执行与摘要联合训练；PPO、Critic、跨段优势，详见 P6。
- `L02 / 2026` [ECHO: Prune to Act, Trace to Learn with Selective Turn Memory in Agentic RL](https://arxiv.org/html/2606.31650)<br />
  来源索引记忆与正信用 token 掩码；持久归档与当前上下文分开，详见 P6。
- `L03 / 2026` [Supersede: Diagnosing and Training the Memory-Update Gap in LLM Agents](https://arxiv.org/abs/2606.27472)<br />
  限长笔记中的当前事实更新；诊断与合成到真实的训练示范，详见 P6。
- `L04 / 2026` [MEMENTO: Learning Context Management for Long Context Processing Agents](https://arxiv.org/abs/2611.00244)<br />
  通过笔记管理长上下文；关注监督起步、奖励与信息保留各自的作用。
- `L05 / 2026` [Forget Freely, Grep Later: Lossless, Index-Free Memory for Long-Horizon LLM Agents](https://arxiv.org/abs/2611.25123)<br />
  持久全文日志加检索的记忆路线；比较信息完整性、检索和工作上下文成本。
- `L06 / 2026` [TRACER: Per-Tool Context Retention for LLM Agents via Consequence-Attributed Reinforcement Learning](https://arxiv.org/abs/2608.29363)<br />
  逐工具决定上下文保留比例；检查后果预测器与重调用成本。
- `L07 / 2026` [Escaping the Context Bottleneck: Active Context Curation for LLM Agents via Reinforcement Learning](https://arxiv.org/abs/2604.11462)<br />
  训练上下文管理者、冻结任务执行器；与双模块共同训练分开。
- `L08 / 2026` [Context Language Models](https://arxiv.org/html/2609.37725)<br />
  将工作上下文视为可编辑对象；关注编辑动作、工具接口和更新策略。
- `L16 / 2026` [When Credit Turns Against Tools: Transition-Aware Credit Assignment in Agentic Reinforcement Learning](https://arxiv.org/abs/2611.11330v1)<br />
  工具类别信用、优势投影、收益对照与监督重放，详见 P7。
- `L17 / 2026` [Coverage, Not Targeting: A Structural Regime in Multi-Turn Agent Credit Assignment](https://arxiv.org/html/2609.02417)<br />
  低验证器密度中的覆盖与集中度控制；首次更新与完整训练分开，详见 P7。
- `L18 / 2026` [VICT: Verifier-Instrumented Credit Tracing for Long-Horizon LLM Agent Reinforcement Learning](https://arxiv.org/abs/2608.28128v1)<br />
  可执行验证器内部证据与步骤追踪；需要白盒、可追溯条件。
- `L19 / 2026` [RTPO: Reverse-Turn Policy Optimization for Stabilizing Agentic RL Training](https://arxiv.org/abs/2608.18682v3)<br />
  逆序轮次优化；关注状态恢复、树形续跑与更新顺序。
- `L20 / 2026` [UniOPSD: Unifying Outcome and Hindsight Feedback for Agentic Reinforcement Learning](https://arxiv.org/abs/2609.34810)<br />
  结果反馈与成功同伴的事后反馈结合；比较新增信息和蒸馏预算。
- `L21 / 2026` [From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2604.09459)<br />
  从推理到 Agent 的信用分配综述；用于查干预定义与可识别性。
- `U003 / 2026` [Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning](https://proceedings.mlr.press/v306/cheng26e.html)<br />
  把多条 rollout 聚合成状态转移图，以到目标的图距离变化估计步骤优势。
- `U030 / 2026` [Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning](https://aclanthology.org/2026.acl-long.583/)<br />
  以结果反馈训练记忆管理与回答代理，学习增、改、删、保留和使用记忆。
- `U031 / 2026` [From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory](https://proceedings.mlr.press/v306/cai26b.html)<br />
  MemoPilot 为冻结玩家训练记忆更新器，用跨轮奖励改善持续交互决策。
- `U044 / 2026` [Turning Multi-Agent Communication into Token-Level Credit with On Policy Self Distillation](https://arxiv.org/abs/2611.16206v1)<br />
  OPSD4MAS 通过信息过滤与 on-policy self-distillation 把未来通信转为 token 级信用。
- `U045 / 2026` [LLM Value Functions for Credit Assignment in Agentic RL](https://arxiv.org/abs/2610.15840v1)<br />
  用冻结 LLM 估计并校准策略相关状态价值，再以价值差分配步骤信用。
- `U290 / 2026` [Credit Without Ground Truth: Claim-Matched Evaluation of Step-Level Credit with Finite Executed Replay](https://arxiv.org/abs/2608.19760v3)<br />
  用两套独立续跑，在 ALFWorld 将信用数值、动作排序、分支修复收益和位置选择增量分开测量。
- `U291 / 2026` [CREDO: Variance-Guided Rubric Evolution for Replay-Corrected Credit Assignment](https://arxiv.org/abs/2609.24174v1)<br />
  冻结裁判把状态转移编码成特征，信用头作初估；按已知概率抽取双边续跑，以加权残差校正预测。
- `U292 / 2026` [The Trace Is the State: Exact Credit Assignment for LLM Agent Teams](https://arxiv.org/abs/2603.06859v4)<br />
  C3 在同一团队前缀替换一条消息，再采样下游；用候选消息的续跑回报差估计信用并作为训练优势。
- `U178 / 2025` [Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978v3)<br />
  把完整轨迹组成宏组，跨轨迹按重复环境状态把动作编入微组，结合两种相对优势估计步骤信用。

下一个查阅页：[P9C：多 LLM 协作训练](/papers/llm-rl/library/multi-llm-cooperation/)。
