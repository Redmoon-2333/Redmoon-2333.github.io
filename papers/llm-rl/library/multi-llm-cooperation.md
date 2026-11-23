---
layout: post
title: P9C：多 LLM 协作训练
permalink: /papers/llm-rl/library/multi-llm-cooperation/
date: 2026-11-24 00:00:00 +0800
topic: llm-rl
categories:
- 论文精读
math: false
reading_prev:
  title: P9B：长程记忆与信用
  url: /papers/llm-rl/library/memory-and-credit/
reading_next:
  title: P9D：工具、过程反馈与多模态
  url: /papers/llm-rl/library/tools-and-multimodal/
---

同一局有多个语言角色时，团队分数怎样交给每位成员，成员更新后又怎样继续一起采样？

[返回 P9 文献导航](/2026/11/23/llm-rl-p9-reading-index/)。先用 [P3](/2026/11/23/llm-rl-p3-credit-assignment/) 的选桥例子分清联合结果和个体反馈。选论文时找两个字段：谁给反馈，谁的参数会变。

## 多 LLM 协作训练（25项）

比较角色交换信息的顺序、各自得到的分数，以及训练时谁能看见全局记录。

- `U020 / 2026` [The Value of Variance: Mitigating Debate Collapse in Multi-Agent Systems via Uncertainty-Driven Policy Optimization](https://proceedings.mlr.press/v306/tang26n.html)<br />
  从成员内部、成员之间和系统输出三个层次刻画不确定性，并用于策略优化。
- `U061 / 2026` [Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in Multi-agent RL](https://arxiv.org/abs/2608.17253v3)<br />
  Co-RL 让参数不共享的模型用队友反馈共同训练，通过队伍异质性减少相关错误。
- `U105 / 2026` [LLM Collaboration with Multi-Agent Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/40487)<br />
  以 MAGRPO 将多轮语言协作表述为合作 MARL。
- `U107 / 2026` [DECOR: Learning to Decompose and Collaborate in Deep Search via Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/chen26gp.html)<br />
  由 Planner、Filter、Answerer 组成深度检索团队，以角色反馈和团队结果联合优化。
- `U108 / 2026` [End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](https://aclanthology.org/2026.acl-long.1399/)<br />
  在异构团队 rollout 间估计相对优势，面向系统成功优化多智能体检索。
- `U109 / 2026` [TeamTR: Trust-Region Fine-Tuning for Multi-Agent LLM Coordination](https://proceedings.mlr.press/v306/xie26o.html)<br />
  每次成员更新后重新采样，并施加成员级信赖域以控制上下文分布漂移。
- `U113 / 2026` [Learning Decentralized LLM Collaboration with Multi-Agent Actor Critic](https://arxiv.org/abs/2601.21972)<br />
  比较集中与去中心 critic 对去中心 LLM 协作训练的影响。
- `U114 / 2026` [Training LLMs for Multi-Agent Collaboration with Verifiable Information Rewards](https://arxiv.org/abs/2610.17893)<br />
  用可验证的信息价值奖励训练证据分享、提问和利用队友信息。
- `U128 / 2026` [Advancing Multi-Agent RAG Systems with Minimalist Reinforcement Learning](https://doi.org/10.65109/QCQC1144)<br />
  规划者拆解多跳问题，执行者检索并处理子问题，MyGO 从采样轨迹中筛选高分结果再优化语言策略。
- `U167 / 2026` [Divide and Cooperate: Role-Decomposed Multi-Agent LLM Training with Cross-Agent Learning Signals](https://arxiv.org/abs/2606.10684v1)<br />
  将检索和回答拆成两个专门角色；生成者遇到证据不足时拒答，其反馈用于训练检索者，检索者反向提供难例。
- `U209 / 2026` [MoRSE: Task-Oriented Multi-Agent System with Mixture of Role-Subtask Experts](https://arxiv.org/abs/2608.09251v1)<br />
  以依赖 DAG 分解任务，按角色及子任务配置 LoRA 专家与语义路由，用分层相对优势拆分路由和执行贡献。
- `U229 / 2026` [RASO: Role-Aware Shared Reflection for Multi-Agent Orchestration in E-Commerce Long-Horizon Planning](https://doi.org/10.1609/icaps.v36i1.42873)<br />
  电商长程规划中，决策角色先写出规则，反思角色根据全局得分及角色级反事实得分检查规则，再共享反馈。
- `U245 / 2026` [SRPO: Setwise Relative Policy Optimization for Multi-Agent LLMs](https://arxiv.org/abs/2609.08452v1)<br />
  把一次环境决策实际消费的多个代理输出视为同一动作，组内共享一个优势，并在联合似然比层面作一次裁剪。
- `U256 / 2026` [TRACER: Turn-level Regret Matching with Inner Reinforcement Credit for Cooperative Multi-LLM Reasoning](https://arxiv.org/abs/2605.28699v1)<br />
  把协作发言拆成控制器决定说或跳过、提议者和评审者生成发言两层，并分层处理信用。
- `U258 / 2026` [Training Small LLMs as Spatial Multi-Agent Policies](https://arxiv.org/abs/2608.01425v1)<br />
  强模型起草符号技能库，随机试跑日志用于合成可行性守卫；各小语言模型学习从可执行技能菜单选项。
- `U018 / 2025` [Advancing Collaborative Debates with Role Differentiation through Multi-Agent Reinforcement Learning](https://aclanthology.org/2025.acl-long.1105/)<br />
  MLC 让序列相关角色嵌入随任务学习，并用角色分化维持成员互补。
- `U019 / 2025` [Reinforce LLM Reasoning through Multi-Agent Reflection](https://proceedings.mlr.press/v267/yuan25l.html)<br />
  DPSDP 把生成—批评—改进表达为多轮决策，利用自生成数据协调训练相关角色。
- `U058 / 2025` [Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs](https://arxiv.org/abs/2510.11062v5)<br />
  AT-GRPO 按智能体和轮次分组，配合工作流 rollout 支持单策略与多策略团队训练。
- `U059 / 2025` [Maestro: Learning to Collaborate via Conditional Listwise Policy Optimization for Multi-Agent LLMs](https://arxiv.org/abs/2511.06134v1)<br />
  Maestro 用并行执行者探索、中央角色合成，以条件列表式策略优化区分决策与理由信号。
- `U060 / 2025` [MarsRL: Advancing Multi-Agent Reasoning System via Reinforcement Learning with Agentic Pipeline Parallelism](https://arxiv.org/abs/2511.11373v1)<br />
  MarsRL 联合训练 Solver、Verifier、Corrector，并以代理流水线处理长轨迹训练。
- `U079 / 2025` [Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/abs/2508.03680v1)<br />
  解耦代理执行与 RL 训练，以统一转移和信用模块支持复杂工作流。
- `U098 / 2025` [MAPoRL2: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](https://aclanthology.org/2025.acl-long.1459/)<br />
  让多个 LLM 先独立回答再讨论，以答案和讨论反馈进行 post-co-training。
- `U099 / 2025` [MARFT: Multi-Agent Reinforcement Fine-Tuning](https://arxiv.org/abs/2504.16129)<br />
  建立适配语言多智能体工作流的博弈建模与强化微调框架。
- `U221 / 2025` [Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System](https://doi.org/10.18653/v1/2025.findings-acl.601)<br />
  生成对话、按任务表现与通信代价打分并筛选后，迭代 SFT 或 DPO，学短而有效的协作通信。
- `U097 / 2024` [Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2410.06101)<br />
  CORY 用先锋与观察者的顺序合作及周期性交换角色进行共同微调。

下一个查阅页：[P9D：工具、过程反馈与多模态](/papers/llm-rl/library/tools-and-multimodal/)。
