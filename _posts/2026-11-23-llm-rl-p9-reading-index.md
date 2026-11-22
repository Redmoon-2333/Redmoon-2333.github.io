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
excerpt: 把182项LLM＋RL交叉文献按11条机制线归位，先读主线再按研究问题查原文，明确目前仍待选择的三个候选。
math: true
slug: llm-rl-p9-reading-index
reading_prev:
  title: E11 · 当前三个候选，下一组实验各要区分什么
  url: /2026/11/23/llm-rl-e11-next-experiments/
---

当前研究集中比较验证器错误、长程压缩与信用覆盖三个切口。P1–P8讲清它们需要的概念与代表方法，实践E1–E11解释已有实验和待验证问题；本篇是按需查阅的文献入口。

## 1. 先读哪些论文

| 想弄懂的问题 | 首读 | 接着读 |
|---|---|---|
| 奖励怎样更新语言策略 | PPO、DeepSeekMath | DeepSeek-R1、DAPO；见P1/P2 |
| 团队得分怎样变成个体信号 | COMA、LLM-MCA、LCA | CREDO、MARFT、TeamTR；见P3 |
| 高分为什么可能偏离目标 | Optimistic Verifiers | Reward Suite Is Leaky、Wrong the Same Way；见P4/P5 |
| 历史太长、事实被更新怎么办 | CompactionRL | ECHO、Supersede；见P6 |
| 关键步骤与保护动作怎样学习 | Coverage, Not Targeting | TACA；见P7 |
| 工具、机器人与多模态如何组合 | ReAct、Search-R1、SayCan | Eureka、ReMAC、cooperative VLA；见P8 |

基础经典方法用于定义，核心近邻用于界定候选。研究优先级与会议名次是不同的排序。

## 2. 这份目录怎样计数

旧合并库307项与新增28项之间，L27与U071是同一作品，因此逻辑合并为334项。其中筛出的直接LLM强化学习条目121项、LLM辅助传统策略或小控制器编排等交叉条目61项，共182项列在下方。其余152项按基础、相邻方法或撤回留史保留，不混入本专题主列表。

同篇多个PDF版本不增加作品数。目录条目表示查阅覆盖；正文深入解释的是各篇承担概念与机制职责的代表论文。原文固定版本、具体任务和指标以相应正文引用为准。

编号U/L用于跨目录查找，不表示阅读次序。下表的“阅读抓手”是简短检索提示，详细方法仍读所链接的原文及P1–P8。

## 不完美验证器与 RLVR（8项）

对应P4、P5。优先关注：训练奖励、真实正确性与跨题错误结构。

| 编号 / 年份 | 原文 | 阅读抓手 |
|---|---|---|
| L09 / 2026 | [Wrong the Same Way: Cross-Prompt Sharing Both Suppresses and Amplifies Reward Hacking in RLVR](https://arxiv.org/abs/2611.13218v1) | 共享参数对跨题错误接受的抑制与放大；可控小模型机制，详见 P4/P5。 |
| L10 / 2026 | [Optimistic Verifiers Get Hacked: Reward Hacking Under an Exact Process Oracle](https://arxiv.org/html/2611.21538v1) | 可执行规则解释的精确参照与乐观代理；合成和公共策略分开，详见 P4。 |
| L11 / 2026 | [Checking the Checkers: A Synthesis of Verifier Noise and Robustness in RLVR](https://arxiv.org/html/2610.17824v1) | 验证器噪声与鲁棒性综述；按错误结构查找训练期干预。 |
| L12 / 2026 | [When the Reward Suite Is Leaky: A Preregistered Causal Contrast of Natural Verifier False Positives in RLVR](https://arxiv.org/abs/2607.11022) | 自然代码测试漏检的配对训练与审计；区分漏洞暴露与留出损伤，详见 P4。 |
| L13 / 2026 | [Shortcuts and Boundaries: Verifiable RL Misalignment and Contextual Safety Learning](https://arxiv.org/html/2611.12865v1) | 可验证奖励与上下文约束的冲突；检查合规反馈与迁移条件。 |
| L14 / 2026 | [LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking](https://arxiv.org/abs/2604.15149) | 规则诱导与实例捷径；考察同构扰动测试依赖的领域知识。 |
| L15 / 2026 | [Quantifying Empirical Compute-Supervision Tradeoffs in RLVR](https://arxiv.org/html/2605.25252) | 计算量与有噪监督的经验比较；区分随机错误与可触发错误。 |
| U067 / 2026 | [Verifier Errors in RLVR: Reward Hacking, Limits of Feedback, and Selective Control](https://arxiv.org/abs/2609.35677v1) | 分析验证器接受错误答案时奖励与真实正确率的分离，并讨论额外审计反馈。 |

## 长程记忆与信用（23项）

对应P3、P6、P7。优先关注：压缩、来源、事实更新、覆盖及可执行回放。

| 编号 / 年份 | 原文 | 阅读抓手 |
|---|---|---|
| L01 / 2026 | [CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents](https://arxiv.org/abs/2607.05378v2) | 执行与摘要联合训练；PPO、Critic、跨段优势，详见 P6。 |
| L02 / 2026 | [ECHO: Prune to Act, Trace to Learn with Selective Turn Memory in Agentic RL](https://arxiv.org/html/2606.31650) | 来源索引记忆与正信用 token 掩码；持久归档与当前上下文分开，详见 P6。 |
| L03 / 2026 | [Supersede: Diagnosing and Training the Memory-Update Gap in LLM Agents](https://arxiv.org/abs/2606.27472) | 限长笔记中的当前事实更新；诊断与合成到真实的训练示范，详见 P6。 |
| L04 / 2026 | [MEMENTO: Learning Context Management for Long Context Processing Agents](https://arxiv.org/abs/2611.00244) | 通过笔记管理长上下文；关注监督起步、奖励与信息保留各自的作用。 |
| L05 / 2026 | [Forget Freely, Grep Later: Lossless, Index-Free Memory for Long-Horizon LLM Agents](https://arxiv.org/abs/2611.25123) | 持久全文日志加检索的记忆路线；比较信息完整性、检索和工作上下文成本。 |
| L06 / 2026 | [TRACER: Per-Tool Context Retention for LLM Agents via Consequence-Attributed Reinforcement Learning](https://arxiv.org/abs/2608.29363) | 逐工具决定上下文保留比例；检查后果预测器与重调用成本。 |
| L07 / 2026 | [Escaping the Context Bottleneck: Active Context Curation for LLM Agents via Reinforcement Learning](https://arxiv.org/abs/2604.11462) | 训练上下文管理者、冻结任务执行器；与双模块共同训练分开。 |
| L08 / 2026 | [Context Language Models](https://arxiv.org/html/2609.37725) | 将工作上下文视为可编辑对象；关注编辑动作、工具接口和更新策略。 |
| L16 / 2026 | [When Credit Turns Against Tools: Transition-Aware Credit Assignment in Agentic Reinforcement Learning](https://arxiv.org/abs/2611.11330v1) | 工具类别信用、优势投影、收益对照与监督重放，详见 P7。 |
| L17 / 2026 | [Coverage, Not Targeting: A Structural Regime in Multi-Turn Agent Credit Assignment](https://arxiv.org/html/2609.02417) | 低验证器密度中的覆盖与集中度控制；首次更新与完整训练分开，详见 P7。 |
| L18 / 2026 | [VICT: Verifier-Instrumented Credit Tracing for Long-Horizon LLM Agent Reinforcement Learning](https://arxiv.org/abs/2608.28128v1) | 可执行验证器内部证据与步骤追踪；需要白盒、可追溯条件。 |
| L19 / 2026 | [RTPO: Reverse-Turn Policy Optimization for Stabilizing Agentic RL Training](https://arxiv.org/abs/2608.18682v3) | 逆序轮次优化；关注状态恢复、树形续跑与更新顺序。 |
| L20 / 2026 | [UniOPSD: Unifying Outcome and Hindsight Feedback for Agentic Reinforcement Learning](https://arxiv.org/abs/2609.34810) | 结果反馈与成功同伴的事后反馈结合；比较新增信息和蒸馏预算。 |
| L21 / 2026 | [From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2604.09459) | 从推理到 Agent 的信用分配综述；用于查干预定义与可识别性。 |
| U003 / 2026 | [Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning](https://proceedings.mlr.press/v306/cheng26e.html) | 把多条 rollout 聚合成状态转移图，以到目标的图距离变化估计步骤优势。 |
| U030 / 2026 | [Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning](https://aclanthology.org/2026.acl-long.583/) | 以结果反馈训练记忆管理与回答代理，学习增、改、删、保留和使用记忆。 |
| U031 / 2026 | [From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory](https://proceedings.mlr.press/v306/cai26b.html) | MemoPilot 为冻结玩家训练记忆更新器，用跨轮奖励改善持续交互决策。 |
| U044 / 2026 | [Turning Multi-Agent Communication into Token-Level Credit with On Policy Self Distillation](https://arxiv.org/abs/2611.16206v1) | OPSD4MAS 通过信息过滤与 on-policy self-distillation 把未来通信转为 token 级信用。 |
| U045 / 2026 | [LLM Value Functions for Credit Assignment in Agentic RL](https://arxiv.org/abs/2610.15840v1) | 用冻结 LLM 估计并校准策略相关状态价值，再以价值差分配步骤信用。 |
| U290 / 2026 | [Credit Without Ground Truth: Claim-Matched Evaluation of Step-Level Credit with Finite Executed Replay](https://arxiv.org/abs/2608.19760v3) | 用两套独立续跑，在 ALFWorld 将信用数值、动作排序、分支修复收益和位置选择增量分开测量。 |
| U291 / 2026 | [CREDO: Variance-Guided Rubric Evolution for Replay-Corrected Credit Assignment](https://arxiv.org/abs/2609.24174v1) | 冻结裁判把状态转移编码成特征，信用头作初估；按已知概率抽取双边续跑，以加权残差校正预测。 |
| U292 / 2026 | [The Trace Is the State: Exact Credit Assignment for LLM Agent Teams](https://arxiv.org/abs/2603.06859v4) | C3 在同一团队前缀替换一条消息，再采样下游；用候选消息的续跑回报差估计信用并作为训练优势。 |
| U178 / 2025 | [Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978v3) | 把完整轨迹组成宏组，跨轨迹按重复环境状态把动作编入微组，结合两种相对优势估计步骤信用。 |

## 多 LLM 协作训练（25项）

对应P3。优先关注：角色交互、共享模型、更新次序与团队信号。

| 编号 / 年份 | 原文 | 阅读抓手 |
|---|---|---|
| U020 / 2026 | [The Value of Variance: Mitigating Debate Collapse in Multi-Agent Systems via Uncertainty-Driven Policy Optimization](https://proceedings.mlr.press/v306/tang26n.html) | 从成员内部、成员之间和系统输出三个层次刻画不确定性，并用于策略优化。 |
| U061 / 2026 | [Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in Multi-agent RL](https://arxiv.org/abs/2608.17253v3) | Co-RL 让参数不共享的模型用队友反馈共同训练，通过队伍异质性减少相关错误。 |
| U105 / 2026 | [LLM Collaboration with Multi-Agent Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/40487) | 以 MAGRPO 将多轮语言协作表述为合作 MARL。 |
| U107 / 2026 | [DECOR: Learning to Decompose and Collaborate in Deep Search via Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/chen26gp.html) | 由 Planner、Filter、Answerer 组成深度检索团队，以角色反馈和团队结果联合优化。 |
| U108 / 2026 | [End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](https://aclanthology.org/2026.acl-long.1399/) | 在异构团队 rollout 间估计相对优势，面向系统成功优化多智能体检索。 |
| U109 / 2026 | [TeamTR: Trust-Region Fine-Tuning for Multi-Agent LLM Coordination](https://proceedings.mlr.press/v306/xie26o.html) | 每次成员更新后重新采样，并施加成员级信赖域以控制上下文分布漂移。 |
| U113 / 2026 | [Learning Decentralized LLM Collaboration with Multi-Agent Actor Critic](https://arxiv.org/abs/2601.21972) | 比较集中与去中心 critic 对去中心 LLM 协作训练的影响。 |
| U114 / 2026 | [Training LLMs for Multi-Agent Collaboration with Verifiable Information Rewards](https://arxiv.org/abs/2610.17893) | 用可验证的信息价值奖励训练证据分享、提问和利用队友信息。 |
| U128 / 2026 | [Advancing Multi-Agent RAG Systems with Minimalist Reinforcement Learning](https://doi.org/10.65109/QCQC1144) | 规划者拆解多跳问题，执行者检索并处理子问题，MyGO 从采样轨迹中筛选高分结果再优化语言策略。 |
| U167 / 2026 | [Divide and Cooperate: Role-Decomposed Multi-Agent LLM Training with Cross-Agent Learning Signals](https://arxiv.org/abs/2606.10684v1) | 将检索和回答拆成两个专门角色；生成者遇到证据不足时拒答，其反馈用于训练检索者，检索者反向提供难例。 |
| U209 / 2026 | [MoRSE: Task-Oriented Multi-Agent System with Mixture of Role-Subtask Experts](https://arxiv.org/abs/2608.09251v1) | 以依赖 DAG 分解任务，按角色及子任务配置 LoRA 专家与语义路由，用分层相对优势拆分路由和执行贡献。 |
| U229 / 2026 | [RASO: Role-Aware Shared Reflection for Multi-Agent Orchestration in E-Commerce Long-Horizon Planning](https://doi.org/10.1609/icaps.v36i1.42873) | 电商长程规划中，决策角色先写出规则，反思角色根据全局得分及角色级反事实得分检查规则，再共享反馈。 |
| U245 / 2026 | [SRPO: Setwise Relative Policy Optimization for Multi-Agent LLMs](https://arxiv.org/abs/2609.08452v1) | 把一次环境决策实际消费的多个代理输出视为同一动作，组内共享一个优势，并在联合似然比层面作一次裁剪。 |
| U256 / 2026 | [TRACER: Turn-level Regret Matching with Inner Reinforcement Credit for Cooperative Multi-LLM Reasoning](https://arxiv.org/abs/2605.28699v1) | 把协作发言拆成控制器决定说或跳过、提议者和评审者生成发言两层，并分层处理信用。 |
| U258 / 2026 | [Training Small LLMs as Spatial Multi-Agent Policies](https://arxiv.org/abs/2608.01425v1) | 强模型起草符号技能库，随机试跑日志用于合成可行性守卫；各小语言模型学习从可执行技能菜单选项。 |
| U018 / 2025 | [Advancing Collaborative Debates with Role Differentiation through Multi-Agent Reinforcement Learning](https://aclanthology.org/2025.acl-long.1105/) | MLC 让序列相关角色嵌入随任务学习，并用角色分化维持成员互补。 |
| U019 / 2025 | [Reinforce LLM Reasoning through Multi-Agent Reflection](https://proceedings.mlr.press/v267/yuan25l.html) | DPSDP 把生成—批评—改进表达为多轮决策，利用自生成数据协调训练相关角色。 |
| U058 / 2025 | [Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs](https://arxiv.org/abs/2510.11062v5) | AT-GRPO 按智能体和轮次分组，配合工作流 rollout 支持单策略与多策略团队训练。 |
| U059 / 2025 | [Maestro: Learning to Collaborate via Conditional Listwise Policy Optimization for Multi-Agent LLMs](https://arxiv.org/abs/2511.06134v1) | Maestro 用并行执行者探索、中央角色合成，以条件列表式策略优化区分决策与理由信号。 |
| U060 / 2025 | [MarsRL: Advancing Multi-Agent Reasoning System via Reinforcement Learning with Agentic Pipeline Parallelism](https://arxiv.org/abs/2511.11373v1) | MarsRL 联合训练 Solver、Verifier、Corrector，并以代理流水线处理长轨迹训练。 |
| U079 / 2025 | [Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/abs/2508.03680v1) | 解耦代理执行与 RL 训练，以统一转移和信用模块支持复杂工作流。 |
| U098 / 2025 | [MAPoRL2: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](https://aclanthology.org/2025.acl-long.1459/) | 让多个 LLM 先独立回答再讨论，以答案和讨论反馈进行 post-co-training。 |
| U099 / 2025 | [MARFT: Multi-Agent Reinforcement Fine-Tuning](https://arxiv.org/abs/2504.16129) | 建立适配语言多智能体工作流的博弈建模与强化微调框架。 |
| U221 / 2025 | [Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System](https://doi.org/10.18653/v1/2025.findings-acl.601) | 生成对话、按任务表现与通信代价打分并筛选后，迭代 SFT 或 DPO，学短而有效的协作通信。 |
| U097 / 2024 | [Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2410.06101) | CORY 用先锋与观察者的顺序合作及周期性交换角色进行共同微调。 |

## 推理 RL、测试时学习与自博弈（24项）

对应P2、P8。优先关注：组优势、探索、奖励来源、数据筛选与共享更新。

| 编号 / 年份 | 原文 | 阅读抓手 |
|---|---|---|
| L22 / 2026 | [When Does Exploration Prevent Collapse? Three Regimes and the Path of the Reward in Self-Consistency Training](https://arxiv.org/abs/2611.01307) | 自一致奖励下探索与坍缩的动力学；归一化前后放探索有不同效果。 |
| L23 / 2026 | [When the Majority Votes Wrong, the Intervention Timing for Test-Time Reinforcement Learning Hides in the Extinction Window](https://arxiv.org/html/2605.19444v1) | 多数答案可能错误时的测试时训练干预；考察少数正确信号的保留。 |
| L24 / 2026 | [Rewarding Reasoning, Not Answers: Fixing and Bounding Test-Time Reinforcement Learning on Medical QA](https://arxiv.org/html/2609.16660) | 小答案空间的测试时奖励；检查过程评分器的领域训练条件。 |
| L25 / 2026 | [Entropy-Regularized Rank-Masked Policy Optimization for Test-Time Reinforcement Learning in Code Generation](https://arxiv.org/html/2609.09135) | 代码测试时训练的共识奖励、秩掩码与熵；与在线多轮任务分开。 |
| L26 / 2026 | [Learning to Solve, Forgetting to Retain: Correct-Set Turnover in RLVR](https://arxiv.org/html/2606.03087v2) | 已答对题集合在共享更新中更替；检查复习采样与保留成本。 |
| L27 / 2026 | [Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL](https://arxiv.org/html/2605.22217) | 自博弈中的数据门控与奖励锚定；归档 U071 为同一作品。 |
| L28 / 2026 | [Enabling Token-wise Importance Sampling in Off-Policy RLVR with Proximal KL Regularization](https://arxiv.org/abs/2610.18738v1) | 离策略 RLVR 的 token 重要性修正与 KL；关注理论条件和数据陈旧度。 |
| U017 / 2026 | [RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments](https://proceedings.mlr.press/v306/zeng26j.html) | RLVE 用可程序生成且答案可验证的环境，根据策略能力调节问题难度。 |
| U037 / 2026 | [Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs](https://aclanthology.org/2026.acl-long.847/) | TPAW 让当前策略与历史检查点合作/竞争，并分别调整回答和玩家的训练权重。 |
| U056 / 2026 | [When Does Closed-Loop Curriculum Adaptation Help GRPO? An Analysis on Countdown](https://arxiv.org/abs/2610.39056v1) | 用难度分桶探测调整课程采样进度与宽度，保持任务有挑战且仍可学习。 |
| U072 / 2026 | [More Convincing, Not More Correct: Self-Play Reward Hacking of Reference-Free LLM Judges](https://arxiv.org/abs/2607.05904v1) | 以独立隐藏答案审计检查自博弈训练是否只提高无参考裁判评分。 |
| U118 / 2026 | [Strat-Reasoner: Reinforcing Strategic Reasoning of LLMs in Multi-Agent Games](https://proceedings.mlr.press/v306/he26u.html) | 把其他角色的推理纳入递归策略思考，用集中 CoT 比较反馈训练语言策略。 |
| U278 / 2026 | [Plan Then Action: High-Level Planning Guidance Reinforcement Learning for LLM Reasoning](https://proceedings.mlr.press/v306/dou26c.html) | 先把推理链压成高层计划做冷启动，再用轨迹组内的奖励同时优化计划质量与答案，更新语言策略。 |
| U038 / 2025 | [Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search](https://proceedings.mlr.press/v267/shen25o.html) | Satori 用带动作的思维链与两阶段训练，将反思和重新搜索内化到单个策略。 |
| U039 / 2025 | [T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](https://proceedings.mlr.press/v267/hou25e.html) | T1 结合含试错与自验证的数据初始化、RL 探索及推理预算扩展。 |
| U073 / 2025 | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948v2) | 用 RL 激励可验证推理能力，并将所得推理模式用于小模型的能力迁移。 |
| U074 / 2025 | [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476v2) | DAPO 讨论大规模语言 RL 的裁剪、采样与训练实现细节，并公开训练系统。 |
| U075 / 2025 | [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783v2) | 区分基座模型能力与 RL 目标的贡献，指出 GRPO 长度相关偏差并提出 Dr. GRPO。 |
| U080 / 2025 | [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335v3) | 同一模型承担出题者与求解者，通过可执行环境验证生成任务与答案来共同演进。 |
| U248 / 2025 | [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](https://arxiv.org/abs/2502.18449v2) | 由开源软件修改历史构造问题与参考补丁，用生成修改与参考修改的相似度反馈训练代码推理模型。 |
| U236 / 2024 | [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267v3) | 以模型生成的偏好反馈训练语言策略，并与使用人类偏好标签的强化学习在相同任务下比较。 |
| U083 / 2023 | [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290v3) | 通过偏好分类损失直接优化策略，作为传统显式奖励模型加 RL 流程的相邻路径。 |
| U084 / 2022 | [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073v1) | 以原则驱动的批评修订构造监督数据，再用 AI 偏好训练奖励模型并进行 RLAIF。 |
| U282 / 2022 | [Training language models to follow instructions with human feedback](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) | 人工示范用于指令微调，人工排序用于训练奖励模型，再用 PPO 依据奖励模型的输出更新语言策略。 |

## 工具、检索与多轮 Agent（25项）

对应P8。优先关注：生成动作与工具观察的边界，检索和环境交互预算。

| 编号 / 年份 | 原文 | 阅读抓手 |
|---|---|---|
| U006 / 2026 | [D-CORE: Incentivizing Task Decomposition in Large Reasoning Models for Complex Tool Use](https://proceedings.mlr.press/v306/xu26bn.html) | D-CORE 先通过自蒸馏强化子任务分解，再用多样性相关 RL 改善复杂工具推理。 |
| U007 / 2026 | [Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](https://aclanthology.org/2026.acl-long.318/) | STEP-HRL 用全局子任务完成信息与局部进度摘要构成紧凑的步骤级转移。 |
| U009 / 2026 | [ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration](https://proceedings.mlr.press/v306/su26d.html) | 训练小型语言调度器选择模型和工具，奖励同时考虑结果、效率与用户偏好。 |
| U032 / 2026 | [Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning](https://proceedings.mlr.press/v306/wang26jh.html) | AWM 生成由代码和数据库驱动的可执行工具环境，为多轮 RL 提供状态与反馈。 |
| U052 / 2026 | [EvoRouter: Cost-Efficient Multi-Agent Routing via Insight-Guided On-Policy Distillation](https://arxiv.org/abs/2610.35690v1) | 用跨任务执行反馈提炼路由经验，再结合特权教师蒸馏与成本感知 RL 训练路由器。 |
| U063 / 2026 | [From Proprietary to Open-Source: Bridging the Distribution Gap via Multi-Agent Protocol Distillation in Agentic Search](https://arxiv.org/abs/2607.24280v1) | MAPD 用结构化、风格归一的协议连接闭源教师与开放学生，再结合蒸馏和 RL。 |
| U064 / 2026 | [One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL](https://arxiv.org/abs/2608.12253v2) | 分析单一 LLM 模拟器的行为模式收缩，并比较行为采样与可训练模拟器种群共同训练。 |
| U068 / 2026 | [Policy and World Modeling Co-Training for Language Agents](https://arxiv.org/abs/2606.02388v2) | PaW 利用 on-policy 转移的下一观测作世界建模辅助监督，并平衡 RL 与预测损失。 |
| U117 / 2026 | [DLM: Unified Decision Language Model for Offline Multi-Agent Sequential Decision Making](https://proceedings.mlr.press/v306/zhang26v.html) | 将离线多智能体轨迹转为对话序列，经过监督训练与轻量奖励 GRPO 两阶段学习。 |
| U132 / 2026 | [AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning](https://openreview.net/forum?id=ZgCCDwcGwn) | 在交互训练早期限制行动轮数，随后逐步延长允许的交互跨度，用同一框架训练多种环境中的语言策略。 |
| U181 / 2026 | [HiRS-Agent: A Hierarchical Multi-Agent System for Reliable Long-Horizon Remote Sensing Task Solving](https://doi.org/10.1145/3767308.3835311) | 遥感长程工作流中，管理角色负责规划、路由和核验，专家角色操作工具，分别接收轨迹奖励和步骤奖励。 |
| U243 / 2026 | [SLMs as Multi-Agent Routers: A Progressive SFT and Reinforcement Learning Approach](https://arxiv.org/abs/2608.00030v1) | 小模型为查询选择专门检索代理并输出调用参数，奖励同时考查主题匹配与实际检索内容相关性。 |
| U270 / 2026 | [Does Reinforcement Fine-Tuning Improve Generalization of LLM Agents? An Empirical Study](https://proceedings.mlr.press/v306/xi26b.html) | 在任务难度、跨环境和连续训练三个设置下比较智能体强化微调的迁移，按观察与行动接口分析环境差异。 |
| U271 / 2026 | [GLARE: Scalable Neuro-Symbolic Reward Shaping for LLM Agents via Group-Level Automata](https://proceedings.mlr.press/v306/yan26k.html) | 将轨迹事件符号化，借助时序逻辑构造自动机，以状态迁移驱动进度奖励，并用组相对策略更新语言智能体。 |
| U005 / 2025 | [Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](https://proceedings.mlr.press/v267/hu25q.html) | GLIDER 用离线层次 RL 连接高层抽象计划与低层控制，研究长程决策和迁移。 |
| U048 / 2025 | [CoDA: A Context-Decoupled Hierarchical Agent with Reinforcement Learning](https://arxiv.org/abs/2512.12716v1) | 同一 LLM 分别承担上下文隔离的 Planner 与 Executor，以轨迹奖励联合优化。 |
| U049 / 2025 | [Learning to Orchestrate Agents in Natural Language with the Conductor](https://arxiv.org/abs/2512.04388v5) | Conductor 用 RL 学习工作池中的通信关系和定向指令，并适配变化的成员集合。 |
| U050 / 2025 | [Controlling Performance and Budget of a Centralized Multi-agent LLM System with Reinforcement Learning](https://arxiv.org/abs/2511.02755v1) | 中央语言控制器按预算选择专家，联合优化任务质量和推理费用。 |
| U076 / 2025 | [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516v5) | 让模型在推理中学习多次检索查询，以结果奖励优化并屏蔽检索内容的训练 token。 |
| U077 / 2025 | [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536v2) | 通过多轮实时代码执行与结果反馈，学习何时、如何将工具插入长推理。 |
| U078 / 2025 | [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958v1) | 系统比较工具使用奖励的类型、尺度、粒度和时间结构，再用 GRPO 训练工具策略。 |
| U228 / 2025 | [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2504.20073v2) | 在数种交互环境研究语言策略训练中奖励变异收缩、梯度尖峰与行为退化，比较轨迹筛选等稳定化方法。 |
| U265 / 2025 | [Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning](https://arxiv.org/abs/2511.14460v2) | 以交互步表征轨迹，将上下文管理、环境接口和策略优化分开，使工具反馈下的多轮训练可以采用不同的信用分配粒度。 |
| U203 / 2024 | [LM2: A Simple Society of Language Models Solves Complex Reasoning](https://arxiv.org/abs/2404.02255v1) | 分解器逐步提出子问题，求解器回答，验证器分类错误。错误类型及其发生位置形成逐步奖励，再用 PPO 更新分解器。 |
| U220 / 2023 | [OpenAGI: When LLM Meets Domain Experts](https://arxiv.org/abs/2304.04370v6) | 由 LLM 按自然语言任务选择并组合专门模型，再利用任务结果提供反馈，强化 LLM 的组合决策。 |

## 过程反馈、安全与多模态（14项）

对应P4、P7、P8。优先关注：反馈的可验证部分、动作空间、视觉观测与策略更新。

| 编号 / 年份 | 原文 | 阅读抓手 |
|---|---|---|
| U069 / 2026 | [Verifiable Process Rewards for Agentic Reasoning](https://arxiv.org/abs/2605.10325v3) | 把搜索、约束或后验 oracle 转为动作级过程验证器，缓解长程任务的稀疏反馈。 |
| U070 / 2026 | [Verifiable Counterfactual Supervision for Process Reward Models](https://arxiv.org/abs/2605.02395v3) | 在符号正确链中注入受控错误并重算后续状态，构造可验证的首错位置监督。 |
| U115 / 2026 | [Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models](https://proceedings.mlr.press/v306/liu26bn.html) | 同一策略以攻防角色在线自博弈，共同适应变化的对手。 |
| U116 / 2026 | [AdvEvo-MARL: Shaping Internalized Safety through Adversarial Co-Evolution in Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/pan26p.html) | 让攻击者与任务防守者共同演进，并用功能组基线稳定更新。 |
| U120 / 2026 | [MARCH: Multi-Agent Reinforced Check for Hallucination](https://aclanthology.org/2026.acl-long.1828/) | 利用 Solver、Proposer、Checker 间的信息隔离改善证据核验闭环。 |
| U124 / 2026 | [Cooperative Multi-Agent Vision-Language-Action Models via Reinforced Fine Tuning](https://arxiv.org/abs/2609.36588) | 以示范采集、离线个体信用过滤和在线潜空间优化改善合作 VLA。 |
| U125 / 2026 | [MRC-MMAgent: Reinforcement Learning for Multi-Role Collaboration in Multimodal Tool-Using Agents](https://arxiv.org/abs/2610.39324) | 把求解、检查和继续执行组织为完整工具交互轨迹，以角色耦合目标优化。 |
| U172 / 2026 | [Enhancing Social Intelligence in LLMs with Hierarchical Reasoning and Utterance-Level Goal Rewarding](https://arxiv.org/abs/2608.05832v1) | 先生成对话策略再生成话语，结合候选目标得分的方差，决定话语奖励强调目标达成还是策略遵循。 |
| U253 / 2026 | [ToMAS: A Pilot Failure-Grounded Theory-of-Mind Benchmark from Multi-Agent LLM Failures](https://arxiv.org/abs/2609.16986v1) | 从真实团队出错轨迹提取需推断队友状态的题目，设置人工筛选、标注一致性与隔离留出集。 |
| U034 / 2025 | [Teaching Language Models to Critique via Reinforcement Learning](https://proceedings.mlr.press/v267/xie25a.html) | CTRL 用固定代码生成器的后续修复效果训练批评器，优化反馈的实际用途。 |
| U035 / 2025 | [S2R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning](https://aclanthology.org/2025.acl-long.1104/) | S2R 先初始化自验证与自修正行为，再用结果级和过程级 RL 强化这些行为。 |
| U036 / 2025 | [RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning](https://proceedings.mlr.press/v305/boyle25a.html) | 将 R1 风格训练扩展到机器人闭环环境，研究小参数语言模型的具身决策。 |
| U122 / 2025 | [MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](https://arxiv.org/abs/2506.00555) | 训练分诊和主治两个 GP 角色，组织多专科意见并用课程处理意见不一致。 |
| U123 / 2025 | [VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ab16cc8f472cf94b282dbff7b99f5f8e-Abstract-Datasets_and_Benchmarks_Track.html) | 以层次具身合作任务评测代理激活、任务规划和轨迹感知，并进行强化微调。 |

## 直接 LLM＋RL 的综述与基准（2项）

对应P9。优先关注：按任务和信息条件寻找近邻方法。

| 编号 / 年份 | 原文 | 阅读抓手 |
|---|---|---|
| U146 / 2026 | [A Survey on Reinforcement Learning Optimization Methods for Multi-Agent Collaboration of Large Language Models](https://doi.org/10.54254/2755-2721/2026.36623) | 按个体奖励微调、联合奖励微调和推理期交互三类组织多语言模型协作研究。 |
| U026 / 2025 | [LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language Models](https://proceedings.mlr.press/v267/abdulhai25a.html) | 提供多轮语言交互 RL 环境与离线/在线方法接口，作为长程代理训练评测入口。 |

## LLM / VLM 辅助奖励设计（21项）

对应P3、P8；E3。优先关注：语言模型产生反馈，下层传统策略训练。

| 编号 / 年份 | 原文 | 阅读抓手 |
|---|---|---|
| U002 / 2026 | [CoRe: Combined Rewards with Vision-Language Model Feedback for Preference-Aligned Reinforcement Learning](https://proceedings.mlr.press/v306/ni26d.html) | 用 VLM 反馈组合显式形式奖励与从视频偏好中学习的残差奖励。 |
| U040 / 2026 | [GARDEN: Generalist Autonomous Reward Design Through Evidence Navigation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2610.37926v1) | 让工具型 LLM 检查代码、训练记录和行为证据，生成并冻结可跨场景使用的奖励程序。 |
| U041 / 2026 | [RM-VLM: From Natural-Language Task Descriptions to Vision-Grounded Reward Machines](https://arxiv.org/abs/2611.08308v1) | LLM 从任务描述构造奖励机，VLM 接地视觉命题并评估转移进度以提供稠密反馈。 |
| U042 / 2026 | [Causal Reward World Models: Zero-shot Reward Design for Automated Skill Generation](https://arxiv.org/abs/2606.23280v1) | 通过离线交互数据学习候选奖励项与目标物理变量的结构先验，再约束零样本奖励生成。 |
| U197 / 2026 | [LLM-ALSO: LLM-Driven Adaptive Learning-Signal Optimization for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2605.29293v1) | Critic LLM 从学习曲线与行为证据诊断，Generator LLM 提议塑形参数，候选在短分支中和不变奖励对照，验证后才合入主训练。 |
| U198 / 2026 | [LLM-Enhanced Multi-Agent Reinforcement Learning for Unified Electric Vehicles-Charging Station-Grid Optimization in Public Charging Systems](https://arxiv.org/abs/2609.13805v1) | 语言模型选择车、充电站与电网各自的状态特征，并按情境为三方目标分配权重，形成加权奖励。 |
| U206 / 2026 | [MARLF: Memory-Augmented Multi-Agent Reinforcement Learning with Large Language Model Feedback](https://doi.org/10.5281/zenodo.19998804) | 把语言模型的稀疏行为评语存入记忆，再用预测与动态加权把反馈变成训练时的稠密奖励。 |
| U217 / 2026 | [Multi-Agent Orchestration with the Common-Sense Reasoning Capabilities of LLMs for Autonomous Driving](https://arxiv.org/abs/2608.20129v2) | 离线语言模型阅读驾驶仿真日志并建议修改奖励，经人工确认后再训练低层策略，运行期由编排器协调控制模块。 |
| U232 / 2026 | [Regime-Conditional Stabilisation of LLM-Augmented Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2607.04470v1) | 动态改写塑形权重会使离策略回放携带旧奖励标签；以分阶段冻结和指数滑动平均抑制权重漂移。 |
| U268 / 2026 | [Autonomous Volt/Var Control in Active Distribution Networks via LLM-Driven Dynamic Reward Shaping](https://www.mdpi.com/2079-9292/15/16/3504) | 根据配电网控制策略的训练表现调整多目标奖励权重，使低层策略在电压约束和网损之间改变优化侧重。 |
| U001 / 2025 | [ELEMENTAL: Interactive Learning from Demonstrations and Vision-Language Models for Reward Design in Robotics](https://proceedings.mlr.press/v267/chen25at.html) | 把语言、视觉示范与逆强化学习结合，分别解决奖励特征表达与权重对齐。 |
| U043 / 2025 | [MAESTRO: Multi-Agent Environment Shaping through Task and Reward Optimization](https://arxiv.org/abs/2511.19253v2) | 把 LLM 放在训练外层，生成交通课程、模板奖励和先验策略正则，部署 MADDPG 控制器。 |
| U101 / 2025 | [ReMAC: Large Language Model-Driven Reward Design for Multi-Agent Manipulation Collaboration](https://sea-workshop.github.io/2025/) | 外层迭代团队与个体奖励群体，内层用 MARL 学习合作操纵策略。 |
| U102 / 2025 | [LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation](https://doi.org/10.1109/LRA.2025.3577527) | LLM 自动生成先验策略与奖励函数，指导多机器人合作策略学习。 |
| U280 / 2025 | [R*: Efficient Reward Design via Reward Structure Evolution and Parameter Alignment Optimization with Large Language Models](https://proceedings.mlr.press/v267/li25v.html) | LLM 参与奖励程序结构变异，多种轨迹评价函数投票筛出偏好样本，再用偏好学习调节奖励参数供控制策略训练。 |
| U151 / 2024 | [Auto MC-Reward: Automated Dense Reward Design with Large Language Models for Minecraft](https://arxiv.org/abs/2312.09238v2) | 奖励程序员生成可执行 Python 奖励，评审器查语法和语义，轨迹分析器总结失败供下一轮改写。 |
| U260 / 2024 | [Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning](https://arxiv.org/abs/2310.12921v2) | 以预训练视觉语言模型计算视觉观测与文字目标的匹配信号，用作零样本奖励训练模拟人形机器人策略。 |
| U081 / 2023 | [Eureka: Human-Level Reward Design via Coding Large Language Models](https://arxiv.org/abs/2310.12931v2) | 用环境代码和训练反馈迭代演化奖励程序，再让低层 RL 获取复杂控制技能。 |
| U082 / 2023 | [Text2Reward: Reward Shaping with Language Models for Reinforcement Learning](https://arxiv.org/abs/2309.11489v3) | 从语言任务与紧凑环境表示生成稠密奖励程序，并允许依据反馈逐步修订。 |
| U210 / 2023 | [Motif: Intrinsic Motivation from Artificial Intelligence Feedback](https://arxiv.org/abs/2310.00166v1) | 令 LLM 比较成对的状态文字描述，用这些偏好学习内在奖励，再训练 RL 智能体探索 NetHack 环境。 |
| U234 / 2023 | [Reward Design with Language Models](https://arxiv.org/abs/2303.00001v1) | 把任务描述和行为表示输入语言模型，将产生的评价作为奖励引导下游策略。 |

## 小控制器编排冻结 LLM（13项）

对应P1、P8；E4、E5。优先关注：路由器、编排器与文本执行器各自的训练对象。

| 编号 / 年份 | 原文 | 阅读抓手 |
|---|---|---|
| U010 / 2026 | [Dynamic Optimizations of LLM Ensembles with Two-Stage Reinforcement Learning Agents](https://proceedings.mlr.press/v306/tekin26a.html) | RL-Focal 将模型子集选择与回答融合分为两个 RL 决策阶段，并考虑错误多样性。 |
| U028 / 2026 | [Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](https://aclanthology.org/2026.acl-long.2130/) | 学习文本片段的本地/远端路由，在隐私泄漏与任务效用之间选择。 |
| U053 / 2026 | [Learning Who Talks to Whom: Role-Aware Communication Topology Scheduling for Efficient LLM Multi-Agent Collaboration](https://arxiv.org/abs/2610.25851v1) | ACTS 根据角色和状态选择可变数量的通信伙伴，并把拓扑纳入团队价值分解。 |
| U106 / 2026 | [Learning to Deliberate: Meta-policy Collaboration for Agentic LLMs with Multi-agent Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/40228) | 学习 Persist、Refine、Concede 等元认知动作，按奖励排序塑造优势。 |
| U119 / 2026 | [Assembling the CREW: A Collaborative Multi-agent Reinforcement Learning Framework for Automated Related Work Generation](https://aclanthology.org/2026.emnlp-main.2052/) | 通过 IPPO 学习检索、传播、写作、批评等操作的组织方式。 |
| U266 / 2026 | [AlphaRouter: Token-level Routing Between SLM and LLM with Reinforcement Learning and Tree Search](https://proceedings.mlr.press/v306/liao26c.html) | 把逐 token 调用大模型或小模型建模为序列决策，通过树搜索形成相邻候选路径，用反事实优势训练路由器。 |
| U273 / 2026 | [LAMaS: A Modular Platform for Learning Low-Latency Multi-Agent Orchestration](https://arxiv.org/abs/2601.10560v3) | 在期望正确率约束下，根据计算图关键路径给算子分摊延迟惩罚，再训练控制器按已完成结果决定是否继续执行。 |
| U287 / 2026 | [Multi-agent Legal Text De-identification Using TRPO and Transformer-based Language Models](https://doi.org/10.1007/s44196-026-01539-x) | 法律文本脱敏系统将待标记句子选择与词元级隐私编辑分配给两个代理，分别学习取舍。 |
| U011 / 2025 | [RLAE: Reinforcement Learning-Assisted Ensemble for LLMs](https://aclanthology.org/2025.emnlp-main.680/) | 把模型集成表达为 MDP，通过 PPO/MAPPO 动态调整上下文相关的集成权重。 |
| U033 / 2025 | [Efficient Integration of External Knowledge to LLM-based World Models via Retrieval-Augmented Generation and Reinforcement Learning](https://aclanthology.org/2025.findings-emnlp.504/) | RAWM 检索相关环境经验，并用 PPO 调整预训练嵌入模型上的小型 MLP 头。 |
| U051 / 2025 | [BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization](https://arxiv.org/abs/2512.23631v2) | BOAD 将子代理设计的探索表达为多臂老虎机，在有限评估预算下寻找协作层次。 |
| U214 / 2025 | [Multi-Agent Collaboration via Evolving Orchestration](https://arxiv.org/abs/2505.19591v2) | 中心调度策略按当前任务状态决定下一步激活哪个代理，从固定组队转向逐步可变的协作路径。 |
| U176 / 2024 | [GPTSwarm: Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823v3) | 以计算图表示语言智能体及其通信边，优化节点行为与智能体间连接并评估任务表现。 |

## 先验、信用与层次控制（22项）

对应P3、P8；E2、E6、E7。优先关注：语言先验、成员反馈、可行性与执行信息。

| 编号 / 年份 | 原文 | 阅读抓手 |
|---|---|---|
| U055 / 2026 | [Anticipating the Consequences of Curriculum Decisions with Large Language Models](https://arxiv.org/abs/2610.04604v1) | 将学习进度与 LLM 对下游收益和可学习性的估计结合，指导课程选择。 |
| U110 / 2026 | [LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/bae26c.html) | LLM 根据状态感知准则设计和修订通信协议，改善成员状态恢复。 |
| U111 / 2026 | [MARLIN: Multi-Agent Reinforcement Learning Guided by Language-Based Inter-Robot Negotiation](https://arxiv.org/abs/2410.14383v5) | 在早期训练中切换语言协商与 RL 控制，用协商计划辅助探索。 |
| U133 / 2026 | [Agentic AI Networking for Heterogeneous Unmanned Aerial Systems in Low-Altitude Wireless Networks](https://arxiv.org/abs/2609.19538v1) | 在无线服务协作仿真中，外环语言模型按服务目标和指标改变效用权重，内环按参数化策略分配资源。 |
| U141 / 2026 | [A hierarchical multi-agent reinforcement learning framework with high-level guidance from large language models](https://doi.org/10.1038/s41598-026-54971-6) | 低频 LLM 指挥器输出战略子目标，经语义奖励塑形和动态动作屏蔽帮助高频 MARL 决策。 |
| U169 / 2026 | [Do LLM-derived graph priors improve multi-agent coordination?](https://arxiv.org/abs/2604.17191v1) | 按局部观察文本推断协调图，用图网络聚合观测后再以 CTDE 更新策略，并比较不同小模型产生的图先验。 |
| U180 / 2026 | [Hierarchical Control in Multi-Agent Games: LLM-based Planning and RL Execution](https://arxiv.org/abs/2606.20014v3) | 集中 LLM 在预训练技能中为队伍做战术选择，低层 RL 技能执行；与行为树和不拆技能的端到端 RL 对照。 |
| U199 / 2026 | [LLM-guided graph neural coordination framework for cooperative multi-agent reinforcement learning](https://doi.org/10.1007/s40747-026-02356-7) | LLM 从战术语义构建团队协调图并给出多帧聚合奖励信号，图网络据此组织局部信息交换。 |
| U200 / 2026 | [LLM-Guided Reinforcement Learning for Adaptive NPC Behavior in Multi-Agent Combat Games](https://arxiv.org/abs/2609.02931v1) | 在游戏仿真中比较共享 PPO 基线与带策略标签的层次版本，评估时让语言模型选择策略标签。 |
| U242 / 2026 | [SkyAgent: A lightweight LLM-driven reinforcement learning framework for adaptive cooperative path planning of two UAVs](https://doi.org/10.1007/s44443-026-00957-3) | 轻量语言规划器定期提议子目标，低层策略在双智能体二维导航仿真中根据当前子目标执行。 |
| U284 / 2026 | [FedMARL-LTI: Federated Multi-Agent Reinforcement Learning with LLM-Compatible Threat Intelligence for Cooperative Cyber Defense](https://doi.org/10.3390/app16168278) | 比较防御团队通过联邦聚合共享模型参数及通过独立通道共享差分隐私语义摘要，评估两类信号对协作的影响。 |
| U288 / 2026 | [Orchestrating Reasoning and Reaction: An Asynchronous Hierarchical Framework for LLM-driven Traffic Signal Control](https://doi.org/10.1145/3770855.3818039) | Astra 由语言模型低频监测全局路况和协调拥堵路段，分散的 MARL 路口策略高频执行。局部异常可请求重新协调。 |
| U015 / 2025 | [Cache-Efficient Posterior Sampling for Reinforcement Learning with LLM-Derived Priors Across Discrete and Continuous Domains](https://aclanthology.org/2025.emnlp-main.560/) | 缓存 LLM 动作先验，并用性能相关的适配减少重复查询，覆盖离散与连续任务。 |
| U047 / 2025 | [SCOPE: Language Models as One-Time Teacher for Hierarchical Planning in Text Environments](https://arxiv.org/abs/2512.09897v1) | 初始化时从示例轨迹抽取 LLM 子目标，预训练轻量规划器，再通过世界模型交互改进。 |
| U057 / 2025 | [LLM-Driven Stationarity-Aware Expert Demonstrations for Multi-Agent Reinforcement Learning in Mobile Systems](https://arxiv.org/abs/2511.19368v1) | RELED 结合考虑非平稳性的 LLM 专家轨迹与自主探索，自适应混合两类学习来源。 |
| U100 / 2025 | [Leveraging Large Language Models for Effective and Explainable Multi-Agent Credit Assignment](https://ifaamas.org/Proceedings/aamas2025/) | 集中 LLM reward-critic 分解环境奖励，并可扩展为显式任务分配。 |
| U103 / 2025 | [YOLO-MARL: You Only LLM Once for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2410.03997) | 在训练前一次性生成策略、状态解释与规划函数，部署普通去中心策略。 |
| U104 / 2025 | [Speaking the Language of Teamwork: LLM-Guided Credit Assignment in Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2502.03723) | 结合 LLM 的个体贡献判断和势函数式奖励塑形，提供稠密个体信号。 |
| U188 / 2025 | [L2M2: A Hierarchical Framework Integrating Large Language Model and Multi-agent Reinforcement Learning](https://doi.org/10.24963/ijcai.2025/12) | LLM 零样本生成长期导航计划与中间目标，MARL 策略负责低层执行；既可联合系统训练，也可对接已有低层策略。 |
| U201 / 2025 | [LLM-Mediated Guidance of MARL Systems](https://arxiv.org/abs/2503.13553v2) | 训练期将稀疏规则或语言指令转成临时任务与动作覆盖，查验早期协调引导是否优于无干预。 |
| U179 / 2023 | [Guiding Pretraining in Reinforcement Learning with Large Language Models](https://proceedings.mlr.press/v202/du23f.html) | LLM 根据当前状态描述提出常识目标，目标实现转成内在探索奖励，先预训练再评估下游任务。 |
| U168 / 2022 | [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691v2) | 将语言模型给出的技能相关度与低层技能的可执行性评分相乘，逐步选出机器人行动。 |

## LLM 与 MARL 交叉综述（5项）

对应P9。优先关注：分类入口；具体结论回到对应原始研究。

| 编号 / 年份 | 原文 | 阅读抓手 |
|---|---|---|
| U054 / 2026 | [The Five Ws of Multi-Agent Communication: Who Talks to Whom, When, What, and Why -- A Survey from MARL to Emergent Language and LLMs](https://arxiv.org/abs/2602.11583v1) | 以通信的对象、内容、时机和目的连接传统 MARL、涌现语言及 LLM 团队研究。 |
| U173 / 2026 | [From Multi-Agent Reinforcement Learning to Agentic AI: A Comprehensive Literature Review of Algorithmic Advances and Decision-Analytic Implications (2020-2025)](https://doi.org/10.66972/ada202734) | 从价值分解、合作策略优化和信用分配等方法梳理 MARL，并讨论与语言智能体决策架构的联系。 |
| U186 / 2026 | [Integration of Multi-agent Reinforcement Learning (MARL) and Large Language Models (LLMs) in Agentic AI Systems: A Narrative Review](https://doi.org/10.1007/s42979-026-05256-2) | 以叙述性综述整理语言推理和 MARL 相结合的分层控制、规划、记忆及多领域案例。 |
| U281 / 2026 | [Toward the Science of Multi-Agent Communication: A Comprehensive Survey](https://yifang99.github.io/Towards-The-Science-of-Agent-Communication/) | 按通信格式和消息内容整理 MARL、涌现语言以及 LLM 协作，在同一分类下考察不同消息所携带的信息。 |
| U126 / 2024 | [LLM-based Multi-Agent Reinforcement Learning: Current and Future Directions](https://arxiv.org/abs/2405.11106) | 早期 LLM 与 MARL 分类入口，串联语言规划、通信和合作决策。 |

## 当前待选问题

C02检验跨题筛查与在线抗捷径，C01改检验保护动作覆盖与恢复诊断，
C03检验结构条件下的聚焦—覆盖分配。它们目前都需要区分性实验，
具体对照与继续条件见[实践E11](/2026/11/23/llm-rl-e11-next-experiments/)。

首次阅读可返回[专题导读](/papers/llm-rl/)，沿统一顺序接着读。
