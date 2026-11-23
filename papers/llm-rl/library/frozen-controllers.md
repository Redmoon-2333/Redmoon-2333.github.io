---
layout: post
title: P9F：冻结模型、控制器与分层决策
permalink: /papers/llm-rl/library/frozen-controllers/
date: 2026-11-24 00:00:00 +0800
topic: llm-rl
categories:
- 论文精读
math: false
reading_prev:
  title: P9E：LLM 与 VLM 设计奖励
  url: /papers/llm-rl/library/reward-design/
reading_next:
  title: P9G：综述与基准
  url: /papers/llm-rl/library/surveys-and-benchmarks/
---

把语言模型当成可调用的帮手。一个小策略可以学习何时检索、复核、发消息或换技能。

[返回 P9 文献导航](/2026/11/23/llm-rl-p9-reading-index/)。先用 [P1](/2026/11/23/llm-rl-p1-learning-object/) 找优化器里真正被更新的参数，再用 [E4](/2026/11/23/llm-rl-e4-routing/) 比较不同调用决策的成本。

## 小控制器编排冻结 LLM（13项）

先分清控制器决定调用什么，还是语言模型自己也接受参数更新。

- `U010 / 2026` [Dynamic Optimizations of LLM Ensembles with Two-Stage Reinforcement Learning Agents](https://proceedings.mlr.press/v306/tekin26a.html)<br />
  RL-Focal 将模型子集选择与回答融合分为两个 RL 决策阶段，并考虑错误多样性。
- `U028 / 2026` [Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](https://aclanthology.org/2026.acl-long.2130/)<br />
  学习文本片段的本地/远端路由，在隐私泄漏与任务效用之间选择。
- `U053 / 2026` [Learning Who Talks to Whom: Role-Aware Communication Topology Scheduling for Efficient LLM Multi-Agent Collaboration](https://arxiv.org/abs/2610.25851v1)<br />
  ACTS 根据角色和状态选择可变数量的通信伙伴，并把拓扑纳入团队价值分解。
- `U106 / 2026` [Learning to Deliberate: Meta-policy Collaboration for Agentic LLMs with Multi-agent Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/40228)<br />
  学习 Persist、Refine、Concede 等元认知动作，按奖励排序塑造优势。
- `U119 / 2026` [Assembling the CREW: A Collaborative Multi-agent Reinforcement Learning Framework for Automated Related Work Generation](https://aclanthology.org/2026.emnlp-main.2052/)<br />
  通过 IPPO 学习检索、传播、写作、批评等操作的组织方式。
- `U266 / 2026` [AlphaRouter: Token-level Routing Between SLM and LLM with Reinforcement Learning and Tree Search](https://proceedings.mlr.press/v306/liao26c.html)<br />
  把逐 token 调用大模型或小模型建模为序列决策，通过树搜索形成相邻候选路径，用反事实优势训练路由器。
- `U273 / 2026` [LAMaS: A Modular Platform for Learning Low-Latency Multi-Agent Orchestration](https://arxiv.org/abs/2601.10560v3)<br />
  在期望正确率约束下，根据计算图关键路径给算子分摊延迟惩罚，再训练控制器按已完成结果决定是否继续执行。
- `U287 / 2026` [Multi-agent Legal Text De-identification Using TRPO and Transformer-based Language Models](https://doi.org/10.1007/s44196-026-01539-x)<br />
  法律文本脱敏系统将待标记句子选择与词元级隐私编辑分配给两个代理，分别学习取舍。
- `U011 / 2025` [RLAE: Reinforcement Learning-Assisted Ensemble for LLMs](https://aclanthology.org/2025.emnlp-main.680/)<br />
  把模型集成表达为 MDP，通过 PPO/MAPPO 动态调整上下文相关的集成权重。
- `U033 / 2025` [Efficient Integration of External Knowledge to LLM-based World Models via Retrieval-Augmented Generation and Reinforcement Learning](https://aclanthology.org/2025.findings-emnlp.504/)<br />
  RAWM 检索相关环境经验，并用 PPO 调整预训练嵌入模型上的小型 MLP 头。
- `U051 / 2025` [BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization](https://arxiv.org/abs/2512.23631v2)<br />
  BOAD 将子代理设计的探索表达为多臂老虎机，在有限评估预算下寻找协作层次。
- `U214 / 2025` [Multi-Agent Collaboration via Evolving Orchestration](https://arxiv.org/abs/2505.19591v2)<br />
  中心调度策略按当前任务状态决定下一步激活哪个代理，从固定组队转向逐步可变的协作路径。
- `U176 / 2024` [GPTSwarm: Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823v3)<br />
  以计算图表示语言智能体及其通信边，优化节点行为与智能体间连接并评估任务表现。

## 先验、信用与层次控制（22项）

先验给出推荐动作，分层控制将目标拆给不同策略；比较时看各层能见哪些信息。

- `U055 / 2026` [Anticipating the Consequences of Curriculum Decisions with Large Language Models](https://arxiv.org/abs/2610.04604v1)<br />
  将学习进度与 LLM 对下游收益和可学习性的估计结合，指导课程选择。
- `U110 / 2026` [LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v306/bae26c.html)<br />
  LLM 根据状态感知准则设计和修订通信协议，改善成员状态恢复。
- `U111 / 2026` [MARLIN: Multi-Agent Reinforcement Learning Guided by Language-Based Inter-Robot Negotiation](https://arxiv.org/abs/2410.14383v5)<br />
  在早期训练中切换语言协商与 RL 控制，用协商计划辅助探索。
- `U133 / 2026` [Agentic AI Networking for Heterogeneous Unmanned Aerial Systems in Low-Altitude Wireless Networks](https://arxiv.org/abs/2609.19538v1)<br />
  在无线服务协作仿真中，外环语言模型按服务目标和指标改变效用权重，内环按参数化策略分配资源。
- `U141 / 2026` [A hierarchical multi-agent reinforcement learning framework with high-level guidance from large language models](https://doi.org/10.1038/s41598-026-54971-6)<br />
  低频 LLM 指挥器输出战略子目标，经语义奖励塑形和动态动作屏蔽帮助高频 MARL 决策。
- `U169 / 2026` [Do LLM-derived graph priors improve multi-agent coordination?](https://arxiv.org/abs/2604.17191v1)<br />
  按局部观察文本推断协调图，用图网络聚合观测后再以 CTDE 更新策略，并比较不同小模型产生的图先验。
- `U180 / 2026` [Hierarchical Control in Multi-Agent Games: LLM-based Planning and RL Execution](https://arxiv.org/abs/2606.20014v3)<br />
  集中 LLM 在预训练技能中为队伍做战术选择，低层 RL 技能执行；与行为树和不拆技能的端到端 RL 对照。
- `U199 / 2026` [LLM-guided graph neural coordination framework for cooperative multi-agent reinforcement learning](https://doi.org/10.1007/s40747-026-02356-7)<br />
  LLM 从战术语义构建团队协调图并给出多帧聚合奖励信号，图网络据此组织局部信息交换。
- `U200 / 2026` [LLM-Guided Reinforcement Learning for Adaptive NPC Behavior in Multi-Agent Combat Games](https://arxiv.org/abs/2609.02931v1)<br />
  在游戏仿真中比较共享 PPO 基线与带策略标签的层次版本，评估时让语言模型选择策略标签。
- `U242 / 2026` [SkyAgent: A lightweight LLM-driven reinforcement learning framework for adaptive cooperative path planning of two UAVs](https://doi.org/10.1007/s44443-026-00957-3)<br />
  轻量语言规划器定期提议子目标，低层策略在双智能体二维导航仿真中根据当前子目标执行。
- `U284 / 2026` [FedMARL-LTI: Federated Multi-Agent Reinforcement Learning with LLM-Compatible Threat Intelligence for Cooperative Cyber Defense](https://doi.org/10.3390/app16168278)<br />
  比较防御团队通过联邦聚合共享模型参数及通过独立通道共享差分隐私语义摘要，评估两类信号对协作的影响。
- `U288 / 2026` [Orchestrating Reasoning and Reaction: An Asynchronous Hierarchical Framework for LLM-driven Traffic Signal Control](https://doi.org/10.1145/3770855.3818039)<br />
  Astra 由语言模型低频监测全局路况和协调拥堵路段，分散的 MARL 路口策略高频执行。局部异常可请求重新协调。
- `U015 / 2025` [Cache-Efficient Posterior Sampling for Reinforcement Learning with LLM-Derived Priors Across Discrete and Continuous Domains](https://aclanthology.org/2025.emnlp-main.560/)<br />
  缓存 LLM 动作先验，并用性能相关的适配减少重复查询，覆盖离散与连续任务。
- `U047 / 2025` [SCOPE: Language Models as One-Time Teacher for Hierarchical Planning in Text Environments](https://arxiv.org/abs/2512.09897v1)<br />
  初始化时从示例轨迹抽取 LLM 子目标，预训练轻量规划器，再通过世界模型交互改进。
- `U057 / 2025` [LLM-Driven Stationarity-Aware Expert Demonstrations for Multi-Agent Reinforcement Learning in Mobile Systems](https://arxiv.org/abs/2511.19368v1)<br />
  RELED 结合考虑非平稳性的 LLM 专家轨迹与自主探索，自适应混合两类学习来源。
- `U100 / 2025` [Leveraging Large Language Models for Effective and Explainable Multi-Agent Credit Assignment](https://ifaamas.org/Proceedings/aamas2025/)<br />
  集中 LLM reward-critic 分解环境奖励，并可扩展为显式任务分配。
- `U103 / 2025` [YOLO-MARL: You Only LLM Once for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2410.03997)<br />
  在训练前一次性生成策略、状态解释与规划函数，部署普通去中心策略。
- `U104 / 2025` [Speaking the Language of Teamwork: LLM-Guided Credit Assignment in Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2502.03723)<br />
  结合 LLM 的个体贡献判断和势函数式奖励塑形，提供稠密个体信号。
- `U188 / 2025` [L2M2: A Hierarchical Framework Integrating Large Language Model and Multi-agent Reinforcement Learning](https://doi.org/10.24963/ijcai.2025/12)<br />
  LLM 零样本生成长期导航计划与中间目标，MARL 策略负责低层执行；既可联合系统训练，也可对接已有低层策略。
- `U201 / 2025` [LLM-Mediated Guidance of MARL Systems](https://arxiv.org/abs/2503.13553v2)<br />
  训练期将稀疏规则或语言指令转成临时任务与动作覆盖，查验早期协调引导是否优于无干预。
- `U179 / 2023` [Guiding Pretraining in Reinforcement Learning with Large Language Models](https://proceedings.mlr.press/v202/du23f.html)<br />
  LLM 根据当前状态描述提出常识目标，目标实现转成内在探索奖励，先预训练再评估下游任务。
- `U168 / 2022` [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691v2)<br />
  将语言模型给出的技能相关度与低层技能的可执行性评分相乘，逐步选出机器人行动。

下一个查阅页：[P9G：综述与基准](/papers/llm-rl/library/surveys-and-benchmarks/)。
