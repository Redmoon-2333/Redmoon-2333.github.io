---
layout: post
title: P9A：验证器与推理训练
permalink: /papers/llm-rl/library/verifiers-and-reasoning/
date: 2026-11-24 00:00:00 +0800
topic: llm-rl
categories:
- 论文精读
math: false
reading_prev:
  title: LLM 与强化学习 P9：当前研究坐标与分层论文阅读目录
  url: /2026/11/23/llm-rl-p9-reading-index/
reading_next:
  title: P9B：长程记忆与信用
  url: /papers/llm-rl/library/memory-and-credit/
---

从训练判分器可能错奖哪些答案读起，再看推理训练怎样利用这些分数。

[返回 P9 文献导航](/2026/11/23/llm-rl-p9-reading-index/)。先读 [P4](/2026/11/23/llm-rl-p4-verifier/) 的判分反例。论文 L10 拆开训练分与真实分，L09 检查同一错误跨题复用时会怎样。

## 不完美验证器与 RLVR（8项）

对照判题器给出的分与任务定义下的正确性；两者不一致时，先找错判的条件。

- `L09 / 2026` [Wrong the Same Way: Cross-Prompt Sharing Both Suppresses and Amplifies Reward Hacking in RLVR](https://arxiv.org/abs/2611.13218v1)<br />
  共享参数对跨题错误接受的抑制与放大；可控小模型机制，详见 P4/P5。
- `L10 / 2026` [Optimistic Verifiers Get Hacked: Reward Hacking Under an Exact Process Oracle](https://arxiv.org/html/2611.21538v1)<br />
  可执行规则解释的精确参照与乐观代理；合成和公共策略分开，详见 P4。
- `L11 / 2026` [Checking the Checkers: A Synthesis of Verifier Noise and Robustness in RLVR](https://arxiv.org/html/2610.17824v1)<br />
  验证器噪声与鲁棒性综述；按错误结构查找训练期干预。
- `L12 / 2026` [When the Reward Suite Is Leaky: A Preregistered Causal Contrast of Natural Verifier False Positives in RLVR](https://arxiv.org/abs/2607.11022)<br />
  自然代码测试漏检的配对训练与审计；区分漏洞暴露与留出损伤，详见 P4。
- `L13 / 2026` [Shortcuts and Boundaries: Verifiable RL Misalignment and Contextual Safety Learning](https://arxiv.org/html/2611.12865v1)<br />
  可验证奖励与上下文约束的冲突；检查合规反馈与迁移条件。
- `L14 / 2026` [LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking](https://arxiv.org/abs/2604.15149)<br />
  规则诱导与实例捷径；考察同构扰动测试依赖的领域知识。
- `L15 / 2026` [Quantifying Empirical Compute-Supervision Tradeoffs in RLVR](https://arxiv.org/html/2605.25252)<br />
  计算量与有噪监督的经验比较；区分随机错误与可触发错误。
- `U067 / 2026` [Verifier Errors in RLVR: Reward Hacking, Limits of Feedback, and Selective Control](https://arxiv.org/abs/2609.35677v1)<br />
  分析验证器接受错误答案时奖励与真实正确率的分离，并讨论额外审计反馈。

## 推理 RL、测试时学习与自博弈（24项）

分清训练时更新参数和推理时多采样，再看任务奖励由谁计算。

- `L22 / 2026` [When Does Exploration Prevent Collapse? Three Regimes and the Path of the Reward in Self-Consistency Training](https://arxiv.org/abs/2611.01307)<br />
  自一致奖励下探索与坍缩的动力学；归一化前后放探索有不同效果。
- `L23 / 2026` [When the Majority Votes Wrong, the Intervention Timing for Test-Time Reinforcement Learning Hides in the Extinction Window](https://arxiv.org/html/2605.19444v1)<br />
  多数答案可能错误时的测试时训练干预；考察少数正确信号的保留。
- `L24 / 2026` [Rewarding Reasoning, Not Answers: Fixing and Bounding Test-Time Reinforcement Learning on Medical QA](https://arxiv.org/html/2609.16660)<br />
  小答案空间的测试时奖励；检查过程评分器的领域训练条件。
- `L25 / 2026` [Entropy-Regularized Rank-Masked Policy Optimization for Test-Time Reinforcement Learning in Code Generation](https://arxiv.org/html/2609.09135)<br />
  代码测试时训练的共识奖励、秩掩码与熵；与在线多轮任务分开。
- `L26 / 2026` [Learning to Solve, Forgetting to Retain: Correct-Set Turnover in RLVR](https://arxiv.org/html/2606.03087v2)<br />
  已答对题集合在共享更新中更替；检查复习采样与保留成本。
- `L27 / 2026` [Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL](https://arxiv.org/html/2605.22217)<br />
  自博弈中的数据门控与奖励锚定；归档 U071 为同一作品。
- `L28 / 2026` [Enabling Token-wise Importance Sampling in Off-Policy RLVR with Proximal KL Regularization](https://arxiv.org/abs/2610.18738v1)<br />
  离策略 RLVR 的 token 重要性修正与 KL；关注理论条件和数据陈旧度。
- `U017 / 2026` [RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments](https://proceedings.mlr.press/v306/zeng26j.html)<br />
  RLVE 用可程序生成且答案可验证的环境，根据策略能力调节问题难度。
- `U037 / 2026` [Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs](https://aclanthology.org/2026.acl-long.847/)<br />
  TPAW 让当前策略与历史检查点合作/竞争，并分别调整回答和玩家的训练权重。
- `U056 / 2026` [When Does Closed-Loop Curriculum Adaptation Help GRPO? An Analysis on Countdown](https://arxiv.org/abs/2610.39056v1)<br />
  用难度分桶探测调整课程采样进度与宽度，保持任务有挑战且仍可学习。
- `U072 / 2026` [More Convincing, Not More Correct: Self-Play Reward Hacking of Reference-Free LLM Judges](https://arxiv.org/abs/2607.05904v1)<br />
  以独立隐藏答案审计检查自博弈训练是否只提高无参考裁判评分。
- `U118 / 2026` [Strat-Reasoner: Reinforcing Strategic Reasoning of LLMs in Multi-Agent Games](https://proceedings.mlr.press/v306/he26u.html)<br />
  把其他角色的推理纳入递归策略思考，用集中 CoT 比较反馈训练语言策略。
- `U278 / 2026` [Plan Then Action: High-Level Planning Guidance Reinforcement Learning for LLM Reasoning](https://proceedings.mlr.press/v306/dou26c.html)<br />
  先把推理链压成高层计划做冷启动，再用轨迹组内的奖励同时优化计划质量与答案，更新语言策略。
- `U038 / 2025` [Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search](https://proceedings.mlr.press/v267/shen25o.html)<br />
  Satori 用带动作的思维链与两阶段训练，将反思和重新搜索内化到单个策略。
- `U039 / 2025` [T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](https://proceedings.mlr.press/v267/hou25e.html)<br />
  T1 结合含试错与自验证的数据初始化、RL 探索及推理预算扩展。
- `U073 / 2025` [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948v2)<br />
  用 RL 激励可验证推理能力，并将所得推理模式用于小模型的能力迁移。
- `U074 / 2025` [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476v2)<br />
  DAPO 讨论大规模语言 RL 的裁剪、采样与训练实现细节，并公开训练系统。
- `U075 / 2025` [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783v2)<br />
  区分基座模型能力与 RL 目标的贡献，指出 GRPO 长度相关偏差并提出 Dr. GRPO。
- `U080 / 2025` [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335v3)<br />
  同一模型承担出题者与求解者，通过可执行环境验证生成任务与答案来共同演进。
- `U248 / 2025` [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](https://arxiv.org/abs/2502.18449v2)<br />
  由开源软件修改历史构造问题与参考补丁，用生成修改与参考修改的相似度反馈训练代码推理模型。
- `U236 / 2024` [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267v3)<br />
  以模型生成的偏好反馈训练语言策略，并与使用人类偏好标签的强化学习在相同任务下比较。
- `U083 / 2023` [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290v3)<br />
  通过偏好分类损失直接优化策略，作为传统显式奖励模型加 RL 流程的相邻路径。
- `U084 / 2022` [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073v1)<br />
  以原则驱动的批评修订构造监督数据，再用 AI 偏好训练奖励模型并进行 RLAIF。
- `U282 / 2022` [Training language models to follow instructions with human feedback](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html)<br />
  人工示范用于指令微调，人工排序用于训练奖励模型，再用 PPO 依据奖励模型的输出更新语言策略。

下一个查阅页：[P9B：长程记忆与信用](/papers/llm-rl/library/memory-and-credit/)。
