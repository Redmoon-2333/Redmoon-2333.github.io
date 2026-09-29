---
topic: reinforcement-learning
layout: post
title: "强化学习 12 · 从经典强化学习到 2026 年的研究版图"
date: 2026-09-26 19:00:00 +0800
categories:
  - 技术实践
series_order: 12
tags:
  - 研究版图
  - LLM
  - RLHF
  - RLVR
excerpt: 检索截止 2026-09-26：经典机制演进、LLM 的五类角色、哪些结论真有可比较证据。
math: true
---
> 前十一章把一个受控世界里的机制走完了：从 Bellman 方程到 Q-learning，从 DQN、REINFORCE 到 PPO，再进入多智能体的非平稳、价值分解、反事实基线与通信。第 11 章收在实验方法上。最后一章换一个尺度——不再问「这个机制怎么工作」，而问「2024–2026 年，这个领域究竟前进了什么，哪些结论真的站得住」。本章按三个问题组织，以可查的一手记录校准既有结论。

## 0 · 核验口径

**原文检索记录日期：2026-09-26，保留不变。** 本章依据已有材料做有限抽查，2026 年的记录只是一部分：151 篇是本地论文库规模，不是全文通读数量；已有阅读记录包括摘要、选定章节，以及 15 篇论文逐页文本的关键词扫描。发表信息来自 arXiv、Crossref、OpenReview、OpenAlex 与会议官网的部分记录；原计划使用的 DBLP、Semantic Scholar 查询受限，未纳入。

发表状态与文本可用性分开记：**published** 表示有明确会议／期刊正式记录或接收记录；**workshop** 单列，不等同主会；**preprint** 只表示可获得预印本，不表示没有正式发表。submitted、rejected、withdrawn 是特定投稿的状态，其中 withdrawn submission 是撤回投稿，不等于已发表论文被撤稿。未确认正式 venue 的条目写「发表状态未核实」。除本次补核的 TD-MPC2、DeepSeek-R1、SayCan、Zero-Shot Planners 及所引用的算法原文外，其余条目的 venue 与实验数字沿用既有记录、未逐条复核；2026 年条目的时效性尤其有限。证据类型仍区分理论、基准实验、演示、分类学分析与综述。

三个冻结问题与一句话结论：

| 问题 | 一句话结论 |
|---|---|
| Q1 经典机制的 2024–2026 演进 | 代表工作包含正式发表与预印本记录，有限检索不能判断整个方向的成熟度 |
| Q2 LLM 在 RL 系统中的五类角色 | 同一个 LLM 可以站在五个位置，被训练的对象完全不同 |
| Q3 哪些结论有可比较证据 | 15 篇样本的关键词命中数不能外推全领域，也不足以建立跨论文排名 |

本章只处理证据：每节先给结论，再给 venue 与证据类型，最后标明这判断到哪里为止。

## 一 · Q1：经典机制的 2024–2026 演进

经典机制在新任务中的延伸，本轮核到的代表有：PPO 类比率与裁剪进入 LLM 的 RLVR（可验证奖励强化学习），模型式方法（model-based：先学一个环境模型，再在模型里规划或推演）有 DreamerV3 与 TD-MPC2 等正式发表工作。本次关键词与题名检索在价值分解、离线 RL、探索与信用分配上的覆盖不均，只触及部分代表工作。

### 1.1 价值分解：有限样本中的代表

经典锚点包括 VDN 的可加分解与 QMIX 的单调混合。已有记录收录了 [Counterfactual value decomposition for cooperative MARL (Liu et al., 2025, Neural Networks)](https://doi.org/10.1016/j.neunet.2025.107692) 等变体；该条只核过元数据，实验部分未核读。本次检索停在题名与摘要层面，覆盖的是代表条目。

### 1.2 PPO 类：从 MAPPO 到 RLVR

合作 MARL 侧可参考 [MAPPO (Yu et al., 2022, NeurIPS 2022, published)](https://doi.org/10.52202/068431-1787) 这一 PPO 基线。LLM 的 RLVR 则提供另一条机制演进线：[DeepSeekMath 首发的 GRPO (Shao et al., 2024, 预印本可用，发表状态未复核)](https://arxiv.org/abs/2402.03300) → [Dr. GRPO (Liu et al., 2025, 预印本可用，发表状态未复核)](https://arxiv.org/abs/2503.20783) 区分两种偏置：**按响应长度归一化对应长度偏置，按组内奖励标准差归一化对应问题难度偏置**，不能混为一谈 → [GSPO (Zheng et al., 2025, 预印本可用，发表状态未复核)](https://arxiv.org/abs/2507.18071) 使用序列级比率与裁剪，讨论 MoE（专家混合，Mixture of Experts）的 RL 稳定性。这里比较的是目标函数的归一化与比率粒度。同线还有 DAPO 的解耦裁剪与动态采样；其正式发表状态本次未复核。

### 1.3 离线 RL：覆盖不足，只能说「未检出」

离线 RL（offline RL）指只用一份事先收集好的固定数据集训练，训练过程中不再与环境交互；数据没覆盖到的状态-动作对，方法再强也学不到。

本轮抽样检出的代表全是预印本，如 [Towards Optimal Offline RL (Li & Kuhn, 2025, preprint)](https://arxiv.org/abs/2503.12283)；唯一核到的发表记录是 workshop 级，即数据集蒸馏用于离线 RL 的一篇（ICML 2024 DMLR Workshop，依据 arXiv comment 判断）。**[未核实]**：本次检索未获得离线 RL 在 2024–2026 的主线 published 记录，这属于方向覆盖不足；本节结论止于「本次检索未检出」。

### 1.4 模型式：DreamerV3 与 TD-MPC2

模型式方法的代表包括 DreamerV3 的 [Mastering diverse control tasks through world models (Hafner et al., 2025, Nature, published)](https://doi.org/10.1038/s41586-025-08744-2)，以及 [TD-MPC2: Scalable, Robust World Models for Continuous Control (Hansen et al., 2024, ICLR 2024, published)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)。TD-MPC2 虽于 2023 年发布预印本，正式会议年份是 2024，属于这里的发表时间窗口。两篇可作不同世界模型路线的阅读入口，DreamerV3 是本次检索到的代表之一。

### 1.5 探索与信用分配

「LLM 能否做探索」可参考 [Can large language models explore in-context? (Krishnamurthy et al., 2024, NeurIPS 2024)](https://doi.org/10.52202/079017-3818)。信用分配记录涉及反事实、Shapley 与上下文内信用等线索，覆盖尚不均衡。GRPO-λ 的既有记录为 ICLR 2026 withdrawn submission（撤回该次投稿），最新状态未复核。

小结：正式发表记录有助于定位版本与出处，方法与实验审查仍以原文为准。

## 二 · Q2：LLM 在 RL 系统中的五类角色

判据固定为四问：LLM 输出什么、谁消费这个输出、参数是否更新、谁被训练。同一个系统可以同时占多个角色，报告时必须分开计。角色混淆最常出现在两处：把「LLM 参与了系统」直接写成「LLM 被训练了」，或者反过来把纯 prompt 编排称作多智能体强化学习（判据同第 6 章易错点一：看有没有策略在训练中被奖励信号改写）。下面每类角色都注明被训练的对象。

**(a) 执行者／行动器**——LLM 在环内出动作，参数不更新。[ReAct (Yao et al., 2023, ICLR 2023, published)](https://arxiv.org/abs/2210.03629) 把推理与行动交替组织；[Reflexion (Shinn et al., 2023, NeurIPS 2023, published)](https://doi.org/10.52202/075280-0377) 用文字反思替代梯度更新。两者都是多任务对照的基准实验。

**(b) 规划器**——LLM 提出子目标，低层策略执行。[SayCan](https://proceedings.mlr.press/v205/ichter23a.html) 属于 CoRL 2022（论文集 2023 年出版），[Language Models as Zero-Shot Planners](https://proceedings.mlr.press/v162/huang22a.html) 已发表于 ICML 2022，不能因有 arXiv 版本就标为未发表。[L2M2](https://doi.org/10.24963/ijcai.2025/12) 是既有记录中的层级 LLM–MARL 例子。共同接口问题是：语言提出的子目标能否被执行器实现，失败后是否有可靠反馈。

**(c) 奖励程序员**——LLM 生成或迭代奖励代码。[Eureka (Ma et al., 2024, ICLR 2024, published)](https://openreview.net/forum?id=IEduRUO55F) 用训练反馈迭代奖励；[Text2Reward (Xie et al., 2024, ICLR 2024, published)](https://arxiv.org/abs/2309.11489) 用语言模型做奖励塑造。两者有正式接收记录与多任务对照，可作为这类接口的案例。[ReMAC (Li et al., 2025, NeurIPS 2025 workshop)](https://openreview.net/forum?id=CWYWhLho0a) 也属这一类，属 NeurIPS 2025 workshop 记录，见第四节。

**(d) critic**——先分两种：LLM-as-critic 读轨迹输出评价或分数、参数不更新；被训练的 critic／reward model 会被更新。前者如 [LLM-MCA (Nagpal et al., 2025, AAMAS 2025, published)](https://doi.org/10.65109/nokw1037)：集中式 LLM 读联合轨迹，输出逐主体 credit 与自然语言解释，基线含 MAPPO/QMIX；后者如过程奖励模型 [Let's Verify Step by Step (Lightman et al., 2024, ICLR 2024, published)](https://arxiv.org/abs/2305.20050) 与 RLHF 中的 reward model。名字都叫 critic，输入、输出与训练目标可以完全不同。

**(e) 被 RL 训练的策略**——LLM 参数通过 RL 更新。RLHF（基于人类反馈的强化学习）的代表是 [InstructGPT](https://doi.org/10.52202/068431-2011)，使用人类偏好训练奖励模型；推理 RL 的代表 [DeepSeek-R1](https://doi.org/10.1038/s41586-025-09422-z) 已有 Nature 2025 正式论文。可验证奖励描述特定训练阶段，不能概括 R1 全流程只用一种奖励。GiGPO、MAPoRL、RAGEN、AgentGym-RL、Agent-R1、MARFT 等提供不同 agent 训练线索，其最新发表状态并未在本次逐一复核。多轮描述时间结构，多智能体描述决策主体数量，两者需要分别判断。

五类角色用于分清接口与训练对象；正式接收、任务数量、对照质量与统计报告是不同维度，不能互相替代。

![LLM 在强化学习系统中的五类角色：执行者、规划器、奖励程序员、评判者、被训练的策略](/assets/img/rl-practice/faro_ch12_roles.png)

## 三 · Q3：哪些结论有可比较证据，哪些只是演示

方法：对 15 篇本地 PDF 的逐页文本做模式扫描——种子/重复次数、误差条、预算（token／模型调用／GPU 时）、baseline，再用关键词统计判断是否与经典 MARL 或无-LLM 方案对照。**这些计数是关键词层面的扫描**：关键词命中不等于确认基线使用，否定结果也可能被不同措辞绕过，标 ⚠ 的条目需要回原文复核。挑这三个要素，是因为它们对应第 11 章的实验规范：预算决定比较是否公平，种子与误差条决定波动是否可见，基线决定改进能归因给谁。

| 历史关键词扫描类别（未经逐篇复核） | 命中数 / 15 |
|---|---:|
| 种子或重复次数相关词 | 4/15 |
| 误差条或标准差相关词 | 6/15 |
| 疑似预算对齐相关表述 | 1/15 |
| token 使用量相关表述 | 1/15 |
| baseline 相关词 | 13/15 |
| 含经典 MARL／无-LLM／人工奖励对照（关键词证据）⚠ | 6/15 |

这些记录适合作为回查线索，不是论文级审计结论：

1. **预算对齐需要人工核查。** 旧扫描命中一个候选，覆盖范围仅此一条；SRPO（arXiv:2609.08452）的存在性、版本与 matched-cost 细节本次未复核。
2. **未命中不等于未报告。** 4/15 与 6/15 是两类关键词命中数；正文、附录、表注及补充材料都可能使用不同措辞。不能据扫描缺词认定某篇论文没有种子或方差。
3. **基线要按训练对象归类。** LLM 系统可能比较单 agent、prompt 编排或多 agent 工作流；LLM 辅助传统 MARL 可能比较 QMIX、MAPPO 或人工奖励。
4. **本次材料不足以建立跨论文排名。** 先逐篇核对环境、模型、训练与推理预算，再判断能否比较；有限样本未检出共同协议不意味着全领域没有统一基准。
5. **演示与负结果要回到原文。** 既有记录摘录了 ToMAS 对未显示训练效应的自述，这类结论应限定到对应配置。对 MASkills、Optima 等的“无种子”“无预算”断言不能仅靠关键词扫描成立。
6. **发表信息与实验证据分开。** GiGPO、MAPoRL、L2M2、MAST 等是后续全文审查入口，不因有 venue 就自动成为“实验支柱”。论文内部的同协议对照也不自动获得跨论文可比性。

## 四 · 边界与更正

本节处理两类事：更正容易误读的说法（4.1–4.4），划定本地玩具实验与文献结果的边界（4.5）。

### 4.1 GRPO 的真实构成

GRPO **保留**了 PPO 的重要性比率、裁剪与逐 token 平均，目标里还含 $-\beta\,\mathrm{D}_{KL}(\pi_\theta\|\pi_{ref})$，**去掉**的只是被学习的 value function（critic）——机制与出处见第 5 章第九节（DeepSeekMath arXiv:2402.03300v3 式 3）。[DeepSeek-R1 预印本 v1 §2.2.1](https://arxiv.org/html/2501.12948v1#S2.SS2.SSS1) 的式 (1) 同样保留这套结构，式 (3) 给出组内标准化优势，这不是“R1 把 KL 系数设为 0”的证据；具体设置随版本与训练阶段变化。所以「GRPO = 无裁剪无 KL」是错的，「GRPO = 完整 PPO」也是错的。

### 4.2 RLHF 与 RLVR 分开

RLHF 的奖励来自可学习 reward model 加人类偏好（InstructGPT，NeurIPS 2022）；RLVR 的奖励来自可验证器（如 DeepSeek-R1 的推理 RL 阶段；正式论文发表于 Nature 2025）。两者失败模式不同：前者有 reward-model 过优化与偏好偏差，后者有可验证性覆盖与长度偏置问题，Dr. GRPO 对后者有明确批评。

### 4.3 agent 训练 ≠ 多智能体训练 ≠ LLM 给 MARL 生成信号

GiGPO、RAGEN、AgentGym-RL、Agent-R1 是**训练 LLM agent 本身**；ReMAC、LLM-MCA、LAMARL 是 **LLM 产出奖励或信用、被训练的是传统 MARL 策略**；MAPoRL、MARFT、SRPO 是**多智能体 LLM 的联合参数优化**。三类的指标与基线互不通用，混用会把结论放错位置。判断归属时先问「谁的参数在更新、指标与基线是什么」：GiGPO 的基线是 prompt-based 与 actor-critic 类方法，LLM-MCA 的基线是 QMIX 与 MAPPO，MAPoRL 的基线是 single-agent LLM 与 vanilla MAS（标准多智能体系统）——三组对照各自成立，但不能互换。

### 4.4 需修正的综述说法（7 条）

| 综述说法 | 核验结论 |
|---|---|
| GiGPO 归入「信用分配器（LLM 辅助 MARL）」 | 归类需修正：它是面向 LLM agent 的 critic-free RL 算法，信用由 episode／step 两级组结构算出，不是 LLM 生成信号；且应补 NeurIPS 2025 published |
| 附录称 Agent-R1「在 arXiv 和 OpenAlex 中未找到可核实记录」 | 结论错误，Agent-R1 存在：arXiv:2511.14460，v1 于 2025-11-18 发布、v2 改题，属预印本技术报告 |
| ReMAC 列为 2025 代表工作 | 证据强度需降级：是 NeurIPS 2025 **workshop** poster，非主会；实验本身合格（5 次独立运行加误差条） |
| Optima 作为 2024／2025 体系化代表 | venue 需修正：OpenReview 显示 ICLR 2025 Rejected，应标「预印本（ICLR 2025 未接收）」 |
| MARFT 标为「2026 / Anonymous under review」 | 年份需修正：arXiv 首发 2025-04-21，已有作者列表，应写「2025 起有 arXiv 版本；当前正式发表状态未核实，不能沿用旧稿的 ICLR 2026 under review 抬头」 |
| TRACER、MASkills 与 LLM-MCA 并列为「LLM 辅助 MARL 的信用分配」 | 归类需修正：两者训练对象都是 LLM 系统，应归入 RL for multi-agent LLM，与 LLM 给 MARL 出信号分开 |
| ToMAS「用 GRPO 做小规模可行性实验」 | 缺关键边界：作者自陈该实验未显示训练效应、不能确立，只能作流水线可行性与负结果证据 |

### 4.5 本地玩具实验与文献结果不可混

本系列的走廊、导航与谜题均在自定义小环境上完成，属于机制演示，**不构成对任何文献结论的复现**。IQL／VDN／QMIX 与通信对照可在各自协议内讨论；COMA、MADDPG／IDDPG 的历史比较已因实现缺陷撤回，缺陷与重训口径见第 11 章章首，按修正实现同协议重训 5 个种子后的现行数字见第 8、9 章，修复前的旧数字只作追溯记录。机制演示同样要先通过实现与评估核验；环境、表示（表格与网络）、基线和预算口径都与原论文不同。小环境的价值在控制变量：转移规则已知时可以精确计算某个已定义的量；它的结论范围也止于该环境、该表示与该预算。第 11 章的实验规范是阅读文献的工具。

## 五 · 读文献时先问什么

1. **预算对齐了吗？** 双方的环境步、rollout、生成 token、模型调用与硬件时间是否同一口径；没对齐的「提升」可能只是花得更多。
2. **种子与方差呢？** 几个随机种子、报不报 SD 或误差条；单次运行的结果波动范围未知。
3. **基线可比吗？** 对照的是同骨干、同环境、同预算的方法，还是另一个基线传统的产物；跨传统、跨论文的排名都不成立。

三问的顺序就是阅读顺序：先看实验设置，再看统计口径，最后才读结论段。

## 参考文献

- The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (Yu et al., 2022, NeurIPS 2022, published). https://doi.org/10.52202/068431-1787
- Training language models to follow instructions with human feedback (Ouyang et al., 2022, NeurIPS 2022, published). https://doi.org/10.52202/068431-2011
- ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023, ICLR 2023, published). https://arxiv.org/abs/2210.03629
- Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023, NeurIPS 2023, published). https://doi.org/10.52202/075280-0377
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（GRPO 首发）(Shao et al., 2024, preprint). https://arxiv.org/abs/2402.03300
- Can large language models explore in-context? (Krishnamurthy et al., 2024, NeurIPS 2024, published). https://doi.org/10.52202/079017-3818
- Eureka: Human-Level Reward Design via Coding Large Language Models (Ma et al., 2024, ICLR 2024, published). https://openreview.net/forum?id=IEduRUO55F
- Text2Reward: Reward Shaping with Language Models for Reinforcement Learning (Xie et al., 2024, ICLR 2024, published). https://arxiv.org/abs/2309.11489
- Let's Verify Step by Step (Lightman et al., 2024, ICLR 2024, published). https://arxiv.org/abs/2305.20050
- Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System (Chen et al., 2024, preprint, ICLR 2025 未接收). https://arxiv.org/abs/2410.08115
- Mastering diverse control tasks through world models（DreamerV3）(Hafner et al., 2025, Nature, published). https://doi.org/10.1038/s41586-025-08744-2
- Counterfactual value decomposition for cooperative multi-agent reinforcement learning (Liu et al., 2025, Neural Networks, published，仅元数据核验). https://doi.org/10.1016/j.neunet.2025.107692
- Towards Optimal Offline Reinforcement Learning (Li & Kuhn, 2025, preprint). https://arxiv.org/abs/2503.12283
- Understanding R1-Zero-Like Training: A Critical Perspective（Dr. GRPO）(Liu et al., 2025, preprint). https://arxiv.org/abs/2503.20783
- Group Sequence Policy Optimization（GSPO）(Zheng et al., 2025, preprint). https://arxiv.org/abs/2507.18071
- DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning (Guo et al., 2025, Nature 645:633–638, published). https://doi.org/10.1038/s41586-025-09422-z （算法式引用预印本 v1：https://arxiv.org/html/2501.12948v1#S2.SS2.SSS1）
- Group-in-Group Policy Optimization for LLM Agent Training（GiGPO）(Feng et al., 2025, NeurIPS 2025, published). https://doi.org/10.52202/085713-1544
- MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning (Park et al., 2025, ACL 2025, published). https://doi.org/10.18653/v1/2025.acl-long.1459
- L2M2: A Hierarchical Framework Integrating Large Language Model and Multi-agent Reinforcement Learning (Geng et al., 2025, IJCAI 2025, published). https://doi.org/10.24963/ijcai.2025/12
- Leveraging Large Language Models for Effective and Explainable Multi-Agent Credit Assignment（LLM-MCA）(Nagpal et al., 2025, AAMAS 2025, published). https://doi.org/10.65109/nokw1037
- Why Do Multi-Agent LLM Systems Fail?（MAST）(Cemri et al., 2025, NeurIPS 2025, published). https://doi.org/10.52202/085713-4082
- MARFT: Multi-Agent Reinforcement Fine-Tuning (Liao et al., 2025, arXiv 版本可用，当前正式发表状态未核实). https://arxiv.org/abs/2504.16129
- Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning (Cheng et al., 2025, preprint, 技术报告). https://arxiv.org/abs/2511.14460
- ReMAC: Large Language Model-Driven Reward Design for Multi-Agent Manipulation Collaboration (Li et al., 2025, NeurIPS 2025 workshop, 非主会). https://openreview.net/forum?id=CWYWhLho0a
- SRPO: Setwise Relative Policy Optimization for Multi-Agent LLMs (Yang et al., 2026, 既有记录，本次未复核). https://arxiv.org/abs/2609.08452
- TD-MPC2: Scalable, Robust World Models for Continuous Control (Hansen, Su & Wang, 2024, ICLR 2024, published). https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html
- Do As I Can, Not As I Say: Grounding Language in Robotic Affordances（SayCan）(Ichter et al., CoRL 2022；PMLR 205:287–318, 2023, published). https://proceedings.mlr.press/v205/ichter23a.html
- Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents (Huang et al., 2022, ICML 2022, PMLR 162:9118–9147, published). https://proceedings.mlr.press/v162/huang22a.html


## 附 · 资料下载

- [复现指南（环境、命令、数据口径）](/assets/attach/rl-practice/复现指南.md)
- [实验数据摘要（逐组配置与结果）](/assets/attach/rl-practice/数据摘要.md)
- [代码与原始日志打包](/assets/attach/rl-practice/rl-practice_code.zip)
