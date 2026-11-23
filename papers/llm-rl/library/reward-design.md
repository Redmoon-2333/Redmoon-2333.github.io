---
layout: post
title: P9E：LLM 与 VLM 设计奖励
permalink: /papers/llm-rl/library/reward-design/
date: 2026-11-24 00:00:00 +0800
topic: llm-rl
categories:
- 论文精读
math: false
reading_prev:
  title: P9D：工具、过程反馈与多模态
  url: /papers/llm-rl/library/tools-and-multimodal/
reading_next:
  title: P9F：冻结模型、控制器与分层决策
  url: /papers/llm-rl/library/frozen-controllers/
---

把语言模型放在裁判或奖励设计者的位置，观察下层策略用这些评分能学到什么。

[返回 P9 文献导航](/2026/11/23/llm-rl-p9-reading-index/)。从 [E3 的双开关任务](/2026/11/23/llm-rl-e3-reward/) 入手，先看程序是否可执行，再和手写规则比较最终成功率。

## LLM / VLM 辅助奖励设计（21项）

核对奖励是程序、偏好还是数值；再读它如何接到下层策略的训练。

- `U002 / 2026` [CoRe: Combined Rewards with Vision-Language Model Feedback for Preference-Aligned Reinforcement Learning](https://proceedings.mlr.press/v306/ni26d.html)<br />
  用 VLM 反馈组合显式形式奖励与从视频偏好中学习的残差奖励。
- `U040 / 2026` [GARDEN: Generalist Autonomous Reward Design Through Evidence Navigation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2610.37926v1)<br />
  让工具型 LLM 检查代码、训练记录和行为证据，生成并冻结可跨场景使用的奖励程序。
- `U041 / 2026` [RM-VLM: From Natural-Language Task Descriptions to Vision-Grounded Reward Machines](https://arxiv.org/abs/2611.08308v1)<br />
  LLM 从任务描述构造奖励机，VLM 接地视觉命题并评估转移进度以提供稠密反馈。
- `U042 / 2026` [Causal Reward World Models: Zero-shot Reward Design for Automated Skill Generation](https://arxiv.org/abs/2606.23280v1)<br />
  通过离线交互数据学习候选奖励项与目标物理变量的结构先验，再约束零样本奖励生成。
- `U197 / 2026` [LLM-ALSO: LLM-Driven Adaptive Learning-Signal Optimization for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2605.29293v1)<br />
  Critic LLM 从学习曲线与行为证据诊断，Generator LLM 提议塑形参数，候选在短分支中和不变奖励对照，验证后才合入主训练。
- `U198 / 2026` [LLM-Enhanced Multi-Agent Reinforcement Learning for Unified Electric Vehicles-Charging Station-Grid Optimization in Public Charging Systems](https://arxiv.org/abs/2609.13805v1)<br />
  语言模型选择车、充电站与电网各自的状态特征，并按情境为三方目标分配权重，形成加权奖励。
- `U206 / 2026` [MARLF: Memory-Augmented Multi-Agent Reinforcement Learning with Large Language Model Feedback](https://doi.org/10.5281/zenodo.19998804)<br />
  把语言模型的稀疏行为评语存入记忆，再用预测与动态加权把反馈变成训练时的稠密奖励。
- `U217 / 2026` [Multi-Agent Orchestration with the Common-Sense Reasoning Capabilities of LLMs for Autonomous Driving](https://arxiv.org/abs/2608.20129v2)<br />
  离线语言模型阅读驾驶仿真日志并建议修改奖励，经人工确认后再训练低层策略，运行期由编排器协调控制模块。
- `U232 / 2026` [Regime-Conditional Stabilisation of LLM-Augmented Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2607.04470v1)<br />
  动态改写塑形权重会使离策略回放携带旧奖励标签；以分阶段冻结和指数滑动平均抑制权重漂移。
- `U268 / 2026` [Autonomous Volt/Var Control in Active Distribution Networks via LLM-Driven Dynamic Reward Shaping](https://www.mdpi.com/2079-9292/15/16/3504)<br />
  根据配电网控制策略的训练表现调整多目标奖励权重，使低层策略在电压约束和网损之间改变优化侧重。
- `U001 / 2025` [ELEMENTAL: Interactive Learning from Demonstrations and Vision-Language Models for Reward Design in Robotics](https://proceedings.mlr.press/v267/chen25at.html)<br />
  把语言、视觉示范与逆强化学习结合，分别解决奖励特征表达与权重对齐。
- `U043 / 2025` [MAESTRO: Multi-Agent Environment Shaping through Task and Reward Optimization](https://arxiv.org/abs/2511.19253v2)<br />
  把 LLM 放在训练外层，生成交通课程、模板奖励和先验策略正则，部署 MADDPG 控制器。
- `U101 / 2025` [ReMAC: Large Language Model-Driven Reward Design for Multi-Agent Manipulation Collaboration](https://sea-workshop.github.io/2025/)<br />
  外层迭代团队与个体奖励群体，内层用 MARL 学习合作操纵策略。
- `U102 / 2025` [LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation](https://doi.org/10.1109/LRA.2025.3577527)<br />
  LLM 自动生成先验策略与奖励函数，指导多机器人合作策略学习。
- `U280 / 2025` [R*: Efficient Reward Design via Reward Structure Evolution and Parameter Alignment Optimization with Large Language Models](https://proceedings.mlr.press/v267/li25v.html)<br />
  LLM 参与奖励程序结构变异，多种轨迹评价函数投票筛出偏好样本，再用偏好学习调节奖励参数供控制策略训练。
- `U151 / 2024` [Auto MC-Reward: Automated Dense Reward Design with Large Language Models for Minecraft](https://arxiv.org/abs/2312.09238v2)<br />
  奖励程序员生成可执行 Python 奖励，评审器查语法和语义，轨迹分析器总结失败供下一轮改写。
- `U260 / 2024` [Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning](https://arxiv.org/abs/2310.12921v2)<br />
  以预训练视觉语言模型计算视觉观测与文字目标的匹配信号，用作零样本奖励训练模拟人形机器人策略。
- `U081 / 2023` [Eureka: Human-Level Reward Design via Coding Large Language Models](https://arxiv.org/abs/2310.12931v2)<br />
  用环境代码和训练反馈迭代演化奖励程序，再让低层 RL 获取复杂控制技能。
- `U082 / 2023` [Text2Reward: Reward Shaping with Language Models for Reinforcement Learning](https://arxiv.org/abs/2309.11489v3)<br />
  从语言任务与紧凑环境表示生成稠密奖励程序，并允许依据反馈逐步修订。
- `U210 / 2023` [Motif: Intrinsic Motivation from Artificial Intelligence Feedback](https://arxiv.org/abs/2310.00166v1)<br />
  令 LLM 比较成对的状态文字描述，用这些偏好学习内在奖励，再训练 RL 智能体探索 NetHack 环境。
- `U234 / 2023` [Reward Design with Language Models](https://arxiv.org/abs/2303.00001v1)<br />
  把任务描述和行为表示输入语言模型，将产生的评价作为奖励引导下游策略。

下一个查阅页：[P9F：冻结模型、控制器与分层决策](/papers/llm-rl/library/frozen-controllers/)。
