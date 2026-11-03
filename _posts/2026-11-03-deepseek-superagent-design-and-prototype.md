---
title: "DeepSeek Super-agent：从单次对话到持续协作"
date: 2026-11-03 00:00:00 +0800
categories: [前沿见闻]
topic: ai-frontiers
tags: [DeepSeek, Super-agent, Agent, 多智能体, 任务编排, 安全]
excerpt: 从事件路由、主动授权、并发与故障恢复出发，解析 DeepSeek 开源 Super-agent 原型的工作机制和安全边界。
---

> RedMoon · 资料截至 2026 年 11 月 3 日。本文分析 DeepSeek 公开的 `dsh-superagent-prototype`，源码固定于 `6337fc823ce20251c8de8085948165c81aaa701e`。[1]

一个团队同时在 GitHub 上讨论几个问题：有人报告按钮错位，有人追问已有修复，CI 也在不断报告测试结果。如果每来一条消息，都由人打开一个新的 Agent 会话、复制上下文、等待结果，再把结果贴回讨论，真正稀缺的就不只是模型能力，还有人的注意力。

DeepSeek 团队提出的 **Super-agent** 试图把这些分散的交互组织成持续协作：系统接收事件，判断哪些需要处理，把工作交给各自的执行会话，并将结果送回原来的讨论位置。开源仓库给出的是面向**单个 GitHub 仓库**的主动式编程 Agent 原型，可选接入飞书；它依赖 DeepSeek Harness 和已配置的模型服务。[1]

本文沿三个问题展开：一条消息怎样变成任务？系统怎样决定是否主动接单？任务持续运行时，状态恢复、权限与人工把关分别由谁负责？

## 一、先分清设计文章与开源原型

Super-agent 的设计文章讨论了一种持续在线的 Agent 系统：用 Router 统筹事件和任务，用 Worker 分别执行工作，并让系统跨会话、跨信道地维持上下文。本文对设计文章的介绍依据注明转载来源的版本；原文入口与转载链接见文末。[2]

开源原型把这一方向落到更具体的场景中。仓库 README 明确说明：这是供参考的研究原型，不是 DeepSeek 官方产品；公开代码经过适配，与团队内部版本及设计文章描述的实现并不完全相同，文章讨论的部分设计仍是未来方向。[1]

| 问题 | 设计文章讨论的方向 | 固定版本原型中可以直接检查的内容 |
| --- | --- | --- |
| 任务组织 | Router 接收并协调事件，Worker 执行具体任务 | Main Session 研判和路由，Ticket 绑定 Executor Session |
| 使用场景 | 跨任务、跨信道的持续协作 | 一个 GitHub 仓库，飞书为可选来源 |
| 上下文 | 按需展开关联对象，形成链式召回 | 评估时提供有界线程窗口及按需读取接口 |
| 执行环境 | 讨论计算、状态和环境的分离 | 每个 Ticket 一个 Session 和工作目录 |
| 系统成熟度 | 架构探索及开放问题 | 研究代码，需自行审查和适配 |

这一区分很重要：文章里的设计可以帮助理解**为什么这样做**，代码和文档才能说明**当前开源版本具体做到了哪里**。

## 二、一条消息怎样变成一项工作

原型使用一组领域对象描述任务从进入到交付的过程。[3]

| 概念 | 通俗理解 | 记录或负责什么 |
| --- | --- | --- |
| Source | 消息来源 | 将 GitHub 或飞书上的事件转换成系统可处理的对象 |
| Trigger | 一次触发 | 一条消息、一次 CI 状态变化等具体事件 |
| Thread | 一段讨论 | 按时间组织相关 Trigger，例如一个 Issue 的讨论 |
| Eval / Decision | 评估与决定 | 判断新事件是接单、忽略、停止，还是未能完成判断 |
| Ticket / Mandate | 工单与授权 | Ticket 表示一项工作；Mandate 保存其范围和授权依据 |
| Assignment / Executor | 派工与执行器 | 把 Ticket 交给具体 Worker Session 执行 |
| Channel | 回写渠道 | 把确认、问题和结果送回对应位置 |

将它们串起来，主路径是：

```text
GitHub / 飞书
    ↓ Source 转换
Trigger 进入 Thread
    ↓
准入 + 路由 / 研判 → Decision
    ↓ 接单
Ticket + Mandate → Assignment → Worker Session
    ↓
Channel 将结果送回原讨论
```

这里的 `Mandate` 不只是任务描述。它还记录了授权强度、来源、所依据的策略版本和匹配规则。这样，系统以后再看到相关消息时，可以把“补充了什么信息”和“是否允许扩大工作范围”分开处理。[3][4]

### 用一个 Issue 走完整条路径

假设已获准的团队成员在 Issue 中写：“@agent，请修复设置页按钮在手机上溢出容器的问题。”下面是按照原型规则整理的示意，不是一次实测记录。

1. **接收**：GitHub Source 发现消息，生成带有来源和发送者的 Trigger。`Intake.receive` 按 Trigger ID 检查重复，再记录它属于哪个 Thread。[5]
2. **准入和路由**：消息明确指向 Agent，且发送者符合策略时，可作为明确请求交给 Main Session 路由。它与根据讨论内容推断出来的主动任务走不同的处理路径。[4]
3. **创建工作**：决定接单后，系统建立 Ticket、Mandate 和 Assignment。Mandate 中的范围应对应“修复手机端按钮溢出”，而不是默认包含整页重新设计。[3]
4. **执行**：Executor 为 Ticket 准备独立 Session 和目录。部署者可以配置准备命令，例如为任务建立 Git worktree；如果没有配置，Worker 从空目录开始。[6]
5. **交付**：Worker 检查问题、修改并验证，必要时创建 pull request。结果或需要人回答的问题通过 Channel 返回团队原来的讨论位置。[1][6]
6. **跟进**：后来 CI 在该 pull request 下报告失败，或者人提出与修复范围有关的补充，订阅关系可以把消息送到已有 Ticket；Worker 继续处理。超出原范围的人工请求应交回路由，不应因为出现在同一讨论中就自动获得更大授权。[4]

上面既有代码能检查的步骤，也有模型要做的判断。系统能够记录范围并把要求交给 Worker，不等于模型一定会正确理解每条消息。

## 三、为什么后续消息不必每次重新派工

一项工作开始后，团队往往会继续在 Issue 或 pull request 中讨论。原型让 Ticket **订阅** Thread：Thread 上的新事件可以扇出给订阅它的 Ticket，成为 Worker 的上下文。一个 Thread 可以对应多个 Ticket，一个 Ticket 也可以跟随多个 Thread。[3]

这与每收到一条评论就重新创建任务不同。对于仍由已有 Ticket 处理的人工跟进，消息先交给对应 Worker；如果它属于当前范围，Worker 可以继续。如果请求超出范围，Worker 可以交回，等没有其他 Ticket 持有该消息后再进入新的评估。[4]

还有两个容易混淆的关系：

| 关系 | 作用 | 是否自动授予新工作权限 |
| --- | --- | --- |
| Subscription，订阅 | 将 Thread 的新事件送到相应 Ticket | 不授予 |
| Correlation，关联 | 为评估提供相关 Thread 的上下文 | 不负责路由，也不授予 |

例如，Worker 为修复按钮溢出而关注另一个相关 Issue。那个 Issue 的讨论可以帮助它理解现有样式，但不能仅因为它读到了讨论，就把另一个功能也纳入自己的任务。

需要特别注意：订阅的 Thread 可以把 bot 和 CI 消息作为上下文送到 Worker。**上下文在授权模型中不授予新权限，并不意味着模型已经免疫恶意指令。**后文会把策略准入与实际安全隔离分开讨论。[4]

## 四、主动性从哪里来，又在哪里停下

系统的主动工作并不是“看到任何问题就修”。授权文档将事件分为明确请求和推断候选：符合来源策略、直接向 Agent 发起的请求，可以作为明确请求；没有直接指向 Agent 的讨论、获准 bot 的消息等，在满足接收配置时可能成为推断候选。[4]

对于推断候选，原型提供三种模式：

| 模式 | 遇到推断出的工作时 |
| --- | --- |
| `off` | 不将推断情况作为候选接单 |
| `observe` | 默认模式；可以评估并记录“本来会接单”，但不实际接单 |
| `auto` | 满足策略和规则的推断候选可以自动接单 |

明确请求在三种模式下都可以按照来源和身份策略处理。`observe` 的用途是让团队先查看系统会选择哪些任务，再决定是否扩大主动工作范围。[4]

### 准入和模型研判分别负责什么

准入层先根据来源、发送者、是否指向 Agent、接收范围等确定候选的强度。之后，推断候选还要由 Main Session 对规则作判断；代码会检查研判结果的结构和条件。[4][7]

固定版本的 `assessmentOutcome` 对推断任务提出了几项关键要求：

1. 每条已启用规则都要有且只有一次判断。
2. 至少有一条已启用规则匹配。
3. 研判结果需要提供意图证据。
4. 需要确认有人要求实现、目标已有明确依据，且没有尚待解决的设计决策。
5. 只有 `auto` 模式允许符合条件的推断候选真正接单；在 `observe` 模式下，正向结果会记为 `would_accept`。[7]

例如，“按钮左右间距都应为 8 像素，请修成一致”有明确目标；“这个页面不太好看”没有给出可执行的目标。默认规则中对视觉修改也有类似区分：定位到元素或发现不一致，不自动等于已确定新的设计方案。[7]

代码检查能够保证研判输出满足指定结构和条件，但“证据是否真的支持该判断”仍依赖模型及后续审查。对高影响动作，不能只靠模型给出的文字说明。

### 策略改变时会发生什么

策略包含规则、接单模式和各 Source 的准入设置。原型为这组配置生成版本标识；接单时的 Mandate 记录所依据的版本。策略在评估期间改变时，旧评估会被丢弃并重新排队；受到影响的运行中 Ticket 会收到策略变化通知。[4][7]

变化不会立即撤销所有正在运行的 Mandate。已结束 Ticket 恢复前则会重新检查相应授权条件：例如推断任务原来依赖 `auto` 模式和某条规则，恢复时这些条件可能已经不再成立。[4]

这是一种对长任务有意义的处理方式：既保留当初为什么接单，也让后续恢复能依据当前策略重新判断。

## 五、Router、Worker 与并行工作

在原型中，Main Session 承担路由和研判职责；每个 Ticket 对应一个 Executor Session，代码中通常称为 Worker。二者分工不同：前者决定工作去向，后者围绕具体任务读取上下文、操作工具并报告结果。[3][6]

```text
                         Main Session
                    接收候选、研判、路由
                    /        |        \
             Ticket A    Ticket B    Ticket C
             Worker A    Worker B    Worker C
             独立会话     独立会话     独立会话
             任务目录     任务目录     任务目录
```

这样组织的直接目的，是让多个任务各自保留会话和工作现场。Worker A 的修复过程不必和 Worker B 的全部工具输出混在同一条对话里；Main Session 可以通过结构化状态了解哪些工作正在进行、等待人回复或准备审阅。

### 并发不是无限启动

执行器用 Worker slot 控制并发。启动准备、唤醒已有 Worker、执行中的 turn，以及 turn 结束后仍在运行的部分后台工作，都会占用 slot。源码默认 `maxWorkers` 为 3，实际部署可以调整。[6][7]

一个有意思的设计是：当普通 slot 已满时，人的跟进或状态请求可以使用**额外的一个 slot**；已有任务的唤醒也优先于新启动。它的目标是避免新工作占满资源后，团队成员无法继续跟进旧任务。这是调度优先级，不是对实际响应时间的保证。[6][8]

### 后台命令与新消息

Worker 的预设会在前台 shell 命令运行 5 秒后将其转为后台任务，方便会话在后续步骤接收新消息。源码还使用 `queue`、`steer`、`inject` 等方式向 Session 投递输入。[6][9]

这让系统具备一定的异步协作能力，但不能把“消息可以投递”理解成任意时刻都能抢占当前模型推理，更不能据此断言设计文章中讨论的全部双工体验已经在开源原型中实现。

### 两个时间间隔不能混为一谈

原型内部的调度 pass 约每 2 秒运行一次，新工作到来时也可请求立即运行；GitHub Source 的默认轮询间隔是 60 秒，GitHub 要求更长间隔时还会遵守更长的值。[10]

因此，“内部每 2 秒检查一次”不意味着 GitHub 上的新评论一定能在 2 秒内被看到。来源发现、排队、模型处理和工具执行是不同阶段。

## 六、上下文怎样保持有用，而不是无限变长

一个长期运行的 Agent 不可能把全部历史对话、每次工具输出和所有旧任务始终塞进同一个上下文窗口。设计文章的转载版本介绍了 **Inline & Spill**：先给出当前判断最相关的信息与引用，过长内容留出继续读取的入口；多个对象之间再通过 ID 连接，按需沿引用逐步回看。文章把后一种多跳查找称为 Chain of Recall。[2]

固定版本原型中，可以直接核对的是一个较具体的机制：对主动候选进行研判时，`MasterAssessor` 提供有界的 Thread 窗口。触发消息、邻近消息、其他窗口消息和摘要预览采用不同长度；需要更多信息时，Main Session 可以按消息、Thread、根消息、父消息、仓库文件、目录或图片继续读取。[11]

可以把它理解成一份带链接的案卷：

```text
候选卡片：当前事件 + 附近讨论 + 关联工单摘要
    ↓ 发现线索不足
按消息 ID / 线程 / 文件路径继续读取
    ↓
结合规则作出接单、忽略或未完成的决定
```

这个机制减少了“一次把所有历史都塞给模型”的需要，也保留了进一步核对的入口。它并不自动保证模型会读取所有关键内容；评估窗口、实际读取和模型判断仍应一起检查。

另外，源码中的 `src/core/memory.ts` 是供内存存储使用的实现名称，不能因为文件名含 `memory`，就推断系统已经实现向量数据库或语义长期记忆。[11][12]

## 七、持久化与恢复：重启之后从哪里继续

持续协作要求系统记住的不只是聊天记录，还包括：事件是否收过、任务是否已接单、消息是否已经发送，以及中断时有哪些工作没有做完。

原型用 SQLite 保存 Trigger、Thread、Decision、Ticket、effect 队列和 outbox 等记录。Engine 在事务内更新领域状态，并将外部工作排入持久化队列；事务提交后，再由对应处理器执行。[12]

| 机制 | 解决的问题 |
| --- | --- |
| Trigger ID 去重 | 相同事件再次进入时避免重复登记 |
| Effect 队列 | 先记下需要执行的工作，再异步处理并重试 |
| Outbox | 记录待发送、已发送或丢弃的消息及其状态 |
| Session、目录与 Assignment | 保存任务执行现场，并在中断后组织继续 |
| 健康状态与面板 | 将持续失败或需要人决定的情况暴露出来 |

例如，工作已经完成，但向讨论区发送结果时网络暂时中断。Outbox 可以保留待发送记录，按退避策略重试；同一 Ticket 发往同一讨论的消息还会按相应规则维持先后顺序。[6][12]

重启时，原型会为被中断的运行中 Assignment 组织恢复，并提示 Worker 先检查目录、Git 历史和 GitHub 状态，再决定是否重复外部动作。[6]

这里有一个工程上重要的边界：**本地去重和可重试队列，不等于所有外部副作用都能严格做到恰好一次。**如果远端已经收到请求，本地却在记下成功前崩溃，恢复时仍需要依靠操作本身的幂等性、远端状态检查或人工复核。

持久化还有运维成本：固定版本的架构文档说明部分历史记录目前不会自动清理。长期使用需要自己考虑数据增长、备份和保留策略。[12]

## 八、安全边界：任务隔离不等于权限隔离

每个 Ticket 有独立目录，适合组织代码修改和并行工作；但部署文档明确说明，Worker 以运行者的操作系统用户身份执行命令，使用 `danger-full-access`，无需逐条命令审批。目录和命令包装本身不是安全边界：Worker 仍可能通过绝对路径执行程序，并读取该账户可访问的内容。[6][13]

因此应把几层控制分开看：

| 控制层 | 原型提供或依赖什么 | 仍需关注什么 |
| --- | --- | --- |
| 谁能发起任务 | Source 策略、允许名单、准入表 | 来源配置过宽会扩大可接单范围 |
| 任务语义范围 | Mandate、匹配规则及 Worker 指令 | 自然语言范围需要模型遵守，不能当作 OS 权限墙 |
| 本机权限 | 运行者的 OS 用户权限 | 使用专用账户，避免放入无关凭证和敏感文件 |
| 仓库操作 | Agent 的 GitHub 身份与仓库授权 | 使用最小必要权限，并由 GitHub 强制分支保护 |
| 合并审批 | 默认交付要求中的提示词 | Host 没有专门的合并决策逻辑；需要保护规则及人工审阅 |
| 外部内容 | 订阅 Thread 的消息、CI 输出等 | 不可信文本可能包含提示注入，需测试并限制可产生的副作用 |

对于推断工作，`mandateInstructions` 会向 Worker 的系统提示中加入范围要求；这属于模型执行时要遵循的指令，并非操作系统强制检查。[8]

合并也是同理。执行文档描述了默认提示词下的交付要求，并建议用 GitHub 分支保护强制审批；提示词可以表达团队流程，真正的仓库访问控制还要由 GitHub 等外部机制实施。[6]

部署指南建议使用独立的 DeepSeek Harness profile、Agent GitHub 身份及专用 OS 账户。管理面板默认监听本机 `127.0.0.1`；本机监听与同源检查不能替代可公开服务所需的账号认证和访问控制。[13]

## 九、它与 LangGraph、MCP、OpenHands 的关系

这几个名字经常一起出现在 Agent 讨论中，但它们处理的问题处于不同层次。[14][15][16]

| 系统或协议 | 主要解决什么 | 与 Super-agent 原型怎样配合或区分 |
| --- | --- | --- |
| Super-agent 原型 | 把外部事件组织为受策略约束的持续任务，并协调 Worker | 关注任务准入、派工、跟进、状态与交付 |
| LangGraph | 用图、状态和 checkpoint 组织可恢复的工作流；支持中断后继续 | 可用来实现某些 Agent 工作流；与任务组织思路并非互斥 |
| MCP | 约定模型应用与工具、资源之间的交互接口 | 可以提供工具接入；协议本身不替应用决定授权、任务路由或沙箱 |
| OpenHands Docker sandbox | 在容器中运行 Agent 的命令与工具 | 关注执行环境隔离；挂载给容器的可写目录仍需审查 |

例如，团队可以借鉴 Super-agent 的 Ticket 和授权模型，同时用某种工作流运行时保存步骤状态，再将代码执行放进具备明确权限配置的隔离环境。每一层都要检查自己的边界，不能因为其中一个环节使用了 Agent 框架或协议，就推断其他环节已经安全。

LangGraph 的 checkpoint 还带来另一项可迁移的经验：中断恢复可能重新运行节点，因此节点中的外部副作用需要考虑幂等性。这与原型重启后检查 GitHub 状态的思路，解决的是相近的恢复问题，但二者的接口和实现不同。[6][14]

## 十、怎样评估一个持续协作 Agent

仓库包含针对准入、领域规则、并发、重启、消息发送和服务管理等方面的测试；其中一部分运行时测试使用模拟 Harness host 或替身命令。它们展示了开发者希望维持的行为契约，但不能直接当成真实项目中的任务成功率。[12]

如果要在自己的仓库评估，可以把结果拆成以下几类：

| 评估问题 | 示例指标或检查方法 |
| --- | --- |
| 该不该接单 | 在已标注事件上统计应接却未接、无需处理却接单的数量 |
| 任务是否做对 | 用真实 Issue 和测试检查修复、回归及人工审阅结果 |
| 是否遵守授权 | 检查是否改动范围外文件、扩大任务或绕过审批 |
| 是否造成打扰 | 统计无用确认、重复评论、无价值 PR 和人工中断 |
| 能否从故障恢复 | 在发送前后、执行中、重启时注入故障，核对重复动作和漏处理 |
| 能否抵抗恶意上下文 | 在 Issue、文档或工具返回中放入越权指令，同时检查正常任务与攻击结果 |

AgentDojo 给安全评估提供了一个参考思路：让 Agent 完成正常任务，同时在其会读取的不可信内容中测试攻击。它是一套独立的评估环境，不是对这个 DeepSeek 原型的现成成绩。[17]

实践中，首先收集 `observe` 模式下的候选和决定，比对团队成员的判断；再从权限窄、改动可审阅的小任务开始。评估时同时记录模型、提示词、策略版本、事件范围及人工验收标准，才能解释结果来自哪里。

## 十一、值得借鉴的设计

这个原型最有启发性的地方，是把 Agent 的持续协作拆成了几项可以分别检查的工程职责：

1. **事件和工作分离**：Trigger 是发生了什么，Ticket 是已经接受的工作，避免每条消息都变成新任务。
2. **授权和上下文分离**：消息可以帮助理解任务，但不因此自动扩大 Mandate。
3. **任务和执行分离**：Ticket 有自己的生命周期，可以在不同 Assignment 中恢复，不把一次模型 turn 当成整个任务。
4. **状态和副作用分离**：先在事务中记录需要做什么，再由持久化队列执行和重试。
5. **主动和打扰分离**：通过 `observe`、规则和面板，先观察系统会做什么，再逐步放开自动接单。

持续 Agent 的能力不只体现在“会不会调用工具”。它还取决于能否判断什么时候行动、能否把任务持续推进，以及出错后是否能让人看懂并接手。

## 参考资料

源码和仓库文档固定到本文分析的 commit。设计文章的原始入口本次访问受到限制，正文所述设计方向参考下列注明来源的转载版本。

1. [DeepSeek：dsh-superagent-prototype README](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/README.md)
2. [DeepSeek：Super-agent 设计探索，知乎原文入口](https://zhuanlan.zhihu.com/p/2099987320443237060)；[注明原始来源的转载](https://blog2.eimoon.com/p/deepseek-super-agent-design/)
3. [领域模型与任务流转](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/docs/001-domain.md)
4. [授权、接单模式和跟进规则](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/docs/002-authorization.md)
5. [Intake：事件接收、去重和扇出](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/src/core/intake.ts)
6. [执行、并发、交付与恢复](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/docs/003-execution.md)
7. [Policy：策略、准入和研判结果校验](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/src/policy.ts)
8. [DSH Executor：授权指令、任务目录和 Worker slot](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/src/executor/dsh.ts)
9. [Session 消息投递与角色配置](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/src/app/sessions.ts)
10. [内部调度 pass](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/src/app/passes.ts)；[GitHub Source 轮询](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/src/sources/github.ts)
11. [MasterAssessor：有界上下文与按需读取](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/src/app/assessment.ts)
12. [架构、测试索引和持久化](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/docs/007-architecture.md)；[内存仓储实现](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/src/core/memory.ts)
13. [部署要求与权限边界](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/docs/006-deployment.md)；[本地管理接口检查](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/src/app/http.ts)
14. [LangGraph：Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)；[Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
15. [Model Context Protocol：Tools，2025-11-25 规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
16. [OpenHands：Docker Sandbox](https://docs.openhands.dev/openhands/usage/sandboxes/docker)
17. [Debenedetti 等：AgentDojo，arXiv:2406.13352](https://arxiv.org/abs/2406.13352)
