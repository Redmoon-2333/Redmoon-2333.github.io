---
title: "REA 技术解析：让 Agent 沿着证据做逆向工程"
date: 2026-10-13 18:10:00 +0800
categories: [前沿见闻]
topic: ai-frontiers
tags: [REA, Agent, MCP, 逆向工程, Electron, 证据驱动]
excerpt: 从源码分层、Evidence 与未解问题管理出发，解析 REA 如何连接 Agent、静态分析和运行证据，并用明确验证范围推进软件重建。
---

> RedMoon · 资料截至 2026 年 10 月 13 日。分析对象为 `morluto/rea`，源码固定于 `d13c9f840d36a321a56917e75c48afcb8d72920e`；同期 npm 稳定版为 `rea-agents 6.4.0`。[1][2]

看到一个应用中的功能，往往很容易描述它做了什么，却很难回答它怎样完成：按钮触发了哪个模块，数据经过了哪些进程，最后又调用了哪个原生函数？当源码缺失，或者只拿得到安装包、压缩后的 JavaScript 和可执行文件，这些问题就需要逆向分析。

REA，全称 Reverse Engineer Anything，把这条调查路径开放给 AI Agent。Agent 通过 MCP 调用 REA，REA 再调用分析器、读取制品或观察运行中的程序，返回带有来源的结果；Agent 根据结果提出下一轮问题，最终解释行为，或写出可以验证的新实现。[1][3]

它值得关注的工程设计，是把**跨工具分析、证据保存、未解问题和验证范围**接到同一个工作流里。理解这套设计，比单看“逆向任何东西”的名字更能判断它在实际项目中的位置。

## 一、REA 在 Agent 系统中的位置

REA 的生产入口是 CLI 和 MCP server，核心实现采用分层的 ESM TypeScript。外部 Agent 负责制定调查策略、解释结果和生成实现；REA 提供分析工具与结果契约。Hopper、Ghidra、IDA 等引擎承担原生代码的反汇编和反编译，JavaScript 路径则通过解析器及应用图分析组织程序关系。[1][3][4]

这可以拆成四个角色：

| 角色 | 负责什么 | 典型输出 |
| --- | --- | --- |
| Agent | 将问题拆成查询，决定下一步，解释证据或生成实现 | 调查结论、重建代码、测试方案 |
| REA 工作流 | 选取分析路径，组织调用，保存与比较结果 | Evidence、应用图、功能路径、未解问题 |
| 分析 provider | 从目标文件或运行环境中提取信息 | 汇编、伪代码、AST 关系、脚本及网络观察 |
| 验证工具与夹具 | 比较原目标和候选实现，检查明确的行为范围 | 差异报告、测试结果、编译产物对照 |

MCP 是 Agent 调用工具的接口。REA 的 CLI 与 MCP 共用 application 层的工作流，因此终端脚本和 Agent 可以进入同一套分析与证据体系。这里的“一套”指共享语义和实现；两种入口的进程与会话生命周期仍有区别。[3][5]

它的使用场景集中在已交付的软件制品、缺源码的功能调查，以及需要观察实际行为的问题。对于拥有完整源码的普通仓库架构分析，项目自身的工作流说明也建议直接使用常规代码阅读工具。[6]

## 二、源码架构：把分析引擎与调查逻辑分开

源码中的依赖方向围绕 provider-neutral 的业务语义组织。下表将仓库架构图压缩成阅读时最有用的路径。[3]

| 层 | 主要位置 | 作用 |
| --- | --- | --- |
| 入口 | `scripts/rea.mjs`、`src/cli.ts`、`src/main.ts` | 分派一次性 CLI 命令或启动 MCP stdio 服务 |
| 接口适配 | `src/server/`、`src/cli/` | 校验请求，映射工具契约，编码结果和错误 |
| 生产组装 | `src/composition/` | 创建 provider、会话、可选能力和资源生命周期 |
| 调查工作流 | `src/application/` | 组合分析、Evidence、比较和功能追踪 |
| 领域语义 | `src/domain/` | 证据身份、应用图、覆盖状态、比较规则和验证判断 |
| 契约 | `src/contracts/` | 定义调用者可见的输入、输出与工具目录 |
| 引擎与平台边界 | `src/ghidra/`、`src/hopper/`、`src/ida/`、`src/browser/` 等 | 适配工具协议、文件格式与平台能力 |
| 进程基础设施 | `src/process/`、`src/windows/`、`native/windows/` | 超时、取消、进程归属、私有工作区与清理 |

以一次 JavaScript 功能追踪为例，MCP handler 先解析 application Evidence 输入，调用共享的工作流函数，再记录输入证据和新结果，最后通过统一的结果编码返回给客户端。功能追踪自身消费已经得到的图，不需要再次读取应用或执行目标。[7][8]

这种分层有两个直接收益。首先，工具名称与调用者需求可以保持稳定，Ghidra、Hopper 等引擎的协议差异留在 adapter 中。其次，证据验证、比较规则和错误语义可以同时服务 CLI 与 MCP，减少两套入口给出不同解释的机会。

### Provider 选择有明确依据

`AnalysisProviderRegistry` 将候选 provider 按 ID 排序，检查可用性、目标支持和分析 profile。自动选择存在唯一可用候选时才绑定；多个可用候选产生 `ambiguous`，要求调用者选择。指定引擎失败时，系统保留该引擎的失败原因和诊断信息。[9]

这个选择机制让调查结果能够回答“是谁分析的、用什么配置分析的”。不同引擎对函数边界、类型、调用关系的恢复可能不同，把 provider 选择记录下来，是解释这些差异的前提。

## 三、Evidence：一次分析究竟保存了什么

REA 将分析结果放入统一的 Evidence record。它保存的不只是结论，还包括目标身份、引擎身份、参数、来源位置，以及这次结果能支持的判断类型。[4][10]

| 字段 | 阅读时要问的问题 |
| --- | --- |
| `subject` | 分析的是哪份制品？SHA-256、格式与架构是什么？ |
| `provider`、`analysis_profile` | 哪个工具和版本产生结果？使用了哪些分析设置？ |
| `operation`、`parameters` | 实际执行了什么查询？ |
| `raw_result` | 是否保留了区别于统一结果的 provider 原始表示？ |
| `normalized_result` | 当前操作的统一结果是什么？ |
| `confidence`、`authority` | 结果是观察、派生还是推断？证据来自哪里？ |
| `locations` | 能否回到文件偏移、代码地址或制品路径？ |
| `limitations`、`relationship_coverage` | 覆盖是否完整？哪些关系尚无法确定？ |
| `evidence_links` | 当前结果依赖哪些已有 Evidence？ |

其中 `confidence` 是 `observed`、`derived`、`inferred` 三种分类，用来区分结果的产生方式。它不是一个经过概率校准的正确率分数。[10]

例如，“某条汇编指令位于某地址”可以来自直接观察；“应用图中存在一条从入口模块到 IPC 包装器的路径”来自图的派生；“这个请求对应某个业务动作”则可能需要进一步推断。把三者分开，读者就知道下一步应该补哪类证据。

### Evidence ID 如何保护结果一致性

`evidence.ts` 对 Evidence 的语义投影计算摘要，得到 `ev_` 前缀的 ID。投影包含目标内容身份、provider、profile、参数、原始与统一结果、覆盖信息及依赖链接；目标名称和展示路径没有作为 subject 身份的一部分加入该投影。导入时会重新计算 ID，语义内容与 ID 不匹配则拒绝。[10]

`EvidenceLedger` 保存不可变记录，重复相同内容可幂等接收，冲突内容被拒绝；导入整个 bundle 时先验证，再提交变更，避免坏输入留下半份会话状态。[11]

这建立的是记录的内容完整性与可追溯性。判断本身是否正确，仍取决于分析器、输入和验证方法。持有完整记录的一方可以生成新的合法记录，因此摘要还需要与来源和信任边界一起解释。

### Unknown 也是正式数据

REA 为尚未解决的问题建立 residual unknown，记录问题、状态、支持与矛盾证据、所需证据类型、建议探针和修订关系。状态包括 `open`、`investigating`、`blocked`、`contradicted` 与 `resolved`。[12]

例如，已经识别到一个 IPC 包装器，但参数是动态字符串，暂时无法确定频道。系统可以保存“频道归属未解决”，而不是把它当成没有 IPC。后续源码范围、运行观察或独立验证能够接着处理这个问题。

## 四、JavaScript 与 Electron：从安装包恢复功能路径

对于 ASAR 或提取后的 JavaScript/Electron 目录，入口是 `analyze_javascript_application`。REA 组织模块、导入、source map、路由、IPC、原生 add-on 关联及配套语义关系，让 Agent 能围绕一个功能沿图追踪。[1][7]

Electron 特别适合解释这个流程。一个渲染页面中的按钮，可能经由 preload 的 `contextBridge` 进入 IPC，再由 main process 调用操作系统能力。只读页面代码，或者只找到一个字符串，往往还没有覆盖整个功能。

以“复制富文本”为调查问题，可以按以下顺序推进：

1. 分析提取后的应用目录或 ASAR，先取得 application graph。
2. 查看模块列表，定位 preload、renderer 与 main process。
3. 以 API、IPC channel、模块或字符串为 seed，追踪相关关系。
4. 检查具体的代码位置，区分明确字面量与动态参数。
5. 当图到达原生边界时，按目标 digest 和 export 转交合适的原生分析路径。
6. 为仍然动态或有歧义的关系补充运行观察或测试。[6][7]

### 静态解析能回答哪些问题

语义图覆盖词法定义与读取、属性读写、闭包捕获、可唯一解析的局部调用、参数和返回值关系，以及受支持的 Promise、事件、定时器、资源获取与释放等模式。[7]

图中一条 `resolved` 的调用关系，表示在当前静态分析范围内找到唯一目标。它是否在某次运行中执行，需要运行证据。动态属性、无法解析的调用、解析恢复和不支持的 API 形状继续保留 unknown；候选边默认不混入确定路径。[7]

这种保守处理对压缩后的 bundle 很重要：变量名很短、别名很多、包装器很深，图能提供调查起点，但一次静态遍历无法自动补齐所有动态行为。

### 运行证据如何与静态图对齐

`reconcile_javascript_runtime` 将静态 application graph 与浏览器、Electron 或 V8 Inspector 观察进行匹配。这个步骤消费已有 Evidence，自身不启动目标或读取新的应用文件。[13]

匹配时，捕获源码的 SHA-256 与位置共同提供较强的对应依据；唯一内容摘要也可建立匹配。只有路径对应时，结论保留为位置推断；多候选保留歧义；同路径字节不一致则报告 mismatch。路径名称不能覆盖摘要冲突。[13]

对 bundle 中的内嵌模块，系统还区分：

| 状态 | 可以如何理解 |
| --- | --- |
| `loaded` | 找到了被接受的运行脚本对应 |
| `resident-in-loaded-asset` | 模块所在的外层资源已被观察到，模块本身尚未独立证明执行 |
| `not-observed` | 当前观察窗口没有得到加载证据，需要结合覆盖范围解释 |

对齐解决的是“观察到的脚本与哪份静态制品相关”。要证明某段代码导致某个结果，仍需要相应的调用、输入输出或干预证据。[13]

## 五、原生二进制：沿着函数、指令和常量继续追踪

对于原生可执行文件或库，REA 通过所选 provider 提供函数分析、伪代码、汇编、字符串、名称、引用和调用关系。Agent 可以从一个特征字符串或已知地址入手，逐步定位调用者、被调用者和常量。[1][9]

三类深度分析 provider 有各自的连接方式：

| Provider | REA 如何连接 | 需要关注的条件 |
| --- | --- | --- |
| Hopper | 启动分析工具并通过内部 Python bridge 适配其 API | 安装、授权、宿主支持与可用操作 |
| Ghidra | Headless Java bridge，分析临时工作区中的目标 | import profile、语言与编译器设置、宿主能力 |
| IDA | 适配外部 IDA MCP，使用已连接 GUI 或拥有的 headless 数据库 | 上游 MCP profile、目标绑定与可用操作 |

这些路径来自源码架构和 provider 文档。工具提供的操作范围按 session 能力发现来判断，不能只因为某个 provider 名称存在就假定它支持所有工具。[3][4][9]

### 伪代码之后为什么还要读汇编

反编译器会推断类型、参数和表达式。调用约定、浮点指令或不完整的函数恢复，都可能影响伪代码表达。REA 保留汇编、字节读取和来源地址，使 Agent 能在伪代码之外继续核对原始行为。[14]

DX-Ball 的 sound-pan 案例正好展示了这条路径。公开记录中，函数伪代码没有完整恢复栈参数与 x87 表达式；调查进一步检查调用者、`FILD`、`FMUL`、`FSUB` 和常量位置，恢复出横坐标、比例与整数转换之间的计算关系。[14]

这里，“发现一个看起来合理的 C 函数”与“证明这段函数重建正确”是两个阶段。后者需要将原始执行结果或编译产物拿来比较。

## 六、让大结果适合 Agent：完整保存，按需读取

大型应用的 Evidence 可能远超模型一次需要阅读的内容。REA 当前 MCP 契约同时处理保存与传输：分析可以完整保留在 session 中，而首次回复返回 summary，后续用 `inspect_analysis_view` 读取某模块、函数 facet 或稳定分页。[4]

一个适合 Agent 的访问顺序是：

```text
目标分析，选择 summary
    → 取得 parent_evidence_id
    → 查看模块或函数清单
    → 选择具体对象及 facet
    → 沿功能路径追踪
    → 必要时导出完整 evidence bundle
```

这段顺序是对工作流的概括，实际参数以当前 server 广告的 schema 为准。[4][6]

代码中的 `ToolResultDelivery` 和 `encodeToolResult` 会计算统一结构和文本表示在 MCP 回复中的实际开销。默认预算围绕所固定 SDK 的 10 MiB 接收缓冲，并给外层消息留出空间。结果超过预算时，系统返回 `resource_constraint` 和已保存 Evidence 的续读入口，而不是把截断文本当作完整成功结果。[4][15]

这里要分清两种成本：回复字节数是一种传输指标，Agent 总 token 和任务耗时则包括工具发现、后续查询、错误、重试与模型处理。项目的公开效率记录也分别讨论这些指标，部分实测中回复缩小并没有对应总 token 或耗时下降。[16]

因此，summary/view 的直接价值是控制阅读范围、保留完整分析并支持继续查询。具体任务是否更省 token、更快，需要在相同问题和条件下测量。

## 七、从解释功能到重建实现：验证必须有明确对象

REA 的 reconstruction obligation ledger 将已有 Evidence 和人工审阅后的要求组织成待验证的重建义务。一个 required obligation 需要有明确实现归属、所需类型与测试案例，以及与原观察具有可比证据类型的 passing verifier。[17]

例如“取消中的请求恰好得到一个终态响应”，就不能靠看到一个 `cancel()` 函数来关闭。需要将请求、取消与最终响应的协议记录对应起来，再用验证结果证明候选实现满足这个要求。

矛盾、重复归属、缺失依赖、未解问题或无法获得的证据类型，会使义务保持开放或失败。输入没有 Evidence、没有义务时，ledger 返回 unknown，而不会因为清单为空就判定完成。[17]

这让重建任务从“代码已经写出来”变成一组可以逐项验收的主张。项目还提供 readiness conformance verifier 检查自己的制品、证据、比较、重放和清理规则；它属于仓库验证器，不是替用户自动证明任意应用完成重建的公开 MCP 工具。[18]

### 三个公开案例，三种不同证据范围

| 案例 | 公开记录支持的结果 | 适用范围 |
| --- | --- | --- |
| DX-Ball 声像计算 | 记录了 3,205 个原始 x86 差分案例，并复现该函数全部 63 个编译字节 | 一段声像计算函数及固定重建 checkpoint |
| Notion 剪贴板 bridge | 静态定位 preload 中暴露的 API 和动态 invoke；再结合包装器源码与独立探针解释路径 | 所选桌面制品、源码范围和独立缓存探针 |
| TH04 弹幕环计算 | REA 检查原始 16 位指令与字节；历史重建记录验证包含该 helper 的 2,139 字节范围 | 所选函数、周围贡献范围与固定历史编译工具链 |

上表引用的是项目公开案例记录，并区分 REA 检查、源码阅读和重建项目已有验证。函数、范围与夹具各有明确边界。[14][19][20]

Notion 案例尤其有参考价值：其精确 IPC channel seed 在静态图里没有匹配，公开说明保留了这处动态传播缺口，再用实际源码范围解释包装器关系。一个诚实暴露缺口的工具结果，仍然可以支持后续有效调查。[19]

## 八、能力范围与 Windows 使用条件

当前仓库文档覆盖原生、JavaScript/Electron、浏览器、.NET、Android、网络记录、固件、WASM、EVM 和 SQLite 等目标。这些能力对应不同 provider 与环境，以下按调查需要归组。[1][21]

| 需要回答的问题 | 相关能力 | 条件与解释范围 |
| --- | --- | --- |
| 安装包里有什么、功能在哪个模块 | 制品清单、ASAR、JS 应用图、静态 .NET | 可先从文件读取和静态分析开始 |
| 原生函数如何计算 | Hopper、Ghidra、IDA | 自备相应引擎；格式、架构和宿主支持分别确认 |
| 网页加载和请求如何关联 | CDP、浏览器脚本与网络观察 | 需要 Chrome 系浏览器，结论受观察窗口影响 |
| Android 包与设备行为 | JADX、ADB、Apktool、JEB 相关能力 | 每条路径需要自己的工具和连接条件 |
| 录制的数据表达什么 | HAR/SAZ、PCAP、逻辑信号、core 文件等 | 读取历史记录；额外工具与宿主限制各自不同 |
| 跨版本变化或重建是否成立 | 证据比较、义务 ledger、验证结果 | 明确成对输入、覆盖范围与比较依据 |

面向 Windows，最直接的起点通常是静态 JavaScript/Electron 或静态 .NET。深度原生分析则需要看具体 provider 的宿主边界。[1][22]

Ghidra 的 Windows P0 文档要求 Windows 10+ x64、固定本地 NTFS 目标与临时卷、兼容 Node.js、自备 Ghidra 12.1.x 及符合其要求的完整 JDK。支持对象是原生、非 managed 的 x86/x86-64 PE 应用或 DLL，提供 25 项只读分析操作；这条路径仍标为实验性，并有路径与文件系统准入约束。[22]

因此，拿到 `.exe` 或 `.dll` 后先判断内容类型很关键：普通原生 PE、托管 PE/CLI 和 NativeAOT 的分析入口不同。扩展名并不能代替格式识别。[5][22]

## 九、怎样开始一次可解释的调查

截至本次核对，GitHub 最新正式 release 与 npm `latest` 都为 `6.4.0`，GitHub 发布时间是 2026 年 10 月 11 日。本文所读 main commit 属于 10 月 13 日，部分 main 能力可能尚未进入该正式包；实际调用前要发现当前工具列表和 schema。[2][6][21]

下面的命令固定稳定包版本，供选择目标后使用。它们是根据文档整理的操作入口，本文的调查没有执行目标应用或更改 Agent 配置。[5][21]

先查看面向 Codex 的配置计划：

```powershell
npx rea-agents@6.4.0 setup --client codex --dry-run
```

希望接入 Agent 时，再运行交互式 setup，审阅它提出的路径与改动：

```powershell
npx rea-agents@6.4.0 setup --client codex
```

只做一次静态应用分析，可以直接走 CLI：

```powershell
npx -y rea-agents@6.4.0 analyze-javascript-application "D:/apps/example" --json
```

`D:/apps/example` 需要替换为选定的提取目录或 ASAR。大型结果应保存并投影为 summary 或局部 view；一次性 CLI 是独立进程，后续命令通过已保存的完整 Evidence 或 bundle 继续，不能沿用上一进程的内存引用。[5]

REA 要求 Node.js 22.x 至少 22.19、24.x 至少 24.11，或 26+；具体能力可能有更高要求。setup 负责选定客户端的配置与工作流文件，工具引擎和依赖按文档另行准备。[1][21]

一次有用的任务描述可以是：

```text
分析这个 Electron 应用的搜索功能。
从实际打包模块追到数据来源和排序逻辑。
给出来源位置，列清动态关系。
为可以重建的部分写验证案例，并保留尚未解释的行为。
```

任务的重点在于指定目标、功能和验收行为。REA 负责提供可查询的制品与运行证据，Agent 据此提出下一步调查，验证则对明确的输入和输出负责。

## 十、对 AI 工程的启发

REA 展示了一种适用于复杂 Agent 工具的设计：**工具返回能够继续调查的事实和来源，Agent 围绕这些证据推进任务。**

这套设计中最有复用价值的部分有四个：用格式和目标身份选择分析路径；让 CLI 与 MCP 共用工作流；将原结果、覆盖范围和 unknown 保存成结构化记录；用显式义务与测试界定“已经完成”的范围。[3][4][17]

它也提示了 Agent 产品的评估方式。工具数量、一次输出长度和生成代码的速度都只能描述局部。真正决定调查质量的，是能否沿来源回查、在动态关系处继续提问、保留矛盾，并用适当的验证关闭具体主张。

## 参考资料

以下源码与文档链接固定到本文分析 commit；release 与 npm 链接用于核对稳定版本。

1. [REA README：定位、能力与工作流](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/README.md)
2. [REA 6.4.0 正式 release](https://github.com/morluto/rea/releases/tag/rea-agents-6.4.0)；[npm 6.4.0 元数据](https://registry.npmjs.org/rea-agents/6.4.0)
3. [源码架构图](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/docs/architecture.mermaid)
4. [MCP runtime contracts](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/docs/mcp-contracts.md)
5. [CLI 与 Evidence 工作流](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/docs/cli.md)
6. [项目的逆向调查工作流说明](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/.agents/skills/reverse-engineer-anything/SKILL.md)
7. [JavaScript 跨层功能、语义与版本工作流](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/docs/javascript-application-workflows.md)
8. [MCP 功能追踪 handler](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/src/server/registerApplicationTools/traceFeature.ts)；[共享追踪工作流](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/src/application/javascript/JavaScriptApplicationWorkflowService.ts)
9. [AnalysisProviderRegistry](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/src/application/binary/AnalysisProviderRegistry.ts)
10. [Evidence schema 与语义身份](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/src/domain/evidence.ts)
11. [EvidenceLedger：记录与原子导入](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/src/application/investigation/EvidenceLedger.ts)
12. [Residual unknown 的数据结构](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/src/domain/residualUnknown.ts)
13. [JavaScript 静态与运行观察对齐](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/docs/javascript-runtime-reconciliation.md)
14. [DX-Ball sound-pan 的公开证据与验证范围](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/website/evidence/dx-ball-sound-pan.md)
15. [MCP 结果交付策略](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/src/server/toolResult.ts)；[实际传输体积编码](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/src/server/toolResultEncoding.ts)
16. [分析效率测量方法与结果解释](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/experiments/analysis-efficiency/README.md)
17. [Reconstruction obligation ledgers](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/docs/reconstruction-obligation-ledgers.md)
18. [Readiness conformance 验证范围](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/docs/reconstruction-readiness.md)
19. [Notion clipboard 的公开证据与动态缺口](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/website/evidence/notion-clipboard.md)
20. [TH04 bullet-ring 的公开证据与历史编译验证](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/website/evidence/th04-bullet-ring.md)
21. [安装、客户端配置与 main/release 边界](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/docs/installation.md)
22. [Windows Ghidra P0 的支持边界](https://github.com/morluto/rea/blob/d13c9f840d36a321a56917e75c48afcb8d72920e/docs/windows-ghidra-p0.md)
