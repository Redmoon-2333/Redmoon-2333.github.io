# personal-site · 训练日志

一个为 GitHub Pages 定制的极简博客骨架。设计主题：「训练日志」——文章即 checkpoint，
签名元素是首页逐字打印的终端日志条，强调色取自 arXiv 标志红。

## 一、发布上线（10 分钟）

1. 在 GitHub 新建**空**仓库，二选一：
   - **用户主站（推荐）**：仓库名 `你的用户名.github.io` → 访问地址 `https://你的用户名.github.io`
   - **项目仓库**：任意名字（如 `blog`）→ 访问地址 `https://你的用户名.github.io/blog/`
2. 按下文发布验证规则完成相关轻量静态检查，由代理判断可发布，并取得用户明确发布/推送授权；仅暂存本次明确文件，检查 staged diff 后提交，再推送（现有站点无需重复初始化远程）：

```bash
cd /d/大学相关/03_个人成长与记录/LLM学习体系/02_博客站点/personal-site
git remote add origin git@github.com:你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

3. 仓库 **Settings → Pages**：
   - 用户主站：Build and deployment → Source 选 `Deploy from a branch` → Branch 选 `main` / `(root)` → Save
4. 改 `_config.yml` 里所有 `YOUR_GITHUB_USERNAME`。
   若用了项目仓库模式，还需设置：`url: "https://你的用户名.github.io"`、`baseurl: "/仓库名"`
5. 等待本次提交对应的 GitHub Pages 自动构建部署成功（不能仅等待固定时长或看到旧页面 200 就视为成功；Jekyll 是 Pages 原生支持，不添加任何 workflow）。随后抓取本次受影响线上页面，核对 HTTP 状态与预期内容，并用真实浏览器截图目检；相关时覆盖桌面与手机。失败及时修复或只回滚本次发布，不涉及他人变更。

## 二、写文章（含分类）

### 栏目体系

站内使用“大板块 → 独立小专题页 → 文章”。每篇必须选择一个 `categories` 和一个标量 `topic`，二者须匹配 `_data/topics.yml`；标签只表达交叉知识点。

| 栏目 | categories 值 | 页面 | 收什么 |
|---|---|---|---|
| 论文 | `论文精读` | `/papers/` | 论文解读系列 |
| 实践 | `技术实践` | `/practice/` | 训练/微调/部署实验记录 |
| 课堂 | `课堂笔记` | `/courses/` | 平时课堂的摘要与思考 |
| 见闻 | `前沿见闻` | `/notes/` | 新模型、新工具、AI 课外学习笔记 |

### 发文模板

在 `_posts/` 里新建文件，命名必须是 `年-月-日-标题.md`：

```markdown
---
title: "我的第一篇：8GB 显存训 LLM"
date: 2026-09-01 10:00:00 +0800
categories: [技术实践]     # 必选一个，不可省略
topic: karpathy            # 必选单值，必须属于所选大板块
# lesson_day: 1           # 课堂文章必填正整数
tags: [miniGPT, LoRA]      # 交叉知识点，不替代专题
excerpt: 一句话摘要，会显示在首页列表里。
math: true                 # 只有需要公式的文章才加这行
---

正文直接写 Markdown……
```

- 文件名里的日期决定文章 URL：`:year/:month/:day/:title`
- 首页显示最新 8 篇及专题；大板块展示专题和文章数，小专题按日期倒序列文，课堂同时显示 Day；归档显示专题，文章含面包屑和返回专题入口。
- RSS 已内置：`/feed.xml`

### 新增专题（一级栏目保持不变）

1. 在 `_data/topics.yml` 的 `items` 中增加唯一英文 `id`、中文 `title`、所属 `section` 与稳定 `url`；然后建立 `<section>/<slug>/index.html`：
   ```markdown
   ---
   layout: topic
   title: 新专题
   topic: new-topic
   permalink: /practice/new-topic/
   ---
   ```
2. 文章写对应 `topic` 与唯一大板块。主导航与专题目录自动读取配置，不增加自定义生成插件。
3. 验证归属、日期排序、数量、空状态和链接；课程标题包含课程名及 Day 序号。

首批专题：课堂为人工智能导论、智能计算系统、人机交互技术、算法设计与分析；实践为 Karpathy 从零构建大模型、LangGraph；论文为大模型基础与经典论文、多智能体强化学习；见闻为模型动态、工具与生态、行业观察。

### 课程资料与审核

不替作者添加未来规划、学习承诺、立 flag 或自我勉励式结尾。正文可以自然结束，不必补一句“接下来我要……”或“做到这些才算学会”。用户原文确有的计划保留。

清点并读取当天全部转写、摘要、PPTX、DOCX、PDF、代码和图片；提取正文、备注、表格、公式，图像需目检。已使用、未使用和无法读取资料写入站点之外的内部整理记录，不放进公开正文。原始课件与转写用于判断覆盖，摘要和旧稿用于辅助，不把误听或课堂类比当科学事实；影响结论的冲突仍需说明，区分课堂讲授、课件延伸和个人补充。

前言可保留 RedMoon 的第一人称与真实口吻；前言之外采用第三人称或无主语的客观记录，直接说明概念、过程、依据与结果。不刻意添加“我觉得”“我会”“对我来说”，也不机械改成“作者认为”。不写文件读取、编辑验收、学习状态等过程汇报。使用 qu-ai-wei、humanizer 和 anti-defensive-writing 去掉模板腔、空泛升华与无关免责声明；不编造经历、删改实质证据或隐藏必要条件。正常参考文献与配套代码入口保留。授权改稿时同步本地笔记的对应段落，不以公开稿覆盖本地独有内容。

每门课独立成文，不再跨课合篇。原课程 `DayN-阅读笔记.md` 与 `08_MyNote/课堂记录/<课程名>/YYYY-MM-DD-DayN.md` 保持知识正文一致；博客仅作公开排版和隐私处理。旧合篇 URL 保留静态分流页，不进入文章集合、首页、RSS 或计数。

发布前做与本次改动相关的 diff、front matter、模板、链接与资源路径等轻量静态检查，由代理判断可发布；不强制本地 Docker、`bundle install` 或 `jekyll build`。只有用户审核且明确要求“发布/推送”后才 commit、push，只暂存本次明确文件并核对 staged diff，禁止 `git add -A`，不夹带他人或已有暂存变更。推送后等待本次 GitHub Pages 部署成功，抓取受影响线上页面核对 HTTP 状态与预期内容，再用真实浏览器截图目检（涉及响应式布局时覆盖桌面与手机）。失败及时修复或只回滚本次发布，不涉及他人变更。整理不代表完成学习。

## 三、传图片

1. 把图片放进 `assets/img/`（建议截图先压一压体积）
2. 文章里引用：`![说明文字](/assets/img/xxx.png)`

> 提示：也可以用图床外链，但自托管更稳、面试演示时不依赖第三方。

## 四、可选本地排错（不阻塞常规发布）

本地 RubyGems 依赖安装与 Jekyll 构建不是发布门槛；Docker 仅供疑难排错时可选使用。常规发布按前述轻量静态检查、明确授权和部署后线上验收执行。

- **WSL2（可选排错）**：
  ```bash
  sudo apt update && sudo apt install ruby-full build-essential zlib1g-dev
  bundle install          # 首次运行
  bundle exec jekyll serve --livereload
  ```
- **Docker**：启动 Docker Desktop，将站点复制到仓库外的隔离构建目录，按当前 Gemfile 执行 `bundle install`、`bundle exec jekyll build`，保留完整日志；静态产物可用本地 HTTP 服务预览。不要替换或提交无关依赖文件。
- 不要求先本地预览再发布；代理完成相关轻量静态检查并判断可发布、且取得用户明确授权后，可直接推送。部署成功后必须在真实浏览器中对受影响线上页面截图目检；桌面/手机、明暗模式、键盘导航、长标题、表格与公式按改动关联检查。

WSL2 可选预览地址：<http://localhost:4000>

## 五、想改设计

全部视觉参数集中在 `assets/css/main.css` 顶部的 `:root` 设计令牌（色板/字体/栏宽），
改几个变量整站换肤；首页打字日志的文案在 `assets/js/main.js` 的 `LINES` 数组。

## 六、后续可扩展

- 评论系统：[giscus](https://giscus.app/)（基于 GitHub Discussions）
- 全文搜索：fuse.js + posts.json 索引
- 友链页 / 项目展示页：复制 `about.md` 的结构即可
