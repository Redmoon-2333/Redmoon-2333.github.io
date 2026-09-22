# CLAUDE.md — personal-site（训练日志博客）

> 面向任何在此仓库工作的 coding agent。改代码前先读完本文件。

## 项目是什么

RedMoon 的个人博客「训练日志」，基于 **Jekyll 4 + GitHub Pages**，中文写作。
- 线上地址：https://redmoon-2333.github.io
- 仓库：https://github.com/Redmoon-2333/Redmoon-2333.github.io （用户主站模式，`baseurl` 必须为空）
- 部署：**push 到 main 即自动构建**，无 CI 配置文件，不要添加 workflow
- ⚠️ **站内所有署名一律用 RedMoon，严禁出现真实姓名**（页眉/页脚/关于页/文章均如此）

## 与学习体系的边界

- `08_MyNote/` 是当前学习笔记源；本仓库 `_posts/` 只保存经过公开化整理的副本。
- 不把博客文章反向写回 Note，不批量发布历史笔记；用户未明确指定内容时不新增公开文章。
- “整理/审阅文章”只生成或检查工作区内容；只有用户明确要求“发布/推送”才执行 commit、push 和线上验证。

## 文件地图

```
_config.yml          站点配置（title/url/分类无关的全局项）
index.html           首页：hero + 打字日志条 + 最近 8 篇
archive.html         全部文章（按年分组 + 标签索引）。⚠️ 是 .html 不是 .md，原因见「坑」
about.md             关于页（纯 Markdown，外层 div 带 markdown="1"）
papers.md            栏目页×4：papers/practice/courses/notes（只有 front matter，
practice.md          专题目录在 _layouts/section.html，文章列表在 _layouts/topic.html）
courses.md
notes.md
404.html
_layouts/            default(外壳) / post(文章) / section(栏目列表)
_includes/           head(SEO+字体+KaTeX条件加载) / header(导航) / footer(页脚)
assets/css/main.css  全部样式。设计令牌集中在顶部 :root，改肤只动那里
assets/js/main.js    打字日志条文案在 LINES 数组；滚动显现；年份填充
assets/img/          文章图片都放这里，文中用 /assets/img/xxx.png 绝对路径引用
_posts/              文章，命名强制 年-月-日-标题.md
Gemfile              实际依赖入口；构建前读取，不替换依赖文件
_data/topics.yml     四个板块与单值专题的统一配置
*/<topic>/index.html 显式专题页面，兼容 GitHub Pages，无自定义生成插件
```

## 发文规范

文件名 `YYYY-MM-DD-slug.md`，front matter：

```yaml
---
title: "标题"
date: 2026-09-01 10:00:00 +0800
categories: [论文精读]      # 四选一：论文精读/技术实践/课堂笔记/前沿见闻
topic: llm-foundations       # 必选单值，与大板块对应
# lesson_day: 1             # 课堂文章必填，正整数
tags: [自由标签]
excerpt: 一句话摘要（首页列表展示）
math: true                  # 仅需公式时加，加载 KaTeX
---
```

新增专题按 README.md 的专题流程；四个一级 categories 保持不变。

## 设计系统（勿破坏）

- 主题「训练日志」：冷白纸 `#FAFAF7` + 墨色 + **arXiv 红 `#B31B1B`** 唯一强调色（暗色模式自动反色，令牌在 main.css 的 prefers-color-scheme 块）
- 字体：Newsreader × Noto Serif SC（标题衬线）/ 系统黑体（正文）/ IBM Plex Mono（元数据）
- 签名元素：首页打字日志条（`#log-text`），动效全部尊重 `prefers-reduced-motion`
- 改动 CSS 后自查：小屏 560px 断点、键盘 focus-visible、对比度

## 已踩过的坑（不要重蹈）

1. **`.md` 文件里写 HTML 缩进 4 空格 = 被 kramdown 当代码块**（archive 页曾因此裸奔标签）。
   含 Liquid 循环/成块 HTML 的页面一律用 `.html` 后缀；`.md` 里只写纯 Markdown。
2. **块级 HTML（`<ul><li>` 等）内部的 Markdown 链接不会被解析**。要么全 Markdown，
   要么容器加 `markdown="1"`（kramdown 专属，about.md 在用）。
3. **GitHub Pages 白名单插件有限**：jekyll-feed/jekyll-seo-tag/paginate 可用；
   `jekyll-archives` 不可用——所以栏目页是手写 Liquid 过滤，别引入 archives。
4. Windows 本机 Ruby 缺 MSYS2 装不上 jekyll，别尝试 `gem install jekyll`，用 Docker。

## 构建验证（改动后必做）

使用 Docker 在隔离副本中按当前 Gemfile 执行 `bundle install` 与 `bundle exec jekyll build`，保留完整日志。不要替换 Gemfile、恢复用户删除的锁文件或提交构建产生的依赖文件。构建目录和缓存放在仓库之外。

构建后验证专题归属、计数、旧网址分流、RSS 和内部链接；Playwright 检查桌面与手机、明暗模式、键盘导航、公式、表格和长标题，并截图目检。

快速静态检查：Python 统计 `{% if %}/{% endif %}`、`{% for %}/{% endfor %}` 数量配对。

## Git 约定

- Conventional Commits（feat/fix/chore/docs），中文描述
- push 即上线，所以**每个 commit 都应是可发布状态**
- 本机 git 凭据管理器已存 GitHub 令牌（Redmoon-2333），push 无需交互

## 课程与专题硬规则（2026-09-22）

- 生成任何学习笔记前，清点并实际读取当天全部转写、摘要、PPTX、DOCX、PDF、代码、图片等；提取正文、备注、表格、公式，图像需目检。已使用、未使用、无法读取资料写入站点之外的内部整理记录，不只依赖自动摘要，不把资料验收表放进正文。
- 转写和原始课件用于核对课堂覆盖，摘要与旧稿用于辅助；材料冲突需说明，不把误听、简化或类比当作科学事实，分清课堂讲授、课件延伸与个人补充。
- 每门课每次课独立成文，不合篇。原课程 `DayN-阅读笔记.md` 与学习体系 `08_MyNote/课堂记录/<课程名>/YYYY-MM-DD-DayN.md`、博客三处知识一致，博客只做公开排版与隐私处理。
- 每篇必选唯一 categories 和标量 topic，二者必须匹配 `_data/topics.yml`；课程增加正整数 lesson_day，标题包含课程名与 Day。普通文章保留原日期及 URL；正文只在用户授权范围内修改。
- 已拆合篇退出 `_posts`，旧 URL 由静态分流页保留，不自动跳转，不进入首页、RSS 或文章计数。
- 用户审核且明确要求发布后才允许 commit、push；只 `git add <本次文件>`，禁止 git add -A。先核对 staged diff，不夹带其他工作区变更；整理不等于完成学习。

## 文章声口（2026-09-22）

- 不替作者添加未来规划、学习承诺、立 flag 或自我勉励式结尾；没有必要的收尾直接结束，不另补口号或升华。用户原文确有的计划不机械删除。
- 前言可保留 RedMoon 的第一人称与真实口吻；前言之外采用第三人称或无主语的客观记录，直接说明概念、过程、依据与结果，不刻意添加“我觉得”“我会”“对我来说”，也不机械改成“作者认为”。此约定替代全文第一视角要求，不编造经历或结果。
- qu-ai-wei、humanizer、anti-defensive-writing 作为编辑步骤，不在正文输出技能诊断或打磨报告。
- 资料清点、提取过程、检查结果、转写时间戳和学习状态声明留在内部记录。正文保留技术条件、引用、实验范围、材料冲突及必要的不确定性，去掉无关免责声明和夸张结论。
- 授权改稿时同步笔记源的对应段落，不以博客全文覆盖本地笔记，不修改原始材料与历史日志。
