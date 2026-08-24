# personal-site · 训练日志

一个为 GitHub Pages 定制的极简博客骨架。设计主题：「训练日志」——文章即 checkpoint，
签名元素是首页逐字打印的终端日志条，强调色取自 arXiv 标志红。

## 一、发布上线（10 分钟）

1. 在 GitHub 新建**空**仓库，二选一：
   - **用户主站（推荐）**：仓库名 `你的用户名.github.io` → 访问地址 `https://你的用户名.github.io`
   - **项目仓库**：任意名字（如 `blog`）→ 访问地址 `https://你的用户名.github.io/blog/`
2. 把本目录推上去：

```bash
cd /d/My_Project/personal-site
git remote add origin git@github.com:你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

3. 仓库 **Settings → Pages**：
   - 用户主站：Build and deployment → Source 选 `Deploy from a branch` → Branch 选 `main` / `(root)` → Save
4. 改 `_config.yml` 里所有 `YOUR_GITHUB_USERNAME`。
   若用了项目仓库模式，还需设置：`url: "https://你的用户名.github.io"`、`baseurl: "/仓库名"`
5. 等 1-2 分钟，GitHub Actions 会自动构建部署（Jekyll 是 Pages 原生支持，无需写任何 workflow）。

## 二、写文章（含分类）

### 栏目体系

站内预设四个栏目（导航栏可见），写文章时在 front matter 里选一个 `categories` 即自动归入：

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
categories: [技术实践]     # 四选一，见上表；可省略（只出现在"全部"页）
tags: [miniGPT, LoRA]      # 自由标签，副级分类
excerpt: 一句话摘要，会显示在首页列表里。
math: true                 # 只有需要公式的文章才加这行
---

正文直接写 Markdown……
```

- 文件名里的日期决定文章 URL：`:year/:month/:day/:title`
- 首页自动显示最新 8 篇（带栏目徽标）；各栏目页按类过滤；`/archive/` 按年份归档全部文章并生成标签索引
- RSS 已内置：`/feed.xml`

### 新增一个栏目（3 步）

1. 新建 `mysection.md`：
   ```markdown
   ---
   layout: section
   title: 我的栏目
   cat: 我的栏目          # 与文章 front matter 的 categories 值完全一致
   desc: 一句话说明这个栏目收什么。
   permalink: /my/
   ---
   ```
2. 在 `_includes/header.html` 导航里照抄一行 `<a href="{{ '/my/' | relative_url }}">我的</a>`
3. 完成——之后 `categories: [我的栏目]` 的文章都会自动归进去

## 三、传图片

1. 把图片放进 `assets/img/`（建议截图先压一压体积）
2. 文章里引用：`![说明文字](/assets/img/xxx.png)`

> 提示：也可以用图床外链，但自托管更稳、面试演示时不依赖第三方。

## 四、本地预览（三选一）

- **WSL2（推荐）**：
  ```bash
  sudo apt update && sudo apt install ruby-full build-essential zlib1g-dev
  bundle install          # 首次运行
  bundle exec jekyll serve --livereload
  ```
- **Docker**：`docker run --rm -p 4000:4000 -v "$PWD":/site jekyll/jekyll jekyll serve`
- **零安装**：直接 push，去 GitHub Pages 地址看效果（慢一点但省事）

预览地址：<http://localhost:4000>

## 五、想改设计

全部视觉参数集中在 `assets/css/main.css` 顶部的 `:root` 设计令牌（色板/字体/栏宽），
改几个变量整站换肤；首页打字日志的文案在 `assets/js/main.js` 的 `LINES` 数组。

## 六、后续可扩展

- 评论系统：[giscus](https://giscus.app/)（基于 GitHub Discussions）
- 全文搜索：fuse.js + posts.json 索引
- 友链页 / 项目展示页：复制 `about.md` 的结构即可
