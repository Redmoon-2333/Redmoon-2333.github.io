# CLAUDE.md — personal-site（训练日志博客）

> 面向任何在此仓库工作的 coding agent。改代码前先读完本文件。

## 项目是什么

RedMoon 的个人博客「训练日志」，基于 **Jekyll 4 + GitHub Pages**，中文写作。
- 线上地址：https://redmoon-2333.github.io
- 仓库：https://github.com/Redmoon-2333/Redmoon-2333.github.io （用户主站模式，`baseurl` 必须为空）
- 部署：**push 到 main 即自动构建**，无 CI 配置文件，不要添加 workflow
- ⚠️ **站内所有署名一律用 RedMoon，严禁出现真实姓名**（页眉/页脚/关于页/文章均如此）

## 文件地图

```
_config.yml          站点配置（title/url/分类无关的全局项）
index.html           首页：hero + 打字日志条 + 最近 8 篇
archive.html         全部文章（按年分组 + 标签索引）。⚠️ 是 .html 不是 .md，原因见「坑」
about.md             关于页（纯 Markdown，外层 div 带 markdown="1"）
papers.md            栏目页×4：papers/practice/courses/notes（只有 front matter，
practice.md          列表逻辑在 _layouts/section.html 里，按 page.cat 过滤 site.categories）
courses.md
notes.md
404.html
_layouts/            default(外壳) / post(文章) / section(栏目列表)
_includes/           head(SEO+字体+KaTeX条件加载) / header(导航) / footer(页脚)
assets/css/main.css  全部样式。设计令牌集中在顶部 :root，改肤只动那里
assets/js/main.js    打字日志条文案在 LINES 数组；滚动显现；年份填充
assets/img/          文章图片都放这里，文中用 /assets/img/xxx.png 绝对路径引用
_posts/              文章，命名强制 年-月-日-标题.md
Gemfile.prod         生产依赖（github-pages gem），仅文档用途
Gemfile              ⚠️ 平时被替换为本地测试用轻量版（jekyll+jekyll-feed+jekyll-seo-tag），
                     见「构建验证」。若发现 Gemfile 与 Gemfile.prod 内容互换，以 Gemfile.prod 为准恢复
```

## 发文规范

文件名 `YYYY-MM-DD-slug.md`，front matter：

```yaml
---
title: "标题"
date: 2026-09-01 10:00:00 +0800
categories: [论文精读]      # 四选一：论文精读/技术实践/课堂笔记/前沿见闻
tags: [自由标签]
excerpt: 一句话摘要（首页列表展示）
math: true                  # 仅需公式时加，加载 KaTeX
---
```

新增栏目的步骤在 README.md「新增一个栏目（3 步）」。

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

```bash
# 用 Docker 真实构建（Gemfile 已是轻量版，直接可用）
docker run --rm -v "//d/My_Project/personal-site://site" -w //site \
  jekyll/jekyll:4 bash -lc "bundle install --quiet && bundle exec jekyll build"
# 检查产物：_site/ 下对应页面无裸标签、Liquid 配对
# 测完恢复：rm -rf _site .jekyll-cache .bundle Gemfile.lock；若 Gemfile 被换过则 mv Gemfile.prod Gemfile
```

快速静态检查：Python 统计 `{% if %}/{% endif %}`、`{% for %}/{% endfor %}` 数量配对。

## Git 约定

- Conventional Commits（feat/fix/chore/docs），中文描述
- push 即上线，所以**每个 commit 都应是可发布状态**
- 本机 git 凭据管理器已存 GitHub 令牌（Redmoon-2333），push 无需交互
