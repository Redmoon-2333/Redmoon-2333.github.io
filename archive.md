---
layout: default
title: 全部文章
permalink: /archive/
---
<h1 class="page-title rise">{{ page.title }}</h1>

{% assign all_tags = site.posts | map: 'tags' | join: ',' | split: ',' | uniq | sort %}
{% if all_tags != empty %}
<nav class="tag-row mono rise" style="--d:.05s" aria-label="标签筛选">
  {% for t in all_tags %}
  <a href="#tag-{{ t | slugify }}">#{{ t }}</a>
  {% endfor %}
</nav>
{% endif %}

{% assign year_prev = '' %}
<ol class="post-list">
{% for post in site.posts %}
  {% assign y = post.date | date: '%Y' %}
  {% if y != year_prev %}
    </ol>
    <h2 class="year mono rise">{{ y }}</h2>
    <ol class="post-list">
    {% assign year_prev = y %}
  {% endif %}
  <li class="post-item reveal-on-scroll">
    <a href="{{ post.url | relative_url }}">
      <p class="eyebrow mono"><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: '%Y-%m-%d' }}</time>{% if post.tags != empty %} · {% for t in post.tags %}<span>#{{ t }}</span> {% endfor %}{% endif %}</p>
      <h3 class="post-item-title">{{ post.title }}</h3>
    </a>
  </li>
{% else %}
  <p class="empty mono">还没有文章。在 _posts/ 里放入你的第一篇 Markdown 即可。</p>
{% endfor %}
</ol>
